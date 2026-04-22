/**
 * NASA FIRMS area-CSV client.
 *
 * Single function: `fetchAreaCsv` — pure HTTP + CSV parse, no DB writes. The
 * matcher (`lib/firms/matcher.ts`) owns persistence.
 *
 * URL form (per https://firms.modaps.eosdis.nasa.gov/api/area/):
 *   https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/<source>/<bbox>/<dayRange>
 *
 * Where:
 *   - <source>     ∈ { VIIRS_NOAA20_NRT, VIIRS_SNPP_NRT, MODIS_NRT, ... }
 *   - <bbox>       = "minLon,minLat,maxLon,maxLat" (degrees, lon/lat order)
 *   - <dayRange>   = "1".."10" (number of days back from today, UTC)
 *
 * Build-without-blocking: `FIRMS_MAP_KEY` is read lazily from `process.env`.
 * If unset we return `{ ok: false, code: "config_missing" }` rather than
 * throwing — the route handler maps that to a typed 503.
 *
 * Rate limiting: NASA documents 5,000 transactions per 10 minutes per key.
 * We layer two safeguards on top:
 *   - in-process token bucket (6 req/min) so a single Vercel instance never
 *     exceeds a sane fraction of the global cap
 *   - exponential backoff with jitter on 429 / 5xx (3 attempts, max ~4 s)
 *
 * The token bucket lives at module scope. Vercel Fluid Compute keeps a warm
 * isolate alive across invocations, so this is meaningful within a single
 * region; multiple regions / cold starts each get their own bucket — that's
 * acceptable because we also call FIRMS at most once per (bucket, source)
 * per cron tick.
 */

export type FirmsSource =
  | "VIIRS_NOAA20_NRT"
  | "VIIRS_SNPP_NRT"
  | "MODIS_NRT";

export type FirmsBbox = readonly [number, number, number, number]; // [minLon, minLat, maxLon, maxLat]

export type FirmsDetection = {
  latitude: number;
  longitude: number;
  brightTi4: number | null;
  brightTi5: number | null;
  scan: number | null;
  track: number | null;
  acqDate: string;       // "YYYY-MM-DD"
  acqTime: string;       // "HHMM"
  satellite: string;
  instrument: string;
  confidence: string | null;
  version: string | null;
  frp: number | null;
  daynight: string | null;
};

export type FirmsFetchOk = {
  ok: true;
  source: FirmsSource;
  bbox: FirmsBbox;
  dayRange: number;
  detections: FirmsDetection[];
  /** True if FIRMS returned the empty-area sentinel rather than rows. */
  emptyArea: boolean;
};

export type FirmsFetchErrCode =
  | "config_missing"
  | "rate_limited"
  | "upstream_error"
  | "network_error"
  | "parse_error"
  | "throttled_local";

export type FirmsFetchErr = {
  ok: false;
  code: FirmsFetchErrCode;
  message: string;
  status?: number;
};

export type FirmsFetchResult = FirmsFetchOk | FirmsFetchErr;

const FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv";

// ---------------------------------------------------------------------------
// In-process token bucket
//
// 6 calls per minute, refilled continuously. A single Vercel instance running
// 1-bucket polls every 15 min uses far less than this; the bucket is the
// safety floor against runaway fan-out (e.g. if `getActiveBuckets` ever grows
// past 50 and we fan out without batching).

const BUCKET_CAPACITY = 6;
const BUCKET_REFILL_PER_MS = 6 / 60_000;

let tokens = BUCKET_CAPACITY;
let lastRefill = Date.now();

function tryConsumeToken(now: number): boolean {
  const elapsed = now - lastRefill;
  if (elapsed > 0) {
    tokens = Math.min(BUCKET_CAPACITY, tokens + elapsed * BUCKET_REFILL_PER_MS);
    lastRefill = now;
  }
  if (tokens >= 1) {
    tokens -= 1;
    return true;
  }
  return false;
}

/** Test-only: reset the bucket between cases. */
export function _resetTokenBucket(): void {
  tokens = BUCKET_CAPACITY;
  lastRefill = Date.now();
}

// ---------------------------------------------------------------------------
// Public API

export type FetchAreaCsvArgs = {
  source: FirmsSource;
  bbox: FirmsBbox;
  /** 1..10 days; FIRMS hard cap. Default: 1 (latest 24h). */
  dayRange?: number;
  /** Override fetch (tests). */
  fetchImpl?: typeof fetch;
  /** Override sleep (tests). */
  sleepMs?: (ms: number) => Promise<void>;
  /** Override `now` for deterministic backoff in tests. */
  now?: () => number;
};

const DEFAULT_DAY_RANGE = 1;
const MAX_DAY_RANGE = 10;

export async function fetchAreaCsv(
  args: FetchAreaCsvArgs,
): Promise<FirmsFetchResult> {
  const dayRange = args.dayRange ?? DEFAULT_DAY_RANGE;
  if (dayRange < 1 || dayRange > MAX_DAY_RANGE) {
    return {
      ok: false,
      code: "parse_error",
      message: `dayRange must be in [1, ${MAX_DAY_RANGE}]; got ${dayRange}`,
    };
  }

  const key = process.env.FIRMS_MAP_KEY;
  if (!key) {
    return {
      ok: false,
      code: "config_missing",
      message: "FIRMS_MAP_KEY is not set (build-without-blocking pattern).",
    };
  }

  const now = args.now ?? Date.now;
  if (!tryConsumeToken(now())) {
    return {
      ok: false,
      code: "throttled_local",
      message: `FIRMS local token bucket exhausted (${BUCKET_CAPACITY}/min cap)`,
    };
  }

  const url = `${FIRMS_BASE}/${encodeURIComponent(key)}/${args.source}/${args.bbox.join(",")}/${dayRange}`;
  const fetchFn = args.fetchImpl ?? fetch;
  const sleep = args.sleepMs ?? defaultSleep;

  let lastErr: FirmsFetchErr | null = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt > 0) {
      const baseDelay = 500 * Math.pow(2, attempt - 1);
      const jitter = Math.random() * 250;
      await sleep(Math.min(4_000, baseDelay + jitter));
    }
    let response: Response;
    try {
      response = await fetchFn(url, {
        headers: { accept: "text/csv" },
        // Hobby function timeout is 60s; FIRMS rarely takes more than 5s.
        // We don't set an explicit signal — Vercel will abort the function
        // first if necessary.
      });
    } catch (err) {
      lastErr = {
        ok: false,
        code: "network_error",
        message: err instanceof Error ? err.message : String(err),
      };
      continue;
    }
    if (response.status === 429) {
      lastErr = {
        ok: false,
        code: "rate_limited",
        status: 429,
        message: "FIRMS returned 429 Too Many Requests",
      };
      continue;
    }
    if (response.status >= 500) {
      lastErr = {
        ok: false,
        code: "upstream_error",
        status: response.status,
        message: `FIRMS upstream ${response.status}`,
      };
      continue;
    }
    if (!response.ok) {
      const text = await safeText(response);
      return {
        ok: false,
        code: "upstream_error",
        status: response.status,
        message: `FIRMS responded ${response.status}: ${text.slice(0, 200)}`,
      };
    }
    const csv = await safeText(response);
    return parseFirmsCsv(csv, args.source, args.bbox, dayRange);
  }
  return lastErr ?? {
    ok: false,
    code: "upstream_error",
    message: "FIRMS request failed after 3 attempts",
  };
}

async function safeText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return "";
  }
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// CSV parser
//
// FIRMS area-CSV format (header row, comma-separated):
//   latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,
//   instrument,confidence,version,bright_ti5,frp,daynight
//
// MODIS uses `brightness` and `bright_t31` instead of bright_ti4/ti5; we map
// both into the canonical `brightTi4` / `brightTi5` slots so downstream code
// is uniform.
//
// Empty-area responses are returned as a single line:
//   "No fire data for the requested area or date range"
// (Or sometimes just the CSV header with no rows.)

export function parseFirmsCsv(
  csv: string,
  source: FirmsSource,
  bbox: FirmsBbox,
  dayRange: number,
): FirmsFetchResult {
  const trimmed = csv.trim();
  if (trimmed.length === 0) {
    return {
      ok: true,
      source,
      bbox,
      dayRange,
      detections: [],
      emptyArea: true,
    };
  }
  // Empty-area sentinels NASA returns vary; both contain "No fire data".
  if (/no fire data/i.test(trimmed) && !trimmed.includes(",")) {
    return {
      ok: true,
      source,
      bbox,
      dayRange,
      detections: [],
      emptyArea: true,
    };
  }
  const lines = trimmed.split(/\r?\n/);
  if (lines.length < 1) {
    return {
      ok: false,
      code: "parse_error",
      message: "FIRMS CSV had no header row",
    };
  }
  const header = lines[0].split(",").map((c) => c.trim().toLowerCase());
  const idx = (name: string): number => header.indexOf(name);
  const latIdx = idx("latitude");
  const lonIdx = idx("longitude");
  if (latIdx < 0 || lonIdx < 0) {
    return {
      ok: false,
      code: "parse_error",
      message: `FIRMS CSV missing latitude/longitude columns; header=${header.join(",")}`,
    };
  }
  const ti4Idx = idx("bright_ti4");
  const ti5Idx = idx("bright_ti5");
  const brightnessIdx = idx("brightness");      // MODIS
  const brightT31Idx = idx("bright_t31");       // MODIS
  const scanIdx = idx("scan");
  const trackIdx = idx("track");
  const acqDateIdx = idx("acq_date");
  const acqTimeIdx = idx("acq_time");
  const satIdx = idx("satellite");
  const instrIdx = idx("instrument");
  const confIdx = idx("confidence");
  const versionIdx = idx("version");
  const frpIdx = idx("frp");
  const daynightIdx = idx("daynight");

  const detections: FirmsDetection[] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().length === 0) continue;
    const cells = line.split(",");
    const lat = num(cells[latIdx]);
    const lon = num(cells[lonIdx]);
    if (lat == null || lon == null) {
      // Malformed row — skip rather than fail the whole batch.
      continue;
    }
    detections.push({
      latitude: lat,
      longitude: lon,
      brightTi4: ti4Idx >= 0 ? num(cells[ti4Idx]) : num(cells[brightnessIdx]),
      brightTi5: ti5Idx >= 0 ? num(cells[ti5Idx]) : num(cells[brightT31Idx]),
      scan: scanIdx >= 0 ? num(cells[scanIdx]) : null,
      track: trackIdx >= 0 ? num(cells[trackIdx]) : null,
      acqDate: acqDateIdx >= 0 ? cells[acqDateIdx]?.trim() ?? "" : "",
      acqTime: acqTimeIdx >= 0 ? cells[acqTimeIdx]?.trim() ?? "" : "",
      satellite: satIdx >= 0 ? cells[satIdx]?.trim() ?? "" : "",
      instrument: instrIdx >= 0 ? cells[instrIdx]?.trim() ?? "" : "",
      confidence: confIdx >= 0 ? cells[confIdx]?.trim() ?? null : null,
      version: versionIdx >= 0 ? cells[versionIdx]?.trim() ?? null : null,
      frp: frpIdx >= 0 ? num(cells[frpIdx]) : null,
      daynight: daynightIdx >= 0 ? cells[daynightIdx]?.trim() ?? null : null,
    });
  }
  return {
    ok: true,
    source,
    bbox,
    dayRange,
    detections,
    emptyArea: detections.length === 0,
  };
}

function num(cell: string | undefined): number | null {
  if (cell === undefined) return null;
  const t = cell.trim();
  if (t === "") return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}
