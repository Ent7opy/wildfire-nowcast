/**
 * /api/aoi/poll — cron entry point.
 *
 * Auth: shared-secret bearer token. Only the GitHub Actions cron (and Vanyo
 * during local debugging) should be calling this.
 *
 * Body (optional):
 *   { "bucket"?: "5x5:W125_N35" }   // poll one specific bucket
 *
 * Behaviour:
 *   - Without `bucket`: enumerate all active buckets, poll each in turn.
 *   - With `bucket`: poll only that bucket.
 *
 * Build-without-blocking:
 *   - `CRON_SECRET` unset → 503 service_unavailable
 *   - `FIRMS_MAP_KEY` unset → 503 (FIRMS client returns config_missing on the
 *     first call; the route surfaces that as 503)
 *   - `DATABASE_URL` unset → 503 via the existing `withDb` helper
 *
 * Per-bucket failures do not abort the whole poll — they are recorded as
 * `partial`/`error` rows in `job_runs`.
 *
 * Runtime: Node.js (default). The `pg` driver and `node:crypto` (used by the
 * dedupe hash) are not Edge-compatible.
 */
import { NextResponse, type NextRequest } from "next/server";
import { z } from "zod";
import { sql } from "drizzle-orm";
import { tryGetDb, type AppDb } from "@/lib/db/client";
import { decodeRows } from "@/lib/db/decode-rows";
import { apiError, dbUnavailableResponse } from "@/lib/api/errors";
import {
  fetchAreaCsv,
  type FirmsBbox,
  type FirmsFetchResult,
  type FirmsSource,
} from "@/lib/firms/client";
import { bucketToBbox, getActiveBuckets } from "@/lib/firms/buckets";
import { matchDetectionsToAois } from "@/lib/firms/matcher";
import { generateBriefForEvent, type GenerateOutcome } from "@/lib/ai/generate";
import { dispatchBrief, type DispatchOutcome } from "@/lib/notify/dispatch";
import { pruneOldDetections } from "@/lib/firms/prune";
import {
  firmsErrorToOutcome,
  runStatusToOutcome,
  type FreshnessOutcome,
} from "@/lib/firms/freshness";

// Test-only injection point: lets the integration suite bypass the live
// FIRMS call without introducing a DI framework. Production leaves it null.
type FirmsFetchFn = (args: {
  source: FirmsSource;
  bbox: FirmsBbox;
  dayRange?: number;
}) => Promise<FirmsFetchResult>;
let testFirmsFetch: FirmsFetchFn | null = null;
export function _setTestFirmsFetch(fn: FirmsFetchFn | null): void {
  testFirmsFetch = fn;
}

// Test-only injection point for the brief generator. Production calls the real
// orchestrator which dials the AI Gateway.
type BriefGenFn = typeof generateBriefForEvent;
let testBriefGen: BriefGenFn | null = null;
export function _setTestBriefGen(fn: BriefGenFn | null): void {
  testBriefGen = fn;
}

// Test-only injection point for the notification dispatcher. Production calls
// the real `dispatchBrief` which dials Resend.
type NotifyDispatchFn = typeof dispatchBrief;
let testNotifyDispatch: NotifyDispatchFn | null = null;
export function _setTestNotifyDispatch(fn: NotifyDispatchFn | null): void {
  testNotifyDispatch = fn;
}

// Test-only re-export — keeps closeJobRun module-private for runtime callers
// while letting unit tests exercise the COALESCE guards directly.
export const _closeJobRunForTest = (
  db: AppDb,
  id: string,
  args: CloseArgs,
): Promise<void> => closeJobRun(db, id, args);

// Hobby max function duration is 60s; we configure to that ceiling so a slow
// FIRMS endpoint doesn't truncate mid-bucket.
export const maxDuration = 60;
// Force the Node runtime (default for App Router) — explicit for clarity.
export const runtime = "nodejs";

const DEFAULT_SOURCE: FirmsSource = "VIIRS_NOAA20_NRT";

const bodySchema = z
  .object({
    bucket: z.string().regex(/^5x5:[EW]\d{3}_[NS]\d{2}$/).optional(),
    source: z
      .enum(["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"])
      .optional(),
  })
  .strict();

type PollBody = z.infer<typeof bodySchema>;

type BucketRunOutcome = {
  bucket: string;
  source: FirmsSource;
  detectionsParsed: number;
  detectionsInserted: number;
  detectionsSkippedIndustrial: number;
  eventsCreated: number;
  eventsUpdated: number;
  /** Stage 3: number of `aoi_briefs` rows generated for events created in this bucket. */
  briefsGenerated: number;
  /**
   * Per-event skip reasons keyed by event id, e.g. `"paused"`, `"already_briefed"`,
   * `"prior_absence"` (pass), `"config_missing"`, etc. Empty when every event
   * either generated a brief or there were no events.
   */
  briefSkipReason: Record<string, string>;
  /** Stage 4: count of `notifications_log` rows with status='sent' from this bucket. */
  notificationsSent: number;
  notificationsFailed: number;
  notificationsSkipped: number;
  /** True if any dispatch returned `config_missing` (RESEND_API_KEY unset). */
  notificationConfigMissing: boolean;
  durationMs: number;
  status: "ok" | "partial" | "error";
  error?: string;
};

export async function POST(req: NextRequest): Promise<NextResponse> {
  const expected = process.env.CRON_SECRET;
  if (!expected) {
    return apiError(
      "service_unavailable",
      "CRON_SECRET is not configured; this is expected during pre-secret setup",
    );
  }

  const auth = req.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) {
    return apiError(
      "validation_failed",
      "Missing Bearer authorization header",
    );
  }
  const token = auth.slice("Bearer ".length).trim();
  if (!constantTimeEquals(token, expected)) {
    return NextResponse.json(
      {
        error: {
          code: "validation_failed" as const,
          message: "Invalid bearer token",
        },
      },
      { status: 401 },
    );
  }

  // Parse body (allow empty body for "poll all").
  let body: PollBody = {};
  try {
    const text = await req.text();
    if (text.trim().length > 0) {
      const parsed = bodySchema.safeParse(JSON.parse(text));
      if (!parsed.success) {
        return apiError(
          "validation_failed",
          "Invalid poll body",
          parsed.error.issues,
        );
      }
      body = parsed.data;
    }
  } catch {
    return apiError("validation_failed", "Body must be valid JSON or empty");
  }

  if (!process.env.FIRMS_MAP_KEY) {
    return apiError(
      "service_unavailable",
      "FIRMS_MAP_KEY is not configured; this is expected during pre-secret setup",
    );
  }

  const db = tryGetDb();
  if (!db) return dbUnavailableResponse();

  const source = body.source ?? DEFAULT_SOURCE;
  const totalStart = Date.now();

  // Parent job_run row.
  const parentRunId = await openJobRun(db, "firms-poll", null, totalStart);

  // Stage 7: 14-day retention sweep. Runs once per poll, before fan-out, so a
  // failure here surfaces quickly and counts land on the parent run row. The
  // matcher only reads `inserted_at >= pollStart` and the brief generator only
  // reads detections inside the event's first/last seen window, both well
  // inside 14 days — see `lib/firms/prune.ts`.
  let detectionsPruned = 0;
  try {
    detectionsPruned = await pruneOldDetections(db);
    if (detectionsPruned > 0) {
      console.info(`[poll] retention sweep removed ${detectionsPruned} firms_detections rows`);
    }
  } catch (err) {
    console.warn(`[poll] retention sweep failed: ${errMessage(err)}`);
  }

  let buckets: string[];
  try {
    if (body.bucket) {
      buckets = [body.bucket];
    } else {
      const active = await getActiveBuckets(db);
      buckets = active.map((b) => b.bucket);
    }
  } catch (err) {
    await closeJobRun(db, parentRunId, {
      status: "error",
      error: errMessage(err),
      finishedAt: new Date(),
    });
    return apiError("internal_error", "Failed to enumerate active buckets");
  }

  const runs: BucketRunOutcome[] = [];
  let totalDetectionsInserted = 0;
  let totalEventsCreated = 0;
  let totalFirmsCalls = 0;
  let totalBriefsGenerated = 0;
  let totalNotificationsSent = 0;
  let anyConfigMissing = false;

  for (const bucket of buckets) {
    const outcome = await runOneBucket(db, bucket, source);
    runs.push(outcome);
    totalDetectionsInserted += outcome.detectionsInserted;
    totalEventsCreated += outcome.eventsCreated;
    totalBriefsGenerated += outcome.briefsGenerated;
    totalNotificationsSent += outcome.notificationsSent;
    if (outcome.notificationConfigMissing) anyConfigMissing = true;
    totalFirmsCalls += 1;
  }

  if (anyConfigMissing) {
    console.warn(
      "[poll] RESEND_API_KEY unset; brief generation succeeded but no emails dispatched.",
    );
  }

  const partial = runs.some((r) => r.status !== "ok");
  await closeJobRun(db, parentRunId, {
    status: runs.length === 0
      ? "ok"
      : partial
        ? "partial"
        : "ok",
    error: null,
    firmsRequestCount: totalFirmsCalls,
    detectionsInserted: totalDetectionsInserted,
    eventsCreated: totalEventsCreated,
    briefsGenerated: totalBriefsGenerated,
    notificationsSent: totalNotificationsSent,
    detectionsPruned,
    finishedAt: new Date(),
  });

  return NextResponse.json({
    runs,
    totalDurationMs: Date.now() - totalStart,
    source,
    bucketCount: buckets.length,
    totalBriefsGenerated,
    totalNotificationsSent,
    detectionsPruned,
  });
}

async function runOneBucket(
  db: AppDb,
  bucket: string,
  source: FirmsSource,
): Promise<BucketRunOutcome> {
  const start = Date.now();
  const childRunId = await openJobRun(db, "firms-poll", bucket, start);
  try {
    const bbox = bucketToBbox(bucket);
    const fetchResult: FirmsFetchResult = await (testFirmsFetch ?? fetchAreaCsv)({
      source,
      bbox,
      dayRange: 1,
    });
    if (!fetchResult.ok) {
      const mapped = firmsErrorToOutcome(fetchResult.code);
      await closeJobRun(db, childRunId, {
        status: "error",
        error: `${fetchResult.code}: ${fetchResult.message}`,
        firmsRequestCount: 1,
        outcome: mapped.outcome,
        retryPending: mapped.retryPending,
        finishedAt: new Date(),
      });
      return {
        bucket,
        source,
        detectionsParsed: 0,
        detectionsInserted: 0,
        detectionsSkippedIndustrial: 0,
        eventsCreated: 0,
        eventsUpdated: 0,
        briefsGenerated: 0,
        briefSkipReason: {},
        notificationsSent: 0,
        notificationsFailed: 0,
        notificationsSkipped: 0,
        notificationConfigMissing: false,
        durationMs: Date.now() - start,
        status: "error",
        error: `${fetchResult.code}: ${fetchResult.message}`,
      };
    }

    const matchOutcome = await matchDetectionsToAois(db, {
      bucket,
      source,
      detections: fetchResult.detections,
    });

    let briefsGenerated = 0;
    const briefSkipReason: Record<string, string> = {};
    let briefError: string | null = null;
    const generatedBriefIds: string[] = [];
    const briefGen = testBriefGen ?? generateBriefForEvent;
    for (const eventId of matchOutcome.createdEventIds) {
      let outcome: GenerateOutcome;
      try {
        outcome = await briefGen(db, eventId);
      } catch (err) {
        outcome = {
          status: "error",
          eventId,
          reason: errMessage(err),
        };
      }
      if (outcome.status === "generated") {
        briefsGenerated += 1;
        if (outcome.briefId) generatedBriefIds.push(outcome.briefId);
      } else if (outcome.status === "skipped") {
        briefSkipReason[eventId] = outcome.reason;
      } else {
        briefSkipReason[eventId] = `error: ${outcome.reason}`;
        briefError = outcome.reason;
      }
    }

    let notificationsSent = 0;
    let notificationsFailed = 0;
    let notificationsSkipped = 0;
    let notificationConfigMissing = false;
    const dispatcher = testNotifyDispatch ?? dispatchBrief;
    for (const briefId of generatedBriefIds) {
      let outcome: DispatchOutcome;
      try {
        outcome = await dispatcher(db, briefId);
      } catch (err) {
        notificationsFailed += 1;
        briefError = briefError ?? errMessage(err);
        continue;
      }
      for (const a of outcome.attempts) {
        if (a.status === "sent") notificationsSent += 1;
        else if (a.status === "failed") notificationsFailed += 1;
        else if (a.status === "config_missing") notificationConfigMissing = true;
        else notificationsSkipped += 1;
      }
    }

    const status: "ok" | "partial" = briefError ? "partial" : "ok";
    const mapped = runStatusToOutcome({ status, error: briefError });

    await closeJobRun(db, childRunId, {
      status,
      error: briefError,
      firmsRequestCount: 1,
      detectionsInserted: matchOutcome.detectionsInserted,
      eventsCreated: matchOutcome.eventsCreated,
      briefsGenerated,
      notificationsSent,
      outcome: mapped.outcome,
      retryPending: mapped.retryPending,
      finishedAt: new Date(),
    });

    return {
      bucket,
      source,
      detectionsParsed: fetchResult.detections.length,
      detectionsInserted: matchOutcome.detectionsInserted,
      detectionsSkippedIndustrial: matchOutcome.detectionsSkippedIndustrial,
      eventsCreated: matchOutcome.eventsCreated,
      eventsUpdated: matchOutcome.eventsUpdated,
      briefsGenerated,
      briefSkipReason,
      notificationsSent,
      notificationsFailed,
      notificationsSkipped,
      notificationConfigMissing,
      durationMs: Date.now() - start,
      status,
      error: briefError ?? undefined,
    };
  } catch (err) {
    const mapped = runStatusToOutcome({ status: "error", error: errMessage(err) });
    await closeJobRun(db, childRunId, {
      status: "error",
      error: errMessage(err),
      firmsRequestCount: 1,
      outcome: mapped.outcome,
      retryPending: mapped.retryPending,
      finishedAt: new Date(),
    });
    return {
      bucket,
      source,
      detectionsParsed: 0,
      detectionsInserted: 0,
      detectionsSkippedIndustrial: 0,
      eventsCreated: 0,
      eventsUpdated: 0,
      briefsGenerated: 0,
      briefSkipReason: {},
      notificationsSent: 0,
      notificationsFailed: 0,
      notificationsSkipped: 0,
      notificationConfigMissing: false,
      durationMs: Date.now() - start,
      status: "error",
      error: errMessage(err),
    };
  }
}

// ---------------------------------------------------------------------------
// job_runs helpers

async function openJobRun(
  db: AppDb,
  jobName: string,
  bucket: string | null,
  startedAtMs: number,
): Promise<string> {
  const startedAt = new Date(startedAtMs).toISOString();
  const result = await db.execute(sql`
    INSERT INTO "job_runs" ("job_name", "bucket", "started_at", "status")
    VALUES (${jobName}, ${bucket}, ${startedAt}, 'running')
    RETURNING "id"
  `);
  const rows = decodeRows<{ id: string | number | bigint }>(result);
  return String(rows[0]?.id ?? "");
}

type CloseArgs = {
  status: "ok" | "partial" | "error";
  error: string | null;
  finishedAt: Date;
  firmsRequestCount?: number;
  detectionsInserted?: number;
  eventsCreated?: number;
  briefsGenerated?: number;
  notificationsSent?: number;
  detectionsPruned?: number;
  /** Stage 8: user-facing taxonomy. Only set on per-bucket child rows. */
  outcome?: FreshnessOutcome;
  /** Stage 8: signals "(retrying)" in the AOI freshness banner. */
  retryPending?: boolean;
};

async function closeJobRun(
  db: AppDb,
  id: string,
  args: CloseArgs,
): Promise<void> {
  if (!id) return;
  // outcome / retry_pending only set on per-bucket child rows (the parent
  // doesn't pass them); use COALESCE so an unspecified arg leaves the
  // existing column value untouched. Today the freshness LATERAL filters
  // on bucket = aoi.region_bucket and parent rows have bucket IS NULL, so
  // a parent overwrite of a child's retry_pending could not be observed —
  // but guarding here means future changes to that filter can't silently
  // clobber a child retry signal.
  const outcome = args.outcome ?? null;
  const retryPending = args.retryPending ?? null;
  await db.execute(sql`
    UPDATE "job_runs"
    SET
      "finished_at" = ${args.finishedAt.toISOString()},
      "status" = ${args.status},
      "error" = ${args.error},
      "firms_request_count" = COALESCE("firms_request_count", 0) + ${args.firmsRequestCount ?? 0},
      "detections_inserted" = COALESCE("detections_inserted", 0) + ${args.detectionsInserted ?? 0},
      "events_created" = COALESCE("events_created", 0) + ${args.eventsCreated ?? 0},
      "briefs_generated" = COALESCE("briefs_generated", 0) + ${args.briefsGenerated ?? 0},
      "notifications_sent" = COALESCE("notifications_sent", 0) + ${args.notificationsSent ?? 0},
      "detections_pruned" = COALESCE("detections_pruned", 0) + ${args.detectionsPruned ?? 0},
      "outcome" = COALESCE(${outcome}, "outcome"),
      "retry_pending" = COALESCE(${retryPending}, "retry_pending")
    WHERE "id" = ${id}
  `);
}

function errMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function constantTimeEquals(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) {
    mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return mismatch === 0;
}
