# Weather context for briefs — source options

Date: 2026-05-07
Author: scout
Trigger: product-review 2026-05-07 §3 — "underdeveloped brief" gap. The Stage 3
schema has a `weather_note: string | null` field that has been hardcoded null
since Stage 3 shipped. The thesis sample brief in `docs/SPEC-A-prime-v1.md`
§LLM brief format leans on language like "RH ~22%, winds 240° @ 28 km/h pushing
activity ENE away from the preserve" — none of which the system can produce
today.

This is a research note, not an ADR. Goal: catalog viable free-tier weather
sources and frame the smallest useful integration.

## 1. What the brief actually wants from weather

Confirmed by reading `lib/ai/schema.ts` and `lib/ai/prompt.ts`:

- `context.weather_note` — free-prose one-liner, currently always null.
- `key_facts.wind_dir_deg`, `wind_speed_kmh`, `wind_toward_aoi` — already in
  the schema, currently always null. The prompt explicitly tells the model
  "wind_dir_deg / wind_speed_kmh / wind_toward_aoi must be null unless wind
  data is provided above (currently they are not)."

So the schema is already prepared for two things:

1. A short prose narrative of conditions (`weather_note`).
2. Three structured wind fields used to derive `wind_toward_aoi` (the only
   weather-derived boolean in the brief).

The thesis brief also alludes to "fuel moisture" and "post-event regrowth" —
those are fuel-model and vegetation-index questions, not pure weather. They
need a separate research pass (FFMC/DMC for FWI; NDVI/dNBR for regrowth) and
are out of scope for a v1.1 weather hook.

Concretely, "fuel moisture relative to historical thresholds" needs at minimum
a 30-year reanalysis baseline (ERA5 or NASA POWER) plus a fuel model — not a
single-call live weather fetch. Triage: defer.

## 2. Public weather sources viable on free tier

| Source | Coverage | Key required | Rate limits | License | Notes |
|---|---|---|---|---|---|
| Open-Meteo (`api.open-meteo.com/v1/forecast`) | Global | None | 10k req/day non-commercial | CC-BY 4.0 | Hourly current + 16-day forecast + ERA5 historical via `archive-api`. Best-fit default. [docs](https://open-meteo.com/en/docs) |
| NOAA NWS API (`api.weather.gov`) | US only | None (User-Agent header) | "reasonable" — not numeric | Public domain | Hourly forecast by lat/lon via `/points/{lat},{lon}` then `/gridpoints/.../forecast/hourly`. [docs](https://www.weather.gov/documentation/services-web-api) |
| NASA POWER (`power.larc.nasa.gov/api/temporal/`) | Global | None | Unspecified, advised <300/min | NASA open data | Daily aggregates and 30-yr climatology. Good for "anomalous low" framing; bad for live wind. [docs](https://power.larc.nasa.gov/docs/services/api/) |
| OpenWeatherMap (free tier) | Global | Yes (`OPENWEATHER_API_KEY`) | 60 calls/min, 1M/month | Custom (proprietary) | Functional but adds a Vanyo blocker for a key. No advantage over Open-Meteo. |
| Copernicus ERA5 / CDS | Global | Yes (`CDSAPI_KEY` — already exists from legacy stack) | Async queue, hours-long | Copernicus license, free non-commercial | Authoritative reanalysis but async batch model — unsuitable for per-tick brief generation. |
| MeteoAlarm | EU | None (CAP feed) | Polite | EUMETNET terms | Alerts (red/orange) not weather measurements. Already noted as a planned source in `docs/pivot-architecture.md` line 348. Different category. |
| CWFIS FFMC/DMC/DC | Canada | None (WFS) | Polite | NRCan open | Fire Weather Index components on a 0.1° grid. Region-specific, like Stage 8's CWFIS perimeter source. |

Discounted: HRRR/GFS gridded NetCDF (correct but ops-heavy — same reason
ADR 0004 deleted the spread stack); Tomorrow.io / WeatherAPI / Visual Crossing
(commercial freemium with low caps and proprietary licenses).

## 3. Smallest useful integrations (per source)

Minimum-viable single field per source — each is a candidate v1.1 micro-feature:

- Open-Meteo `current_weather` → `wind_speed_10m`, `wind_direction_10m`. One
  call per AOI per brief; populates the three `key_facts` wind fields and
  derives `wind_toward_aoi` from `bearing_from_aoi_deg`. Highest leverage.
- Open-Meteo `hourly` → 48-hour `precipitation_sum`. Adds "Last 48h: 0 mm
  precipitation" to `weather_note`. Cheap.
- Open-Meteo `hourly` → `relative_humidity_2m` current. Adds "RH 22%" to
  `weather_note` (the exact phrasing the spec example uses).
- NOAA NWS hourly forecast (US only) → same fields, different caller, two-call
  protocol (`/points` then `/forecast/hourly`).
- CWFIS FFMC current → "FFMC 91 (extreme)" for Canada AOIs only. Compelling
  because it is the language Canadian fire managers actually use.
- NASA POWER 30-yr climatology → "August precip is at 12% of 30-yr norm."
  Anomaly framing; needs a join with a live source.

## 4. Architectural shape

Mirror Stage 8's authority-perimeter pattern (`lib/ai/authority/sources.ts`):

- New `lib/ai/weather/sources.ts` — for v1.1, one global source (Open-Meteo)
  is enough. Region bucketing only matters if we later add CWFIS/NWS for
  region-specific terminology.
- New `lib/ai/weather/fetch.ts` — exports `fetchWeatherContext(lat, lon)`
  returning `WeatherContext | null`. Failure path returns null and the brief
  ships without weather; same build-without-blocking discipline as the FIRMS
  client and Stage 8 authority fetch.
- Path A (orchestrator pre-fetch) for the same reason Stage 8 chose Path A:
  AI SDK v6 `generateObject` does not accept tools. The orchestrator in
  `lib/ai/generate.ts` calls `fetchWeatherContext` before `generateObject`
  and folds the result into `BriefContext.weather`.

Schema question: keep `weather_note: string | null` (LLM-formatted) or change
to structured `{wind_dir_deg, wind_speed_kmh, precip_48h_mm, rh_pct}`?

- For: structured shape lets the renderer build the compact key_facts card
  consistently and bypasses LLM hallucination on numbers. The wind fields
  already exist structured in `key_facts`, so duplication is partial.
- Against: changing the v1 schema means bumping `SCHEMA_VERSION`, which is
  frozen by contract in `lib/ai/schema.ts`. The cheapest path is: populate
  the existing structured wind fields in `key_facts`, then have the LLM
  format the prose `weather_note` from values we hand it in the user prompt.
  Same pattern as Stage 3 already uses for distance/bearing.

Recommendation: keep schema v1; populate the existing wind fields and pass a
formatted weather summary into the prompt for the model to copy into
`weather_note`. Schema bump only if we later add fuel-moisture / anomaly
fields, which need their own ADR.

## 5. Frame for the next product-reviewer pass

- **Source to start with:** Open-Meteo. Global, no key, CC-BY, single call per
  AOI per brief, well-documented. No Vanyo blocker (no registration).
- **Minimum-viable fields:** `wind_direction_10m` + `wind_speed_10m` (current)
  and `precipitation_sum` (last 48h hourly aggregate). Three numbers, one
  HTTP call. Populates the three structured wind fields in `key_facts`,
  derives `wind_toward_aoi` from existing `bearing_from_aoi_deg`, and gives
  the LLM enough material to write a meaningful `weather_note`.
- **Stage shape:** Stage 10-ish, mirroring Stage 8 in size (one source, one
  fetcher, one orchestrator wiring change, schema unchanged). Test with PGlite
  + a mocked `fetch`. Spec the failure path: timeout > 3s → null → brief ships
  without weather.
- **Out of scope for the same stage:** historical/anomaly framing (needs ERA5
  + 30-yr baselines), fuel moisture (FFMC for Canada, FWI generally), and
  MeteoAlarm alert integration (already a separate planned ingest).

Open question for product reviewer: do we want NOAA NWS as a US-specific
override, or is one global source via Open-Meteo good enough for v1.1? The
Stage 8 authority pattern chose region-specific (NIFC, CWFIS) — the analogous
choice for weather would be (NWS for US, Open-Meteo elsewhere). Defaulting to
global keeps the diff small and matches the "global free-tier" north star.

## Sources cited

- `docs/SPEC-A-prime-v1.md` §LLM brief format (lines 192–280)
- `lib/ai/schema.ts`, `lib/ai/prompt.ts`
- `lib/ai/authority/sources.ts` (pattern reference)
- `docs/pivot-architecture.md` lines 338–348, 471 (legacy weather/MeteoAlarm)
- Open-Meteo: https://open-meteo.com/en/docs and https://open-meteo.com/en/docs/historical-weather-api
- NOAA NWS API: https://www.weather.gov/documentation/services-web-api
- NASA POWER: https://power.larc.nasa.gov/docs/services/api/
- Copernicus CDS: https://cds.climate.copernicus.eu/api-how-to
- CWFIS FWI: https://cwfis.cfs.nrcan.gc.ca/datamart
- MeteoAlarm CAP feed: https://feeds.meteoalarm.org/
