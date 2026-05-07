# Antimeridian (lon=180) latent matcher bug — options

Status: research only. No code change in this PR. Fix recommendation lands in a
follow-up PR.

## 1. The bug

`regionBucketFromLonLat` (`lib/geo/region-bucket.ts:25`) computes the bucket SW
corner with `Math.floor(lon / 5) * 5`. For `lon=180` exactly, this yields
`swLon=180`, producing the bucket key `5x5:E180_N…`.

`bucketToBbox` (`lib/firms/buckets.ts:42-56`) then inverts the key. At
line 53 it clamps the NE corner: `const neLon = Math.min(180, swLon + TILE_DEG)`.
For `swLon=180` this resolves to `neLon = min(180, 185) = 180`. The returned
bbox is `[180, lat, 180, lat+5]` — zero width along longitude.

Downstream consequence:

- `app/api/aoi/poll/route.ts` calls `bucketToBbox` to drive
  `fetchAreaCsv(source, bbox, …)`. FIRMS treats a degenerate bbox as either
  empty or a `400`; either way no detections come back for that bucket.
- The matcher `findAoiMatches` (`lib/firms/matcher.ts:264-354`) joins by
  `region_bucket = $bucket`, so even if an unrelated bucket's detections
  geographically intersected the AOI's polygon, they would not be considered.

Net effect: an AOI whose centroid is exactly `lon=180` is silently invisible
to the watch loop forever. No error surfaces, no `job_runs` warning, no
notification regression — it just stays dark.

## 2. Likelihood / impact

Likelihood is near-zero in practice:

- AOI creation accepts arbitrary user polygons; the centroid calculation is
  the only path producing the lon coordinate that feeds
  `regionBucketFromLonLat`. A computed centroid landing on exactly the IEEE-754
  representation of `180.0` requires either a hand-crafted polygon symmetric
  about the date line, or upstream normalization that emits `180` rather than
  `-180`.
- More likely: a date-line-spanning polygon (e.g. an Aleutian island chain or
  a Russian Far East / Kamchatka stewardship parcel) whose naive centroid
  collapses across the seam. That is a separate, larger bug class (see
  Out of Scope) but the symptom can present as a centroid at `±180`.

Impact, when it does happen, is silent darkness. No alarm bells, no partial
match — total miss. Stewardship-tier users near the date line (Aleutians,
Pacific Islands, Russian Far East, Chukotka) are the realistic affected
cohort. Population is small, probability is small, but the failure mode
("we promised to watch your land and we never told you it burned") is the
worst kind of failure for the product thesis.

## 3. Three fix options

### Option A — Normalize lon=180 → -180 in `regionBucketFromLonLat`

In `lib/geo/region-bucket.ts`, before the `Math.floor`, coerce
`lon === 180` (or `lon >= 180`) to `-180`. This pushes the centroid into the
`5x5:W180_N…` bucket, whose `bucketToBbox` returns `[-180, lat, -175, lat+5]`
— a normal 5° wide tile on the western side of the date line.

LOC: ~3 lines of code + ~30 lines of test. Well within the budget.

Risk: low. The only externally observable behavior change is for
`lon === 180`, which today produces a degenerate bucket — any change is
strictly an improvement. Existing AOIs whose stored `region_bucket` is
`5x5:E180_…` would need a one-time backfill (or be rebucketed on next
update). Search of seed/fixture data to confirm none exist is part of the
follow-up PR.

### Option B — Wrap east edge in `bucketToBbox`

Make `bucketToBbox` return an array of bboxes when `swLon + 5 > 180`,
producing both `[180, lat, 180, lat+5]` (degenerate, drop) and
`[-180, lat, -175, lat+5]`. Matcher and poll route would need to accept
`FirmsBbox | FirmsBbox[]` and issue multiple FIRMS calls per bucket.

LOC: well above 200. Touches `app/api/aoi/poll/route.ts`, `lib/firms/client.ts`
signatures, every test fixture that constructs a bbox by hand. This is the
"correct" model for true antimeridian-spanning tiles but it is the wrong
shape of fix for a single-point bug.

### Option C — Reject lon=180 at AOI creation time

Validate in `lib/db/aoi-repository.ts` (or whichever creation path computes
the centroid) and reject with a clear 400 like
`"AOI centroid lands exactly on the antimeridian; nudge the polygon a few
metres east or west."`

Doesn't fix existing data. Pushes a UX failure onto the user for a
mathematically arbitrary boundary. Acceptable as a guard-rail layered on top
of A; not acceptable as the only fix.

## 4. Recommendation

**Option A.** Smallest viable change, no public API churn, no migration
unless the audit in step 1 of the follow-up PR finds an existing
`5x5:E180_…` row. Keeps the bucket grid uniformly 5° wide. Option B is the
"right" model only when we actually need to support polygons that span the
date line — that is a different problem class (see §6) and should not be
absorbed into this fix.

If the follow-up PR uncovers callers that depend on `regionBucketFromLonLat`
emitting `5x5:E180_…` (none expected from the grep above — only tests,
docs, briefs reference these helpers), fall back to Option C as a
guard-rail. Option B stays out of scope until polygon-spanning support
lands.

## 5. Test strategy for the Option A fix

In `tests/region-bucket.test.ts` and `tests/firms-buckets.test.ts`:

- `regionBucketFromLonLat(180, 0)` returns `5x5:W180_N00`, identical to
  `regionBucketFromLonLat(-180, 0)`.
- `bucketToBbox` round-trips: bucket from `(180, 0)` produces a non-degenerate
  bbox `[-180, 0, -175, 5]`.
- Boundary: `regionBucketFromLonLat(179.999, 0)` stays in `5x5:E175_N00`
  (unchanged behavior).
- Boundary: `regionBucketFromLonLat(-180, 0)` stays in `5x5:W180_N00`.
- Integration (testcontainer): an AOI with centroid `(180, 52)` and a FIRMS
  detection at `(-179.5, 52)` produce a match. Today they would not.

## 6. Out of scope

- Multi-bbox-per-bucket support (Option B's deeper change). Would require
  `FirmsBbox | FirmsBbox[]` plumbing through poll route and client.
- AOI polygons that genuinely cross the date line. The centroid calculation,
  the PostGIS `polygon` column, and the FIRMS `bbox` query are all naive
  about the seam. A Kamchatka-to-Aleutians parcel is a separate problem
  whose fix likely requires reprojecting to a shifted-longitude CRS or
  splitting the polygon at lon=±180.
- Backfill of any existing rows with `region_bucket LIKE '5x5:E180\_%'`.
  Belongs to the follow-up implementation PR, not this research note.
