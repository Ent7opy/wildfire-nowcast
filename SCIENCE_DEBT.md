# Science Debt

Tracked stage-gap WARNINGs and known deviations from `science_grade` quality.
Each entry records the limitation, its impact, the mitigation path, and the target stage.

Per `AGENTS.md`: WARNINGs must include a mitigation action and a target stage.
A WARNING cannot replace a STOP/BLOCKER.

---

## Open Items

### SD-01 — Large-batch neutral-score fallback
**File:** `api/fires/repo.py` lines ~354-381 (persistence) and ~523-549 (weather)
**Stage gap:** `mvp_operational` → `science_grade`

**Limitation:** When a scoring batch exceeds the configured threshold (≈5 000 records),
fire detections receive neutral default scores (`persistence=0.5`, `weather=0.3`) rather
than computed values. This protects against OOM but silently under-scores large wildfire
complexes during their most critical phase.

**Impact:** Active fires that span a large area (e.g., complex multi-front events) may
appear lower-priority than smaller but correctly scored events. Decision-support outputs
for large AOIs are less reliable.

**Mitigation:** Replace the neutral-score fallback with chunked batch processing so all
records receive real scores regardless of batch size. Target chunk size: 1 000 records.

**Target stage:** `science_grade`

---

### SD-02 — Weather bias correction bypassed for location-based forecasts
**File:** `ml/spread/service.py` (`_annotate_fallback_info`, bias-corrector resolution)
**Stage gap:** `mvp_operational` → `science_grade`

**Limitation:** When a forecast is triggered by lat/lon without a named `region_name`,
the regional weather bias corrector is skipped and a WARNING is logged. Two users querying
overlapping areas get forecasts of different accuracy depending on how they queried.
The UI now surfaces `weather_bias_corrected: false` as a badge (see P1.2/P2.1), so users
are informed — but the underlying accuracy gap remains.

**Impact:** Location-based forecasts may diverge from region-tuned forecasts by a
non-trivial margin, especially in regions with known GFS bias (e.g., coastal terrain,
elevated plateaus).

**Mitigation:** Snap location-based queries to the nearest calibrated region using a
spatial lookup (region boundary polygons or centroid-nearest-neighbour). Fall back to
global-uncorrected only if no region is within a configurable distance threshold.

**Target stage:** `science_grade`

