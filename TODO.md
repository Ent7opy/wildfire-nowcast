# Wildfire Nowcast - Backend TODOs

> Generated from code review scan focusing on bugs, inconsistencies, and potential issues in a non-profit wildfire monitoring and forecasting system (world-wide).

## Table of Contents
- [ML Module](#ml-module)
- [Ingest Module](#ingest-module)
- [Data Consistency Issues](#data-consistency-issues)
- [Performance Concerns](#performance-concerns)
- [Missing Error Handling](#missing-error-handling)
---

## ML Module

### ML-001: Division by Zero Risk in Wind Scale Calculation
**File:** `ml/spread/heuristic_v0.py` (line 305)
**Issue:** `wind_scale = 1.0 + (wind_speed * self.config.wind_influence_km_h_per_ms / (self.config.base_spread_km_h + 1e-6))`
The epsilon `1e-6` is small; if `base_spread_km_h` is 0 and `wind_speed` is large, this could still cause numerical issues.
**Risk:** Division by near-zero resulting in overflow or unexpected spread calculations.
**Fix:** Add explicit check for `base_spread_km_h <= 0` and handle appropriately.

### ML-002: Spread Model Configuration Not Validated
**File:** `ml/spread/heuristic_v0.py` (lines 26-62)
**Issue:** `HeuristicSpreadV0Config` uses dataclass with defaults but no validation that values are reasonable (e.g., `max_kernel_size` must be odd).
**Risk:** Invalid configurations could cause runtime errors or incorrect predictions.
**Fix:** Add `__post_init__` validation to ensure constraints like `max_kernel_size >= 7` and odd.

### ML-003: FFT Convolution Can Produce Numerical Artifacts
**File:** `ml/spread/heuristic_v0.py` (line 183)
**Issue:** `fftconvolve` is used with `mode="same"`, which can introduce boundary artifacts.
**Risk:** Edge cells in the forecast grid may have incorrect probability values.
**Fix:** Consider padding or using direct convolution for small kernels; document boundary behavior.

### ML-004: Missing Model Version in Forecast Output
**File:** `ml/spread/service.py` (lines 151-156)
**Issue:** Model name is logged but not consistently stored in the forecast output metadata.
**Risk:** Difficult to trace which model version produced a forecast for debugging/auditing.
**Fix:** Add model name and version to `SpreadForecast` metadata attributes.

### ML-005: Calibrator Missing Horizon Warning Is Too Permissive
**File:** `ml/spread/service.py` (lines 384-389)
**Issue:** When calibration is missing for some horizons, only a warning is logged but raw probabilities are returned.
**Risk:** Users may unknowingly receive uncalibrated predictions.
**Fix:** Make this behavior configurable or include a prominent flag in the response indicating uncalibrated horizons.

### ML-006: Denoiser Inference Fails Silently on Missing Features
**File:** `ml/denoiser_inference.py` (lines 99-104)
**Issue:** Missing feature columns are filled with NaN and a warning is logged, but the model may not handle NaN values correctly.
**Risk:** Model predictions could be incorrect or fail silently.
**Fix:** Ensure the model can handle NaN values or raise an error if critical features are missing.

### ML-007: Weather Bias Corrector Path Resolution Inconsistent
**File:** `ml/spread/service.py` (lines 261-290) vs `ml/weather_bias_correction.py`
**Issue:** Path resolution logic is duplicated between spread service and weather bias correction module with slightly different fallback orders.
**Risk:** Inconsistent behavior when resolving model artifacts.
**Fix:** Centralize path resolution logic in a shared utility.

---

## Ingest Module

### INGEST-001: FIRMS Ingest Doesn't Handle API Rate Limits
**File:** `ingest/firms_ingest.py` (lines 99-113)
**Issue:** The code fetches data from FIRMS API but doesn't explicitly handle HTTP 429 (rate limit) responses.
**Risk:** Could be blocked by FIRMS API during bulk backfills.
**Fix:** Add explicit rate limit handling with exponential backoff.

### INGEST-002: Weather Ingest Fallback Cycle May Use Stale Data
**File:** `ingest/weather_ingest.py` (lines 832-867)
**Issue:** When primary GFS cycle fails, falls back to previous 6h cycle without checking data age.
**Risk:** Forecasts may use significantly outdated weather data without warning.
**Fix:** Add maximum age check for fallback data; warn if data is too old.

### INGEST-003: NetCDF Files Not Closed on Exception
**File:** `ingest/weather_ingest.py` (lines 452-517)
**Issue:** While there's a `finally` block to close the dataset, similar patterns in other functions may not be as robust.
**Risk:** File handle leaks during batch processing.
**Fix:** Audit all xarray dataset usages for proper resource cleanup; consider using context managers.

### INGEST-004: Scoring Updates Not Atomic
**File:** `ingest/firms_ingest.py` (lines 114-122)
**Issue:** False source masking, persistence scores, landcover scores, weather scores, and fire likelihood are updated in separate transactions.
**Risk:** Partial scoring updates could leave detections in inconsistent states if the process crashes.
**Fix:** Consider wrapping all scoring updates in a single database transaction per batch.

### INGEST-005: Denoiser Subprocess Call Assumes UV Availability
**File:** `ingest/firms_ingest.py` (lines 237-288)
**Issue:** The denoiser is invoked via `uv run`, assuming uv is in PATH.
**Risk:** In containerized environments without uv, this will fail.
**Fix:** Make the denoiser invocation configurable or use Python directly with proper module path.

### INGEST-006: Weather Bbox Default Hardcoded to Europe
**File:** `ingest/config.py` (lines 116-119)
**Issue:** Default weather bbox is `5.0, 35.0, 20.0, 47.0` (Europe/Mediterranean).
**Risk:** For a world-wide wildfire system, this default is inappropriate for other regions.
**Fix:** Either require explicit bbox configuration or use a truly global default.

### INGEST-007: No Validation of Downloaded GRIB File Integrity
**File:** `ingest/weather_ingest.py` (lines 190-305)
**Issue:** Downloaded GRIB files are not validated for corruption before processing.
**Risk:** Partial/corrupted downloads could cause cryptic errors during GRIB parsing.
**Fix:** Add checksum validation or file size checks; retry on validation failure.

### INGEST-008: Terrain Cache Key Doesn't Include Resolution
**File:** `api/forecast/repo.py` (implied by `find_cached_terrain`)
**Issue:** If terrain resolution changes, the cache may return incompatible data.
**Risk:** Grid misalignment between terrain and other data sources.
**Fix:** Include resolution and other relevant metadata in cache key computation.

---

## Data Consistency Issues

### DATA-001: Fire Likelihood Scoring Uses Different NULL Handling
**File:** `api/fires/repo.py` (lines 395-400)
**Issue:** `compute_fire_likelihood` is called with explicit None handling that converts NULLs to default values (0.5), but this logic is duplicated and could become inconsistent.
**Risk:** Different parts of the codebase may handle missing scores differently.
**Fix:** Centralize NULL-to-default conversion in the scoring function itself.

### DATA-002: GridSpec.from_bbox Has Ambiguous Parameter Order
**File:** `api/core/grid.py` (lines 47-88)
**Issue:** The method accepts either a bbox tuple OR individual coordinates, which can be confusing.
**Risk:** Developers may pass arguments in wrong order.
**Fix:** Consider separate factory methods: `from_bbox_tuple()` and `from_bounds()`.

### DATA-003: Weather Data Timezone Handling Fragile
**File:** `api/fires/scoring.py` (lines 461-464), `ingest/weather_ingest.py` (multiple)
**Issue:** Timezone-aware datetimes are converted to naive UTC by stripping tzinfo, which can be error-prone.
**Risk:** Timezone-related bugs during DST transitions or when comparing timestamps.
**Fix:** Use `numpy.datetime64` with explicit UTC handling consistently throughout.

### DATA-004: Coordinate Precision Loss in Grid Calculations
**File:** `api/core/grid.py` (lines 77-78)
**Issue:** `math.floor()` and floating point arithmetic can introduce precision errors for edge cases.
**Risk:** Grid cells at boundaries may be misaligned or dropped.
**Fix:** Use decimal.Decimal for critical coordinate calculations or add epsilon handling.

---

## Performance Concerns

### PERF-001: Fire Detection Queries Not Paginated
**File:** `api/fires/repo.py` (lines 45-159)
**Issue:** `list_fire_detections_bbox_time` can return up to 10,000 rows (the limit parameter max), but all are loaded into memory at once.
**Risk:** Large result sets could cause memory pressure.
**Fix:** Consider streaming results for large queries or implementing cursor-based pagination.

### PERF-002: Persistence Score Computation Inefficient for Large Batches
**File:** `api/fires/scoring.py` (lines 87-201)
**Issue:** For each detection in a batch, the algorithm queries the database for spatial-temporal clustering using `ST_DWithin`.
**Risk:** N+1 query pattern for large batches; each detection triggers a database query.
**Fix:** Batch the clustering computation or use a single query with window functions.

### PERF-003: Weather Data Loading Loads Entire Dataset
**File:** `ml/spread_features.py` (lines 152-222)
**Issue:** `xr.open_dataset(path)` loads the entire weather NetCDF file, then slices to the window.
**Risk:** Large weather files could cause memory issues.
**Fix:** Use `chunks` parameter for lazy loading or pre-subset files by region.

### PERF-004: JIT Pipeline Synchronous Blocking
**File:** `api/routes/forecast.py` (lines 118-185)
**Issue:** The `/forecast/jit` endpoint enqueues a job but doesn't provide an async polling mechanism with websockets/SSE.
**Risk:** UI must poll repeatedly; inefficient for long-running operations.
**Fix:** Consider implementing Server-Sent Events (SSE) or WebSocket for job status updates.

---

## Missing Error Handling

### ERR-001: Redis Connection Failure Not Handled
**File:** `api/main.py` (lines 29-32)
**Issue:** `FastAPILimiter.init(redis)` is called at startup but failures are not caught.
**Risk:** Application startup may fail or hang if Redis is unavailable.
**Fix:** Add try-catch and graceful degradation (disable rate limiting if Redis is down).

### ERR-002: Missing Weather Data Returns Fallback Without Notification
**File:** `ml/spread_features.py` (lines 135-140, 214-216)
**Issue:** When weather data is missing, a calm fallback is returned, but the API response doesn't indicate this.
**Risk:** Users receive forecasts based on zero-wind assumptions without knowing.
**Fix:** Add metadata flag to forecast response indicating fallback weather was used.

### ERR-003: Terrain Loading Failure Silently Uses Empty Terrain
**File:** `ml/spread_features.py` (lines 386-408)
**Issue:** If terrain loading fails, empty terrain (zeros) is used without raising an error.
**Risk:** Spread forecasts may be incorrect due to missing slope/aspect data.
**Fix:** Make terrain loading failure more visible; add warning to forecast metadata.

### ERR-004: Missing Industrial Sources Table Handling
**File:** `api/fires/scoring.py` (lines 19-84)
**Issue:** `mask_false_sources` assumes the `industrial_sources` table exists and is populated.
**Risk:** If the table is empty or missing, all detections pass through unmasked.
**Fix:** Log warning if table is empty; add health check for required reference data.

---

## Documentation & Code Quality

### DOC-001: Missing Type Hints in Some Functions
**Files:** Various
**Issue:** Some functions lack complete type hints, making static analysis difficult.
**Fix:** Add comprehensive type annotations, especially for public APIs.

### DOC-002: TODO Comment in Production Code
**File:** `ml/spread/service.py` (line 76)
**Issue:** `# TODO: Implement cluster-to-bbox resolution once clustering logic exists.`
**Fix:** Create a tracking issue and reference it in the comment.

### DOC-003: Configuration Documentation Out of Sync
**Issue:** Some configuration options are documented in code comments but not in user-facing documentation.
**Fix:** Ensure all configuration options are documented in `docs/` or README.

---

## Testing Gaps

### TEST-001: Edge Cases Not Covered
- Empty bboxes (degenerate cases)
- Invalid datetime formats
- Network timeouts during external API calls
- Concurrent access to JIT job queue

### TEST-002: Integration Test Gaps
- Full end-to-end FIRMS ingest to forecast pipeline
- Weather fallback cycle scenarios
- Denoiser failure during FIRMS ingest

---

## Recommendations Summary

### Immediate Actions (High Priority)
1. Fix timezone handling in forecast endpoint (API-001, CRIT-001)
2. Add bbox validation to prevent invalid queries (API-001)
3. Fix potential division by zero in spread model (ML-001)
4. Add Redis connection failure handling (ERR-001)
5. Document weather fallback behavior (ERR-002)

### Short Term (Medium Priority)
1. Add rate limiting to expensive endpoints (API-002)
2. Centralize NULL handling in fire likelihood scoring (DATA-001)
3. Add model version to forecast metadata (ML-004)
4. Fix race condition in JIT pipeline (API-006)
5. Add GRIB file integrity validation (INGEST-007)

### Long Term (Lower Priority)
1. Implement distributed locking for cache operations (API-006)
2. Add streaming/pagination for large queries (PERF-001)
3. Optimize persistence score computation (PERF-002)
4. Implement SSE/WebSocket for job status (PERF-004)
5. Comprehensive audit of timezone handling (DATA-003)

---

*Last updated: 2026-01-30*
*Scan scope: api/, ml/, ingest/, ui/ modules*
*Project: Wildfire Nowcast & Forecast (World-wide non-profit wildfire monitoring system)*
