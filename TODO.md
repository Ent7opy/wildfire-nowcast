# Wildfire Nowcast - Backend TODOs

> Generated from code review scan focusing on bugs, inconsistencies, and potential issues in a non-profit wildfire monitoring and forecasting system (world-wide).

## Table of Contents
- [Ingest Module](#ingest-module)
- [Data Consistency Issues](#data-consistency-issues)
- [Performance Concerns](#performance-concerns)
- [Missing Error Handling](#missing-error-handling)
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
