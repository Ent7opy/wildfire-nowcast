# Task: Centralise Magic Numbers into `constants.py`

**Location:** `forecast/worker.py`, `routes/forecast.py`, `routes/fires.py:18-35`
**Impact:** Low — clean code / maintainability
**Maturity target:** `mvp_operational`

## Problem

Several values are hardcoded in multiple places:

- `DEFAULT_HORIZONS_HOURS = [24, 48, 72]` — duplicated between `forecast/worker.py` and `routes/forecast.py`
- `FIRE_DETECTION_BASE_COLUMNS` — hardcoded list in `routes/fires.py:18-35`
- Likely others (score weights, bbox size limits, minimum detection confidence thresholds)

When these change, they must be updated in multiple files. A mismatch between worker and route defaults is a silent bug.

## Proposed Solution

Create `api/constants.py`:

```python
# api/constants.py

# Forecast horizons
DEFAULT_HORIZONS_HOURS: list[int] = [24, 48, 72]
MAX_HORIZON_HOURS: int = 72

# Fire detection
FIRE_DETECTION_BASE_COLUMNS: list[str] = [
    "detection_id",
    "latitude",
    "longitude",
    "detection_time",
    "confidence",
    "frp",
    "satellite",
    "is_noise",
    "review_status",
    # ... complete list here
]

# Bounding box
MAX_BBOX_AREA_DEG2: float = 25.0  # 5° × 5° hard cap

# Scoring
PERSISTENCE_SCORE_WEIGHT: float = 0.35
LANDCOVER_SCORE_WEIGHT: float = 0.25
WEATHER_SCORE_WEIGHT: float = 0.40

# Archive
MAX_ARCHIVE_RANGE_DAYS: int = 7  # also set via env var; this is the default
```

All modules import from `api.constants` rather than defining local literals.

## Acceptance Criteria

- [ ] `api/constants.py` created and contains at minimum: `DEFAULT_HORIZONS_HOURS`, `FIRE_DETECTION_BASE_COLUMNS`, `MAX_BBOX_AREA_DEG2`
- [ ] `forecast/worker.py` and `routes/forecast.py` both import `DEFAULT_HORIZONS_HOURS` from `api.constants`
- [ ] `routes/fires.py` imports `FIRE_DETECTION_BASE_COLUMNS` from `api.constants`
- [ ] Audit finds no remaining magic number literals in route or worker files for the values listed above
- [ ] Values that are also configurable via env vars remain env-var-driven at runtime — the constant serves as the default only

## Notes

- Do not move values that are genuinely config (i.e. change between environments) into this file — those belong in `api/config.py` (Pydantic settings). `constants.py` is for values that are fixed by design, not by deployment
- A quick `grep` pass after implementation is the verification step: `grep -rn "\[24, 48, 72\]" .` should return zero results outside of `constants.py`
