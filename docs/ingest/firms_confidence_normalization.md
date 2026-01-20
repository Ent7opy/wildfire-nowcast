# FIRMS Confidence Normalization

## Overview

FIRMS confidence values are normalized to create a weak prior (0-1 scale) for fire likelihood scoring. This ensures that confidence alone cannot determine whether a detection is a real wildfire.

## Sensor-Specific Confidence Semantics

### MODIS (Terra/Aqua)

MODIS confidence is **categorical** with three levels:

| Category | Raw Value | Interpretation |
|----------|-----------|----------------|
| Low (l) | 10 | Low confidence detections; high false positive rate |
| Nominal (n) | 50 | Standard confidence for most fire detections |
| High (h) | 90 | High confidence; typically large or intense fires |

**Scale:** 0-100 (after categorical-to-numeric mapping)

**Characteristics:**
- Coarse granularity (3 discrete levels)
- Categorical interpretation mapped to numeric values
- Most detections fall in nominal or high categories

### VIIRS (S-NPP/NOAA-20)

VIIRS confidence is **numeric** and continuous:

| Range | Interpretation |
|-------|----------------|
| 0-30 | Low confidence; high false positive rate |
| 30-70 | Nominal confidence; typical fire detections |
| 70-100 | High confidence; well-validated detections |

**Scale:** 0-100 (continuous)

**Characteristics:**
- Fine granularity (continuous values)
- Direct quality metric from detection algorithm
- More nuanced than MODIS categorical approach

## Normalization Strategy

Both MODIS and VIIRS use a 0-100 scale, enabling a unified normalization approach:

```python
normalized_confidence = raw_confidence / 100.0  # Maps to [0, 1]
```

### Edge Cases

- **Missing confidence (NULL):** Returns neutral prior (0.5)
- **Out of range values:** Clamped to [0, 100] before normalization
- **Unknown sensor:** Same normalization applied

## Confidence as Weak Prior

Normalized confidence is weighted to ensure it contributes **at most 20%** to the composite Fire Likelihood Score:

```python
confidence_prior = normalized_confidence * 0.2  # Max contribution: 0.2
```

### Design Rationale

1. **Prevents over-reliance on confidence:** Even perfect confidence (100) contributes only 0.2 to a 0-1 likelihood score
2. **Requires multiple signals:** Fire likelihood must be supported by other factors (persistence, land cover, weather, etc.)
3. **Handles missing data gracefully:** Missing confidence defaults to neutral (0.1 contribution), not 0

## Implementation

See `api/fires/service.py` for:
- `normalize_firms_confidence()`: Raw confidence → normalized prior [0, 1]
- `compute_confidence_prior()`: Applies 20% weight constraint

## Usage in Fire Likelihood Scoring

The confidence prior will be combined with other signals in the composite Fire Likelihood Score (task WN-FIRE-006):

- **Confidence:** ≤20% weight (weak prior, this task)
- **Persistence:** ~30% weight (multi-detection requirement)
- **Land cover:** ~20% weight (plausibility check)
- **Weather:** ~20% weight (meteorological plausibility)
- **Multi-sensor bonus:** ~10% weight (cross-validation)

Total: 100% weight → Fire Likelihood Score [0, 1]

## References

- NASA FIRMS documentation: https://firms.modaps.eosdis.nasa.gov/
- MODIS confidence: Derived from fire detection algorithm quality flags
- VIIRS confidence: Based on detection algorithm quality metrics and background characterization
