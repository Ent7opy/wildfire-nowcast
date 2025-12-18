# Hotspot Denoiser Weak-Labeling Strategy (ML v1)

## Goal
Define a repeatable, realistic weak-labeling approach to create “imperfect but useful” labels for training a first-version classifier that separates likely real wildfire detections from likely artefacts/permanent hotspots.

## Key Constraint
- **Labels** may use future context (e.g., persistence over days) for training set creation.
- **Model features** at inference time must NOT use future information (no leakage).

## 1. FIRMS Signals and Artefact Patterns

### FIRMS Fields (Canonical CSV Column Names)
We rely on the following fields as ingested by our pipeline:
- `latitude`, `longitude`: Decimal degrees.
- `acq_date`, `acq_time`: Used to create UTC acquisition timestamps.
- `confidence`: Numeric (0-100) or text-mapped.
    - **Normalization**: `high=90`, `nominal=60`, `low=30`. When only categorical levels are available, we map them to these numeric proxies for consistency in thresholds.
- `frp`: Fire Radiative Power (MW).
- `brightness`: Brightness temperature (I-4 channel, Kelvin).
- `bright_t31`: Brightness temperature (I-5 channel, Kelvin).
- `satellite`, `instrument`: Source platform identifiers.
- `scan`, `track`: Pixel dimensions/resolution metadata.

### Artefact Patterns
Common false-positive patterns in our Area of Interest (AOI):
- **Gas flares / industrial heat sources**: Often static, highly persistent, and typically exhibit high brightness but low spatial growth.
- **Coastlines / water adjacency**: Sun-glint on water can trigger false detections, often with low confidence and isolated spatial distribution.
- **Repeating static hotspots**: Heat sources like power plants or urban heat islands that trigger at the same coordinates repeatedly.
- **Single-frame noise**: Transient sensor artefacts that appear once and never recur or show spatial clustering.

## 2. Label Definitions (3-Way)

We use a 3-way labeling scheme to ensure high-precision training data.

| Label | Definition | Rationale |
| :--- | :--- | :--- |
| **POSITIVE** | Likely real wildfire. | High-precision signals with growth/persistence or clustering. |
| **NEGATIVE** | Likely noise or permanent hotspot. | Clear industrial sources or extremely long static persistence. |
| **UNKNOWN** | Ambiguous or uncertain. | Detections that don't meet high-precision criteria. Excluded from training in v1. |

**Rationale**: Training on high-precision POS/NEG labels reduces label noise, which is critical for a first-version model.

## 3. Labeling Rules (v1 Heuristics)

These rules use **future context** and **spatiotemporal clustering** to assign labels.

### Logic Definitions
- **Distance**: Haversine distance.
- **Clustering**: A set of detections where each is within 2km and 24 hours of at least one other detection in the set.
- **Persistence**: A cluster or detection is considered "persistent" if it is present on ≥ 2 distinct days within a 72-hour window.
- **Adjacency**: Defined as the 8-neighborhood in the snapped `0.01°` grid index space (Δrow, Δcol ∈ {−1, 0, 1}, excluding (0,0)).
- **Industrial Radius**: `INDUSTRIAL_RADIUS_KM = 2.0` (Parameter to be tuned during QA).

### Rule Table

| Rule Name | Condition | Label | Notes |
| :--- | :--- | :--- | :--- |
| **Industrial Mask** | Distance < `INDUSTRIAL_RADIUS_KM` from a known industrial/flare catalog. | **NEGATIVE** | Strongest negative signal. |
| **Chronic Static** | Detection in the **same** grid cell on ≥ 20 distinct days within a 90-day window AND no detections in adjacent cells. | **NEGATIVE** | High precision for permanent heat sources. |
| **Cluster Growth** | Part of a spatiotemporal cluster (≥ 3 detections) showing expansion into **adjacent cells** on subsequent days. | **POSITIVE** | Classic wildfire behavior (requires future context). |
| **Persistent Cluster** | A **Persistent Cluster** (present on ≥ 2 days in 72h) that is NOT Chronic Static and NOT near an Industrial Mask. | **POSITIVE** | High-precision fire signal. |
| **Low-Conf Singleton** | `confidence < 30` (or "low") AND no other detections within 5km/24h. | **NEGATIVE** | Likely sensor noise. |
| **Static Persistence** | 5-19 distinct days in 30 days in the same cell AND no adjacent cell detections. | **UNKNOWN** | Could be smoldering/peat; safer to exclude from NEGATIVE. |
| **High-Conf Event** | `confidence > 80` AND `frp > 10` but no clustering/growth. | **UNKNOWN** | High risk of industrial leakage outside mask; safer as UNKNOWN. |
| **Default** | Any detection not meeting the above rules. | **UNKNOWN** | Safely ignored during training. |

## 4. Training Data Scope

### Geographic Extent
- **Primary AOI**: [Chosen Region - e.g., Western US / California]
- **Secondary AOI**: [Chosen Region - e.g., South-Eastern Europe / Balkans]

### Time Span
- **Duration**: 12 months (e.g., 2024 full year).
- **Reason**: Captures full seasonality (fire season vs. winter artefacts).

### Holdout Strategy
- **Temporal Split**: 
    - **Training**: First 11 months.
    - **Evaluation (Holdout)**: Last 1 month.
- **Note**: Avoid random row splits to prevent leakage from the same fire event across multiple days.

## 5. Target Label Balance
- Aim for at least **5,000 POSITIVE** and **2,000 NEGATIVE** samples.
- **Fallback Plan**:
    - If **NEGATIVE** count is low: expand industrial mask sources (e.g., refineries, power plants) and/or lengthen the time window.
    - If **POSITIVE** count is low: relax cluster requirements from ≥ 3 to ≥ 2 detections, provided they are NOT Chronic Static and NOT near an Industrial Mask.

## 6. QA Checks for Label Sanity

- **Class Distribution**: Monitor ratio of POS : NEG : UNKNOWN.
- **Manual Visual Audit**: Plot a sample of ~100–200 labeled points on a map (e.g., Folium) to verify heuristics.
- **Leakage Audit**: Confirm features for `denoised_score` inference do NOT include future-looking logic.
- **FRP Distribution**: Verify POS labels have significantly higher mean FRP than NEG labels.

## 7. Known Weaknesses and Biases
- **Low Recall**: Intentionally misses smaller/ambiguous fires to ensure high training precision.
- **Non-Wildfire Heat**: Volcanic activity or geothermal regions (hot springs) may be mislabeled unless explicitly masked.
- **Cloud Masking**: Persistence rules might be interrupted by cloud cover.

## 8. Leakage Rules

| Logic Type | Allowed for Labeling? | Allowed for Inference? | Notes |
| :--- | :--- | :--- | :--- |
| Single detection metrics | Yes | Yes | `confidence`, `frp`, etc. |
| Historical persistence | Yes | Yes | **Conditional**: Only if queried from live DB (e.g., past 7 days). |
| Future persistence | **Yes** | **NO** | Strictly for labeling only. |
| Future growth | **Yes** | **NO** | Strictly for labeling only. |
| Static industrial catalog | Yes | Yes | If catalog is available at runtime. |

---
**Definition of Done (DoD)**:
- [X] Label scheme documented using actual FIRMS column names.
- [X] Heuristics split into Industrial Mask vs. Chronic Static with precise grid-based adjacency.
- [X] Clustering (24h) and Persistence (72h) windows defined and consistent.
- [X] Dataset scope placeholders defined for implementation choice.
- [X] Fallback plan for label balance included.
- [X] QA, Leakage rules, and known limitations clear.
