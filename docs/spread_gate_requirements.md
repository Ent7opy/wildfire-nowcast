# Spread Gate Requirements

This document is the authoritative specification for what a spread model must satisfy to pass
the champion-challenger gate and be promoted to each maturity stage. It consolidates hard stop
definitions, stage warning format, science debt register schema, and the `science_grade`
promotion checklist.

See also:
- [Spread Maturity Policy](spread_maturity_policy.md) — stage definitions and governance rules
- [Spread Data Sources](spread_data_sources.md) — authoritative source declarations

---

## 1. Gate Report Contract

Every run of `eval_spread_champion_challenger.py` writes a `gate_report.json` with the
following top-level fields. All fields are required; absence of any field is a pipeline bug.

| Field | Type | Description |
|-------|------|-------------|
| `pass` | bool | `true` only when all hard stops are clear, gate metrics pass, and `weighted_bss_improvement > 0` |
| `recommend_challenger` | bool | Whether the challenger beat the champion on primary/secondary metrics |
| `maturity_stage` | string | `"mvp_operational"` or `"science_grade"` |
| `hard_stops` | array | Zero or more hard stop objects (see §2); non-empty forces `pass: false` |
| `stage_warnings` | array | Stage-gap warnings (see §3); do not block promotion but must have tracking IDs |
| `promotion_decision` | string | `"promote_challenger"` or `"hold_challenger"` |
| `science_debt_register` | array | Open science debt items (see §4) |
| `decision` | object | Full metric comparison output from `compute_recommendation` |

---

## 2. Hard Stops

Hard stops unconditionally set `gate_report.pass = false` and block promotion, regardless of
metric scores. They cannot be overridden by warnings or science debt items.

Each hard stop object has this schema:

```jsonc
{
  "id": "STOP-XXX-NNN",          // unique stop ID
  "message": "...",              // what went wrong (machine-readable, stable)
  "mitigation": "...",           // what the operator must do to clear it
  "target_stage": "..."          // maturity stage at which this stop was triggered
}
```

### 2.1 Defined Hard Stops

#### STOP-STAGE-001 — Invalid maturity stage

| Field | Value |
|-------|-------|
| **Trigger** | `gate.maturity_stage` is not one of `mvp_operational`, `science_grade` |
| **Effect** | Evaluation aborts; no metric comparison is attempted |
| **Mitigation** | Set `gate.maturity_stage` to a valid value in the eval config |

---

#### STOP-GEO-001 — Geospatial alignment failure

| Field | Value |
|-------|-------|
| **Trigger** | The eval region grid fails the canonical analysis-grid contract: CRS ≠ EPSG:4326, or cell size ≠ 0.01° |
| **Effect** | Data collection is skipped entirely; gate report is written with this hard stop |
| **Mitigation** | Re-project the region grid to EPSG:4326 at `DEFAULT_CELL_SIZE_DEG` (0.01°) before running eval |
| **Why** | Train-time features are computed in the canonical spatial frame. A mismatched CRS or resolution means the challenger's metric comparison references a spatially different input from training, making results scientifically invalid |
| **Implementation** | `ml/spread_features.py:assert_grid_alignment` — raises `ValueError` with the STOP-GEO-CRS or STOP-GEO-RES prefix, caught in `run_eval` and wired here |

---

#### STOP-SOURCE-001 — Missing authoritative source declaration

| Field | Value |
|-------|-------|
| **Trigger** | The eval config `data_sources` block is absent or missing one or more of: `fires`, `weather`, `terrain`, `fuels` |
| **Effect** | Gate report sets `pass: false`; metric comparison result is ignored |
| **Mitigation** | Add a `data_sources` block to the eval config (see [Spread Data Sources §5](spread_data_sources.md)) |
| **Why** | Undeclared input provenance means a promoted model cannot be traced to its training inputs. The gate cannot validate that inference will use the same sources as training |

Example:

```yaml
data_sources:
  fires: "nasa_firms_viirs_nrt"
  weather: "noaa_gfs_025deg"
  terrain: "srtm30_dem_derived"
  fuels: "esa_worldcover_10m_ndvi+ecmwf_ecland_lfmc+nfdrs_dfmc"
```

---

#### STOP-CAL-001 — Missing calibrator artifact (LearnedSpreadModelV3)

| Field | Value |
|-------|-------|
| **Trigger** | Challenger is `LearnedSpreadModelV3` and `challenger.model_params.calibrator_run_dir` is absent or does not contain `calibrator.pkl` |
| **Effect** | Gate report sets `pass: false` |
| **Mitigation** | Run the calibration step and provide `calibrator_run_dir` pointing to the directory containing `calibrator.pkl` |
| **Why** | V3 probability outputs are post-hoc calibrated; raw logits are not valid probabilities and must not be compared against the champion |

---

#### STOP-CONTRACT-001 — Feature contract mismatch

| Field | Value |
|-------|-------|
| **Trigger** | The challenger's `runtime_contract.json` (or fallback `feature_schema.json`) channel list diverges from `CANONICAL_V2_CHANNELS` by name, order, or count |
| **Effect** | Gate report sets `pass: false`; metric comparison is treated as scientifically invalid |
| **Mitigation** | Re-export the challenger so its feature schema matches `CANONICAL_V2_CHANNELS`, or update `CANONICAL_V2_CHANNELS` and retrain |
| **Why** | A channel mismatch means train-time and infer-time tensors have different physical meanings at the same index positions. Any metric scores produced under such a mismatch are fabricated |
| **Implementation** | `ml/spread/runtime_contract.py:validate_channel_alignment` — both name AND order must match; reordering is treated as a hard stop, not a warning |

Canonical channel list (v2/v3, 18 channels, in required order):

```
fire_t0, fire_t-6h, fire_t-12h,
u10, v10, t2m, rh2m, precip_24h,
slope_deg, aspect_sin, aspect_cos, elevation_m, ruggedness, tpi,
ndvi, lfmc, dfmc,
region_id_embedding_input
```

---

## 3. Stage Warnings

Stage warnings are stage-aware quality flags. They **never override hard stops**. A gate report
can pass with open warnings, but each warning must carry a `tracking_id` so it can be resolved
before `science_grade` promotion.

Each stage warning object has this schema:

```jsonc
{
  "id": "WARN-XXX-NNN",         // unique warning ID
  "tracking_id": "...",         // stable slug used to track resolution (required)
  "warning": "...",             // human-readable description of the gap
  "mitigation": "...",          // what must be done to clear this warning
  "target_stage": "..."         // the maturity stage by which this must be resolved
}
```

### Defined Stage Warnings

| ID | tracking_id | Trigger | Target stage |
|----|-------------|---------|--------------|
| `WARN-MVP-BSS-001` | `spread-science-debt-bss-positive-skill` | `weighted_bss_improvement ≤ 0` | `mvp_operational` |
| `WARN-GATE-001` | `spread-science-debt-gate-regression` | Primary/secondary gate did not pass | (current stage) |
| `WARN-FIRE-001` | — | Cross-sensor VIIRS inter-calibration not validated | `science_grade` |
| `WARN-WX-001` | — | Bias correction optional | `science_grade` |
| `WARN-WX-002` | — | No high-resolution weather source for non-CONUS domains | `science_grade` |
| `WARN-TERR-001` | — | DEM provenance not declared per region | `science_grade` |

Warnings sourced from `docs/spread_data_sources.md` (WARN-FIRE-001, WARN-WX-*, WARN-TERR-001)
are raised by the ingest/feature pipeline, not by the gate report directly. They must be resolved
before `science_grade` promotion.

---

## 4. Science Debt Register

The science debt register tracks deferred quality obligations that are not enforced at
`mvp_operational` but **must be closed before `science_grade` promotion**.

Each science debt item has this schema:

```jsonc
{
  "debt_id": "SCI-DEBT-XXX",        // unique debt ID
  "tracking_id": "...",             // stable slug for external tracking (required)
  "description": "...",             // what is deferred
  "target_stage": "science_grade",  // always science_grade for MVP debts
  "exit_criteria": "..."            // what observable condition closes this debt
}
```

### Open debts at `mvp_operational`

| debt_id | tracking_id | Description | Exit criteria |
|---------|-------------|-------------|---------------|
| `SCI-DEBT-EXT-GT` | `spread-science-debt-ext-ground-truth` | External ground-truth verification not enforced in MVP gate | Validated against authoritative external ground-truth dataset |
| `SCI-DEBT-DM-SAL` | `spread-science-debt-dm-sal-governance` | DM significance and SAL threshold governance deferred | DM significance and SAL thresholds enforced in promotion policy |
| `SCI-DEBT-DRIFT` | `spread-science-debt-drift-monitoring` | Reliability/calibration drift monitoring not yet mandatory | Drift monitors and alert thresholds operational in production |

These are automatically written into every `gate_report.json` for `mvp_operational` evaluations
by `_build_stage_governance` in `eval_spread_champion_challenger.py`.

---

## 5. Metric Gate Thresholds

The `compute_recommendation` function applies these thresholds (config-overridable via
`gate.*` in the eval YAML):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bss_improvement_min` | 0.03 | Weighted BSS improvement over champion must exceed this |
| `bss_horizon_floor` | −0.005 | No single horizon may regress below this BSS floor |
| `sal_regression_max` | 0.05 | SAL composite may not degrade by more than this |
| `dm_pvalue_max` | 0.05 | Diebold-Mariano test p-value must be below this |
| `max_pr_auc_drop` | 0.01 | PR-AUC may not drop by more than this |
| `max_iou_drop` | 0.02 | IoU (at 0.3 and 0.5 thresholds) may not drop by more than this |

Additionally, SAL improvement must hold on at least ⌈2/3⌉ of evaluated horizons.

These thresholds govern the `decision.pass` field. A failing gate sets `WARN-GATE-001` and
`promotion_decision = "hold_challenger"`, but does not constitute a hard stop by itself.

---

## 6. `science_grade` Promotion Checklist

The following conditions must all be true before a spread model may be promoted to
`science_grade`. This is the authoritative definition of what `science_grade` means.

### Hard stop clearance (all required)

- [ ] `STOP-GEO-001` — region grid passes CRS + resolution contract
- [ ] `STOP-SOURCE-001` — all four source categories declared in eval config
- [ ] `STOP-CONTRACT-001` — challenger feature schema exactly matches `CANONICAL_V2_CHANNELS`
- [ ] `STOP-CAL-001` — calibrator artifact present (V3 models only)
- [ ] `gate_report.pass = true` and `promotion_decision = "promote_challenger"`

### Science debt closure (all required)

- [ ] `SCI-DEBT-EXT-GT` — external ground-truth validation completed and documented
- [ ] `SCI-DEBT-DM-SAL` — DM significance and SAL thresholds locked in promotion policy
- [ ] `SCI-DEBT-DRIFT` — drift monitoring operational in production with defined alert thresholds

### Stage-gap warning resolution (all required)

- [ ] `WARN-FIRE-001` — cross-sensor VIIRS inter-calibration validated on held-out season
- [ ] `WARN-WX-001` — bias corrector artifact validated and enforced for all promoted models
- [ ] `WARN-WX-002` — high-resolution weather source available for all operational domains
- [ ] `WARN-TERR-001` — DEM provenance declared per region in `terrain_features_metadata`

### Metric bar (all required)

- [ ] `weighted_bss_improvement > 0.03` across all evaluation horizons
- [ ] No single horizon BSS regression below −0.005
- [ ] SAL improvement on ≥ ⌈2/3⌉ horizons
- [ ] DM test p < 0.05 on all horizons
- [ ] PR-AUC drop < 0.01; IoU drop < 0.02

### Lineage and provenance (all required)

- [ ] All four `data_sources` keys declared and match actual inference sources
- [ ] `SpreadForecast.probabilities` carries complete lineage attributes (see
  [Spread Data Sources §6](spread_data_sources.md))
- [ ] `runtime_contract.json` present in model artifact directory
- [ ] Champion-challenger eval run against a held-out season not used in training

---

## 7. Gate Report Lifecycle

```
eval_spread_champion_challenger.py
  │
  ├─ pre-flight geo check ──────────────────── STOP-GEO-001 (skips data collection if triggered)
  │
  ├─ _collect_comparison_arrays()
  │    └─ build_spread_inputs() per reference time
  │
  ├─ compute_recommendation()   ──────────────── metric gate thresholds (§5)
  │
  └─ _build_stage_governance()
       ├─ STOP-STAGE-001
       ├─ STOP-GEO-001
       ├─ STOP-SOURCE-001
       ├─ STOP-CAL-001
       ├─ STOP-CONTRACT-001
       ├─ WARN-MVP-BSS-001
       ├─ WARN-GATE-001
       └─ science_debt_register (mvp_operational only)
            ├─ SCI-DEBT-EXT-GT
            ├─ SCI-DEBT-DM-SAL
            └─ SCI-DEBT-DRIFT

Output: gate_report.json, summary.json, summary.csv, decision.md
```
