# Task: Review Queue — Close the learning loop into denoiser retraining

**Location:** `ml/denoiser/label_v2.py`, `api/fires/repo.py`, `ml/denoiser_inference_v2.py`
**Impact:** High — without this, operator feedback is audit log only; the model never improves from human corrections
**Maturity target:** `science_grade`
**Depends on:** Tasks 11–14 (queue must be producing trustworthy labels before feeding them into training)

## Problem

Operator resolutions (`resolved_notes = "confirmed_fire" | "marked_noise"`) are stored in `denoiser_review_queue` but are never read by the labeling pipeline. The `label_v2.py` labeling step derives training labels exclusively from authoritative perimeters and rule-based heuristics. Human corrections from the review queue are wasted signal.

This means:
- The model never learns from the cases it was most uncertain about
- Systematic misclassifications (e.g. a specific region where the model consistently over-suppresses) persist indefinitely
- The review queue never shrinks over time as the model improves

## Proposed Solution

### Step 1: Map resolved queue items to training labels

Add a query in `label_v2.py` (or a new `label_review_queue.py` module) that reads resolved review queue items and emits them as `denoiser_labels_v2` rows:

```python
# Pseudo-logic
SELECT
    rq.event_id,
    rq.fire_detection_id,
    CASE
        WHEN rq.resolved_notes = 'confirmed_fire' THEN 1
        WHEN rq.resolved_notes = 'marked_noise'   THEN 0
    END AS label,
    rq.resolved_at,
    rq.resolved_by,
    'review_queue' AS label_source
FROM denoiser_review_queue rq
WHERE rq.status = 'resolved'
  AND rq.resolved_notes IN ('confirmed_fire', 'marked_noise')
  AND rq.resolved_by NOT LIKE 'auto:%'  -- exclude perimeter auto-closes (already covered by perimeter labels)
  AND NOT EXISTS (
      SELECT 1 FROM denoiser_labels_v2 dl
      WHERE dl.event_id = rq.event_id
        AND dl.label_source = 'review_queue'
  )
```

Emit these as `denoiser_labels_v2` rows with `label_source = 'review_queue'` and a configurable `label_weight` (default: 0.8 — slightly lower than authoritative perimeter labels at 1.0, to account for operator error rate).

### Step 2: Conflict resolution with existing labels

If an event already has a label from an authoritative perimeter AND a conflicting operator label, the perimeter wins. Log the conflict for QA inspection — these are cases where either the operator was wrong or the perimeter boundary is imprecise.

Conflict log schema (new table or append to existing audit log):
```
label_conflict(event_id, perimeter_label, operator_label, resolved_by, created_at)
```

### Step 3: Operator accuracy tracking

Track per-operator (by `resolved_by` value) the agreement rate between their labels and eventual ground truth (authoritative perimeters, ingested post-resolution):

```python
# A background job run weekly
operator_accuracy = (
    confirmed_fires_later_covered_by_perimeter / total_confirmed_fire_labels
)
noise_accuracy = (
    marked_noise_not_covered_by_perimeter / total_marked_noise_labels
)
```

Store per-operator accuracy in a `operator_label_quality` table. Use this to weight labels in training: high-accuracy operators get weight 1.0, low-accuracy operators get weight 0.5.

Initially, `resolved_by = "operator"` (no identity) — all operator labels share a single accuracy score. This is a known limitation; individual accountability requires auth (out of scope for now).

### Step 4: Systematic miss detection

A batch job (weekly or post-retrain) queries for geographic clusters where operator confirmations significantly outnumber model passes — meaning the model is systematically suppressing real fires in a specific region or condition. Surface these as `WARNING` log entries with the cluster centroid and condition summary. These should trigger a model recalibration investigation, not an automatic fix.

## Acceptance Criteria

- [ ] Resolved review queue items with `resolved_notes IN ('confirmed_fire', 'marked_noise')` are emitted as `denoiser_labels_v2` rows with `label_source = 'review_queue'`
- [ ] Auto-resolved items (`resolved_by LIKE 'auto:%'`) are excluded — they are already covered by perimeter labels
- [ ] Label weight for review queue labels is configurable (default 0.8); perimeter labels remain weight 1.0
- [ ] Conflicts between operator labels and perimeter labels are logged to an audit table; perimeter label wins
- [ ] A weekly job computes per-`resolved_by` accuracy against subsequent perimeter ground truth and writes to `operator_label_quality`
- [ ] The `denoiser-label-v2` pipeline Makefile target ingests review queue labels as part of its standard run (not a separate optional step)
- [ ] A test asserts that a resolved `confirmed_fire` queue item produces a label=1 row in `denoiser_labels_v2`
- [ ] A test asserts that a resolved `marked_noise` queue item produces a label=0 row
- [ ] A test asserts that auto-resolved items do NOT produce duplicate label rows

## Notes

- **Do not implement this before Tasks 11–13 are complete.** The quality of training labels depends on operators having enough context to make correct decisions. Feeding low-quality labels into the model is worse than not feeding them at all.
- The `label_weight` field must already exist or be added to `denoiser_labels_v2` — check the schema before implementing
- This task targets `science_grade` maturity. It does not need to be live for `mvp_operational`. Add a `WARN` log if resolved queue labels exist but the feedback loop is not yet enabled, so the gap is visible during ops review.
- Systematic miss detection (Step 4) is the lowest priority item in this task — implement Steps 1–3 first
