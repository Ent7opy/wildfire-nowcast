# Task: Review Queue — Phase 1 triage UX (labels, sort, split sections)

**Location:** `ui/src/components/ReviewQueuePanel.tsx`, `api/routes/internal.py`
**Impact:** High — makes the queue operationally usable without any new data
**Maturity target:** `mvp_operational`

## Problem

The Review Queue presents a flat chronological list of 200 items mixing two fundamentally different situations under opaque ML-internal labels. An operator has no way to know what needs attention first, what "HARD BYPASS" means, or why they should trust their own decision.

Three specific problems, all fixable without new data sources:

1. **Labels are ML jargon.** `fail_closed_hard_bypass` and `fail_closed_or_uncertainty` mean nothing to an incident commander. They need plain-language operational framing.
2. **Sort order is wrong.** Chronological recency is not operational urgency. A high-FRP HARD BYPASS event sitting in the queue for 3 hours is more dangerous than a fresh low-FRP uncertainty event.
3. **The two queue types are visually identical.** HARD BYPASS items are safety escalations (assume fire, confirm and act). UNCERTAINTY items are judgment calls (model couldn't decide, human must). They warrant different visual treatment and different operator mindsets.

## Proposed Solution

### 1. Plain-language label translation

Replace the reason chip text:

| Raw reason | Display label | Colour / tone |
|---|---|---|
| `fail_closed_hard_bypass` | **High-Energy Alert** | Red / urgent |
| `fail_closed_or_uncertainty` | **Model Uncertain** | Amber / caution |

Add a tooltip or subtitle on each chip explaining the operational meaning in one sentence:
- High-Energy Alert: *"Exceptionally high fire energy or confirmed forest conditions — treated as fire until reviewed."*
- Model Uncertain: *"Classifier score was borderline — human judgment required."*

### 2. Urgency-based sort

Sort order (descending priority):
1. `fail_closed_hard_bypass` items first
2. Within each group, sort by `frp_max DESC` (highest energy first)
3. Within same FRP tier, sort by `created_at ASC` (oldest unresolved first — most overdue)

This can be done client-side on the existing payload data; no API change required.

### 3. Two-section layout

Split the panel into two collapsible sections:
- **🔴 High-Energy Alerts** (HARD BYPASS items) — collapsed count shown in header
- **🟡 Uncertain Detections** (UNCERTAINTY items) — collapsed count shown in header

Both sections open by default. Empty state for each section should say something useful, e.g. *"No high-energy alerts pending"*.

### 4. Score display cleanup

The raw `event_score` float (e.g. `0.547`) is meaningless to operators. Replace it with a plain-language confidence band:

| Score range | Display |
|---|---|
| < 0.35 | Low confidence |
| 0.35–0.45 | Below threshold |
| 0.45–0.55 | Borderline |
| 0.55–0.65 | Above threshold |
| > 0.65 | High confidence |

For HARD BYPASS items the score is forced to `1.0` by the model — do not display a score band for these; it would be misleading.

## Acceptance Criteria

- [ ] `fail_closed_hard_bypass` items render as "High-Energy Alert" in red
- [ ] `fail_closed_or_uncertainty` items render as "Model Uncertain" in amber
- [ ] Each label has a one-sentence tooltip explaining operational meaning
- [ ] Items are sorted by: HARD BYPASS first → FRP descending → oldest first within tier
- [ ] Panel has two collapsible sections with counts in headers
- [ ] Raw `event_score` float is replaced with a plain-language confidence band
- [ ] HARD BYPASS items do not display a score band
- [ ] Existing "Confirm Fire" / "Mark as Noise" actions are unchanged

## Notes

- All changes in this task are purely UI — no API changes, no database changes
- Do not remove the raw values from the payload; they may be useful in a details/expand view later
- This task is a prerequisite for Task 12 (context enrichment) and Task 13 (decision panel)
