# Task: Fix confidence display bug in Review Queue panel

**Location:** `ui/src/components/ReviewQueuePanel.tsx`, `ui/src/types/api.ts`
**Impact:** High — destroys operator trust immediately; trivial fix
**Maturity target:** `mvp_operational`

## Problem

Confidence values in the Review Queue render as e.g. `5000%` instead of `50%`. The raw FIRMS confidence value (a float, e.g. `50.0`) is being multiplied by 100 somewhere it has already been normalised, or the percentage symbol is appended to an already-scaled value.

This is a display-only bug but it is the first thing any operator sees, and it makes the entire panel look broken.

## Proposed Solution

Locate where `confidence_max` from `payload_json` is formatted for display in `ReviewQueuePanel.tsx`. The value stored in the database is the raw FIRMS confidence float (0–100 range). It should be displayed as-is with a `%` suffix, not multiplied again.

If the value is stored as a 0–1 fraction anywhere in the pipeline, that needs to be confirmed first before deciding where to apply the fix — check `ml/denoiser_inference_v2.py` where `payload_json` is assembled to confirm the stored unit.

## Acceptance Criteria

- [ ] Confidence renders correctly (e.g. raw value `50.0` → display `50%`, not `5000%`)
- [ ] Fix is applied at the display layer, not by mutating stored data
- [ ] Verified against at least one real queue item in the database
- [ ] No other numeric fields in the panel (FRP, score) are affected

## Notes

- Do not change the stored value in `payload_json` — only fix the display formatting
- While here, confirm that `event_score` (0–1 float) is also displayed correctly and not multiplied by 100
