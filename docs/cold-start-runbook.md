# Cold-Start Measurement Runbook

How to measure SPEC-A-prime acceptance criterion #2 — *cold start to first watch ≤ 5 min on a clean browser* — against the production deployment, with a real polygon and a real human.

## 1. Goal

Measure the wall-clock time, in seconds, for a brand-new stewardship user to:

1. Land on `/`
2. Sign up
3. Create their first AOI
4. Receive the watch-confirmed email

The 5-minute target covers steps 1–4. Step 5 (backfill match → brief generation) and step 6 (brief email) are observed but not gated by the 5-min budget — they depend on whether FIRMS reported any detection inside the polygon in the prior 24 h.

Each segment is timed independently so the slowest leg is identifiable.

## 2. Pre-conditions

- **Browser**: Chromium-based, fresh private/incognito window. Disable ad-blockers and tracking-protection extensions for the session. Open DevTools → Network panel before navigating.
- **Account**: A brand-new identity. Either Clerk test mode (`+clerk_test` email convention) or a real fresh inbox you control. Do not re-use a prior session.
- **Polygon**: A real GeoJSON polygon from a stewardship-relevant public source. Suggested sources (operator must confirm URLs and licensing before use):
  - The Nature Conservancy preserve outlines — TNC operates a public ArcGIS portal; locate a preserve feature service and export one preserve as GeoJSON.
  - Natura 2000 site polygons — the European Environment Agency publishes the Natura 2000 dataset; download a single SCI/SPA polygon.
  - Fallback: a hand-drawn polygon over a known active-fire region, captured in geojson.io.
- **Region**: Pick an AOI overlapping a region with current or recent fire activity (e.g. summer Mediterranean, dry-season tropics, late-summer western US). For backfill testing specifically, use a polygon over a region known to have had detections in the last 24 h. Cross-check with the FIRMS public map before starting.
- **Resend**: Production mode, with a real inbox you can monitor in real time (have it open in another tab).
- **Server state**: `master` HEAD with Stage 9 deployed to Vercel. Confirm the deploy is green before T₀.

## 3. Timed steps

Start a stopwatch (or use `Date.now()` in DevTools console) at T₀.

| # | Event | Clock | Capture |
|---|---|---|---|
| 1 | Visit `/` | T₀ start | Page-load time from Network panel |
| 2 | Sign-up complete (Clerk redirect to app) | T₁ | T₁ − T₀ |
| 3 | AOI POST returns 201 | T₂ | T₂ − T₁; request/response from Network panel |
| 4 | Watch-confirmed email visible in inbox | T₃ | T₃ − T₂ (Resend → inbox latency) |
| 5 | First backfill poll triggers + matches | T₄ | Vercel logs for `aoi-backfill` job_run |
| 6 | Brief email visible in inbox | T₅ | T₅ − T₄ |

**Acceptance #2 pass condition**: T₃ − T₀ ≤ 300 s.

If step 5 yields zero matches, stop the cold-start clock at T₃ and record "no detections in 24 h window — backfill brief not exercised." That is not a failure of acceptance #2.

## 4. Capture format

Record everything in UTC, second precision. Required artifacts:

- Timestamp table (markdown, one row per step above).
- Screenshot of DevTools Network panel showing the AOI POST request + response headers + timing.
- Screenshot of inbox showing the watch-confirmed email with full received-time header.
- Vercel production logs for the time window `[T₀ − 30s, T₅ + 30s]`, filtered to the project. Save as `.log` attachment.
- If anything errored, the full request ID from the Vercel log line.

## 5. Post-measurement

Write up the result in `pm/research-log/YYYY-MM-DD-cold-start-measurement.md` using this template:

```markdown
# Cold-start measurement, <date>

- **AOI source**: <link or path to GeoJSON, with licensing note>
- **Region bucket**: <e.g. Iberian Peninsula, Aug fire season>
- **Active-fire context**: <FIRMS detections inside polygon in prior 24 h: count + source>
- **Browser / OS**: <Chromium version, OS>

## Timings (UTC)

| Step | Timestamp | Δ from prior | Δ from T₀ |
| ... | ... | ... | ... |

## Total cold-start (T₀ → T₃)

<seconds> — **PASS / FAIL** vs 300 s target.

## Surprises / breakage

<free-form>

## Artifacts

- `signals/<date>-cold-start-network.png`
- `signals/<date>-cold-start-vercel.log`
- ...
```

Raw artifacts go in `pm/signals/`.

## 6. Failure modes to watch for

- **Watch-confirmed email never arrives** — check Resend dashboard for the send event and the `notifications_log` table for a row with the AOI ID. If Resend shows delivered but inbox empty, suspect spam routing.
- **Backfill silently fails** — query `job_runs` where `job_name = 'aoi-backfill'` for the AOI ID. A missing row means the cron didn't fire; an `error` status means the job ran and failed (capture the error column).
- **Stage 5 JIT email lookup misses** — symptom: brief generated but addressed to `<userId>@pending.invalid`. Surfaces in `notifications_log` with reason `no_recipient_pending`. Means the Clerk → DB user-email reconciliation didn't run before the brief was queued.
- **ICNF / Mediterranean authority polygon null** — acceptable per Stage 8 deferral; brief still generates without the perimeter overlay. Note in writeup but not a failure.
- **Brief never generated** — check `aoi_briefs` for a row with the AOI ID. If absent, the gate rejected the candidate detection set; check the gate-decision log for the rejection reason.
