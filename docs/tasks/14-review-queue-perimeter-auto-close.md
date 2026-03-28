# Task: Review Queue — Auto-close items matched to authoritative perimeters

**Location:** `ingest/orchestrator.py`, `api/routes/internal.py`, `ml/denoiser_inference_v2.py`
**Impact:** Medium — reduces queue volume for the clearest cases; frees operator attention for genuine uncertainty
**Maturity target:** `mvp_operational`

## Problem

Some items in the Review Queue have a clear answer available that doesn't require human judgment: if an authoritative fire perimeter (NIFC, WFIGS, CWFIS, Copernicus EMS) already covers the detection location, the detection is confirmed by a source more authoritative than the denoiser model. These items should not occupy operator time.

Currently, perimeter ingestion and denoiser review are independent pipelines. A HARD BYPASS event sitting in the queue might already have an authoritative perimeter intersecting it, but no code path makes that connection.

## Proposed Solution

### Auto-close on perimeter match

After each perimeter ingest cycle (NIFC, WFIGS, CWFIS, Copernicus), run a spatial cross-check:

```sql
-- Find open review queue items whose event centroid falls within a freshly ingested perimeter
SELECT rq.id, rq.event_id, p.source, p.perimeter_id
FROM denoiser_review_queue rq
JOIN fire_events fe ON fe.event_id = rq.event_id
JOIN fire_perimeters p ON ST_Within(fe.centroid_geom, p.geometry)
WHERE rq.status = 'open'
  AND p.ingested_at > NOW() - INTERVAL '2 hours'  -- only newly ingested perimeters
```

For each matched item, auto-resolve:
```python
resolved_by = f"auto:perimeter:{perimeter_source}"   # e.g. "auto:perimeter:wfigs"
resolved_notes = "confirmed_fire"
status = "resolved"
```

Log each auto-close with the perimeter source and perimeter ID for auditability.

### Where to wire this

Add a post-ingest hook in `orchestrator.py` after each perimeter ingest job completes. It should be a lightweight function: one spatial query, bulk update, structured log output. It must not block or slow down the ingest pipeline itself — run it as a fire-and-forget step after the perimeter commit.

### UI changes

In the Review Queue panel, auto-resolved items should be distinguishable from operator-resolved items if the operator views resolved history. Display `resolved_by` as *"Auto-confirmed: WFIGS perimeter"* rather than *"operator"*.

## Acceptance Criteria

- [ ] After each perimeter ingest cycle, open queue items with centroids inside a new perimeter are auto-resolved as `confirmed_fire`
- [ ] `resolved_by` is set to `auto:perimeter:<source>` (e.g. `auto:perimeter:wfigs`), not `operator`
- [ ] Auto-close is logged with perimeter source and perimeter ID
- [ ] Auto-close does not run synchronously inside the perimeter ingest transaction — it runs post-commit
- [ ] A test confirms that an open queue item with a centroid inside an ingested perimeter gets auto-resolved
- [ ] A test confirms that an item whose centroid is outside the perimeter does NOT get auto-resolved
- [ ] The auto-close mechanism works for all four perimeter sources: NIFC, WFIGS, CWFIS, Copernicus EMS

## Notes

- This is strictly additive — it does not change how items enter the queue, only how they exit
- Do not auto-close based on proximity to a perimeter; require actual spatial intersection (`ST_Within` or `ST_Intersects`) to avoid false positives at perimeter edges
- UNCERTAINTY items are eligible for auto-close just as HARD BYPASS items are — the perimeter is the authoritative signal regardless of the ML reason
- This is separate from the learning loop (Task 15) — auto-closes should feed into training labels the same way operator confirmations do
