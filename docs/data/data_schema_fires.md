## Fire Detection Data Schema

This document describes the relational + spatial schema backing fire detections in Postgres/PostGIS. It complements `db-migrations.md` and should be the reference for API, ML, and UI contributors who need to understand how detection data is stored and queried.

---

### Tables

#### `ingest_batches`
Tracks each source import (e.g., FIRMS VIIRS daily file).

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `BIGSERIAL` | Primary key. |
| `source` | `VARCHAR(64)` | Short identifier such as `firms_viirs` or `firms_modis`. |
| `source_uri` | `TEXT` | Original file path/URL for traceability. |
| `started_at` / `completed_at` | `TIMESTAMPTZ` | Optional timestamps for ingest lifecycle. |
| `status` | `VARCHAR(32)` | Optional ingest pipeline state (`pending`, `succeeded`, `failed`, ...). |
| `record_count` | `INTEGER` | Number of detections loaded in this batch (mirrors `records_inserted`). |
| `records_fetched` | `INTEGER` | Raw CSV rows retrieved from the provider. |
| `records_inserted` | `INTEGER` | Rows that landed in `fire_detections`. |
| `records_skipped_duplicates` | `INTEGER` | Rows skipped because they already existed for the same source/time/location. |
| `metadata` | `JSONB` | Free-form details (e.g., processing options, checksum). |
| `created_at` | `TIMESTAMPTZ` | Defaults to `now()` for auditing. |

#### `fire_detections`
Stores every detection point with the key FIRMS-style attributes plus ML/ingest metadata.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | `BIGSERIAL` | Primary key. |
| `geom` | `geometry(Point, 4326)` | Canonical spatial column (used for bbox/AOI queries). |
| `lat` / `lon` | `DOUBLE PRECISION` | Denormalized for quick numeric access; constrained to valid ranges. |
| `acq_time` | `TIMESTAMPTZ` | Combined acquisition timestamp (UTC). |
| `sensor` | `VARCHAR(32)` | Sensor family identifier (e.g., `VIIRS`). |
| `source` | `VARCHAR(64)` | Dataset variant or provider label. |
| `dedupe_hash` | `VARCHAR(64)` | Rounded lat/lon + timestamp hash; unique per `source` for idempotent ingest. |
| `confidence`, `brightness`, `bright_t31`, `frp`, `scan`, `track` | `DOUBLE PRECISION` | High-value FIRMS metrics stored directly for filtering/sorting. |
| `raw_properties` | `JSONB` | All other raw fields (e.g., `satellite`, `daynight`, collection flags). Keeps ingestion lossless without inflating columns. |
| `denoised_score` | `DOUBLE PRECISION` | Output of hotspot denoiser (`0–1`). |
| `is_noise` | `BOOLEAN` | Optional denoiser classification flag. |
| `ingest_batch_id` | `BIGINT` FK | Optional link back to `ingest_batches.id` (on delete → `SET NULL`). |
| `created_at` | `TIMESTAMPTZ` | When the record was inserted (defaults to `now()`). |

---

### Indexing & Query Patterns

- **Spatial**: `ix_fire_detections_geom` (GiST on `geom`) accelerates bbox/contains/intersects filters coming from the API/UI map and ML AOIs.
- **Temporal**: `ix_fire_detections_acq_time` (B-tree) supports recent-history queries (`WHERE acq_time BETWEEN ...`).
- **Deduplication**: `uq_fire_detections_source_dedupe_hash` enforces uniqueness for a `(source, rounded lat/lon, acq_time)` tuple so the FIRMS importer can safely re-request overlapping windows.
- Combined filters (time window + bbox) leverage both indexes via PostgreSQL's bitmap index scans; keep predicates sargable to benefit (no `date_trunc` on the column, for example).

Example query for API/UI:

```sql
SELECT id,
       acq_time,
       sensor,
       confidence,
       frp,
       raw_properties
FROM fire_detections
WHERE acq_time BETWEEN now() - interval '24 hours' AND now()
  AND geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
  AND ST_Intersects(
        geom,
        ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
      )
ORDER BY acq_time DESC
LIMIT 500;
```

ML workflows can link detections back to their `ingest_batch_id` for reproducibility or use `raw_properties` to expose additional FIRMS metadata without schema churn.

---

### Future Extensions

- Additional FIRMS numeric columns can be promoted out of `raw_properties` if they become hot query fields.
- `ingest_batches` can store retry metadata or checksums once ingest pipelines are implemented.
- If AOI-based summarization becomes common, consider materialized views or partitioning by acquisition date to keep queries predictable at scale.

