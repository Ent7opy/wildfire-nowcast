from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from api.db import get_engine
from ml.denoiser.eventize import EventizeParams, eventize_detections

_TEST_SOURCE = "TEST_EVENTIZE_ASSOC"


@pytest.fixture(scope="module")
def eventize_schema_available() -> None:
    check_stmt = text(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ANY(:tables)
        """
    )
    required = ["fire_fronts", "fire_events", "fire_event_memberships"]

    with get_engine().begin() as conn:
        present = int(conn.execute(check_stmt, {"tables": required}).scalar_one() or 0)
    if present != len(required):
        pytest.skip("Eventize v2 schema not available. Run migrations before integration tests.")


@pytest.fixture
def test_batch(eventize_schema_available):
    with get_engine().begin() as conn:
        batch_id = int(
            conn.execute(
                text(
                    """
                    INSERT INTO ingest_batches (source, source_uri, started_at, status)
                    VALUES (:source, :uri, NOW(), 'running')
                    RETURNING id
                    """
                ),
                {"source": _TEST_SOURCE, "uri": "test://eventize"},
            ).scalar_one()
        )

    yield batch_id

    with get_engine().begin() as conn:
        conn.execute(text("DELETE FROM fire_detections WHERE ingest_batch_id = :batch_id"), {"batch_id": batch_id})
        conn.execute(text("DELETE FROM fire_events WHERE source = :source"), {"source": _TEST_SOURCE})
        conn.execute(text("DELETE FROM fire_fronts WHERE source = :source"), {"source": _TEST_SOURCE})
        conn.execute(text("DELETE FROM ingest_batches WHERE id = :batch_id"), {"batch_id": batch_id})


def _insert_detection(
    *,
    batch_id: int,
    acq_time: datetime,
    lat: float,
    lon: float,
    suffix: str,
    false_source_masked: bool = False,
    persistence_score: float = 0.2,
) -> int:
    stmt = text(
        """
        INSERT INTO fire_detections (
            geom,
            lat,
            lon,
            acq_time,
            sensor,
            source,
            confidence,
            brightness,
            frp,
            scan,
            track,
            ingest_batch_id,
            confidence_score,
            dedupe_hash,
            false_source_masked,
            persistence_score
        )
        VALUES (
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
            :lat,
            :lon,
            :acq_time,
            'VIIRS',
            :source,
            80.0,
            320.0,
            12.0,
            1.0,
            1.0,
            :batch_id,
            0.8,
            :dedupe_hash,
            :false_source_masked,
            :persistence_score
        )
        RETURNING id
        """
    )
    with get_engine().begin() as conn:
        return int(
            conn.execute(
                stmt,
                {
                    "lat": lat,
                    "lon": lon,
                    "acq_time": acq_time,
                    "source": _TEST_SOURCE,
                    "batch_id": batch_id,
                    "dedupe_hash": f"eventize_assoc_{batch_id}_{suffix}",
                    "false_source_masked": bool(false_source_masked),
                    "persistence_score": float(persistence_score),
                },
            ).scalar_one()
        )


def _fetch_ids(batch_id: int) -> dict[int, tuple[str | None, str | None]]:
    stmt = text(
        """
        SELECT id, front_id, event_id
        FROM fire_detections
        WHERE ingest_batch_id = :batch_id
        ORDER BY id
        """
    )
    with get_engine().begin() as conn:
        rows = conn.execute(stmt, {"batch_id": batch_id}).mappings().all()
    return {int(r["id"]): (r["front_id"], r["event_id"]) for r in rows}


@pytest.mark.integration
def test_eventize_idempotency(test_batch: int) -> None:
    t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    _insert_detection(batch_id=test_batch, acq_time=t0, lat=42.0, lon=23.0, suffix="idem_1")
    _insert_detection(
        batch_id=test_batch,
        acq_time=t0 + timedelta(minutes=20),
        lat=42.001,
        lon=23.001,
        suffix="idem_2",
    )

    params = EventizeParams()
    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    first = _fetch_ids(test_batch)

    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    second = _fetch_ids(test_batch)

    assert first == second


@pytest.mark.integration
def test_eventize_event_stability_with_future_append(test_batch: int) -> None:
    t0 = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc)
    d1 = _insert_detection(batch_id=test_batch, acq_time=t0, lat=43.0, lon=24.0, suffix="stable_1")
    d2 = _insert_detection(
        batch_id=test_batch,
        acq_time=t0 + timedelta(hours=2),
        lat=43.002,
        lon=24.002,
        suffix="stable_2",
    )

    params = EventizeParams(event_link_radius_m=15000.0, event_max_gap_days=11)
    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    before = _fetch_ids(test_batch)

    _insert_detection(
        batch_id=test_batch,
        acq_time=t0 + timedelta(days=1),
        lat=43.003,
        lon=24.003,
        suffix="stable_3",
    )
    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    after = _fetch_ids(test_batch)

    assert before[d1][1] == after[d1][1]
    assert before[d2][1] == after[d2][1]


@pytest.mark.integration
def test_eventize_static_source_separation(test_batch: int) -> None:
    t0 = datetime(2026, 2, 3, 9, 0, tzinfo=timezone.utc)
    dynamic_id = _insert_detection(
        batch_id=test_batch,
        acq_time=t0,
        lat=44.0,
        lon=25.0,
        suffix="split_dynamic",
        false_source_masked=False,
        persistence_score=0.2,
    )
    static_id = _insert_detection(
        batch_id=test_batch,
        acq_time=t0 + timedelta(minutes=10),
        lat=44.0005,
        lon=25.0005,
        suffix="split_static",
        false_source_masked=True,
        persistence_score=0.95,
    )

    params = EventizeParams(strict_static_split=True)
    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    ids = _fetch_ids(test_batch)

    assert ids[dynamic_id][1] is not None
    assert ids[static_id][1] is not None
    assert ids[dynamic_id][1] != ids[static_id][1]


@pytest.mark.integration
def test_eventize_continuity_chain_merges_event(test_batch: int) -> None:
    t0 = datetime(2026, 2, 4, 6, 0, tzinfo=timezone.utc)
    ids = [
        _insert_detection(batch_id=test_batch, acq_time=t0, lat=45.0000, lon=26.0000, suffix="chain_1"),
        _insert_detection(
            batch_id=test_batch,
            acq_time=t0 + timedelta(days=2),
            lat=45.0450,
            lon=26.0000,
            suffix="chain_2",
        ),
        _insert_detection(
            batch_id=test_batch,
            acq_time=t0 + timedelta(days=4),
            lat=45.0900,
            lon=26.0000,
            suffix="chain_3",
        ),
    ]

    params = EventizeParams(event_link_radius_m=6000.0, event_max_gap_days=11)
    eventize_detections(get_engine(), batch_id=test_batch, params=params)
    mapped = _fetch_ids(test_batch)

    event_ids = {mapped[did][1] for did in ids}
    assert len(event_ids) == 1
