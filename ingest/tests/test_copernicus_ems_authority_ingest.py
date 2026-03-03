from __future__ import annotations

import ingest.copernicus_ems_authority_ingest as mod


def test_build_records_from_aois_uses_wkt_extent() -> None:
    detail = {
        "code": "EMSR999",
        "eventTime": "2025-08-12T09:40:00",
        "activationTime": "2025-08-12T13:56:00",
        "lastUpdate": "2025-08-13T12:00:00",
        "closed": True,
        "aois": [
            {
                "name": "Test AOI",
                "number": 1,
                "extent": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
            }
        ],
    }
    rows = mod._build_records(detail, run_id="run_x", source_last_edit=None)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_object_id"] == "EMSR999:AOI:1"
    assert row["tier"] == "silver"
    assert row["is_authoritative"] is True
    assert row["poly_featurestatus"] == "Certified"
    assert row["geom_wkt"].startswith("POLYGON")


def test_build_records_falls_back_to_activation_extent() -> None:
    detail = {
        "code": "EMSR1000",
        "eventTime": "2025-07-10T10:00:00",
        "activationTime": "2025-07-10T12:00:00",
        "closed": False,
        "extent": "POLYGON ((2 2, 3 2, 3 3, 2 2))",
        "aois": [],
    }
    rows = mod._build_records(detail, run_id="run_y", source_last_edit=None)
    assert len(rows) == 1
    assert rows[0]["source_object_id"] == "EMSR1000:ACTIVATION_EXTENT"
    assert rows[0]["poly_featurestatus"] == "Approved"
