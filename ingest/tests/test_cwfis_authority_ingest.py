from __future__ import annotations

import ingest.cwfis_authority_ingest as mod


def test_to_multipolygon_geojson_from_polygon() -> None:
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    out = mod._to_multipolygon_geojson(geom)
    assert out is not None
    assert out["type"] == "MultiPolygon"
    assert len(out["coordinates"]) == 1


def test_extract_record_builds_authoritative_nbac_row() -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]]]},
        "properties": {
            "__gid": 123,
            "year": 2024,
            "nfireid": 456,
            "admin_area": "AB",
            "ag_sdate": "2024-08-01Z",
            "capdate": "2024-08-15Z",
        },
    }
    row = mod._extract_record(
        feature,
        source_profile="cwfis_nbac_historical",
        source_layer="public:nbac",
        run_id="run_x",
        profile=mod.CWFIS_PROFILES["cwfis_nbac_historical"],
    )
    assert row is not None
    assert row["source_object_id"] == "123"
    assert row["tier"] == "gold"
    assert row["is_authoritative"] is True
    assert row["poly_featurestatus"] == "Certified"
