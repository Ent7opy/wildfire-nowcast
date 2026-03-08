from ingest.firms_client import parse_detection_rows


def _base_row() -> dict[str, str]:
    return {
        "latitude": "58.68555",
        "longitude": "17.12984",
        "acq_date": "2026-02-15",
        "acq_time": "0",
        "instrument": "VIIRS",
        "confidence": "n",
        "frp": "0.82",
        "scan": "0.46",
        "track": "0.63",
        "daynight": "N",
    }


def test_parse_detection_rows_maps_viirs_thermal_aliases() -> None:
    row = {
        **_base_row(),
        "bright_ti4": "302.1",
        "bright_ti5": "259.32",
    }
    detections, summary = parse_detection_rows([row], "VIIRS_SNPP_NRT", ingest_batch_id=1)
    assert len(detections) == 1
    det = detections[0]
    assert det.brightness == 302.1
    assert det.bright_t31 == 259.32
    assert det.confidence == 50.0  # mapped from "n"
    assert summary.brightness_missing == 0


def test_parse_detection_rows_prefers_primary_modis_columns_over_aliases() -> None:
    row = {
        **_base_row(),
        "brightness": "330.0",
        "bright_t31": "295.0",
        "bright_ti4": "302.1",
        "bright_ti5": "259.32",
    }
    detections, _ = parse_detection_rows([row], "MODIS_NRT", ingest_batch_id=1)
    assert len(detections) == 1
    det = detections[0]
    assert det.brightness == 330.0
    assert det.bright_t31 == 295.0


def test_parse_detection_rows_handles_invalid_viirs_thermal_aliases() -> None:
    row = {
        **_base_row(),
        "bright_ti4": "not_a_number",
        "bright_ti5": "also_bad",
    }
    detections, summary = parse_detection_rows([row], "VIIRS_SNPP_NRT", ingest_batch_id=1)
    assert len(detections) == 1
    det = detections[0]
    assert det.brightness is None
    assert det.bright_t31 is None
    assert summary.brightness_missing == 1
