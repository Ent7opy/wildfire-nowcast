import pytest

from ingest.config import FIRMSIngestSettings


def test_firms_window_defaults() -> None:
    cfg = FIRMSIngestSettings(FIRMS_MAP_KEY="dummy")
    assert cfg.firms_initial_lookback_minutes == 360
    assert cfg.firms_incremental_lookback_minutes == 30


def test_firms_window_validation_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        FIRMSIngestSettings(
            FIRMS_MAP_KEY="dummy",
            FIRMS_INITIAL_LOOKBACK_MINUTES=0,
        )

    with pytest.raises(ValueError):
        FIRMSIngestSettings(
            FIRMS_MAP_KEY="dummy",
            FIRMS_INCREMENTAL_LOOKBACK_MINUTES=-1,
        )


def test_firms_watermark_grace_rejects_negative() -> None:
    with pytest.raises(ValueError):
        FIRMSIngestSettings(
            FIRMS_MAP_KEY="dummy",
            FIRMS_WATERMARK_GRACE_MINUTES=-5,
        )
