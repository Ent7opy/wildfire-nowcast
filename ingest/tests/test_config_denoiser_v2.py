import pytest

from ingest.config import FIRMSIngestSettings


def test_denoiser_v2_pipeline_fields_defaults() -> None:
    cfg = FIRMSIngestSettings(
        FIRMS_MAP_KEY="dummy",
    )

    assert cfg.denoiser_pipeline_version == "v1"
    assert cfg.denoiser_shadow_mode is False
    assert 0.0 <= cfg.denoiser_strong_filter_threshold <= 1.0
    assert 0.0 <= cfg.denoiser_downweight_threshold <= 1.0


def test_denoiser_pipeline_version_validation() -> None:
    cfg = FIRMSIngestSettings(
        FIRMS_MAP_KEY="dummy",
        DENOISER_PIPELINE_VERSION="v2",
    )
    assert cfg.denoiser_pipeline_version == "v2"


def test_denoiser_pipeline_version_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        FIRMSIngestSettings(
            FIRMS_MAP_KEY="dummy",
            DENOISER_PIPELINE_VERSION="v3",
        )


def test_denoiser_uncertainty_band_validates_order() -> None:
    with pytest.raises(ValueError):
        FIRMSIngestSettings(
            FIRMS_MAP_KEY="dummy",
            DENOISER_UNCERTAINTY_BAND_LOW=0.6,
            DENOISER_UNCERTAINTY_BAND_HIGH=0.4,
        )
