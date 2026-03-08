from types import SimpleNamespace
from unittest.mock import patch

from ingest.firms_ingest import _assert_batch_denoiser_complete


def _cfg(version: str, shadow_mode: bool) -> SimpleNamespace:
    return SimpleNamespace(
        denoiser_pipeline_version=version,
        denoiser_shadow_mode=shadow_mode,
    )


@patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
def test_v2_shadow_mode_checks_v2_columns_only(mock_count) -> None:
    _assert_batch_denoiser_complete(101, config=_cfg("v2", True))
    cols = tuple(mock_count.call_args.kwargs["columns"])
    assert "event_id" in cols
    assert "front_id" in cols
    assert "denoised_score" not in cols
    assert "is_noise" not in cols


@patch("ingest.firms_ingest.repository.count_rows_with_null_columns_for_batch", return_value=0)
def test_v2_live_mode_checks_v2_and_legacy_columns(mock_count) -> None:
    _assert_batch_denoiser_complete(101, config=_cfg("v2", False))
    cols = tuple(mock_count.call_args.kwargs["columns"])
    assert "event_id" in cols
    assert "denoised_score" in cols
    assert "is_noise" in cols
