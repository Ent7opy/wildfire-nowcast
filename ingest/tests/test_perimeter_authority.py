"""Tests for perimeter authority tier ranking and conflict resolution."""

from __future__ import annotations

import ingest.perimeter_authority as mod


def test_tier_rank_ordering() -> None:
    """Gold < silver < bronze < blocked (lower rank = higher authority)."""
    assert mod.tier_rank("gold") < mod.tier_rank("silver")
    assert mod.tier_rank("silver") < mod.tier_rank("bronze")
    assert mod.tier_rank("bronze") < mod.tier_rank("blocked")


def test_tier_rank_none_is_lowest() -> None:
    assert mod.tier_rank(None) == mod.DEFAULT_RANK
    assert mod.tier_rank(None) > mod.tier_rank("blocked")


def test_tier_rank_unknown_is_lowest() -> None:
    assert mod.tier_rank("unknown_tier") == mod.DEFAULT_RANK


def test_tier_rank_case_insensitive() -> None:
    assert mod.tier_rank("Gold") == mod.tier_rank("gold")
    assert mod.tier_rank("SILVER") == mod.tier_rank("silver")


def test_should_overwrite_equal_tier() -> None:
    """Same tier can overwrite (e.g., NIFC re-publishing its own perimeter)."""
    assert mod.should_overwrite("gold", "gold") is True
    assert mod.should_overwrite("silver", "silver") is True


def test_should_overwrite_higher_authority() -> None:
    """Higher authority (lower rank) can overwrite lower authority."""
    assert mod.should_overwrite("gold", "silver") is True
    assert mod.should_overwrite("gold", "bronze") is True
    assert mod.should_overwrite("silver", "bronze") is True


def test_should_not_overwrite_lower_authority() -> None:
    """Lower authority (higher rank) must not overwrite higher authority."""
    assert mod.should_overwrite("silver", "gold") is False
    assert mod.should_overwrite("bronze", "gold") is False
    assert mod.should_overwrite("bronze", "silver") is False
    assert mod.should_overwrite("blocked", "gold") is False


def test_should_overwrite_none_existing() -> None:
    """When existing tier is None (no existing record), always overwrite."""
    assert mod.should_overwrite("gold", None) is True
    assert mod.should_overwrite("blocked", None) is True


def test_should_overwrite_none_incoming() -> None:
    """When incoming tier is None (unknown), it gets DEFAULT_RANK."""
    assert mod.should_overwrite(None, "gold") is False
    assert mod.should_overwrite(None, "blocked") is False


def test_fire_perimeters_source_tier_nifc_is_gold() -> None:
    assert mod.FIRE_PERIMETERS_SOURCE_TIER["NIFC"] == "gold"


def test_log_authority_conflict_emits_warning() -> None:
    """Ensure the log function does not raise and emits a message."""
    # Smoke test: should not raise.
    mod.log_authority_conflict(
        source="test_source",
        source_id="test_id_123",
        incoming_tier="bronze",
        existing_tier="gold",
    )


def test_nifc_parse_feature_includes_authority_tier() -> None:
    """Verify _parse_feature in nifc_perimeters_ingest includes authority_tier."""
    from ingest.nifc_perimeters_ingest import _parse_feature

    feature = {
        "attributes": {
            "attr_IrwinID": "test-irwin-123",
            "attr_FireDiscoveryDateTime": 1704067200000,  # 2024-01-01 UTC
            "attr_IncidentName": "Test Fire",
            "attr_CalculatedAcres": 100.0,
        },
        "geometry": {
            "rings": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
    }
    parsed = _parse_feature(feature)
    assert parsed is not None
    assert parsed["authority_tier"] == "gold"
    assert parsed["source"] == "NIFC"
