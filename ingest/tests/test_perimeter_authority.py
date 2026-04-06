"""Tests for perimeter authority tier ranking and conflict resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

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


def test_record_authority_conflict_executes_insert() -> None:
    """Verify record_authority_conflict calls conn.execute with expected params."""
    mock_conn = MagicMock()

    mod.record_authority_conflict(
        mock_conn,
        table_name="authoritative_perimeters",
        source="wfigs_current",
        source_id="OBJ-42",
        incoming_tier="bronze",
        existing_tier="gold",
        outcome="rejected",
        run_id="run_001",
        details={"reason": "test"},
    )

    mock_conn.execute.assert_called_once()
    args, kwargs = mock_conn.execute.call_args
    # First positional arg is the SQL text object.
    params = args[1]
    assert params["table_name"] == "authoritative_perimeters"
    assert params["source"] == "wfigs_current"
    assert params["source_id"] == "OBJ-42"
    assert params["incoming_tier"] == "bronze"
    assert params["existing_tier"] == "gold"
    assert params["outcome"] == "rejected"
    assert params["run_id"] == "run_001"
    assert '"reason"' in params["details"]


def test_authority_aware_upsert_rejects_lower_authority() -> None:
    """authority_aware_upsert should reject rows with lower authority than existing."""
    mock_conn = MagicMock()
    # Simulate an existing gold row for the lookup SELECT.
    mock_result = MagicMock()
    mock_result.fetchone.return_value = ("gold",)
    mock_conn.execute.return_value = mock_result

    fake_insert = MagicMock()
    rows = [
        {
            "source_profile": "test_profile",
            "source_layer": "test_layer",
            "source_object_id": "id_1",
            "tier": "bronze",
            "run_id": "run_x",
        },
    ]

    upserted, rejected = mod.authority_aware_upsert(
        mock_conn,
        insert_stmt=fake_insert,
        rows=rows,
        source_label="Test",
    )

    assert upserted == 0
    assert rejected == 1
    # The INSERT statement should never have been executed (no accepted rows).
    for c in mock_conn.execute.call_args_list:
        assert c[0][0] is not fake_insert


def test_authority_aware_upsert_accepts_equal_or_higher() -> None:
    """authority_aware_upsert should accept rows with equal or higher authority."""
    mock_conn = MagicMock()

    # First call: tier lookup returns silver; second call: the bulk INSERT.
    lookup_result = MagicMock()
    lookup_result.fetchone.return_value = ("silver",)
    insert_result = MagicMock()
    insert_result.rowcount = 1
    mock_conn.execute.side_effect = [
        lookup_result,   # SELECT tier (record_authority_conflict also calls execute)
        MagicMock(),     # INSERT into perimeter_authority_conflicts (accepted audit)
        insert_result,   # bulk INSERT ... ON CONFLICT
    ]

    fake_insert = MagicMock()
    rows = [
        {
            "source_profile": "test_profile",
            "source_layer": "test_layer",
            "source_object_id": "id_1",
            "tier": "gold",
            "run_id": "run_y",
        },
    ]

    upserted, rejected = mod.authority_aware_upsert(
        mock_conn,
        insert_stmt=fake_insert,
        rows=rows,
        source_label="Test",
    )

    assert upserted == 1
    assert rejected == 0


def test_authority_aware_upsert_new_row_no_existing() -> None:
    """When no existing row, accept without audit record."""
    mock_conn = MagicMock()

    lookup_result = MagicMock()
    lookup_result.fetchone.return_value = None  # no existing row
    insert_result = MagicMock()
    insert_result.rowcount = 1
    mock_conn.execute.side_effect = [lookup_result, insert_result]

    fake_insert = MagicMock()
    rows = [
        {
            "source_profile": "p",
            "source_layer": "l",
            "source_object_id": "id_new",
            "tier": "silver",
            "run_id": "run_z",
        },
    ]

    upserted, rejected = mod.authority_aware_upsert(
        mock_conn,
        insert_stmt=fake_insert,
        rows=rows,
        source_label="Test",
    )

    assert upserted == 1
    assert rejected == 0
    # Only 2 execute calls: SELECT tier + bulk INSERT (no audit for brand-new row).
    assert mock_conn.execute.call_count == 2


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
