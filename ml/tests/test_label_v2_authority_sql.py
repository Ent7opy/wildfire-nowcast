import pytest

from ml.denoiser.label_v2 import _build_perimeter_sql


def test_build_perimeter_sql_authoritative_contains_governance_filters() -> None:
    positive_sql, far_low_sql = _build_perimeter_sql(perimeter_source="authoritative_perimeters")
    positive_text = str(positive_sql)
    far_low_text = str(far_low_sql)

    assert "JOIN authoritative_perimeters ap" in positive_text
    assert "ap.poly_featurestatus IN ('Approved', 'Certified')" in positive_text
    assert "ap.poly_featureaccess = 'Public'" in positive_text
    assert "ap.poly_isvisible = 'Yes'" in positive_text
    assert "COALESCE(ap.attr_isquarantined, 0) = 0" in positive_text
    assert "ap.tier IN ('silver', 'gold')" in positive_text
    assert "FROM authoritative_perimeters ap" in far_low_text


def test_build_perimeter_sql_legacy_uses_fire_perimeters() -> None:
    positive_sql, far_low_sql = _build_perimeter_sql(perimeter_source="fire_perimeters")
    assert "JOIN fire_perimeters fp" in str(positive_sql)
    assert "FROM fire_perimeters fp" in str(far_low_sql)


def test_build_perimeter_sql_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unsupported perimeter_source"):
        _build_perimeter_sql(perimeter_source="unknown")
