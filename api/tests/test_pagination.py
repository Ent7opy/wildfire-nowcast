"""Unit tests for api.pagination."""

import base64
import json
from datetime import datetime, timezone

import pytest

from api.pagination import build_page, decode_cursor, encode_cursor


# ---------------------------------------------------------------------------
# encode_cursor / decode_cursor round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_plain_fields():
    cursor = encode_cursor(id=42, offset=100)
    data = decode_cursor(cursor)
    assert data == {"id": 42, "offset": 100}


def test_roundtrip_datetime_becomes_aware():
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    cursor = encode_cursor(t=dt, id=7)
    data = decode_cursor(cursor)
    assert data["t"] == dt
    assert data["t"].tzinfo is not None


def test_roundtrip_naive_datetime_gets_utc():
    naive = datetime(2024, 6, 1, 12, 0, 0)
    cursor = encode_cursor(t=naive, id=3)
    data = decode_cursor(cursor)
    assert data["t"].tzinfo == timezone.utc


def test_cursor_is_opaque_string():
    cursor = encode_cursor(id=1)
    assert isinstance(cursor, str)
    assert len(cursor) > 0


def test_decode_invalid_cursor_raises_value_error():
    with pytest.raises(ValueError, match="Invalid cursor"):
        decode_cursor("not-valid-base64!!!")


def test_decode_invalid_timestamp_raises_value_error():
    bad = base64.urlsafe_b64encode(json.dumps({"t": "not-a-date"}).encode()).decode()
    with pytest.raises(ValueError, match="Invalid cursor timestamp"):
        decode_cursor(bad)


def test_no_t_field_not_parsed_as_datetime():
    cursor = encode_cursor(id=99)
    data = decode_cursor(cursor)
    assert "t" not in data


# ---------------------------------------------------------------------------
# build_page
# ---------------------------------------------------------------------------

class _Row(dict):
    """dict subclass — verifies build_page doesn't require plain dict input."""


def _make_rows(n: int) -> list[_Row]:
    return [_Row(id=i, val=f"v{i}") for i in range(n)]


def test_build_page_no_more():
    rows = _make_rows(3)
    page = build_page(rows, limit=5, cursor_fn=lambda r: encode_cursor(id=r["id"]))
    assert page["has_more"] is False
    assert page["next_cursor"] is None
    assert len(page["data"]) == 3
    assert page["limit"] == 5


def test_build_page_has_more():
    rows = _make_rows(6)
    page = build_page(rows, limit=5, cursor_fn=lambda r: encode_cursor(id=r["id"]))
    assert page["has_more"] is True
    assert page["next_cursor"] is not None
    assert len(page["data"]) == 5
    assert decode_cursor(page["next_cursor"])["id"] == 4


def test_build_page_empty():
    page = build_page([], limit=10, cursor_fn=lambda r: encode_cursor(id=r["id"]))
    assert page["has_more"] is False
    assert page["next_cursor"] is None
    assert page["data"] == []


def test_build_page_data_are_plain_dicts():
    rows = _make_rows(2)
    page = build_page(rows, limit=5, cursor_fn=lambda r: encode_cursor(id=r["id"]))
    for item in page["data"]:
        assert type(item) is dict
