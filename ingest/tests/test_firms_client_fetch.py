from __future__ import annotations

import httpx
import pytest

from ingest.firms_client import FIRMSClientError, fetch_csv_rows


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "", url: str = "https://example.test/firms") -> None:
        self.status_code = status_code
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", self.url)
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("error", request=request, response=response)


def test_fetch_csv_rows_raises_for_invalid_map_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ingest.firms_client.httpx.get",
        lambda *args, **kwargs: _FakeResponse(text="Invalid MAP_KEY."),
    )

    with pytest.raises(FIRMSClientError, match="Invalid MAP_KEY"):
        fetch_csv_rows(
            map_key="bad-key",
            source="VIIRS_SNPP_NRT",
            bbox="-180,-90,180,90",
            day_range=1,
            timeout_seconds=5.0,
            max_retries=0,
        )


def test_fetch_csv_rows_raises_for_unexpected_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ingest.firms_client.httpx.get",
        lambda *args, **kwargs: _FakeResponse(text="foo,bar\n1,2\n"),
    )

    with pytest.raises(FIRMSClientError, match="missing columns"):
        fetch_csv_rows(
            map_key="any-key",
            source="VIIRS_SNPP_NRT",
            bbox="-180,-90,180,90",
            day_range=1,
            timeout_seconds=5.0,
            max_retries=0,
        )


def test_fetch_csv_rows_accepts_valid_header_with_zero_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ingest.firms_client.httpx.get",
        lambda *args, **kwargs: _FakeResponse(
            text="latitude,longitude,acq_date,acq_time,confidence,frp\n"
        ),
    )

    rows = fetch_csv_rows(
        map_key="any-key",
        source="VIIRS_SNPP_NRT",
        bbox="-180,-90,180,90",
        day_range=1,
        timeout_seconds=5.0,
        max_retries=0,
    )
    assert rows == []
