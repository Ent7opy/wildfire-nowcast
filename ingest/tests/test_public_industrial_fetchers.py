from __future__ import annotations

from scripts.fetch_br_ibama_public_api import (
    ProbeResult,
    _is_machine_accessible_arcgis,
    _is_machine_accessible_wfs,
    _looks_like_block_page,
)
from scripts.fetch_cn_mee_public_permits import (
    parse_detail_coords,
    parse_license_rows,
    parse_temp_report_key,
    parse_total_pages,
)


def test_cn_parsers_extract_expected_fields() -> None:
    html = """
    <input type="hidden" name="tempReportKey"  value="062744af007e4e128acedf821c44ee60" />
    <script>var totalPages = 37671;</script>
    <tr>
      <td title="山东省">山东省</td>
      <td title="潍坊市">潍坊市</td>
      <td title="91370783797337187J001C">91370783797337187J001C</td>
      <td title="寿光市中冶水务有限公司">寿光市中冶水务有限公司</td>
      <td title="污水处理及其再生利用">污水处理及其再生利用</td>
      <td title="2025-12-16至2030-12-15">2025-12-16至2030-12-15</td>
      <td title="2025-12-16">2025-12-16</td>
      <td title="重点管理">重点管理</td>
      <td><a href="/perxxgkinfo/xkgkAction!xkgk.action?xkgk=getxxgkContent&dataid=57dcc4151b8647f088c4dd710e4baf35">view</a></td>
    </tr>
    """

    assert parse_temp_report_key(html) == "062744af007e4e128acedf821c44ee60"
    assert parse_total_pages(html) == 37671

    rows = parse_license_rows(html)
    assert len(rows) == 1
    row = rows[0]
    assert row.dataid == "57dcc4151b8647f088c4dd710e4baf35"
    assert row.permit_id == "91370783797337187J001C"
    assert row.facility_name == "寿光市中冶水务有限公司"


def test_cn_detail_coords_parser() -> None:
    detail = '<input type="hidden" id="longitude" value="118.75955"/><input type="hidden" id="latitude" value="37.18323"/>'
    coords = parse_detail_coords(detail)
    assert coords is not None
    assert coords.longitude == 118.75955
    assert coords.latitude == 37.18323


def test_br_probe_block_and_access_classification() -> None:
    blocked = ProbeResult(
        url="https://example.test",
        status_code=403,
        content_type="text/html",
        body_sha256="abc",
        sample="<html>Attention Required! | Cloudflare</html>",
        error=None,
        fetched_at="2026-03-01T00:00:00+00:00",
    )
    assert _looks_like_block_page(blocked.sample)
    assert not _is_machine_accessible_wfs(blocked)
    assert not _is_machine_accessible_arcgis(blocked)

    open_wfs = ProbeResult(
        url="https://example.test/wfs",
        status_code=200,
        content_type="text/xml",
        body_sha256="def",
        sample="<wfs:WFS_Capabilities version=\"1.1.0\"></wfs:WFS_Capabilities>",
        error=None,
        fetched_at="2026-03-01T00:00:00+00:00",
    )
    assert _is_machine_accessible_wfs(open_wfs)

    open_arcgis = ProbeResult(
        url="https://example.test/arcgis",
        status_code=200,
        content_type="application/json",
        body_sha256="ghi",
        sample='{"currentVersion":10.8,"folders":[],"services":[]}',
        error=None,
        fetched_at="2026-03-01T00:00:00+00:00",
    )
    assert _is_machine_accessible_arcgis(open_arcgis)
