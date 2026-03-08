from __future__ import annotations

import ingest.coverage_mask_from_authoritative as mod


def test_jurisdiction_specs_include_three_authorities() -> None:
    specs = mod._JURISDICTION_SPECS
    profiles = {s.authority_profile for s in specs}
    assert profiles == {"wfigs_us", "cwfis_ca", "copernicus_eu"}
    assert len(specs) == 3


def test_validity_windows_match_policy() -> None:
    by_profile = {s.authority_profile: s for s in mod._JURISDICTION_SPECS}

    us = by_profile["wfigs_us"]
    assert us.valid_from.isoformat() == "2025-01-01T00:00:00+00:00"
    assert us.valid_to.isoformat() == "2025-08-31T23:59:59+00:00"

    ca = by_profile["cwfis_ca"]
    assert ca.valid_to.isoformat() == "2024-12-04T23:59:59+00:00"

    eu = by_profile["copernicus_eu"]
    assert eu.valid_from.isoformat() == "2025-08-08T00:00:00+00:00"
    assert eu.valid_to.isoformat() == "2025-08-15T23:59:59+00:00"
