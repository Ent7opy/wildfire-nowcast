"""Taxonomy crosswalk helpers for industrial thermal potential classification (TPC)."""

from __future__ import annotations

from typing import Any


_ALLOWED_TAXONOMIES = {"NACE", "NAICS", "ANZSIC", "GBT4754", "fuel_type", "other"}

# TPC range [0, 1], where 1.0 is maximal thermal signature potential.
_FUEL_TPC = {
    "coal": 0.95,
    "gas": 0.80,
    "oil": 0.85,
    "petcoke": 0.90,
    "biomass": 0.65,
    "waste": 0.70,
    "nuclear": 0.40,
    "hydro": 0.10,
    "wind": 0.05,
    "solar": 0.05,
}


def normalize_taxonomy(raw: str | None) -> str:
    value = (raw or "").strip()
    if value in _ALLOWED_TAXONOMIES:
        return value
    upper = value.upper()
    if upper in _ALLOWED_TAXONOMIES:
        return upper
    if value.lower() in {"fuel", "fuel_type"}:
        return "fuel_type"
    return "other"


def _nace_tpc(code: str) -> float:
    code = code.strip()
    if code.startswith("24"):
        return 1.00  # basic metals / steel
    if code.startswith("23.51") or code.startswith("23"):
        return 0.95  # cement / mineral
    if code.startswith("19"):
        return 0.90  # refinery / coke
    if code.startswith("20"):
        return 0.80  # chemicals
    if code.startswith("35"):
        return 0.70  # utilities
    return 0.50


def _naics_tpc(code: str) -> float:
    digits = "".join(ch for ch in code if ch.isdigit())
    if digits.startswith("331"):
        return 1.00  # primary metals
    if digits.startswith("327"):
        return 0.95  # nonmetallic mineral (cement etc.)
    if digits.startswith("324"):
        return 0.90  # petroleum refining
    if digits.startswith("211"):
        return 0.85  # oil and gas extraction
    if digits.startswith("221"):
        return 0.70  # utilities
    return 0.50


def _anzsic_tpc(code: str) -> float:
    digits = "".join(ch for ch in code if ch.isdigit())
    if digits.startswith("2110"):
        return 1.00  # iron and steel
    if digits.startswith("060"):
        return 0.90  # coal mining and associated processing
    if digits.startswith("1811"):
        return 0.85  # industrial gas
    if digits.startswith("2231"):
        return 0.75  # boilers / heavy metal containers
    return 0.50


def _gbt_tpc(code: str) -> float:
    digits = "".join(ch for ch in code if ch.isdigit())
    if digits.startswith("31"):
        return 1.00  # ferrous metals
    if digits.startswith("30"):
        return 0.95  # non-ferrous / materials
    if digits.startswith("26"):
        return 0.90  # chemical raw materials
    if digits.startswith("25"):
        return 0.85  # petroleum processing / coking
    if digits.startswith("44"):
        return 0.70  # power and heat
    return 0.50


def _fuel_tpc(code: str) -> float:
    token = code.strip().lower()
    for key, value in _FUEL_TPC.items():
        if key in token:
            return value
    return 0.65


def infer_thermal_potential_class(
    *,
    sector_code: str | None,
    sector_taxonomy: str | None,
    facility_type: str | None = None,
) -> float:
    taxonomy = normalize_taxonomy(sector_taxonomy)
    code = (sector_code or "").strip()

    if taxonomy == "NACE":
        return _nace_tpc(code)
    if taxonomy == "NAICS":
        return _naics_tpc(code)
    if taxonomy == "ANZSIC":
        return _anzsic_tpc(code)
    if taxonomy == "GBT4754":
        return _gbt_tpc(code)
    if taxonomy == "fuel_type":
        return _fuel_tpc(code or (facility_type or ""))

    fallback = (facility_type or "").lower()
    if any(token in fallback for token in ("steel", "smelt", "cement", "kiln", "flare", "refinery")):
        return 0.90
    return 0.50


def as_iso3(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if len(token) == 3 and token.isalpha():
        return token
    return None
