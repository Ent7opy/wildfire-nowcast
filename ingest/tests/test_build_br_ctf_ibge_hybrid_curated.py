from __future__ import annotations

from scripts.build_br_ctf_ibge_hybrid_curated import (
    _find_column,
    _norm_cnpj,
    _norm_col,
    _norm_mun,
    _parse_dt,
    _to_category_code,
)


def test_norm_col_and_find_column() -> None:
    cols = ["Razão Social", "CNPJ/CPF", "Código Município IBGE", "Situação"]
    assert _norm_col("Razão Social") == "razaosocial"
    assert _find_column(cols, ["razaosocial"]) == "Razão Social"
    assert _find_column(cols, ["cnpjcpf"]) == "CNPJ/CPF"
    assert _find_column(cols, ["inexistente"]) is None


def test_normalizers() -> None:
    assert _norm_cnpj("12.345.678/0001-99") == "12345678000199"
    assert _norm_cnpj("123") == ""
    assert _norm_mun("3550308") == "3550308"
    assert _norm_mun("35503") == ""


def test_category_and_date_parsing() -> None:
    assert _to_category_code("4.0") == "4"
    assert _to_category_code("Categoria 5") == "5"
    assert _parse_dt("2025-01-31") == "2025-01-31"
    assert _parse_dt("31/01/2025") == "2025-01-31"
    assert _parse_dt("") == ""
