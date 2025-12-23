"""Helpers to safely compose SQL that can't be parameter-bound.

Table / schema identifiers cannot be passed as SQL bind parameters. If a caller
needs to choose a table name at runtime (e.g. via CLI), we must validate that
the input is strictly an identifier and not SQL syntax.
"""

from __future__ import annotations

import re

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_table_reference(table_ref: str) -> str:
    """
    Validate a SQL table reference used as an identifier (not a value).

    Accepts either:
      - table
      - schema.table

    Intentionally rejects quotes, whitespace, punctuation, and any SQL syntax.
    Returns the normalized reference (stripped).
    """
    if table_ref is None:
        raise ValueError("label_table must be a non-empty string")

    table_ref = table_ref.strip()
    if not table_ref:
        raise ValueError("label_table must be a non-empty string")

    parts = table_ref.split(".")
    if not (1 <= len(parts) <= 2):
        raise ValueError(
            "label_table must be either <table> or <schema>.<table> (no quoting)"
        )

    for part in parts:
        if not _SQL_IDENTIFIER_RE.fullmatch(part):
            raise ValueError(
                "label_table contains invalid characters; only letters, digits, and '_' "
                "are allowed, and it must start with a letter or '_'"
            )

    return table_ref

