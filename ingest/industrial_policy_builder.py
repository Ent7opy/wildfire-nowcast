"""Build and activate industrial masking policy rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

from ingest.repository import get_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs" / "industrial_policy_global_v1.yaml"


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Policy config must be a mapping")
    return payload


def build_policy(
    *,
    policy_version: str,
    strict_no_go: bool,
    gold_buffer_m: float,
    silver_buffer_min_m: float,
    silver_buffer_max_m: float,
    active_from: datetime | None,
    active_to: datetime | None,
) -> dict[str, Any]:
    stmt = text(
        """
        INSERT INTO industrial_mask_policies (
            policy_version,
            strict_no_go,
            gold_buffer_m,
            silver_buffer_min_m,
            silver_buffer_max_m,
            active_from,
            active_to,
            created_at,
            updated_at
        ) VALUES (
            :policy_version,
            :strict_no_go,
            :gold_buffer_m,
            :silver_buffer_min_m,
            :silver_buffer_max_m,
            COALESCE(:active_from, NOW()),
            :active_to,
            NOW(),
            NOW()
        )
        ON CONFLICT (policy_version) DO UPDATE SET
            strict_no_go = EXCLUDED.strict_no_go,
            gold_buffer_m = EXCLUDED.gold_buffer_m,
            silver_buffer_min_m = EXCLUDED.silver_buffer_min_m,
            silver_buffer_max_m = EXCLUDED.silver_buffer_max_m,
            active_from = EXCLUDED.active_from,
            active_to = EXCLUDED.active_to,
            updated_at = NOW()
        """
    )

    with get_engine().begin() as conn:
        conn.execute(
            stmt,
            {
                "policy_version": policy_version,
                "strict_no_go": bool(strict_no_go),
                "gold_buffer_m": float(gold_buffer_m),
                "silver_buffer_min_m": float(silver_buffer_min_m),
                "silver_buffer_max_m": float(silver_buffer_max_m),
                "active_from": active_from,
                "active_to": active_to,
            },
        )

    return {
        "policy_version": policy_version,
        "strict_no_go": bool(strict_no_go),
        "gold_buffer_m": float(gold_buffer_m),
        "silver_buffer_min_m": float(silver_buffer_min_m),
        "silver_buffer_max_m": float(silver_buffer_max_m),
        "active_from": active_from.isoformat() if active_from else None,
        "active_to": active_to.isoformat() if active_to else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update industrial masking policy")
    parser.add_argument("--config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--policy-version", default=None)
    parser.add_argument("--strict-no-go", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gold-buffer-m", type=float, default=None)
    parser.add_argument("--silver-buffer-min-m", type=float, default=None)
    parser.add_argument("--silver-buffer-max-m", type=float, default=None)
    parser.add_argument("--active-from", default=None)
    parser.add_argument("--active-to", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_config(Path(args.config).expanduser().resolve())

    policy_version = str(args.policy_version or config.get("policy_version") or "").strip()
    if not policy_version:
        raise SystemExit("policy_version is required")

    strict_no_go = (
        bool(args.strict_no_go)
        if args.strict_no_go is not None
        else bool(config.get("strict_no_go", True))
    )
    gold_buffer_m = float(args.gold_buffer_m if args.gold_buffer_m is not None else config.get("gold_buffer_m", 375.0))
    silver_buffer_min_m = float(
        args.silver_buffer_min_m
        if args.silver_buffer_min_m is not None
        else config.get("silver_buffer_min_m", 750.0)
    )
    silver_buffer_max_m = float(
        args.silver_buffer_max_m
        if args.silver_buffer_max_m is not None
        else config.get("silver_buffer_max_m", 1000.0)
    )

    active_from = _parse_dt(args.active_from or config.get("active_from"))
    active_to = _parse_dt(args.active_to or config.get("active_to"))

    if silver_buffer_max_m < silver_buffer_min_m:
        raise SystemExit("silver_buffer_max_m must be >= silver_buffer_min_m")

    summary = build_policy(
        policy_version=policy_version,
        strict_no_go=strict_no_go,
        gold_buffer_m=gold_buffer_m,
        silver_buffer_min_m=silver_buffer_min_m,
        silver_buffer_max_m=silver_buffer_max_m,
        active_from=active_from,
        active_to=active_to,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
