"""CLI utilities for model registry register/promote/rollback flows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.model_registry import (  # noqa: E402
    list_active_models,
    promote_model,
    register_model,
    rollback_model,
)


def _load_metrics(value: str | None) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}

    raw = value.strip()
    if raw.startswith("@"):
        file_path = Path(raw[1:])
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    path_candidate = Path(raw)
    if path_candidate.exists() and path_candidate.is_file():
        with path_candidate.open("r", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(raw)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model registry operations")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="Register a model artifact")
    register.add_argument("--family", required=True, help="Model family: denoiser|spread")
    register.add_argument("--artifact", required=True, help="Artifact URI/path")
    register.add_argument(
        "--metrics",
        default=None,
        help="Metrics JSON string or path (or @path).",
    )
    register.add_argument("--status", default="registered", help="Registry status")
    register.add_argument("--model-id", default=None, help="Optional explicit model_id")
    register.add_argument(
        "--id-only",
        action="store_true",
        help="Print only model_id to stdout.",
    )

    promote = sub.add_parser("promote", help="Promote model to active champion")
    promote.add_argument("--family", required=True, help="Model family: denoiser|spread")
    promote.add_argument("--model-id", required=True, help="Registered model_id")
    promote.add_argument("--by", default=None, help="Promoted-by identifier")
    promote.add_argument("--notes", default=None, help="Promotion notes")

    rollback = sub.add_parser("rollback", help="Rollback to previous promotion")
    rollback.add_argument("--family", required=True, help="Model family: denoiser|spread")
    rollback.add_argument("--by", default=None, help="Operator identifier")
    rollback.add_argument("--notes", default="rollback", help="Rollback notes")

    sub.add_parser("active", help="Show active promoted models")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "register":
        metrics_json = _load_metrics(args.metrics)
        model_id = register_model(
            family=args.family,
            artifact_uri=args.artifact,
            metrics_json=metrics_json,
            status=args.status,
            model_id=args.model_id,
        )
        if args.id_only:
            print(model_id)
            return
        _print_json({
            "action": "register",
            "model_id": model_id,
            "family": args.family,
            "artifact": args.artifact,
        })
        return

    if args.command == "promote":
        promoted_by = args.by or os.getenv("USER") or os.getenv("USERNAME")
        active = promote_model(
            family=args.family,
            model_id=args.model_id,
            promoted_by=promoted_by,
            notes=args.notes,
        )
        _print_json({"action": "promote", "active": active})
        return

    if args.command == "rollback":
        promoted_by = args.by or os.getenv("USER") or os.getenv("USERNAME")
        active = rollback_model(
            family=args.family,
            promoted_by=promoted_by,
            notes=args.notes,
        )
        _print_json({"action": "rollback", "active": active})
        return

    if args.command == "active":
        _print_json({"models": list_active_models()})
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
