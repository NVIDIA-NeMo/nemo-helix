#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate an audit-spec ``audit.md`` file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _markdown import AuditMarkdownError  # noqa: E402
from _schema import AuditEnvironmentError, AuditSpecError, item_counts, load_audit_spec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True, help="audit.md file to validate")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    try:
        spec = load_audit_spec(args.audit)
    except AuditEnvironmentError as exc:
        _print({"valid": None, "error_type": "environment", "error": str(exc)}, compact=args.compact)
        return 2
    except (AuditMarkdownError, AuditSpecError) as exc:
        _print({"valid": False, "error_type": "audit_spec", "error": str(exc)}, compact=args.compact)
        return 1

    _print(
        {
            "valid": True,
            "schema": spec["schema"],
            "agent": spec["agent"],
            "status": spec["status"],
            "item_count": len(spec["items"]),
            "item_counts": item_counts(spec),
        },
        compact=args.compact,
    )
    return 0


def _print(payload: dict, *, compact: bool) -> None:
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":") if compact else None, indent=None if compact else 2)
    )


if __name__ == "__main__":
    raise SystemExit(main())
