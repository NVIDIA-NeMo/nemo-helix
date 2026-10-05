# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect whether Harbor can judge this repository.

Standard library only, and safe to import when Harbor is absent. Everything that
touches Harbor itself lives in ``_judge.py``, which ``discover.py`` imports only
after this module reports Harbor available.

This is the provider gate. A second provider adds its own probe next to this one
rather than changing ``discover.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path

from _checks import ADVISORY, FAIL, PASS, WARN, CheckResult, check

PROVIDER = "harbor"


def probe() -> dict:
    """Report the runtime without importing Harbor.

    ``find_spec`` answers whether Harbor is importable without paying the import
    or running its top-level code, which matters when the answer is no.
    """
    importable = False
    version = None
    try:
        importable = importlib.util.find_spec("harbor") is not None
    except (ImportError, ValueError):
        importable = False
    if importable:
        try:
            version = importlib.metadata.version("harbor")
        except importlib.metadata.PackageNotFoundError:
            version = None

    executable = Path(sys.executable).parent / "harbor"
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "harbor_importable": importable,
        "harbor_version": version,
        "harbor_cli": str(executable) if executable.is_file() else shutil.which("harbor"),
    }


def probe_checks(runtime: dict) -> list[CheckResult]:
    """Turn the probe into the checks that gate the validation ladder."""
    checks: list[CheckResult] = []
    if runtime["harbor_importable"]:
        checks.append(
            check(
                "harbor",
                "runtime",
                PASS,
                "Harbor {} is importable, so Harbor judges this report.".format(runtime["harbor_version"] or "?"),
            )
        )
    else:
        checks.append(
            check(
                "harbor",
                "runtime",
                FAIL,
                "Harbor is not importable, so nothing in this report is proven.",
                hint="Install Harbor into the interpreter running this script, then run discovery again.",
            )
        )
    if runtime["harbor_cli"] is None:
        checks.append(
            check(
                "harbor-cli",
                "runtime",
                WARN,
                "No harbor executable exists on PATH, so the CLI round trip cannot run.",
                severity=ADVISORY,
            )
        )
    return checks


def is_available(runtime: dict) -> bool:
    """Return whether the ladder can run against this runtime."""
    return bool(runtime["harbor_importable"])
