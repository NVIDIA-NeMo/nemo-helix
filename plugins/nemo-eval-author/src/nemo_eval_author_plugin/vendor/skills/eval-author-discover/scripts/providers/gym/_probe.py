# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect whether a working NeMo Gym CLI is installed.

Standard library only. Discovery never imports Gym: Judge and Solve run the
``gym`` command, which brings its own interpreter. So the Probe asks that command
for its version, and Gym can live in its own environment, apart from Harbor.

The Probe passes when Harbor or Gym is installed. Gym alone is enough to explore,
and to judge and solve Gym manifests. Harbor still judges every task directory,
including the Gym extension tasks that discovery converts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _checks import ADVISORY, PASS, WARN, CheckResult, check

# `gym env validate <name>` and `gym env test <name>`, which Judge and Solve run, need this version.
MIN_VERSION = (0, 6, 0)
_VERSION_PATTERN = re.compile(r"NeMo Gym v?(\d+)\.(\d+)\.(\d+)")
_VERSION_TIMEOUT_SEC = 60
INSTALL_HINT = "Install NeMo Gym 0.6.0 (`pip install nemo-gym==0.6.0`, Python 3.13.14 or newer), then rerun discovery."


def probe() -> dict:
    """Report the Gym CLI next to this interpreter, or else on PATH, and the version it reports."""
    beside = Path(sys.executable).parent / "gym"
    cli = str(beside) if beside.is_file() else shutil.which("gym")
    version = _version(cli) if cli else None
    return {
        "gym_available": version is not None and _parse(version) >= MIN_VERSION,
        "gym_version": version,
        "gym_cli": cli if version else None,
    }


def _version(cli: str) -> str | None:
    """Return the version ``gym --version`` reports, or None when the command is not NeMo Gym or fails."""
    try:
        # Gym writes a Hydra outputs/ folder into its working directory, so give it a throwaway one.
        with tempfile.TemporaryDirectory(prefix="eval-author-gym-") as scratch:
            completed = subprocess.run(
                [cli, "--version"],
                cwd=scratch,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=_VERSION_TIMEOUT_SEC,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    found = _VERSION_PATTERN.search(completed.stdout) if completed.returncode == 0 else None
    return ".".join(found.groups()) if found else None


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def is_available(runtime: dict) -> bool:
    """Return whether a NeMo Gym CLI new enough for Judge and Solve is installed."""
    return bool(runtime["gym_available"])


def unavailable_reason(runtime: dict) -> str:
    """Say why Gym cannot judge manifests: missing, or too old."""
    if runtime["gym_version"]:
        return "NeMo Gym {} is older than {}.".format(runtime["gym_version"], ".".join(map(str, MIN_VERSION)))
    return "NeMo Gym is not installed."


def probe_checks(runtime: dict, *, report_missing: bool) -> list[CheckResult]:
    """Report Gym availability; report its absence only when it matters."""
    if runtime["gym_available"]:
        return [
            check(
                "gym",
                "runtime",
                PASS,
                "NeMo Gym {} is installed at {}.".format(runtime["gym_version"], runtime["gym_cli"]),
                severity=ADVISORY,
            )
        ]
    if not report_missing:
        return []
    return [
        check(
            "gym",
            "runtime",
            WARN,
            unavailable_reason(runtime),
            severity=ADVISORY,
            hint=INSTALL_HINT,
        )
    ]
