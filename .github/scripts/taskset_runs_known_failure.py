# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Is a failed auth-workload run of the taskset-runs e2e the known NVBug 7083853, and nothing else?

The nightly's ``auth-workload`` row runs Jobs under workload token exchange. Until 7083853 is fixed on ``main``,
its image-build Job can't read the task (``403 Not allowed to read entities``). The row reports that failure
instead of failing on it, so it can't hide a regression elsewhere. Known only when all of these hold:

- the JUnit has at least one failure, and every failure is ``[image-build]`` with that 403;
- the evidence recorded the failed image-build Job with that 403;
- the evidence recorded the Job pods, every one running as ``nemo-helix-jobs`` with the workload token, so the
  failure came from the token-exchange path.

Anything else (a setup failure, no evidence, another error) is not known, and the row stays red.

Usage: taskset_runs_known_failure.py <junit.xml> <evidence-dir> <marker-out>
Exits 0 and writes the marker JSON when known; exits 1 otherwise.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

BUG = "NVBug 7083853"
BUG_URL = "https://nvbugspro.nvidia.com/bug/7083853"
DENIED = re.compile(r"PermissionDeniedError: HTTP 403: Not allowed to read entities")
WORKLOAD_SERVICE_ACCOUNT = "nemo-helix-jobs"
EVIDENCE_NAME = re.compile(r"taskset-runs-e2e-[0-9a-f]{32}\.json$")


def failure_messages(junit: str) -> list[str]:
    """Each failed or errored test's message."""
    root = ET.fromstring(junit)
    return [
        f"{node.get('message', '')}\n{node.text or ''}"
        for case in root.iter("testcase")
        for node in case
        if node.tag in {"failure", "error"}
    ]


def _sections(evidence: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    return (section for section in evidence.values() if isinstance(section, Mapping))


def known_failure(junit: str, evidence: Sequence[Mapping[str, Any]]) -> str | None:
    """``BUG`` when the run failed only by it, as the module docstring defines; else None."""
    messages = failure_messages(junit)
    if not messages or not all("[image-build]" in m and DENIED.search(m) for m in messages):
        return None
    sections = [section for e in evidence for section in _sections(e) if section.get("image_build_failures")]
    if not sections:
        return None
    for section in sections:
        failures = section["image_build_failures"]
        pods = section.get("job_pods") or []
        if not all(DENIED.search(str(f.get("error"))) for f in failures):
            return None
        if not pods or not all(
            p.get("service_account") == WORKLOAD_SERVICE_ACCOUNT and p.get("workload_token") for p in pods
        ):
            return None
    return BUG


def main(argv: Sequence[str]) -> int:
    junit_path, evidence_dir, marker = (Path(arg) for arg in argv)
    junit = junit_path.read_text(encoding="utf-8") if junit_path.is_file() else ""
    evidence = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(evidence_dir.rglob("*.json"))
        if EVIDENCE_NAME.search(path.name)
    ]
    bug = known_failure(junit, evidence) if junit else None
    if bug is None:
        print("not a known failure")
        return 1
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"row": "auth-workload", "bug": bug, "url": BUG_URL}), encoding="utf-8")
    print(f"::warning::auth-workload failed only by the known {bug} ({BUG_URL}); reported, not failed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
