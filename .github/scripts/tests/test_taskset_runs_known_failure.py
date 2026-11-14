# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the nightly auth-workload row's known-failure classifier.

The row passes only when every failure is NVBug 7083853's image-build 403 under workload identity; each test below
breaks one of those conditions and checks the row stays red.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "taskset_runs_known_failure.py"
_spec = importlib.util.spec_from_file_location("taskset_runs_known_failure", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
known = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = known
_spec.loader.exec_module(known)

DENIED = "PermissionDeniedError: HTTP 403: Not allowed to read entities in workspace 'ws'"


def junit(*messages: str) -> str:
    cases = "".join(f'<testcase><failure message="{m}">trace</failure></testcase>' for m in messages)
    return f'<testsuites><testsuite>{cases}<testcase name="ok"/></testsuite></testsuites>'


def pod(service_account: str = "nemo-helix-jobs", token: bool = True) -> dict[str, Any]:
    return {"name": "p", "service_account": service_account, "workload_token": token}


def evidence(error: str = DENIED, pods: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    failures = [{"job": "task-image-build-x-1", "status": "error", "error": error}]
    return {
        "invocation": "i",
        "test_1": {"image_build_failures": failures, "job_pods": [pod()] if pods is None else pods},
    }


IMAGE_BUILD_DENIED = f"AssertionError: [image-build] task-image-build-x-1: {DENIED}"


def test_the_image_build_403_under_workload_identity_is_known() -> None:
    assert known.known_failure(junit(IMAGE_BUILD_DENIED), [evidence()]) == "NVBug 7083853"


@pytest.mark.parametrize(
    ("junit_xml", "runs"),
    [
        (junit(), [evidence()]),  # nothing failed: not this row's business
        (junit(IMAGE_BUILD_DENIED, "AssertionError: [dispatch] run-1 still running"), [evidence()]),
        (junit("AssertionError: [submit] exited 1"), [evidence()]),
        (junit(IMAGE_BUILD_DENIED), []),  # no evidence
        (junit(IMAGE_BUILD_DENIED), [evidence(error="RuntimeError: out of disk")]),
        (junit(IMAGE_BUILD_DENIED), [evidence(pods=[])]),  # identity not shown
        (junit(IMAGE_BUILD_DENIED), [evidence(pods=[pod(service_account="default")])]),
        (junit(IMAGE_BUILD_DENIED), [evidence(pods=[pod(), pod(token=False)])]),
    ],
    ids=[
        "green",
        "another-failure",
        "other-stage",
        "no-evidence",
        "other-error",
        "no-pods",
        "submitter-sa",
        "no-token",
    ],
)
def test_anything_else_is_not_known(junit_xml: str, runs: list[dict[str, Any]]) -> None:
    assert known.known_failure(junit_xml, runs) is None


def test_main_writes_the_marker_only_when_known(tmp_path: Path) -> None:
    (tmp_path / "junit.xml").write_text(junit(IMAGE_BUILD_DENIED))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / f"taskset-runs-e2e-{'0' * 32}.json").write_text(json.dumps(evidence()))
    marker = tmp_path / "logs" / "known-failure.json"

    assert known.main([str(tmp_path / "junit.xml"), str(tmp_path / "logs"), str(marker)]) == 0
    assert json.loads(marker.read_text())["bug"] == "NVBug 7083853"

    marker.unlink()
    assert known.main([str(tmp_path / "missing.xml"), str(tmp_path / "logs"), str(marker)]) == 1
    assert not marker.exists()


if __name__ == "__main__":
    sys.exit(
        pytest.main(
            [
                __file__,
                "-q",
                "-o",
                "addopts=",
                "-c",
                os.devnull,
                "-p",
                "no:cacheprovider",
                "--confcutdir",
                str(Path(__file__).parent),
            ]
        )
    )
