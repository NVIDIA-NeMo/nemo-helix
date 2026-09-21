# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the orphaned-test-gate check."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_test_gates.py"
_spec = importlib.util.spec_from_file_location("check_test_gates", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
check_test_gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_test_gates)

GATED = '''
import os
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("{var}"), reason="opt-in")

def test_thing():
    pass
'''


def _tree(tmp_path: Path, *, gate: str, workflow: str = "") -> tuple[Path, Path]:
    tests = tmp_path / "pkg" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_gated.py").write_text(GATED.format(var=gate), encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text(workflow or "jobs:\n  build:\n    steps: []\n", encoding="utf-8")
    return tmp_path, workflows


def test_a_gate_no_workflow_sets_is_reported(tmp_path: Path) -> None:
    root, workflows = _tree(tmp_path, gate="RUN_NOTHING_SETS_THIS")

    orphans = check_test_gates.orphaned_gates(root, workflows)

    assert "RUN_NOTHING_SETS_THIS" in orphans
    assert orphans["RUN_NOTHING_SETS_THIS"][0].name == "test_gated.py"


def test_a_gate_a_workflow_sets_is_accepted(tmp_path: Path) -> None:
    # The whole point: turning the variable on in CI is the fix, and the check has to notice.
    root, workflows = _tree(
        tmp_path, gate="RUN_SOMETHING", workflow="jobs:\n  build:\n    env:\n      RUN_SOMETHING: '1'\n"
    )

    assert check_test_gates.orphaned_gates(root, workflows) == {}


def test_an_allowed_gate_is_not_reported(tmp_path: Path) -> None:
    allowed = next(iter(check_test_gates.ALLOWED_UNSET))
    root, workflows = _tree(tmp_path, gate=allowed)

    assert check_test_gates.orphaned_gates(root, workflows) == {}


def test_installed_packages_are_not_our_gates(tmp_path: Path) -> None:
    # A dependency's own test suite is full of opt-in gates we neither own nor can set.
    vendored = tmp_path / ".venv" / "lib" / "site-packages" / "dep" / "tests"
    vendored.mkdir(parents=True)
    (vendored / "test_dep.py").write_text(GATED.format(var="SOME_DEP_FLAG"), encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    assert check_test_gates.orphaned_gates(tmp_path, workflows) == {}


def test_a_file_that_does_not_parse_does_not_fail_the_check(tmp_path: Path) -> None:
    # Fixtures and templates share the test_*.py name. Refusing to run over them would make the
    # check fail for a reason unrelated to gates.
    tests = tmp_path / "tests"
    tests.mkdir(parents=True)
    (tests / "test_broken.py").write_text("def (:\n", encoding="utf-8")
    (tests / "test_binary.py").write_bytes(b"\xa4\xa4\xa4")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    assert check_test_gates.orphaned_gates(tmp_path, workflows) == {}


def test_an_infrastructure_probe_is_not_a_gate(tmp_path: Path) -> None:
    # `skipif(not docker_available())` skips visibly when the infra is absent and runs when present,
    # so it is not the failure mode this guards. Only env-var opt-ins are.
    tests = tmp_path / "tests"
    tests.mkdir(parents=True)
    (tests / "test_probe.py").write_text(
        "import pytest\n\npytestmark = pytest.mark.skipif(not _docker(), reason='no docker')\n", encoding="utf-8"
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    assert check_test_gates.orphaned_gates(tmp_path, workflows) == {}


if __name__ == "__main__":
    # Mirrors test_ngc_metadata.py: `-c os.devnull` and `--confcutdir` detach this from the
    # repository's pytest.ini and conftest, which pull in workspace packages a standalone script
    # does not install.
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
                "-c",
                os.devnull,
                "-p",
                "no:cacheprovider",
                "--confcutdir",
                str(Path(__file__).parent),
            ]
        )
    )
