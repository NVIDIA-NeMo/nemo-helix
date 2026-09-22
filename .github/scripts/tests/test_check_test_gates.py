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
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_test_gates.py"
_spec = importlib.util.spec_from_file_location("check_test_gates", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
check_test_gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_test_gates)

GATED = """
import os
import pytest

pytestmark = pytest.mark.skipif(not {read}, reason="opt-in")

def test_thing():
    pass
"""

#: The three spellings of one gate. A check that knows only the first would miss the others, and the
#: tests they guard would keep skipping in CI while the check reported nothing.
ENV_READS = (
    'os.environ.get("{var}")',
    'os.environ["{var}"]',
    'os.getenv("{var}")',
)


def _tree(tmp_path: Path, *, gate: str, workflow: str = "", read: str = ENV_READS[0]) -> tuple[Path, Path]:
    tests = tmp_path / "pkg" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_gated.py").write_text(GATED.format(read=read.format(var=gate)), encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text(workflow or "jobs:\n  build:\n    steps: []\n", encoding="utf-8")
    return tmp_path, workflows


@pytest.mark.parametrize("read", ENV_READS)
def test_a_gate_no_workflow_sets_is_reported(tmp_path: Path, read: str) -> None:
    root, workflows = _tree(tmp_path, gate="RUN_NOTHING_SETS_THIS", read=read)

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
    (vendored / "test_dep.py").write_text(GATED.format(read=ENV_READS[0].format(var="SOME_DEP_FLAG")), encoding="utf-8")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    assert check_test_gates.orphaned_gates(tmp_path, workflows) == {}


def test_the_other_pytest_filename_pattern_is_scanned(tmp_path: Path) -> None:
    # pytest.ini sets `python_files = test_*.py *_test.py`. Scanning one pattern would leave gates in
    # the other invisible here while pytest still skipped those tests.
    tests = tmp_path / "pkg" / "tests"
    tests.mkdir(parents=True)
    (tests / "thing_test.py").write_text(
        GATED.format(read=ENV_READS[0].format(var="RUN_OTHER_PATTERN")), encoding="utf-8"
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    assert "RUN_OTHER_PATTERN" in check_test_gates.orphaned_gates(tmp_path, workflows)


def test_the_generated_sdk_is_skipped_but_first_party_sdk_tooling_is_not(tmp_path: Path) -> None:
    """The exclusion is the generated tree, not every directory that happens to be called ``sdk``.

    ``sdk/python`` is Stainless output -- a gate there would not be ours to set. But
    ``tools/nemo-platform-sdk-tools/tests/sdk`` is first-party, and a component-name match would
    exempt it too.
    """
    generated = tmp_path / "sdk" / "python" / "nemo-platform" / "tests"
    generated.mkdir(parents=True)
    (generated / "test_generated.py").write_text(
        GATED.format(read=ENV_READS[0].format(var="RUN_GENERATED")), encoding="utf-8"
    )
    first_party = tmp_path / "tools" / "nemo-platform-sdk-tools" / "tests" / "sdk"
    first_party.mkdir(parents=True)
    (first_party / "thing_test.py").write_text(
        GATED.format(read=ENV_READS[0].format(var="RUN_FIRST_PARTY")), encoding="utf-8"
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yaml").write_text("jobs: {}\n", encoding="utf-8")

    orphans = check_test_gates.orphaned_gates(tmp_path, workflows)

    assert "RUN_GENERATED" not in orphans
    assert "RUN_FIRST_PARTY" in orphans


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
