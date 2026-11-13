# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the JUnit no-skips check."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_junit_no_skips.py"
_spec = importlib.util.spec_from_file_location("check_junit_no_skips", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
check_junit_no_skips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_junit_no_skips)


def _report(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(f'<?xml version="1.0" encoding="utf-8"?>{body}', encoding="utf-8")
    return path


def _pytest_report(tmp_path: Path, *, tests: int, skipped: int) -> Path:
    """The layout pytest's ``--junitxml`` writes: one suite under a ``testsuites`` root."""
    return _report(
        tmp_path,
        f'<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" '
        f'skipped="{skipped}" tests="{tests}"></testsuite></testsuites>',
    )


def test_a_report_with_every_test_run_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _pytest_report(tmp_path, tests=2, skipped=0)

    assert check_junit_no_skips.main([str(report), "--label", "sandboxed Gym e2e"]) == 0
    assert capsys.readouterr().out == "sandboxed Gym e2e: 2 tests, none skipped\n"


def test_a_skipped_test_fails_and_says_how_many(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _pytest_report(tmp_path, tests=2, skipped=1)

    assert check_junit_no_skips.main([str(report), "--label", "sandboxed Gym e2e"]) == 1
    assert "sandboxed Gym e2e ran 2 tests and skipped 1" in capsys.readouterr().err


def test_a_report_with_no_tests_fails(tmp_path: Path) -> None:
    # A marker that selects nothing must not pass as "nothing skipped".
    assert check_junit_no_skips.main([str(_pytest_report(tmp_path, tests=0, skipped=0))]) == 1


def test_a_bare_testsuite_root_is_counted(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testsuite name="pytest" skipped="0" tests="3"></testsuite>')

    assert check_junit_no_skips.counts(report) == (3, 0)


def test_several_suites_are_summed(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="0"/><testsuite tests="3" skipped="1"/></testsuites>',
    )

    assert check_junit_no_skips.counts(report) == (5, 1)


def test_a_nested_suite_is_not_counted_twice(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="0"><testsuite tests="2" skipped="0"/></testsuite></testsuites>',
    )

    assert check_junit_no_skips.counts(report) == (2, 0)


@pytest.mark.parametrize("body", ["", "<testsuites><testsuite", '<testsuites><testsuite tests="two"/></testsuites>'])
def test_an_unreadable_report_fails_instead_of_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], body: str
) -> None:
    path = tmp_path / "report.xml"
    path.write_text(body, encoding="utf-8")

    assert check_junit_no_skips.main([str(path)]) == 1
    assert "cannot read JUnit report" in capsys.readouterr().err


def test_a_missing_report_fails_instead_of_raising(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # pytest that died before writing its report must not look like a lane with nothing to check.
    assert check_junit_no_skips.main([str(tmp_path / "absent.xml")]) == 1
    assert "cannot read JUnit report" in capsys.readouterr().err


if __name__ == "__main__":
    # Mirrors test_check_test_gates.py: `-c os.devnull` and `--confcutdir` detach this from the
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
