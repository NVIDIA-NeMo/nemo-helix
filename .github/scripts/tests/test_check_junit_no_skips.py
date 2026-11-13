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


def test_a_skipped_test_fails_and_is_named_with_its_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # What pytest writes for a skipif gate: the suite's count and the case's element.
    report = _report(
        tmp_path,
        '<testsuites><testsuite name="pytest" skipped="1" tests="2">'
        '<testcase classname="e2e.test_evaluator_plugin" name="test_a" />'
        '<testcase classname="e2e.test_evaluator_plugin" name="test_b">'
        '<skipped type="pytest.skip" message="NHX_E2E_SANDBOXED_GYM is not set">'
        "e2e/test_evaluator_plugin.py:9: NHX_E2E_SANDBOXED_GYM is not set</skipped></testcase>"
        "</testsuite></testsuites>",
    )

    assert check_junit_no_skips.main([str(report), "--label", "sandboxed Gym e2e"]) == 1
    assert capsys.readouterr().err == (
        "sandboxed Gym e2e: expected no skips, found:\n"
        "  e2e.test_evaluator_plugin::test_b (pytest.skip): NHX_E2E_SANDBOXED_GYM is not set\n"
    )


def test_an_xfail_fails_and_reads_as_an_xfail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # pytest counts an xfail as a skip. The message must not send the reader looking for a skip gate.
    report = _report(
        tmp_path,
        '<testsuites><testsuite name="pytest" skipped="1" tests="2">'
        '<testcase classname="e2e.test_evaluator_plugin" name="test_a" />'
        '<testcase classname="e2e.test_evaluator_plugin" name="test_b">'
        '<skipped type="pytest.xfail" message="known product bug" /></testcase>'
        "</testsuite></testsuites>",
    )

    assert check_junit_no_skips.main([str(report)]) == 1
    assert "e2e.test_evaluator_plugin::test_b (pytest.xfail): known product bug" in capsys.readouterr().err


def test_a_report_with_no_tests_fails_and_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A marker that selects nothing must not pass as "nothing skipped".
    report = _pytest_report(tmp_path, tests=0, skipped=0)

    assert check_junit_no_skips.main([str(report), "--label", "builder e2e"]) == 1
    assert capsys.readouterr().err == "builder e2e: the report holds no tests\n"


def test_the_expected_number_of_tests_passes(tmp_path: Path) -> None:
    report = _pytest_report(tmp_path, tests=2, skipped=0)

    assert check_junit_no_skips.main([str(report), "--expect-tests", "2"]) == 0


@pytest.mark.parametrize("tests", [1, 3])
def test_any_other_number_of_tests_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str], tests: int) -> None:
    # A test the lane's marker stops selecting is deselected, not skipped: only the count shows it.
    report = _pytest_report(tmp_path, tests=tests, skipped=0)

    assert check_junit_no_skips.main([str(report), "--label", "sandboxed Gym e2e", "--expect-tests", "2"]) == 1
    assert capsys.readouterr().err == f"sandboxed Gym e2e: expected 2 tests, the report holds {tests}\n"


def test_a_bare_testsuite_root_is_counted(tmp_path: Path) -> None:
    report = _report(tmp_path, '<testsuite name="pytest" skipped="0" tests="3"></testsuite>')

    assert check_junit_no_skips.read_report(report) == (3, [])


def test_several_suites_are_summed(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="0"/><testsuite name="b" tests="3" skipped="1"/></testsuites>',
    )

    assert check_junit_no_skips.read_report(report) == (5, ["suite 'b' counts 1 skipped but names no test"])


def test_a_nested_suite_is_not_counted_twice(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="0"><testsuite tests="2" skipped="0"/></testsuite></testsuites>',
    )

    assert check_junit_no_skips.read_report(report) == (2, [])


def test_a_skip_counted_only_by_a_nested_suite_fails(tmp_path: Path) -> None:
    # The parent totals the nested suite's tests but not its skip, and no case records it.
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="0"><testsuite name="inner" tests="2" skipped="1"/>'
        "</testsuite></testsuites>",
    )

    assert check_junit_no_skips.read_report(report) == (2, ["suite 'inner' counts 1 skipped but names no test"])
    assert check_junit_no_skips.main([str(report)]) == 1


def test_a_skipped_case_in_a_nested_suite_is_named(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="1" skipped="0"><testsuite tests="1" skipped="0">'
        '<testcase classname="m" name="t"><skipped message="gate" /></testcase>'
        "</testsuite></testsuite></testsuites>",
    )

    assert check_junit_no_skips.read_report(report) == (1, ["m::t (skipped): gate"])


def test_skips_recorded_only_as_cases_are_counted(tmp_path: Path) -> None:
    # No `skipped` attribute on the suite: some writers record a skip only on the case itself.
    report = _report(
        tmp_path,
        '<testsuite tests="2"><testcase classname="m" name="a"><skipped/></testcase>'
        '<testcase classname="m" name="b"/></testsuite>',
    )

    assert check_junit_no_skips.read_report(report) == (2, ["m::a (skipped): no reason given"])


def test_a_skip_recorded_both_ways_is_counted_once(tmp_path: Path) -> None:
    # What pytest writes: the suite's count and the case's element describe the same skip.
    report = _report(
        tmp_path,
        '<testsuites><testsuite tests="2" skipped="1"><testcase classname="m" name="a"><skipped/></testcase>'
        '<testcase classname="m" name="b"/></testsuite></testsuites>',
    )

    assert check_junit_no_skips.read_report(report) == (2, ["m::a (skipped): no reason given"])


def test_a_skip_outside_any_case_is_counted(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        '<testsuite tests="1" skipped="0"><testcase classname="m" name="a"/><skipped message="setup"/></testsuite>',
    )

    assert check_junit_no_skips.read_report(report) == (1, ["outside any test case (skipped): setup"])


@pytest.mark.parametrize(
    "body",
    [
        "",
        "<testsuites><testsuite",
        '<testsuites><testsuite tests="two"/></testsuites>',
        '<?xml version="1.0" encoding="no-such-encoding"?><testsuites/>',
    ],
)
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
