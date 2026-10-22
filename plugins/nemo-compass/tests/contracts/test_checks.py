# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_compass_plugin.contracts.checks import (
    CheckResult,
    advisories,
    format_report,
    make_check_result,
    required_failures,
)


def test_make_check_result_maps_outcome_and_severity_to_status() -> None:
    passed = make_check_result("ready", "runtime", True, "required", "ready", "not ready")
    failed = make_check_result("ready", "runtime", False, "required", "ready", "not ready", hint="fix it")
    warned = make_check_result("remote", "runtime", False, "advisory", "reachable", "unreachable")

    assert passed.status == "pass"
    assert passed.hint is None
    assert failed.status == "fail"
    assert failed.hint == "fix it"
    assert warned.status == "warn"


def test_filters_return_only_blockers_or_warnings() -> None:
    results = [
        CheckResult(name="pass", group="profile", status="pass", severity="required", message="ok"),
        CheckResult(name="fail", group="profile", status="fail", severity="required", message="broken"),
        CheckResult(name="warn", group="platform", status="warn", severity="advisory", message="offline"),
    ]

    assert [result.name for result in required_failures(results)] == ["fail"]
    assert [result.name for result in advisories(results)] == ["warn"]


def test_format_report_sorts_groups_and_prints_nonpassing_hints() -> None:
    report = format_report(
        [
            CheckResult(
                name="remote",
                group="platform",
                status="warn",
                severity="advisory",
                message="platform unreachable",
                hint="start the platform",
            ),
            CheckResult(
                name="profile",
                group="profile",
                status="pass",
                severity="required",
                message="profile found",
                hint="not printed",
            ),
        ]
    )

    assert report == ("Platform\n  ⚠ platform unreachable\n      hint: start the platform\nProfile\n  ✓ profile found")
