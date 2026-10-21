# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-neutral readiness result construction and presentation."""

from typing import Literal

from pydantic import BaseModel

CheckStatus = Literal["pass", "warn", "fail"]
CheckSeverity = Literal["required", "advisory"]


class CheckResult(BaseModel):
    """One required or advisory readiness check."""

    name: str
    group: str
    status: CheckStatus
    severity: CheckSeverity
    message: str
    hint: str | None = None


def make_check_result(
    name: str,
    group: str,
    ok: bool,
    severity: CheckSeverity,
    pass_message: str,
    fail_message: str,
    *,
    hint: str | None = None,
) -> CheckResult:
    """Build a passing, blocking, or advisory result from a boolean probe."""
    if ok:
        status: CheckStatus = "pass"
    elif severity == "required":
        status = "fail"
    else:
        status = "warn"
    return CheckResult(
        name=name,
        group=group,
        status=status,
        severity=severity,
        message=pass_message if ok else fail_message,
        hint=None if ok else hint,
    )


def format_report(results: list[CheckResult]) -> str:
    """Format checks into deterministic grouped terminal output."""
    marks: dict[CheckStatus, str] = {"pass": "✓", "warn": "⚠", "fail": "✗"}
    lines: list[str] = []
    for group in sorted({result.group for result in results}):
        lines.append(group.capitalize())
        for result in (item for item in results if item.group == group):
            lines.append(f"  {marks[result.status]} {result.message}")
            if result.hint and result.status != "pass":
                lines.append(f"      hint: {result.hint}")
    return "\n".join(lines)


def required_failures(results: list[CheckResult]) -> list[CheckResult]:
    """Return required failures that block a command."""
    return [result for result in results if result.status == "fail" and result.severity == "required"]


def advisories(results: list[CheckResult]) -> list[CheckResult]:
    """Return non-blocking warnings."""
    return [result for result in results if result.status == "warn"]
