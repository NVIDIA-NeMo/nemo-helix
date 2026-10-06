# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness result construction and presentation.

One check is one named verdict, and a report is a list of them. Statuses and
severities are plain strings rather than enums, so the JSON a report emits needs no
conversion step and stays readable to whatever reads it next.

Uses ``dataclass`` rather than ``pydantic.BaseModel`` so this module carries no
dependency of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PASS = "pass"
WARN = "warn"
FAIL = "fail"

REQUIRED = "required"
ADVISORY = "advisory"

_MARKS = {PASS: "\u2713", WARN: "\u26a0", FAIL: "\u2717"}


@dataclass
class CheckResult:
    """One required or advisory readiness check.

    ``proven`` records whether Harbor judged this result or the skill merely
    observed it. An observed result never counts as evidence that a suite runs,
    so the report keeps the distinction rather than flattening it.
    """

    name: str
    group: str
    status: str
    severity: str
    message: str
    hint: str | None = None
    proven: bool = field(default=True)

    def as_dict(self) -> dict:
        """Return the JSON-serializable form."""
        return {
            "name": self.name,
            "group": self.group,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
            "proven": self.proven,
        }


def check(
    name: str,
    group: str,
    status: str,
    message: str,
    *,
    severity: str = REQUIRED,
    hint: str | None = None,
    proven: bool = True,
) -> CheckResult:
    """Build one check result."""
    return CheckResult(
        name=name,
        group=group,
        status=status,
        severity=severity,
        message=message,
        hint=hint,
        proven=proven,
    )


def format_report(results: list[CheckResult]) -> str:
    """Format checks into deterministic grouped terminal output."""
    lines: list[str] = []
    for group in sorted({result.group for result in results}):
        lines.append(group.capitalize())
        for result in (item for item in results if item.group == group):
            suffix = "" if result.proven else "  (observed, not proven)"
            lines.append("  {} {}{}".format(_MARKS[result.status], result.message, suffix))
            if result.hint and result.status != PASS:
                lines.append("      hint: {}".format(result.hint))
    return "\n".join(lines)


def required_failures(results: list[CheckResult]) -> list[CheckResult]:
    """Return required failures that block a command."""
    return [result for result in results if result.status == FAIL and result.severity == REQUIRED]


def advisories(results: list[CheckResult]) -> list[CheckResult]:
    """Return non-blocking warnings."""
    return [result for result in results if result.status == WARN]
