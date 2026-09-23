# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit-report collector: request -> token claims -> expected -> observed."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field


@dataclass
class Row:
    case_id: str
    group: str
    description: str
    method: str
    path: str
    identity: str
    claims: str
    expected: str
    observed: int
    passed: bool
    phase: str
    notes: str = ""


@dataclass
class ReportCollector:
    rows: list[Row] = field(default_factory=list)

    def record(self, row: Row) -> None:
        self.rows.append(row)

    def as_json(self) -> list[dict]:
        return [asdict(r) for r in sorted(self.rows, key=lambda r: r.case_id)]

    def render(self) -> str:
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
        total = len(self.rows)
        passed = sum(r.passed for r in self.rows)
        lines = [
            "# Authz E2E verification report (real OIDC, signed JWTs)",
            "",
            f"Generated {ts} by `e2e/authz_oidc` — {passed}/{total} cases passed.",
            "",
            "Identity for every request is an RS256-signed JWT minted by the in-harness",
            "OIDC issuer and validated by the platform via JWKS discovery",
            "(`auth.allow_unsigned_jwt=false`; no `X-NHX-Principal-*` headers anywhere).",
            "",
        ]
        groups: dict[str, list[Row]] = {}
        for row in sorted(self.rows, key=lambda r: r.case_id):
            groups.setdefault(row.group, []).append(row)
        for group, rows in groups.items():
            lines += [
                f"## {group} ({rows[0].phase} phase)" if all(r.phase == rows[0].phase for r in rows) else f"## {group}",
                "",
            ]
            lines += [
                "| case | request | identity (claims) | expected | observed | result |",
                "|------|---------|-------------------|----------|----------|--------|",
            ]
            for r in rows:
                result = "PASS" if r.passed else "**FAIL**"
                req = f"`{r.method} {r.path}`"
                lines.append(
                    f"| {r.case_id} | {req} | {r.identity}: {r.claims} | {r.expected} | {r.observed} | {result} |"
                )
            lines.append("")
            for r in rows:
                if r.notes or not r.passed:
                    note = r.notes or r.description
                    lines.append(f"- **{r.case_id}** — {r.description}. {note if r.notes else ''}".rstrip())
            lines.append("")
        return "\n".join(lines)
