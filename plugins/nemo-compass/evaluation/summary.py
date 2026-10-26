# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the local insights YAML as compact markdown (for $GITHUB_STEP_SUMMARY)."""

from pathlib import Path

import yaml


def render_summary_md(insights_path: Path, subject: str) -> str:
    """Markdown summary of ``{"insights": [...]}`` at *insights_path*.

    One bullet per insight: bold title, status, trace-ref count, first
    description line. Missing/empty file renders an explicit zero section so a
    CI summary never silently omits a subject.
    """
    records: list[dict] = []
    if insights_path.exists():
        raw = yaml.safe_load(insights_path.read_text(encoding="utf-8")) or {}
        records = list(raw.get("insights", []))
    lines = [f"## Insights — {subject} ({len(records)})", ""]
    if not records:
        lines.append("_No insights._")
    for rec in records:
        refs = rec["evidence"] if "evidence" in rec else rec.get("trace_refs") or []
        first_desc = str(rec.get("description", "")).splitlines()[0] if rec.get("description") else ""
        lines.append(f"- **{rec.get('title', '(untitled)')}** — {rec.get('status', '?')}, {len(refs)} trace refs")
        if first_desc:
            lines.append(f"  {first_desc}")
    return "\n".join(lines) + "\n"
