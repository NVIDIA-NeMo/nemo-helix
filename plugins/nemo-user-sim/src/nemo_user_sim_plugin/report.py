# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joins a simulate job's ``summary`` and ``scores`` results into one row per conversation."""

from collections import Counter
from typing import Any

SEGMENTS = ("persona_age_bin", "persona_education_level", "user_interaction_style", "probe_type")


def conversations(summary: dict[str, Any], scores: dict[str, dict] | None) -> list[dict[str, Any]]:
    rows = []
    for conversation in summary["conversations"]:
        score = (scores or {}).get(conversation["trajectory_id"], {})
        rows.append(
            {
                **conversation,
                "passed": score.get("passed"),
                "quality": score.get("quality"),
                "reason": score.get("reason"),
                "picked": score.get("picked"),
            }
        )
    return rows


def failure_rates(rows: list[dict[str, Any]], segment: str) -> dict[str, float]:
    """Share of scored conversations that didn't pass, per value of ``segment``."""
    scored = [row for row in rows if row["passed"] is not None]
    totals = Counter(row.get(segment) for row in scored)
    failed = Counter(row.get(segment) for row in scored if not row["passed"])
    return {str(value): failed[value] / count for value, count in totals.most_common()}


def session_link(row: dict[str, Any], base_url: str, workspace: str) -> str | None:
    """The conversation's Studio transcript, when it was published to Intake."""
    if row.get("publication") != "published":
        return None
    return f"{base_url.rstrip('/')}/studio/workspaces/{workspace}/intake/sessions/{row['trajectory_id']}"
