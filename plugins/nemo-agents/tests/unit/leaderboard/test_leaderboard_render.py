# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for usage leaderboard terminal rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from nemo_agents_plugin.leaderboard.render import render_entries, render_leaderboard
from nemo_agents_plugin.leaderboard.types import (
    AgentLeaderboard,
    AgentLeaderboardEntry,
    AgentLeaderboardWarning,
)


def _entry(
    entry_id: str,
    *,
    task_name: str = "workspace-basic-mcp",
    runtime_image: str | None = "nhx-nat-workspace-basic-mcp:latest",
    token_count: int | None = 2000,
    compute_units: float,
    created_at: datetime | None = None,
) -> AgentLeaderboardEntry:
    return AgentLeaderboardEntry(
        entry_id=entry_id,
        task_name=task_name,
        runtime_image=runtime_image,
        token_count=token_count,
        compute_units=compute_units,
        created_at=created_at,
    )


def test_render_entries_wide_layout_contains_full_columns():
    rendered = render_entries(
        (
            _entry(
                "run-1",
                compute_units=20,
                created_at=datetime(2026, 5, 1, 11, 0, 0, tzinfo=timezone.utc),
            ),
        ),
        compact=False,
        width=120,
    )

    assert "Usage Leaderboard" in rendered
    assert "Task" in rendered
    assert "Runtime Image" in rendered
    assert "Tokens" in rendered
    assert "CU/Token" in rendered
    assert "Created" in rendered
    assert "workspace-basic-mcp" in rendered
    assert "nhx-nat-workspace" in rendered


def test_render_entries_compact_layout_omits_wide_only_columns():
    rendered = render_entries(
        (_entry("run-1", compute_units=20),),
        compact=True,
        width=80,
    )

    assert "Usage Leaderboard" in rendered
    assert "Task" in rendered
    assert "Compute Units" in rendered
    assert "Runtime Image" not in rendered
    assert "CU/Token" not in rendered
    assert "Created" not in rendered


def test_render_entries_auto_compact_on_narrow_width():
    rendered = render_entries(
        (_entry("run-1", compute_units=20),),
        compact=None,
        width=80,
    )

    assert "Compute Units" in rendered
    assert "CU/Token" not in rendered


def test_render_leaderboard_appends_warnings():
    leaderboard = AgentLeaderboard(
        entries=(_entry("run-1", compute_units=20),),
        warnings=(AgentLeaderboardWarning(message="Skipped invalid report", source_path="path/to/bad.json"),),
    )

    rendered = render_leaderboard(leaderboard, compact=True, width=80)

    assert "Warnings:" in rendered
    assert "Skipped invalid report (path/to/bad.json)" in rendered


def test_render_entries_empty_state():
    rendered = render_entries((), compact=True, width=80)

    assert "No entries" in rendered


def test_render_entries_uses_scientific_notation_for_tiny_compute_units_per_token():
    rendered = render_entries(
        (_entry("run-1", token_count=19_840, compute_units=0.952),),
        compact=False,
        width=120,
    )

    assert "4.80e-05" in rendered


def test_render_entries_preserves_tiny_non_zero_compute_units():
    rendered = render_entries(
        (
            _entry("run-1", compute_units=0.004),
            _entry("run-2", compute_units=-0.004),
        ),
        compact=False,
        width=120,
    )

    assert "<0.01" in rendered
    assert ">-0.01" in rendered


def test_render_entries_handles_missing_token_and_runtime_image_fields():
    rendered = render_entries(
        (_entry("run-1", runtime_image=None, token_count=None, compute_units=20),),
        compact=False,
        width=120,
    )

    assert "—" in rendered
