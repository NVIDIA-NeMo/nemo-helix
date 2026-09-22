# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for usage leaderboard internal data shapes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nemo_agents_plugin.leaderboard.types import (
    AgentLeaderboard,
    AgentLeaderboardEntry,
    AgentLeaderboardWarning,
)


def test_entry_compute_units_per_token():
    entry = AgentLeaderboardEntry(
        entry_id="run-1",
        task_name="workspace-basic-mcp",
        compute_units=8.0,
        token_count=2,
    )

    assert entry.compute_units_per_token == 4.0


def test_entry_compute_units_per_token_none_when_inputs_not_positive():
    entry = AgentLeaderboardEntry(
        entry_id="run-1",
        task_name="workspace-basic-mcp",
        compute_units=0.0,
        token_count=2,
    )

    assert entry.compute_units_per_token is None


def test_entry_keeps_optional_metadata(tmp_path: Path):
    created_at = datetime(2026, 4, 30, 12, 0, 0)
    source_path = tmp_path / "report.json"
    entry = AgentLeaderboardEntry(
        entry_id="run-1",
        task_name="workspace-basic-mcp",
        compute_units=0.2,
        compute_units_formula_version="usage_report_v0_compute_units",
        token_count=1234,
        runtime_image="nhx-nat-workspace-basic-mcp:latest",
        created_at=created_at,
        source_path=str(source_path),
        source_dir="/tmp/workspace-basic-mcp",
        run_count=2,
        raw_report={"schema_version": "v0"},
    )

    assert entry.created_at == created_at
    assert entry.source_path == str(source_path)
    assert entry.source_dir == "/tmp/workspace-basic-mcp"
    assert entry.compute_units_formula_version == "usage_report_v0_compute_units"
    assert entry.token_count == 1234
    assert entry.runtime_image == "nhx-nat-workspace-basic-mcp:latest"
    assert entry.run_count == 2
    assert entry.raw_report == {"schema_version": "v0"}


def test_leaderboard_container_defaults_warnings_to_empty():
    entry = AgentLeaderboardEntry(
        entry_id="run-1",
        task_name="workspace-basic-mcp",
        compute_units=0.2,
    )

    leaderboard = AgentLeaderboard(entries=(entry,))

    assert leaderboard.entries == (entry,)
    assert leaderboard.warnings == ()


def test_leaderboard_container_keeps_warnings(tmp_path: Path):
    source_path = tmp_path / "report.json"
    entry = AgentLeaderboardEntry(
        entry_id="run-1",
        task_name="workspace-basic-mcp",
        compute_units=0.2,
    )
    warning = AgentLeaderboardWarning(
        message="Missing created_at; using report file ordering fallback.",
        source_path=str(source_path),
    )

    leaderboard = AgentLeaderboard(entries=(entry,), warnings=(warning,))

    assert leaderboard.warnings == (warning,)
