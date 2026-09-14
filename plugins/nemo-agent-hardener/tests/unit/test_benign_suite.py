# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benign-suite read/write helpers."""

from __future__ import annotations

from pathlib import Path

from nemo_agent_hardener_plugin.jobs import benign_suite


def test_read_suite_missing_file_returns_empty(tmp_path: Path) -> None:
    assert benign_suite.read_suite(tmp_path / "nope.csv") == []


def test_write_then_read_round_trips_in_column_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "requests.csv"
    suite = [
        {"tool": "clock", "payload": "what time is it?", "label": "benign", "rationale": "basic", "persona": "user"},
        {"tool": "clock", "payload": "date please", "label": "benign", "rationale": "basic", "persona": ""},
    ]
    benign_suite.write_suite(csv_path, suite)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "tool,payload,label,rationale,persona"
    assert benign_suite.read_suite(csv_path) == suite


def test_read_suite_skips_rows_missing_tool_or_payload(tmp_path: Path) -> None:
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text(
        "tool,payload,label,rationale,persona\nclock,valid,benign,r,p\n,no-tool,benign,r,p\nclock,,benign,r,p\n",
        encoding="utf-8",
    )
    suite = benign_suite.read_suite(csv_path)
    assert [row["payload"] for row in suite] == ["valid"]
