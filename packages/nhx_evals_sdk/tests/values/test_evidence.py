# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading a trace file as ATIF."""

from __future__ import annotations

import json
from pathlib import Path

from nhx_evals_sdk.values.evidence import read_atif

ATIF_PAYLOAD = {
    "schema_version": "ATIF-v1.7",
    "session_id": "s1",
    "agent": {"name": "codex", "version": "1.0"},
    "steps": [
        {"step_id": 1, "source": "user", "message": "solve it", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"step_id": 2, "source": "agent", "message": "204", "timestamp": "2026-01-01T00:00:01+00:00"},
    ],
}


def test_a_valid_atif_file_parses(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(ATIF_PAYLOAD), encoding="utf-8")

    trajectory = read_atif(path)

    assert trajectory is not None
    assert [step.step_id for step in trajectory.steps] == [1, 2]


def test_json_that_is_not_atif_is_not_mistaken_for_it(tmp_path: Path) -> None:
    # The reason this function exists: a producer names the file, so only its contents identify it.
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"not": "atif"}), encoding="utf-8")

    assert read_atif(path) is None


def test_a_truncated_file_is_refused_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text('{"schema_version": "ATIF-v1.7", "steps": [', encoding="utf-8")

    assert read_atif(path) is None


def test_an_absent_file_is_refused_rather_than_raising(tmp_path: Path) -> None:
    assert read_atif(tmp_path / "missing.json") is None


def test_a_directory_is_refused_rather_than_raising(tmp_path: Path) -> None:
    assert read_atif(tmp_path) is None
