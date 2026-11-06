# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local bundle layout. The cluster is not required to read these files."""

from __future__ import annotations

import json
from pathlib import Path

from config import Step

MEASUREMENT_FIELDS = (
    "kind",
    "count",
    "duration_seconds",
    "service",
    "rate_per_second",
    "timeout_seconds",
    "workspaces",
)


def step_dir(run_dir: Path, step_id: str) -> Path:
    return run_dir / "steps" / step_id


def record_path(run_dir: Path, step_id: str, job_name: str) -> Path:
    return step_dir(run_dir, step_id) / f"{job_name}.record.json"


def history_path(run_dir: Path, step_id: str, job_name: str) -> Path:
    return step_dir(run_dir, step_id) / f"{job_name}.jsonl"


def marker_path(run_dir: Path, step_id: str) -> Path:
    return step_dir(run_dir, step_id) / "complete.json"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not an object")
    return {str(key): value for key, value in loaded.items()}


def marker_matches(run_dir: Path, step: Step, *, workspaces: int | None = None) -> bool:
    """A step is finished only when its marker stores the loaded measurement."""
    path = marker_path(run_dir, step.id)
    if not path.is_file():
        return False
    stored = read_json(path)
    expected = step.measurement()
    if workspaces is not None:
        expected = {**expected, "workspaces": workspaces}
    return all(_marker_value(stored, field) == expected[field] for field in MEASUREMENT_FIELDS)


def _marker_value(stored: dict[str, object], field: str) -> object:
    if field == "workspaces":
        return stored.get(field, 1)
    return stored.get(field)


def cohort_path(run_dir: Path, step_id: str) -> Path:
    return step_dir(run_dir, step_id) / "cohort.json"
