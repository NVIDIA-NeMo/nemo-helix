# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from config import ConfigError, load_sweep, select_step

SWEEP = Path(__file__).resolve().parents[1] / "sweep.yaml"


def test_default_sweep_lists_each_measurement() -> None:
    sweep = load_sweep(SWEEP)
    assert [step.id for step in sweep.steps] == [
        "exit-01",
        "exit-10",
        "exit-25",
        "idle-01",
        "idle-10",
        "idle-25",
        "files-01",
        "files-10",
        "files-25",
        "secrets-01",
        "secrets-10",
        "secrets-25",
        "auth-01",
        "auth-10",
        "auth-25",
    ]
    assert sweep.steps[0].kind == "exit"
    assert sweep.steps[3].duration_seconds == 300
    assert sweep.steps[6].service == "files"
    assert sweep.steps[6].rate_per_second == 5


def test_unknown_step_lists_configured_ids() -> None:
    sweep = load_sweep(SWEEP)
    with pytest.raises(ConfigError, match="exit-01") as caught:
        select_step(sweep, "missing")
    assert "auth-25" in str(caught.value)


def test_duplicate_step_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
run_key: stress
payload_bytes: 8
pods:
  exit: {cpu_request: 100m, memory_request: 256Mi, cpu_limit: 500m, memory_limit: 512Mi}
  idle: {cpu_request: 100m, memory_request: 256Mi, cpu_limit: 500m, memory_limit: 512Mi}
  hammer: {cpu_request: 500m, memory_request: 512Mi, cpu_limit: "1", memory_limit: 1Gi}
steps:
  - {id: exit-01, kind: exit, count: 1}
  - {id: exit-01, kind: exit, count: 2}
"""
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_sweep(path)


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
run_key: stress
payload_bytes: 8
pods:
  exit: {cpu_request: 100m, memory_request: 256Mi, cpu_limit: 500m, memory_limit: 512Mi}
  idle: {cpu_request: 100m, memory_request: 256Mi, cpu_limit: 500m, memory_limit: 512Mi}
  hammer: {cpu_request: 500m, memory_request: 512Mi, cpu_limit: "1", memory_limit: 1Gi}
steps:
  - {id: exit-01, kind: batch, count: 1}
"""
    )
    with pytest.raises(ConfigError, match="kind"):
        load_sweep(path)
