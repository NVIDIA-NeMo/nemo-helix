# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from config import ConfigError, format_catalog, load_sweep, select_step, sweep_files

SWEEP = Path(__file__).resolve().parents[1] / "sweep.yaml"


def test_default_sweep_lists_each_measurement() -> None:
    sweep = load_sweep(SWEEP)
    assert [step.id for step in sweep.steps] == [
        "exit-01",
        "exit-10",
        "exit-25",
        "exit-50",
        "exit-100",
        "exit-500",
        "barrier-10",
        "barrier-10-w2",
        "barrier-100",
        "barrier-100-w2",
        "files-01",
        "files-10",
        "files-25",
        "secrets-01",
        "secrets-10",
        "secrets-25",
        "secrets-100",
        "secrets-300",
        "auth-01",
        "auth-10",
        "auth-25",
    ]
    assert sweep.image == "nvcr.io/0921617854601259/nemo-helix-dev/nhx-tasks:70cc13d4055922c48de00f383a2dbabfc622ad88"
    assert sweep.steps[0].kind == "exit"
    assert sweep.step("exit-50").count == 50
    assert sweep.step("exit-100").count == 100
    assert sweep.step("exit-500").count == 500
    barrier = sweep.step("barrier-10")
    assert barrier.kind == "barrier"
    assert barrier.count == 10
    assert barrier.timeout_seconds == 180
    assert barrier.workspaces == 1
    assert sweep.step("barrier-10-w2").workspaces == 2
    assert sweep.step("barrier-100").count == 100
    assert sweep.step("barrier-100").timeout_seconds == 600
    assert sweep.step("barrier-100-w2").workspaces == 2
    assert sweep.step("files-01").service == "files"
    assert sweep.step("files-01").rate_per_second == 5
    assert sweep.step("secrets-100").count == 100
    assert sweep.step("secrets-300").count == 300
    assert sweep.step("secrets-300").duration_seconds == 60
    assert sweep.step("secrets-300").rate_per_second == 5


def test_catalog_lists_the_sweep_and_its_steps() -> None:
    text = format_catalog("sweep.yaml", load_sweep(SWEEP))
    assert text.splitlines()[0] == "sweep.yaml"
    assert "run_key: stress" in text
    assert "image: nvcr.io/0921617854601259/nemo-helix-dev/nhx-tasks:70cc13d4055922c48de00f383a2dbabfc622ad88" in text
    assert "exit-01 exit count=1" in text
    assert "barrier-10 barrier count=10 timeout=180s workspaces=1" in text
    assert "barrier-100-w2 barrier count=100 timeout=600s workspaces=2" in text
    assert "files-01 hammer count=1 duration=60s service=files rate=5/s" in text
    assert "auth-25" in text


def test_sweep_files_includes_the_shipped_sweep() -> None:
    assert SWEEP in sweep_files(SWEEP.parent)


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
