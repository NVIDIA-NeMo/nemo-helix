# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A node's own environment reaches its ``ray start``, and never the driver."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nhx.rl.tasks.training.backends.nemo_rl import ray_bootstrap
from nhx.rl.tasks.training.backends.nemo_rl.ray_bootstrap import RayClusterBootstrap

NODE_ENV = {"NCCL_IB_HCA": "=mlx5_21,=mlx5_22"}


def _bootstrap(tmp_path: Path, rank: int, node_env: dict[str, str]) -> RayClusterBootstrap:
    return RayClusterBootstrap(rank=rank, world_size=2, master_addr="head", log_dir=tmp_path, node_env=node_env)


def _record_ray_start(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ray_bootstrap.subprocess, "run", run)
    return calls


def _start(bootstrap: RayClusterBootstrap) -> Callable[[], object]:
    return bootstrap._start_head_process if bootstrap.rank == 0 else bootstrap._start_worker_process


@pytest.mark.parametrize("rank", [0, 1])
def test_ray_start_gets_the_node_env_on_top_of_the_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setenv("NHX_TEST_KEEP", "1")
    calls = _record_ray_start(monkeypatch)

    assert _start(_bootstrap(tmp_path, rank, dict(NODE_ENV)))() is not None

    assert calls[0]["env"]["NCCL_IB_HCA"] == "=mlx5_21,=mlx5_22"
    assert calls[0]["env"]["NHX_TEST_KEEP"] == "1"


@pytest.mark.parametrize("rank", [0, 1])
def test_ray_start_inherits_the_process_env_without_a_node_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    calls = _record_ray_start(monkeypatch)

    assert _start(_bootstrap(tmp_path, rank, {}))() is not None

    assert calls[0]["env"] is None


def test_the_driver_never_gets_the_node_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NeMo-RL copies the driver's environment to every worker, replacing the other nodes' own values."""
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(ray_bootstrap, "read_subprocess_output", lambda proc, buffer: None)
    envs: list[dict[str, str]] = []

    def popen(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        envs.append(kwargs["env"])
        return SimpleNamespace(wait=lambda: 0, poll=lambda: 0, returncode=0, pid=7)

    monkeypatch.setattr(ray_bootstrap.subprocess, "Popen", popen)

    assert _bootstrap(tmp_path, 0, dict(NODE_ENV))._run_driver("driver.py", []) == 0

    assert "NCCL_IB_HCA" not in envs[0]
