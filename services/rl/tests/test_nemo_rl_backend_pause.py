# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the backend runs the driver: the environment it gets, and how its exit code is read."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nhx.customization_common.service.context import NHXJobContext
from nhx.customization_common.training.errors.exceptions import CustomizerTrainingError
from nhx.rl.entities.values import TrainingType
from nhx.rl.tasks.training.protocol import LibraryConfig, TrainingPaused

# NeMo-RL is only installed in the training image, and backend imports its checkpoint
# module at module scope. Stub just that import, then remove the stubs so they do not
# leak into other tests on the same worker.
_stubbed: list[str] = []
if importlib.util.find_spec("nemo_rl") is None:  # pragma: no cover - env dependent
    _checkpoint_mod = types.ModuleType("nemo_rl.utils.checkpoint")
    setattr(_checkpoint_mod, "CheckpointingConfig", dict)
    setattr(_checkpoint_mod, "CheckpointManager", object)
    for _name, _module in (
        ("nemo_rl", types.ModuleType("nemo_rl")),
        ("nemo_rl.utils", types.ModuleType("nemo_rl.utils")),
        ("nemo_rl.utils.checkpoint", _checkpoint_mod),
    ):
        if _name not in sys.modules:
            sys.modules[_name] = _module
            _stubbed.append(_name)

from nhx.rl.tasks.training.backends.nemo_rl import backend as backend_module  # noqa: E402

for _name in _stubbed:
    del sys.modules[_name]


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    *,
    num_nodes: int = 1,
    driver_env: dict[str, str | None] | None = None,
) -> SimpleNamespace:
    """Run the backend with a stub bootstrap. ``driver_env`` records NCCL_IB_HCA as the driver would see it."""

    def run_with_driver(script: str, args: list[str]) -> int:
        if driver_env is not None:
            driver_env["NCCL_IB_HCA"] = os.environ.get("NCCL_IB_HCA")
        return exit_code

    bootstrap = SimpleNamespace(
        progress=None,
        node_env={},
        driver_output=deque(["Traceback (most recent call last):", "RuntimeError: driver failed"]),
        run_with_driver=run_with_driver,
        terminate_driver=lambda signum: None,
    )
    monkeypatch.setattr(backend_module, "create_bootstrap_from_env", lambda: bootstrap)
    backend = backend_module.NemoRLBackend(cast(NHXJobContext, SimpleNamespace(job_id="job-1")))
    config: Any = SimpleNamespace(
        parallelism=SimpleNamespace(num_gpus_per_node=1, num_nodes=num_nodes),
        integrations=None,
        model=SimpleNamespace(name="out"),
        training=SimpleNamespace(training_type=TrainingType.DPO),
    )
    library = LibraryConfig(config_dict={}, config_path=tmp_path / "nemo_rl_config.yaml")
    progress: Any = SimpleNamespace(report_running=lambda *args, **kwargs: None)

    backend.execute_training(config, library, progress)
    return bootstrap


def test_a_driver_that_exits_for_a_pause_raises_training_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(TrainingPaused):
        _execute(tmp_path, monkeypatch, 75)


def test_any_other_failure_is_still_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure is classified from the driver's output and keeps the original error."""
    with pytest.raises(CustomizerTrainingError) as raised:
        _execute(tmp_path, monkeypatch, 1)

    assert raised.value.detail == "RuntimeError: driver failed"


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_a_pods_rdma_devices_go_to_its_ray_start_and_never_to_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_nodes: int
) -> None:
    """NeMo-RL copies the driver's environment to every node, where the head's devices do not exist."""
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    monkeypatch.setattr(backend_module, "get_nccl_ib_env", lambda: {"NCCL_IB_HCA": "=mlx5_21,=mlx5_22"})
    driver_env: dict[str, str | None] = {}

    bootstrap = _execute(tmp_path, monkeypatch, 0, num_nodes=num_nodes, driver_env=driver_env)

    assert bootstrap.node_env == ({"NCCL_IB_HCA": "=mlx5_21,=mlx5_22"} if num_nodes > 1 else {})
    assert driver_env["NCCL_IB_HCA"] is None
