# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the backend reads the driver's exit code."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
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


def _execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int) -> None:
    bootstrap = SimpleNamespace(
        progress=None,
        driver_output=deque(["Traceback (most recent call last):", "RuntimeError: driver failed"]),
        run_with_driver=lambda script, args: exit_code,
        terminate_driver=lambda signum: None,
    )
    monkeypatch.setattr(backend_module, "create_bootstrap_from_env", lambda: bootstrap)
    backend = backend_module.NemoRLBackend(SimpleNamespace(job_id="job-1"))  # type: ignore[arg-type]
    config: Any = SimpleNamespace(
        parallelism=SimpleNamespace(num_gpus_per_node=1, num_nodes=1),
        integrations=None,
        model=SimpleNamespace(name="out"),
        training=SimpleNamespace(training_type=TrainingType.DPO),
    )
    library = LibraryConfig(config_dict={}, config_path=tmp_path / "nemo_rl_config.yaml")
    progress: Any = SimpleNamespace(report_running=lambda *args, **kwargs: None)

    backend.execute_training(config, library, progress)


def test_a_driver_that_exits_for_a_pause_raises_training_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(TrainingPaused):
        _execute(tmp_path, monkeypatch, 75)


def test_any_other_failure_is_still_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="driver failed"):
        _execute(tmp_path, monkeypatch, 1)
