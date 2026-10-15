# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Automodel training runner."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()

from nhx.automodel.entities.values import TrainingType  # noqa: E402
from nhx.automodel.tasks.training.runner import TrainingRunner  # noqa: E402
from nhx.automodel.tasks.training.schemas import (  # noqa: E402
    DistillationConfig,
    ModelConfig,
    TrainingMetrics,
    TrainingStepConfig,
)
from nhx.customization_common.service.context import NHXJobContext  # noqa: E402


def _job_context(storage_path: Path) -> NHXJobContext:
    return NHXJobContext(
        workspace="default",
        job_id="job-1",
        attempt_id="attempt-0",
        step="training",
        task="task-1",
        jobs_url=None,
        files_url=None,
        storage_path=storage_path,
        config_path=storage_path / "config.json",
    )


def _config() -> TrainingStepConfig:
    return TrainingStepConfig(
        model=ModelConfig(path="/run/scratch/job/model"),
        dataset=TrainingStepConfig.DatasetConfig(path="/run/scratch/job/dataset"),
        training=TrainingStepConfig.TrainingConfig(
            training_type=TrainingType.DISTILLATION,
            kd=DistillationConfig(teacher_model=ModelConfig(path="/run/scratch/job/teacher_model")),
        ),
        schedule=TrainingStepConfig.ScheduleConfig(),
        batch=TrainingStepConfig.BatchConfig(),
        optimizer=TrainingStepConfig.OptimizerConfig(),
        parallelism=TrainingStepConfig.ParallelismConfig(),
        output_model="trained-model",
        workspace_path="/run/scratch/job/training",
        output_path="/run/scratch/job/output_model",
    )


def test_normalizes_legacy_job_storage_paths_to_runtime_mount(tmp_path: Path) -> None:
    storage = tmp_path / "job"
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._job_ctx = _job_context(storage)

    normalized = runner._normalize_storage_paths(_config())

    assert normalized.model.path == str(storage / "model")
    assert normalized.dataset.path == str(storage / "dataset")
    assert normalized.training.kd is not None
    assert normalized.training.kd.teacher_model.path == str(storage / "teacher_model")
    assert normalized.workspace_path == str(storage / "training")
    assert normalized.output_path == str(storage / "output_model")


def runner_with_duration(tmp_path: Path, duration: float | None) -> TrainingRunner:
    runner = TrainingRunner.__new__(TrainingRunner)
    runner._config = SimpleNamespace(seed=1)
    runner._progress = SimpleNamespace(
        training_duration_seconds=duration,
        report_running=lambda *args, **kwargs: None,
        report_completed=lambda *args, **kwargs: None,
        report_error=lambda *args, **kwargs: None,
    )
    runner._dist_ctx = SimpleNamespace(is_coordinator=True, sync_point=lambda name: None)
    runner._workspace_path = tmp_path
    runner._output_path = tmp_path / "output"
    runner._backend = MagicMock()
    return runner


def written_duration(tmp_path: Path) -> float | None:
    payload = json.loads((tmp_path / "customizer_training_result.json").read_text())
    return payload["training_duration_seconds"]


def test_result_json_copies_the_recorded_training_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, 17.5)
    runner._compile_config_phase = lambda: MagicMock()  # type: ignore[method-assign]
    runner._training_phase = lambda library: TrainingMetrics()  # type: ignore[method-assign]
    runner._backend.find_checkpoints.return_value = {}
    runner._backend.process_checkpoints.return_value = None

    result = runner.run()

    assert result.training_duration_seconds == 17.5
    assert written_duration(tmp_path) == 17.5


def test_result_json_keeps_the_recorded_duration_when_training_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, 9.25)
    runner._compile_config_phase = lambda: MagicMock()  # type: ignore[method-assign]
    runner._training_phase = MagicMock(side_effect=RuntimeError("train failed"))  # type: ignore[method-assign]

    result = runner.run()

    assert result.success is False
    assert result.training_duration_seconds == 9.25
    assert written_duration(tmp_path) == 9.25


def test_result_json_omits_duration_when_training_never_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nhx.automodel.tasks.training.runner.get_gpu_info", lambda: None)
    runner = runner_with_duration(tmp_path, None)
    runner._compile_config_phase = MagicMock(side_effect=RuntimeError("compile failed"))  # type: ignore[method-assign]

    result = runner.run()

    assert result.training_duration_seconds is None
    assert written_duration(tmp_path) is None
