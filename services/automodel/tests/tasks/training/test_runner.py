# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Automodel training runner."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules["nemo_automodel"] = MagicMock()
sys.modules["nemo_automodel._transformers"] = MagicMock()
sys.modules["nemo_automodel._transformers.registry"] = MagicMock()

from nmp.automodel.entities.values import TrainingType  # noqa: E402
from nmp.automodel.tasks.training.runner import TrainingRunner  # noqa: E402
from nmp.automodel.tasks.training.schemas import DistillationConfig, ModelConfig, TrainingStepConfig  # noqa: E402
from nmp.customization_common.service.context import NMPJobContext  # noqa: E402


def _job_context(storage_path: Path) -> NMPJobContext:
    return NMPJobContext(
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
