# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resume or rerun continues the job's tracker run instead of starting another."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from nemo_helix_plugin.integrations import WandbIntegration
from nhx.customization_common.service.context import NHXJobContext
from nhx.rl.app.jobs.training.schemas import TrainingStepConfig
from nhx.rl.tasks.training.integrations import build_mlflow_config, build_wandb_config, find_logger_mlflow_run_id


def _job_ctx(tmp_path: Path, job_uid: str | None) -> NHXJobContext:
    return NHXJobContext(
        workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step="training",
        task="task-abc",
        jobs_url=None,
        files_url=None,
        storage_path=tmp_path,
        config_path=tmp_path / "config.json",
        job_uid=job_uid,
    )


def _customizer_config(**integrations: Any) -> TrainingStepConfig:
    config = SimpleNamespace(
        integrations=SimpleNamespace(wandb=integrations.get("wandb"), mlflow=integrations.get("mlflow")),
        workspace_path="/workspace",
        model=SimpleNamespace(name="model"),
        output_model="out",
    )
    return cast(TrainingStepConfig, config)


@pytest.mark.parametrize("job_uid", ["platform-job-abc123", None])
def test_wandb_run_id_is_the_job_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_uid: str | None) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "key")

    config = build_wandb_config(
        _customizer_config(wandb=WandbIntegration(project="p")), _job_ctx(tmp_path, job_uid), "nemo_rl"
    )

    assert config is not None
    assert config.get("id") == job_uid
    assert config.get("resume") == ("allow" if job_uid else None)


def test_mlflow_tags_the_job_uid(tmp_path: Path) -> None:
    config = build_mlflow_config(
        _customizer_config(
            mlflow=SimpleNamespace(
                tracking_uri="http://mlflow",
                tags={"job_uid": "user"},
                description=None,
                experiment_name=None,
                run_name=None,
            )
        ),
        _job_ctx(tmp_path, "platform-job-abc123"),
        "nemo_rl",
    )

    assert config is not None
    assert config["tags"]["job_uid"] == "platform-job-abc123"


def test_the_run_lookup_only_runs_when_mlflow_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up: list[Any] = []
    monkeypatch.setattr(
        "nhx.rl.tasks.training.integrations.find_mlflow_run_id",
        lambda config: looked_up.append(config) or "run-1",
    )
    mlflow = {"tracking_uri": "http://mlflow", "experiment_name": "exp", "tags": {"job_uid": "uid"}}

    assert find_logger_mlflow_run_id({"mlflow_enabled": False, "mlflow": mlflow}) is None
    assert find_logger_mlflow_run_id({"mlflow_enabled": True, "mlflow": mlflow}) == "run-1"
    assert looked_up == [mlflow]
