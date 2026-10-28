# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_helix_plugin.integrations import IntegrationsSpec, MlflowIntegration, WandbIntegration
from nhx.customization_common.integrations import (
    IntegrationRuntimeContext,
    build_mlflow_config,
    build_wandb_config,
    find_mlflow_run_id,
)
from nhx.customization_common.service.context import NHXJobContext


@pytest.fixture
def job_ctx(tmp_path: Path) -> NHXJobContext:
    return NHXJobContext(
        workspace="test-workspace",
        job_id="job-123",
        attempt_id="attempt-1",
        step="training",
        task="task-abc123",
        jobs_url="http://jobs.example.com",
        files_url="http://files.example.com",
        storage_path=tmp_path / "job-storage",
        config_path=tmp_path / "config.json",
    )


def _runtime_ctx(
    job_ctx: NHXJobContext,
    *,
    wandb: WandbIntegration | None = None,
    mlflow: MlflowIntegration | None = None,
    output_name: str = "output-model-name",
    workspace_path: str = "/workspace",
    model_name: str = "meta/llama-test",
    framework: str = "automodel",
) -> IntegrationRuntimeContext:
    return IntegrationRuntimeContext(
        wandb=wandb,
        mlflow=mlflow,
        output_name=output_name,
        workspace_path=workspace_path,
        model_name=model_name,
        job_ctx=job_ctx,
        framework=framework,
    )


class TestBuildMlflowConfig:
    def test_returns_none_without_mlflow(self, job_ctx: NHXJobContext) -> None:
        assert build_mlflow_config(_runtime_ctx(job_ctx)) is None

    def test_missing_tracking_uri_warns_and_disables(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ctx = _runtime_ctx(
            job_ctx,
            mlflow=MlflowIntegration(experiment_name="exp-no-uri"),
        )
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        caplog.set_level("WARNING")

        assert build_mlflow_config(ctx) is None
        assert "MLflow integration is configured but no tracking URI is set" in caplog.text

    def test_config_tracking_uri_takes_precedence(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ctx = _runtime_ctx(
            job_ctx,
            mlflow=MlflowIntegration(
                tracking_uri="http://config-mlflow.example.com:5000",
                experiment_name="configured-experiment",
                tags={"user_tag": "user_value"},
                description="run-description",
            ),
        )
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow.example.com:5000")

        result = build_mlflow_config(ctx)

        assert result is not None
        assert result["tracking_uri"] == "http://config-mlflow.example.com:5000"
        assert result["experiment_name"] == "configured-experiment"
        assert result["run_name"] == "job-123"
        assert result["tags"]["service"] == "nemo-helix"
        assert result["tags"]["framework"] == "automodel"
        assert result["tags"]["workspace"] == "test-workspace"
        assert result["tags"]["job"] == "job-123"
        assert result["tags"]["task"] == "task-abc123"
        assert result["tags"]["model_name"] == "meta/llama-test"
        assert result["tags"]["user_tag"] == "user_value"
        assert result["tags"]["mlflow.note.content"] == "run-description"

    def test_from_integrations_spec_factory(self, job_ctx: NHXJobContext, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow.example.com:5000")
        integrations = IntegrationsSpec.model_validate({"mlflow": {"experiment_name": "configured-experiment"}})
        ctx = IntegrationRuntimeContext.from_integrations_spec(
            integrations=integrations,
            output_name="output-model-name",
            workspace_path="/workspace",
            model_name="meta/llama-test",
            job_ctx=job_ctx,
            framework="automodel",
        )
        result = build_mlflow_config(ctx)
        assert result is not None
        assert result["tracking_uri"] == "http://env-mlflow.example.com:5000"


class TestBuildWandbConfig:
    def test_returns_none_without_wandb(self, job_ctx: NHXJobContext) -> None:
        assert build_wandb_config(_runtime_ctx(job_ctx)) is None

    def test_missing_api_key_and_base_url_warns_and_disables(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ctx = _runtime_ctx(job_ctx, wandb=WandbIntegration(project="proj"))
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        caplog.set_level("WARNING")

        assert build_wandb_config(ctx) is None
        assert "WandB API key is not set" in caplog.text

    def test_base_url_without_api_key_warns_and_activates(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        ctx = _runtime_ctx(
            job_ctx,
            wandb=WandbIntegration(project="proj", base_url="https://wandb.internal"),
        )
        caplog.set_level("WARNING")

        result = build_wandb_config(ctx)

        assert result is not None
        assert result["settings"]["base_url"] == "https://wandb.internal"
        assert "base_url only" in caplog.text

    def test_builds_full_config(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-api-key")
        ctx = _runtime_ctx(
            job_ctx,
            wandb=WandbIntegration(
                project="my-project",
                name="my-run",
                entity="my-team",
                tags=["tag-a"],
                notes="notes",
            ),
            workspace_path="/tmp/workspace",
        )

        result = build_wandb_config(ctx)

        assert result is not None
        assert result["project"] == "my-project"
        assert result["name"] == "my-run"
        assert result["entity"] == "my-team"
        assert result["notes"] == "notes"
        assert result["dir"] == "/tmp/workspace/wandb"

    def test_wandb_dir_uses_ephemeral_for_output_model(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-api-key")
        ctx = _runtime_ctx(
            job_ctx,
            wandb=WandbIntegration(project="proj"),
            workspace_path="/var/run/scratch/job/output_model",
        )

        result = build_wandb_config(ctx)

        assert result is not None
        assert result["dir"] == "/var/run/scratch/job/ephemeral/wandb"

    def test_wandb_dir_uses_ephemeral_for_training_workspace(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-api-key")
        ctx = _runtime_ctx(
            job_ctx,
            wandb=WandbIntegration(project="proj"),
            workspace_path="/var/run/scratch/job/training",
        )

        result = build_wandb_config(ctx)

        assert result is not None
        assert result["dir"] == "/var/run/scratch/job/ephemeral/wandb"


class TestTrackerRunContinuity:
    """A resume or rerun continues the job's tracker run instead of starting another."""

    @pytest.fixture
    def job_with_uid(self, job_ctx: NHXJobContext) -> NHXJobContext:
        return dataclasses.replace(job_ctx, job_uid="platform-job-abc123")

    def test_wandb_run_id_is_the_job_uid(self, job_with_uid: NHXJobContext, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-api-key")

        config = build_wandb_config(_runtime_ctx(job_with_uid, wandb=WandbIntegration(project="p")))

        assert config is not None
        assert config["id"] == "platform-job-abc123"
        assert config["resume"] == "allow"

    def test_wandb_without_a_job_uid_starts_a_new_run(
        self, job_ctx: NHXJobContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-api-key")

        config = build_wandb_config(_runtime_ctx(job_ctx, wandb=WandbIntegration(project="p")))

        assert config is not None
        assert "id" not in config and "resume" not in config

    def test_mlflow_tags_the_job_uid_over_user_tags(self, job_with_uid: NHXJobContext) -> None:
        mlflow = MlflowIntegration(tracking_uri="http://mlflow", tags={"job_uid": "from-user"})

        config = build_mlflow_config(_runtime_ctx(job_with_uid, mlflow=mlflow))

        assert config is not None
        assert config["tags"]["job_uid"] == "platform-job-abc123"

    def test_find_mlflow_run_id_returns_the_newest_tagged_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        searches: list[dict[str, Any]] = []

        class _Client:
            def __init__(self, tracking_uri: str) -> None:
                assert tracking_uri == "http://mlflow"

            def get_experiment_by_name(self, name: str) -> SimpleNamespace:
                return SimpleNamespace(experiment_id=f"id-of-{name}")

            def search_runs(self, experiment_ids: list[str], **kwargs: Any) -> list[SimpleNamespace]:
                searches.append({"experiment_ids": experiment_ids, **kwargs})
                return [SimpleNamespace(info=SimpleNamespace(run_id="run-1"))]

        monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client)

        run_id = find_mlflow_run_id(
            {"tracking_uri": "http://mlflow", "experiment_name": "exp", "tags": {"job_uid": "platform-job-abc123"}}
        )

        assert run_id == "run-1"
        assert searches[0]["experiment_ids"] == ["id-of-exp"]
        assert searches[0]["filter_string"] == "tags.job_uid = 'platform-job-abc123'"
        assert searches[0]["max_results"] == 1

    @pytest.mark.parametrize(
        "config",
        [None, {"tracking_uri": "http://mlflow", "experiment_name": "exp", "tags": {}}],
        ids=["mlflow-off", "no-job-uid"],
    )
    def test_find_mlflow_run_id_does_not_search_without_a_job_uid(
        self, config: dict[str, Any] | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("MLflow should not be called")

        monkeypatch.setattr("mlflow.tracking.MlflowClient", fail)

        assert find_mlflow_run_id(config) is None

    def test_find_mlflow_run_id_starts_a_new_run_when_the_lookup_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Client:
            def __init__(self, tracking_uri: str) -> None:
                raise ConnectionError("tracking server unavailable")

        monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client)

        config = {"tracking_uri": "http://mlflow", "experiment_name": "exp", "tags": {"job_uid": "platform-job-abc123"}}
        assert find_mlflow_run_id(config) is None
