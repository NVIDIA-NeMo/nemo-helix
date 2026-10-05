# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nemo_helix_plugin.integrations import IntegrationsSpec
from nhx.customization_common.service.context import NHXJobContext
from nhx.unsloth.integrations.hf_bridge import apply_integrations_to_sft_config


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


class TestApplyIntegrationsToSftConfig:
    def test_none_integrations(self, job_ctx: NHXJobContext, tmp_path: Path) -> None:
        report_to, kwargs, env = apply_integrations_to_sft_config(
            integrations=None,
            job_ctx=job_ctx,
            output_name="out",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )
        assert report_to == ["none"]
        assert kwargs == {}
        assert env == {}

    def test_wandb_only_when_api_key_present(
        self,
        job_ctx: NHXJobContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        integrations = IntegrationsSpec.model_validate(
            {
                "wandb": {
                    "project": "my-project",
                    "name": "run-001",
                    "entity": "my-team",
                    "notes": "notes",
                    "tags": ["sft"],
                },
            },
        )

        report_to, kwargs, env = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )

        assert report_to == ["wandb"]
        assert kwargs == {"run_name": "run-001"}
        assert env["WANDB_PROJECT"] == "my-project"
        assert env["WANDB_ENTITY"] == "my-team"
        assert env["WANDB_NOTES"] == "notes"
        assert "service:nemo-helix" in env["WANDB_TAGS"]
        assert env["WANDB_DIR"] == str(tmp_path / "wandb")
        assert "MLFLOW_RUN_NAME" not in env

    def test_wandb_dir_outside_output_model_upload_tree(
        self,
        job_ctx: NHXJobContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        integrations = IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}})
        output_model = Path("/var/run/scratch/job/output_model")

        _, _, env = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=output_model,
            model_name="meta/llama",
        )

        assert env["WANDB_DIR"] == "/var/run/scratch/job/ephemeral/wandb"

    def test_wandb_skipped_without_api_key(
        self,
        job_ctx: NHXJobContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        integrations = IntegrationsSpec.model_validate({"wandb": {"project": "my-project"}})

        report_to, kwargs, env = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )

        assert report_to == ["none"]
        assert kwargs == {}
        assert env == {}

    def test_mlflow_only_sets_training_run_name(
        self,
        job_ctx: NHXJobContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        integrations = IntegrationsSpec.model_validate(
            {
                "mlflow": {
                    "tracking_uri": "http://mlflow:5000",
                    "experiment_name": "exp-1",
                    "name": "run-001",
                    "tags": {"team": "nlp"},
                },
            },
        )

        report_to, kwargs, env = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )

        assert report_to == ["mlflow"]
        assert kwargs == {"run_name": "run-001"}
        assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
        assert env["MLFLOW_EXPERIMENT_NAME"] == "exp-1"
        assert "MLFLOW_RUN_NAME" not in env
        tags = json.loads(env["MLFLOW_TAGS"])
        assert tags["service"] == "nemo-helix"
        assert tags["team"] == "nlp"

    def test_both_backends_wandb_name_wins(
        self,
        job_ctx: NHXJobContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        integrations = IntegrationsSpec.model_validate(
            {
                "wandb": {"project": "p", "name": "w-run"},
                "mlflow": {"tracking_uri": "http://mlflow:5000", "name": "m-run"},
            },
        )

        report_to, kwargs, env = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )

        assert report_to == ["wandb", "mlflow"]
        assert kwargs == {"run_name": "w-run"}
        assert env["WANDB_PROJECT"] == "p"
        assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"

    def test_conflicting_run_names_logs_warning(
        self,
        job_ctx: NHXJobContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        integrations = IntegrationsSpec.model_validate(
            {
                "wandb": {"project": "p", "name": "w-run"},
                "mlflow": {"tracking_uri": "http://mlflow:5000", "name": "m-run"},
            },
        )
        caplog.set_level("WARNING")

        _, kwargs, _ = apply_integrations_to_sft_config(
            integrations=integrations,
            job_ctx=job_ctx,
            output_name="my-output",
            workspace_path=tmp_path,
            model_name="meta/llama",
        )

        assert kwargs == {"run_name": "w-run"}
        assert "differ" in caplog.text
