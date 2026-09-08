# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock

import nemo_scaled_evals_plugin.jobs.evaluation_execution as evaluation_job_module
import nemo_scaled_evals_plugin.jobs.task_image_build as build_job_module
import nemo_scaled_evals_plugin.tasks.evaluation_execution as evaluation_task_module
import pytest
from nemo_platform_plugin.jobs.providers import CPUExecutionProvider, SubprocessExecutionProvider
from nemo_scaled_evals_plugin.jobs.evaluation_execution import EvaluationExecutionJob
from nemo_scaled_evals_plugin.jobs.naming import (
    evaluation_execution_job_name,
    task_image_build_job_name,
)
from nemo_scaled_evals_plugin.jobs.specs import EvaluationExecutionSpec, TaskImageBuildSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import TaskImageBuildJob
from pydantic import ValidationError
from scaled_evals.api.settings import settings


def _build_spec() -> TaskImageBuildSpec:
    return TaskImageBuildSpec(
        task_id="task_1",
        revision=2,
        build_attempt=3,
        backend="buildkit",
        object_key="tasks/task_1/revisions/2.tar.gz",
    )


def _evaluation_spec() -> EvaluationExecutionSpec:
    return EvaluationExecutionSpec(
        evaluation_id="eval_1",
        execution_number=4,
        runtime="sandbox_k8s",
        deadline_seconds=7200,
    )


def test_specs_and_deterministic_names() -> None:
    assert _build_spec().model_dump()["payload"] == {}
    assert _evaluation_spec().deadline_seconds == 7200
    with pytest.raises(ValidationError):
        TaskImageBuildSpec.model_validate(
            {
                **_build_spec().model_dump(),
                "build_attempt": 0,
                "credentials": {"token": "secret"},
            }
        )
    with pytest.raises(ValidationError):
        EvaluationExecutionSpec.model_validate({**_evaluation_spec().model_dump(), "deadline_seconds": 0})

    build_name = task_image_build_job_name("task_1", 2, 3)
    assert build_name == task_image_build_job_name("task_1", 2, 3)
    assert build_name != task_image_build_job_name("task_1", 2, 4)
    long_name = evaluation_execution_job_name("EVAL/" + "x" * 100, 4)
    assert len(long_name) <= 63 - len("job-fileset-")
    assert len(f"job-fileset-{long_name}") <= 63
    assert re.fullmatch(r"^[a-z](?!.*--)[a-z0-9\-@.+_]{1,62}(?<!-)$", long_name)


@pytest.mark.asyncio
async def test_compile_outputs_use_current_platform_job_models(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "platform_jobs_postgres_password_secret", "scaled-evals-postgres-password")
    monkeypatch.setattr(
        settings,
        "platform_jobs_credentials_encryption_key_secret",
        "scaled-evals-credentials-encryption-key",
    )
    monkeypatch.setattr(settings, "platform_jobs_registry_auth_secret", "scaled-evals-registry-auth")
    build = await TaskImageBuildJob.compile(
        workspace="default",
        spec=_build_spec(),
        entity_client=object(),
        job_name=None,
        async_sdk=object(),
        profile="build-profile",
        options={"scaled_evals": {"application_image": "registry.example/scaled-evals:test"}},
    )
    evaluation = await EvaluationExecutionJob.compile(
        workspace="default",
        spec=_evaluation_spec(),
        entity_client=object(),
        job_name=None,
        async_sdk=object(),
        profile="evaluation-profile",
        options={"scaled_evals": {"application_image": "registry.example/scaled-evals:test"}},
    )

    build_step = build.steps[0]
    assert build_step.name == "task-image-build"
    assert build_step.executor.provider == "cpu"
    assert build_step.executor.model_dump(exclude_unset=True)["provider"] == "cpu"
    assert build_step.executor.profile == "build-profile"
    assert build_step.executor.container.image == "registry.example/scaled-evals:test"
    assert build_step.executor.container.command == ["nemo_scaled_evals_plugin.tasks.task_image_build"]
    assert {
        item.name: item.from_secret.name if item.from_secret else None for item in build_step.environment or []
    } == {
        "PGPASSWORD": "scaled-evals-postgres-password",
        "CREDENTIALS_ENCRYPTION_KEY": "scaled-evals-credentials-encryption-key",
        "TASK_IMAGE_REGISTRY_AUTH_JSON": "scaled-evals-registry-auth",
    }
    assert build_step.config == _build_spec().model_dump(mode="json")

    evaluation_step = evaluation.steps[0]
    assert evaluation_step.name == "evaluation-execution"
    assert isinstance(evaluation_step.executor, CPUExecutionProvider)
    assert evaluation_step.executor.profile == "evaluation-profile"
    assert evaluation_step.executor.container.command == ["nemo_scaled_evals_plugin.tasks.evaluation_execution"]
    assert {
        item.name: item.from_secret.name if item.from_secret else None for item in evaluation_step.environment or []
    } == {
        "PGPASSWORD": "scaled-evals-postgres-password",
        "CREDENTIALS_ENCRYPTION_KEY": "scaled-evals-credentials-encryption-key",
        "TASK_IMAGE_REGISTRY_AUTH_JSON": "scaled-evals-registry-auth",
    }
    assert evaluation_step.executor.resources.requests.cpu == "50m"
    assert evaluation_step.executor.resources.limits.memory == "1Gi"
    assert evaluation_step.config["deadline_seconds"] == 7200

    local_build = await TaskImageBuildJob.compile(
        workspace="default",
        spec=_build_spec(),
        entity_client=object(),
        job_name=None,
        async_sdk=object(),
        options={"scaled_evals": {"provider": "subprocess"}},
    )
    assert isinstance(local_build.steps[0].executor, SubprocessExecutionProvider)
    assert local_build.steps[0].executor.model_dump(exclude_unset=True)["provider"] == "subprocess"
    assert local_build.steps[0].executor.command == [
        "python",
        "-m",
        "nemo_scaled_evals_plugin.tasks.task_image_build",
    ]


def test_direct_run_reuses_existing_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = MagicMock()
    worker.run.return_value = True
    worker_factory = MagicMock(return_value=worker)
    monkeypatch.setattr(build_job_module, "TaskBuildWorker", worker_factory)
    assert TaskImageBuildJob().run(_build_spec().model_dump()) == {"status": "completed"}
    worker_factory.assert_called_once_with(worker_id="scaled-evals-build-task_1-r2-a3")
    backend_job = worker.run.call_args.args[0]
    assert (backend_job.task_id, backend_job.revision, backend_job.attempt) == ("task_1", 2, 3)
    assert backend_job.credentials == {}
    worker.run.return_value = False
    assert TaskImageBuildJob().run(_build_spec().model_dump()) == {"status": "failed"}

    dispatcher = MagicMock()
    monkeypatch.setattr(evaluation_job_module, "Dispatcher", lambda: dispatcher)
    assert EvaluationExecutionJob().run(_evaluation_spec().model_dump()) == {
        "status": "completed",
        "evaluation_id": "eval_1",
        "execution_number": 4,
    }
    dispatcher.run.assert_called_once_with("eval_1", maintain_claim=False, expected_execution_number=4)


def test_jobs_are_discovered_from_plugin_entry_points() -> None:
    from nemo_platform_plugin.discovery import discover, discover_controllers, discover_jobs
    from nemo_scaled_evals_plugin.controller import ScaledEvalsJobsController

    discover.cache_clear()
    jobs = discover_jobs()
    assert jobs["scaled-evals.task-image-build"] is TaskImageBuildJob
    assert jobs["scaled-evals.evaluation-execution"] is EvaluationExecutionJob
    assert discover_controllers()["scaled-evals-jobs"] is ScaledEvalsJobsController


def test_evaluation_task_creates_in_cluster_kubeconfig(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "kubeconfig"
    monkeypatch.setenv("KUBECONFIG", str(path))
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT_HTTPS", "6443")
    monkeypatch.setenv("POD_NAMESPACE", "test-namespace")

    evaluation_task_module._ensure_in_cluster_kubeconfig()

    config = json.loads(path.read_text())
    assert config["clusters"][0]["cluster"]["server"] == "https://10.0.0.1:6443"
    assert config["contexts"][0]["context"]["namespace"] == "test-namespace"
    assert config["users"][0]["user"]["tokenFile"].endswith("/serviceaccount/token")
