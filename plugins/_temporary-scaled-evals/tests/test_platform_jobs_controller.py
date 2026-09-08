# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus

pytest.importorskip("scaled_evals")

from nemo_scaled_evals_plugin.controller import ScaledEvalsJobsController
from scaled_evals.api.repositories.build_repository import TaskBuildJob
from scaled_evals.api.settings import settings


@pytest.mark.asyncio
async def test_controller_submits_deterministic_reference_only_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = ScaledEvalsJobsController()
    jobs = AsyncMock()
    controller._jobs = cast(Any, jobs)
    response = MagicMock()
    response.data.return_value = SimpleNamespace(id="platform-job-id")
    jobs.create_job.return_value = response
    build = TaskBuildJob(
        task_id="task_1",
        revision=2,
        backend="prebuilt",
        payload={"image_ref": "registry.example/task@sha256:abc"},
        credentials={"REGISTRY_PASSWORD": "must-not-leak"},
        object_key="tasks/task_1/revisions/2/task.tar.gz",
        attempt=1,
    )
    bind_build = MagicMock(return_value=True)
    record_evaluation = MagicMock()
    monkeypatch.setattr(controller, "_claim_build", lambda: build)
    monkeypatch.setattr(controller, "_bind_build", bind_build)
    monkeypatch.setattr(
        controller,
        "_claim_evaluation",
        lambda: {"id": "eval_1", "previous_status": "queued", "status": "provisioning"},
    )
    monkeypatch.setattr(
        controller,
        "_load_evaluation",
        lambda _evaluation_id: {
            "id": "eval_1",
            "status": "provisioning",
            "runtime": "sandbox_k8s",
            "current_execution": 3,
        },
    )
    monkeypatch.setattr(controller, "_record_evaluation_job", record_evaluation)
    resolved_settings = settings._resolve()
    monkeypatch.setattr(resolved_settings, "platform_jobs_image", "registry.example/scaled-evals@sha256:def")

    await controller._submit_one_build()
    build_request = jobs.create_job.await_args_list[0].kwargs["body"]
    assert build_request.name == "scaled-evals-build-task_1-r2-a1"
    assert build_request.spec["task_id"] == "task_1"
    assert "credentials" not in build_request.spec
    assert build_request.platform_spec.steps[0].executor.container.image == ("registry.example/scaled-evals@sha256:def")
    bind_build.assert_called_once_with(build, build_request.name)

    await controller._submit_one_evaluation()
    evaluation_request = jobs.create_job.await_args_list[1].kwargs["body"]
    assert evaluation_request.name == "scaled-evals-evaluation-eval_1-e3"
    assert evaluation_request.spec["evaluation_id"] == "eval_1"
    assert evaluation_request.spec["execution_number"] == 3
    assert "credentials" not in evaluation_request.spec
    record_evaluation.assert_called_once_with(
        "eval_1",
        3,
        evaluation_request.name,
        "platform-job-id",
    )


@pytest.mark.asyncio
async def test_controller_preserves_sandbox_teardown_during_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = ScaledEvalsJobsController()
    jobs = AsyncMock()
    controller._jobs = cast(Any, jobs)
    jobs.get_job_status.side_effect = [
        MagicMock(data=lambda: SimpleNamespace(status=PlatformJobStatus.ACTIVE)),
        MagicMock(data=lambda: SimpleNamespace(status=PlatformJobStatus.COMPLETED)),
    ]
    completed = MagicMock()
    monkeypatch.setattr(
        controller,
        "_list_cancelled_evaluations",
        lambda: [
            {"id": "with-sandbox", "dispatch_job_name": "job-1", "backend_handle": {"pod": "sandbox"}},
            {"id": "not-started", "dispatch_job_name": "job-2", "backend_handle": None},
            {"id": "finished", "dispatch_job_name": "job-3", "backend_handle": None},
        ],
    )
    monkeypatch.setattr(controller, "_complete_cancel_teardown", completed)

    await controller._cancel_evaluation_jobs()

    jobs.cancel_job.assert_not_awaited()
    completed.assert_called_once_with({"id": "finished", "dispatch_job_name": "job-3", "backend_handle": None})
