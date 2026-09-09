# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus

pytest.importorskip("scaled_evals")

import nemo_scaled_evals_plugin.controller as controller_module
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
async def test_controller_settles_every_cancelled_evaluation_it_inspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"id": "running-sandbox", "dispatch_job_name": "job-1", "backend_handle": {"pod": "sandbox"}},
        {"id": "orphaned-sandbox", "dispatch_job_name": "job-2", "backend_handle": {"pod": "sandbox"}},
        {"id": "queued", "dispatch_job_name": "job-3", "backend_handle": None},
        {"id": "finished", "dispatch_job_name": "job-4", "backend_handle": None},
        {"id": "vanished", "dispatch_job_name": "job-5", "backend_handle": None},
    ]
    statuses = {
        "job-1": PlatformJobStatus.ACTIVE,
        "job-2": PlatformJobStatus.ERROR,
        "job-3": PlatformJobStatus.PENDING,
        "job-4": PlatformJobStatus.COMPLETED,
    }

    async def _get_job_status(*, workspace: str, name: str) -> Any:
        if name not in statuses:
            raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://jobs")))
        return SimpleNamespace(data=lambda: SimpleNamespace(status=statuses[name]))

    cancelled: list[str] = []

    async def _cancel_job(*, workspace: str, name: str) -> None:
        cancelled.append(name)

    repo = MagicMock()
    controller = ScaledEvalsJobsController()
    controller._jobs = cast(Any, SimpleNamespace(get_job_status=_get_job_status, cancel_job=_cancel_job))
    monkeypatch.setattr(controller, "_list_cancelled_evaluations", lambda: rows)
    monkeypatch.setattr(controller_module, "pooled_connection", lambda *a, **k: nullcontext(MagicMock()))
    monkeypatch.setattr(controller_module, "EvaluationRepository", lambda conn: repo)

    await controller._cancel_evaluation_jobs()

    # A queued Job is stopped before it pulls an image and runs a cancelled
    # evaluation. A running one is left alone because its task owns sandbox
    # teardown, and cancelling it would race a sandbox launch.
    assert cancelled == ["job-3"]
    # A dead Job still holding a sandbox has no task left to release it. It must
    # terminalize, or it pins the head of the teardown window forever and hides
    # the leaked runtime.
    assert [call.args[0] for call in repo.record_cancel_teardown_failure.call_args_list] == ["orphaned-sandbox"]
    assert "job-2" in repo.record_cancel_teardown_failure.call_args.args[1]
    assert [call.args[0] for call in repo.record_cancel_teardown_succeeded.call_args_list] == [
        "finished",
        "vanished",
    ]


@pytest.mark.asyncio
async def test_controller_heartbeat_survives_an_unprocessable_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the module global, not the resolved singleton: other tests reset
    # the lazy settings instance, so its identity is not stable across the suite.
    monkeypatch.setattr(
        controller_module,
        "settings",
        SimpleNamespace(platform_build_jobs_enabled=True, platform_evaluation_jobs_enabled=True),
    )
    controller = ScaledEvalsJobsController()
    calls: list[str] = []

    def _phase(name: str, *, fails: bool = False) -> Any:
        async def _run() -> None:
            calls.append(name)
            if fails:
                raise RuntimeError("poison-pill row")

        return _run

    monkeypatch.setattr(controller, "_submit_one_build", _phase("_submit_one_build", fails=True))
    monkeypatch.setattr(controller, "_reconcile_builds", _phase("_reconcile_builds"))
    monkeypatch.setattr(controller, "_submit_one_evaluation", _phase("_submit_one_evaluation"))
    monkeypatch.setattr(controller, "_reconcile_one_evaluation", _phase("_reconcile_one_evaluation"))
    monkeypatch.setattr(controller, "_cancel_evaluation_jobs", _phase("_cancel_evaluation_jobs"))
    heartbeats: list[int] = []
    monkeypatch.setattr(controller, "_heartbeat", lambda: heartbeats.append(1))

    await controller.reconcile()

    # A failing phase must not abort the pass, skip the heartbeat, or report the
    # controller unhealthy: readiness gates the API, so either would take the
    # control plane offline over one bad row.
    assert calls == [
        "_submit_one_build",
        "_reconcile_builds",
        "_submit_one_evaluation",
        "_reconcile_one_evaluation",
        "_cancel_evaluation_jobs",
    ]
    assert heartbeats == [1]
    assert controller.is_healthy

    def _unreachable_database() -> None:
        raise RuntimeError("database is unreachable")

    monkeypatch.setattr(controller, "_heartbeat", _unreachable_database)
    await controller.reconcile()

    assert not controller.is_healthy
