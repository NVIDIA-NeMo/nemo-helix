# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus

pytest.importorskip("scaled_evals")

import nemo_scaled_evals_plugin.controller as controller_module
from nemo_scaled_evals_plugin.controller import ScaledEvalsJobsController
from scaled_evals.api.repositories.build_repository import TaskBuildJob
from scaled_evals.api.settings import settings

_CREATED = datetime(2026, 9, 29, tzinfo=UTC)


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
    controller._entities = MagicMock()
    submitter = controller.submitter
    write_inputs = MagicMock()
    monkeypatch.setattr(submitter, "_write_inputs", write_inputs)
    monkeypatch.setattr(
        submitter,
        "_claim",
        lambda _evaluation_id: {"id": "eval_1", "previous_status": "queued", "status": "provisioning"},
    )
    monkeypatch.setattr(
        submitter,
        "_load",
        lambda _evaluation_id: {
            "id": "eval_1",
            "status": "provisioning",
            "runtime": "sandbox_k8s",
            "current_execution": 3,
        },
    )
    monkeypatch.setattr(submitter, "_record", record_evaluation)
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
    # The launch inputs are stored before the Job that reads them exists.
    write_inputs.assert_called_once_with("eval_1", 3, evaluation_request.name)
    assert evaluation_request.spec["inputs_workspace"] == settings.entity_store_workspace
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
        "job-1": HelixJobStatus.ACTIVE,
        "job-2": HelixJobStatus.ERROR,
        "job-3": HelixJobStatus.PENDING,
        "job-4": HelixJobStatus.COMPLETED,
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
async def test_an_active_job_keeps_its_reconcile_lease_so_the_drain_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-running Job must not hand its row back to the same drain pass.

    The claim orders by `dispatch_claimed_at`, which releasing the reconcile
    lease does not change, so a released row sorts first again. Releasing it
    made the drain re-claim the head of the queue until the batch ran out and
    never reach the rows behind it.
    """
    rows = [
        {"id": "ev_1", "current_execution": 1, "dispatch_job_name": "job-1"},
        {"id": "ev_2", "current_execution": 1, "dispatch_job_name": "job-2"},
    ]
    leased: set[str] = set()

    def _claim() -> dict[str, Any] | None:
        # Stands in for claim_stale_dispatch_job: the first unleased row in a
        # fixed order, so a released row is handed straight back.
        for row in rows:
            if row["id"] not in leased:
                leased.add(str(row["id"]))
                return row
        return None

    inspected: list[str] = []

    async def _get_job_status(*, workspace: str, name: str) -> Any:
        inspected.append(name)
        return SimpleNamespace(data=lambda: SimpleNamespace(status=HelixJobStatus.ACTIVE))

    repo = MagicMock()
    # Releasing really does make the row claimable again, which is the whole
    # mechanism of the spin. Without this the fake could never reproduce it.
    repo.release_dispatch_reconcile_claim.side_effect = lambda evaluation_id, **_: leased.discard(evaluation_id)
    controller = ScaledEvalsJobsController()
    controller._jobs = cast(Any, SimpleNamespace(get_job_status=_get_job_status))
    monkeypatch.setattr(controller, "_claim_stale_evaluation", _claim)
    monkeypatch.setattr(controller_module, "pooled_connection", lambda *a, **k: nullcontext(MagicMock()))
    monkeypatch.setattr(controller_module, "EvaluationRepository", lambda conn: repo)

    await controller._drain(controller._reconcile_one_evaluation)()

    # Each row inspected once, so the pass reached the end of the queue instead
    # of spending the whole batch on its head.
    assert inspected == ["job-1", "job-2"]
    repo.release_dispatch_reconcile_claim.assert_not_called()


@pytest.mark.asyncio
async def test_controller_heartbeat_survives_an_unprocessable_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the module global, not the resolved singleton: other tests reset
    # the lazy settings instance, so its identity is not stable across the suite.
    monkeypatch.setattr(
        controller_module,
        "settings",
        SimpleNamespace(
            platform_jobs_phase_batch_size=20,
            platform_jobs_phase_budget_seconds=5.0,
        ),
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
    for name in (
        "_cleanup_one_execution",
        "_build_one_evidence",
        "_build_one_archive",
        "_submit_benchmark_archives",
        "_cleanup_one_benchmark_archive",
    ):
        monkeypatch.setattr(controller, name, _phase(name))
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
        "_cleanup_one_execution",
        "_build_one_evidence",
        "_build_one_archive",
        "_submit_benchmark_archives",
        "_cleanup_one_benchmark_archive",
    ]
    assert heartbeats == [1]
    assert controller.is_healthy

    def _unreachable_database() -> None:
        raise RuntimeError("database is unreachable")

    monkeypatch.setattr(controller, "_heartbeat", _unreachable_database)
    await controller.reconcile()

    assert not controller.is_healthy


@pytest.mark.asyncio
async def test_controller_drains_a_bounded_batch_per_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controller_module,
        "settings",
        SimpleNamespace(platform_jobs_phase_batch_size=3, platform_jobs_phase_budget_seconds=60.0),
    )
    controller = ScaledEvalsJobsController()

    # Submitting one row per pass caps a fanned-out benchmark at roughly six
    # rows a minute, so a pass has to keep claiming until the queue is empty.
    queued = 0

    async def _two_queued_rows() -> bool:
        nonlocal queued
        queued += 1
        return queued < 3

    await controller._drain(_two_queued_rows)()
    assert queued == 3

    # The batch size bounds a backlog so one phase cannot starve the others or
    # delay the heartbeat that gates API readiness.
    endless = 0

    async def _endless_queue() -> bool:
        nonlocal endless
        endless += 1
        return True

    await controller._drain(_endless_queue)()
    assert endless == 3

    # Slow rows hit the wall-clock budget before the count limit.
    monkeypatch.setattr(controller_module.settings, "platform_jobs_phase_budget_seconds", 0.0)
    endless = 0
    await controller._drain(_endless_queue)()
    assert endless == 1

    # A raising row ends the batch and surfaces to reconcile(), which logs it.
    # Each step fails its own row first, so continuing would turn one Jobs
    # outage into a batch of failed rows per pass.
    async def _poison_pill_row() -> bool:
        raise RuntimeError("poison-pill row")

    with pytest.raises(RuntimeError):
        await controller._drain(_poison_pill_row)()


def test_immediate_submission_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_scaled_evals_plugin.submitter as submitter_module

    client = MagicMock(side_effect=RuntimeError("platform unreachable"))
    monkeypatch.setattr(submitter_module, "get_async_nemo_client", client)

    # The committed row is the handoff: a failure here must not fail the API
    # request, because the controller submits the row on its next pass.
    submitter_module.submit_evaluation_now("eval_1")
    client.assert_called_once()


def test_controller_tears_down_orphaned_executions(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup = {"id": 7, "evaluation_id": "eval_1", "execution_number": 2, "runtime": "sandbox_k8s"}
    claims = iter([cleanup, None])
    monkeypatch.setattr(controller_module, "pooled_connection", lambda *a, **k: nullcontext(MagicMock()))
    monkeypatch.setattr(
        controller_module,
        "ExecutionCleanupRepository",
        lambda _conn: SimpleNamespace(claim_one=lambda **_kwargs: next(claims)),
    )
    teardown = MagicMock()
    monkeypatch.setattr(controller_module, "teardown_orphaned_execution", teardown)
    controller = ScaledEvalsJobsController()

    assert "cleanup_executions" in dict(controller._phases())
    assert controller._cleanup_execution() is True
    assert controller._cleanup_execution() is False
    teardown.assert_called_once_with(
        cleanup,
        worker_id=controller._worker_id,
        connect=controller_module.pooled_connection,
    )


@pytest.mark.asyncio
async def test_controller_owns_evidence_and_archive_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller_module, "pooled_connection", lambda *a, **k: nullcontext(MagicMock()))
    controller = ScaledEvalsJobsController()
    controller._jobs = cast(Any, AsyncMock())
    phases = dict(controller._phases())
    for name in ("build_evidence", "build_archives", "submit_benchmark_archives", "cleanup_benchmark_archives"):
        assert name in phases

    dispatcher = MagicMock()
    dispatcher.claim_next_evidence.side_effect = ["eval_1", None]
    dispatcher.claim_next_archive.side_effect = [None]
    controller._dispatcher = dispatcher
    await phases["build_evidence"]()
    await phases["build_archives"]()
    dispatcher.build_evidence.assert_called_once_with("eval_1")
    dispatcher.build_archive.assert_not_called()

    monkeypatch.setattr(
        controller,
        "_list_claimable_benchmark_archives",
        lambda: [
            {"benchmark_run_id": "run_1", "generation": "aaaaaaaa-1", "attempts": 0},
            {"benchmark_run_id": "run_2", "generation": "bbbbbbbb-2", "attempts": 2},
        ],
    )
    create_job = AsyncMock(
        side_effect=[None, ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://jobs")))]
    )
    monkeypatch.setattr(controller.submitter, "create_job", create_job)
    await phases["submit_benchmark_archives"]()
    assert [call.args[0] for call in create_job.call_args_list] == [
        "scaled-evals-benchmark-archive-run_1-aaaaaaaa-a1",
        "scaled-evals-benchmark-archive-run_2-bbbbbbbb-a3",
    ]
    assert create_job.call_args.args[2].benchmark_run_id == "run_2"

    repo = MagicMock()
    repo.claim_cleanup.side_effect = ["run_1", None]
    monkeypatch.setattr(controller_module, "BenchmarkArchiveRepository", lambda _conn: repo)
    await phases["cleanup_benchmark_archives"]()
    dispatcher.cleanup_benchmark_archives.assert_called_once_with("run_1")


def test_submitter_stores_immutable_launch_inputs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_scaled_evals_plugin.submitter as submitter_module
    from nemo_helix_plugin.entities.base import EntityConflictError

    row = {"id": "eval_1", "name": "e", "status": "provisioning", "current_execution": 3, "created_at": _CREATED}
    repo = MagicMock()
    repo.load_for_dispatch.return_value = row
    monkeypatch.setattr(submitter_module, "pooled_connection", lambda: nullcontext(MagicMock()))
    monkeypatch.setattr(submitter_module, "EvaluationRepository", lambda _conn: repo)
    entities = MagicMock()
    submitter = submitter_module.EvaluationSubmitter(AsyncMock(), "w", entities)

    submitter._write_inputs("eval_1", 3, "job-e3")
    entity = entities.create.call_args.args[0]
    assert (entity.name, entity.evaluation_id, entity.execution_number) == ("job-e3", "eval_1", 3)
    # State columns are dropped, and what remains is JSON-safe.
    assert entity.inputs == {"id": "eval_1", "name": "e", "created_at": _CREATED.isoformat()}

    # A resubmission finds the entity an earlier attempt wrote.
    entities.create.side_effect = EntityConflictError("exists")
    submitter._write_inputs("eval_1", 3, "job-e3")
