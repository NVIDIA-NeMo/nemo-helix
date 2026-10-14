# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded reconciliation between scaled-evals queues and Platform Jobs."""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any

from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities.base import SyncEntityClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_scaled_evals_plugin.jobs.benchmark_archive_build import BenchmarkArchiveBuildJob
from nemo_scaled_evals_plugin.jobs.naming import benchmark_archive_job_name, task_image_build_job_name
from nemo_scaled_evals_plugin.jobs.specs import BenchmarkArchiveBuildSpec, TaskImageBuildSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import TaskImageBuildJob
from nemo_scaled_evals_plugin.submitter import EvaluationSubmitter
from scaled_evals.api import entity_store
from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveStore
from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.db import pooled_connection
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.repositories.ops_repository import OperationsRepository
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskBuildJob, TaskStore
from scaled_evals.dispatch.worker import Dispatcher, _retry_delay_seconds, teardown_orphaned_execution

LOG = logging.getLogger(__name__)
_ACTIVE_JOB_STATUSES = {
    HelixJobStatus.CREATED,
    HelixJobStatus.PENDING,
    HelixJobStatus.ACTIVE,
    HelixJobStatus.PAUSED,
    HelixJobStatus.PAUSING,
    HelixJobStatus.RESUMING,
    HelixJobStatus.CANCELLING,
}
# Statuses in which no task has executed yet, so no sandbox can exist.
_PRELAUNCH_JOB_STATUSES = {
    HelixJobStatus.CREATED,
    HelixJobStatus.PENDING,
}
# Outlasts the slowest teardown (OpenSandbox's cleanup timeout plus a minute, or
# kubectl's 60 s waits) so another controller does not re-claim a live one.
CLEANUP_CLAIM_TIMEOUT_SECONDS = 600.0
# The evaluation Job builds its own evidence and archive as soon as it records a
# terminal status. The backstop leaves unclaimed rows alone this long so it does
# not take that work from a live Job.
FINALIZE_GRACE_SECONDS = 60.0
# Evidence builds do not renew their lease, so the backstop treats a Job's claim
# as live for this long. Archive builds hold a row lock and are never taken over.
FINALIZE_CLAIM_TIMEOUT_SECONDS = 600.0


class ScaledEvalsJobsController(NemoController):
    """Submit and reconcile scaled-evals work through Platform Jobs."""

    name = "scaled-evals-jobs"
    dependencies = ["jobs"]

    def __init__(self) -> None:
        self._jobs: AsyncJobsClient | None = None
        self._submitter: EvaluationSubmitter | None = None
        self._entities: SyncEntityClient | None = None
        self._worker_id = f"scaled-evals-jobs:{socket.gethostname()}:{time.time_ns()}"
        self._healthy = True
        self._background: dict[str, asyncio.Task[None]] = {}
        self._dispatcher = Dispatcher(
            connect=pooled_connection,
            worker_id=self._worker_id,
            claim_timeout=FINALIZE_CLAIM_TIMEOUT_SECONDS,
        )

    @property
    def jobs(self) -> AsyncJobsClient:
        if self._jobs is None:
            raise RuntimeError("scaled-evals Jobs controller has not started")
        return self._jobs

    @property
    def submitter(self) -> EvaluationSubmitter:
        if self._submitter is None:
            self._submitter = EvaluationSubmitter(self.jobs, self._worker_id, self._entities)
        return self._submitter

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    async def on_startup(self) -> None:
        client = get_async_nemo_client(as_service="scaled-evals", internal=True)
        self._jobs = AsyncJobsClient.from_client(client)
        if settings.platform_jobs_provider == "cpu" and not settings.platform_jobs_image:
            raise RuntimeError("SCALED_EVALS_PLATFORM_JOBS_IMAGE is required for the cpu Platform Jobs provider")
        self._entities = entity_store.entity_client()

    async def list_objects(self) -> list:
        """Return no objects because reconciliation is queue-oriented."""
        return []

    async def reconcile_one(self, obj: object) -> None:
        """Reject item reconciliation; this controller overrides reconcile()."""
        raise NotImplementedError

    async def reconcile(self) -> None:
        """Perform one bounded pass over build, evaluation, and cancellation work."""
        for name, phase in self._phases():
            try:
                await phase()
            except Exception:
                # One unprocessable row must not abort the pass, skip the
                # heartbeat, or mark this controller unhealthy. Readiness gates
                # the API, so a poison-pill row would otherwise take it offline.
                LOG.exception("scaled-evals Platform Jobs phase %s failed", name)
        try:
            await asyncio.to_thread(self._heartbeat)
        except Exception:
            self._healthy = False
            LOG.exception("scaled-evals Platform Jobs heartbeat failed")
        else:
            self._healthy = True

    def _phases(self) -> list[tuple[str, Callable[[], Awaitable[None]]]]:
        """Return the reconciliation phases, in execution order."""
        return [
            ("submit_build", self._drain(self._submit_one_build)),
            ("reconcile_builds", self._reconcile_builds),
            ("submit_evaluation", self._drain(self._submit_one_evaluation)),
            ("reconcile_evaluation", self._drain(self._reconcile_one_evaluation)),
            ("cancel_evaluations", self._cancel_evaluation_jobs),
            ("submit_benchmark_archives", self._submit_benchmark_archives),
            self._in_background("cleanup_executions", self._cleanup_one_execution),
            self._in_background("build_evidence", self._build_one_evidence),
            self._in_background("build_archives", self._build_one_archive),
            self._in_background("cleanup_benchmark_archives", self._cleanup_one_benchmark_archive),
        ]

    def _drain(self, step: Callable[[], Awaitable[bool]]) -> Callable[[], Coroutine[Any, Any, None]]:
        """Wrap a single-row step so one pass drains a bounded batch of rows."""

        async def drained() -> None:
            # A raising step ends the batch and is logged by reconcile(). Each
            # step marks its own row failed before raising, so continuing here
            # would turn a Jobs outage into a batch of failed rows per pass
            # instead of one. Progress is still one row per pass in that case.
            deadline = time.monotonic() + settings.platform_jobs_phase_budget_seconds
            for _ in range(settings.platform_jobs_phase_batch_size):
                claimed = await step()
                if not claimed or time.monotonic() >= deadline:
                    return

        return drained

    async def _submit_one_build(self) -> bool:
        """Submit one claimed build; return False only when the queue is empty."""
        job = await asyncio.to_thread(self._claim_build)
        if job is None:
            return False
        name = task_image_build_job_name(job.task_id, job.revision, job.attempt)
        bound = await asyncio.to_thread(self._bind_build, job, name)
        if not bound:
            return True
        spec = TaskImageBuildSpec.model_validate(
            {
                "task_id": job.task_id,
                "revision": job.revision,
                "build_attempt": job.attempt,
                "backend": job.backend,
                "object_key": job.object_key,
                "payload": job.payload,
            }
        )
        try:
            await self.submitter.create_job(name, TaskImageBuildJob, spec)
        except ConflictError:
            LOG.debug("Platform build job %s already exists", name)
        except Exception as exc:
            await asyncio.to_thread(self._fail_build, job, name, f"Platform Job submission failed: {exc}")
            raise
        return True

    async def _submit_one_evaluation(self) -> bool:
        """Submit one queued or orphaned evaluation; return False only when none remain."""
        return await self.submitter.submit()

    async def _reconcile_builds(self) -> None:
        for row in await asyncio.to_thread(self._list_builds):
            name = str(row["job_name"])
            try:
                status = (await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)).data()
            except NotFoundError:
                await asyncio.to_thread(
                    self._fail_build_row,
                    row,
                    "Platform build job disappeared before recording a result",
                )
                continue
            if status.status in _ACTIVE_JOB_STATUSES:
                await asyncio.to_thread(self._heartbeat_build, row)
            elif status.status in {
                HelixJobStatus.ERROR,
                HelixJobStatus.CANCELLED,
                HelixJobStatus.COMPLETED,
            }:
                detail = f"Platform build job ended as {status.status.value} without recording a ready task revision"
                await asyncio.to_thread(self._fail_build_row, row, detail)

    async def _reconcile_one_evaluation(self) -> bool:
        """Repair one stale evaluation; return False only when none are stale."""
        row = await asyncio.to_thread(self._claim_stale_evaluation)
        if row is None:
            return False
        name = str(row["dispatch_job_name"])
        try:
            status = (await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)).data()
        except NotFoundError:
            await asyncio.to_thread(self._fail_evaluation_job, row, "Platform evaluation job was not found")
            return True
        if status.status in _ACTIVE_JOB_STATUSES:
            # Hold the lease rather than releasing it. The claim orders by
            # dispatch_claimed_at, which a release does not change, so a
            # released row sorts first again and the drain re-claims this same
            # row instead of advancing. The lease lapses after claim_timeout,
            # which is soon enough for a job that is running normally.
            return True
        if status.status == HelixJobStatus.COMPLETED:
            detail = "Platform evaluation job completed without recording a terminal evaluation status"
        else:
            detail = f"Platform evaluation job ended as {status.status.value}"
        await asyncio.to_thread(self._fail_evaluation_job, row, detail)
        return True

    def _in_background(
        self, name: str, step: Callable[[], Awaitable[bool]]
    ) -> tuple[str, Callable[[], Awaitable[None]]]:
        """Return a phase that drains ``step`` in the background, one drain at a time.

        Teardowns can wait minutes on kubectl or OpenSandbox, and archive builds
        on object storage. Inside the pass they would delay submissions and the
        heartbeat, and a heartbeat older than ``platform_jobs_controller_stale_seconds``
        fails API readiness.
        """

        async def start() -> None:
            task = self._background.get(name)
            if task is None or task.done():
                task = asyncio.create_task(self._drain(step)(), name=f"scaled-evals-{name}")
                task.add_done_callback(_log_background_failure)
                self._background[name] = task

        return name, start

    async def _cleanup_one_execution(self) -> bool:
        """Tear down one orphaned execution runtime; return False only when none remain."""
        return await asyncio.to_thread(self._cleanup_execution)

    def _cleanup_execution(self) -> bool:
        cleanup = EvaluationStore().claim_cleanup(
            worker_id=self._worker_id, claim_timeout=CLEANUP_CLAIM_TIMEOUT_SECONDS
        )
        if cleanup is None:
            return False
        teardown_orphaned_execution(cleanup, worker_id=self._worker_id)
        return True

    # ponytail: evidence and archives are built by the evaluation Job itself.
    # These phases are the backstop for rows no Job finalizes (Jobs that died,
    # pre-launch cancels, controller-failed runs, archive rebuild requests). They
    # build in a controller thread, so a backlog competes with the controller's
    # CPU and connection pool. Upgrade path: a finalize Job per backlog row.
    async def _build_one_evidence(self) -> bool:
        evaluation_id = await asyncio.to_thread(self._dispatcher.claim_next_evidence, min_age=FINALIZE_GRACE_SECONDS)
        if evaluation_id is None:
            return False
        await asyncio.to_thread(self._dispatcher.build_evidence, evaluation_id)
        return True

    async def _build_one_archive(self) -> bool:
        evaluation_id = await asyncio.to_thread(self._dispatcher.claim_next_archive, min_age=FINALIZE_GRACE_SECONDS)
        if evaluation_id is None:
            return False
        await asyncio.to_thread(self._dispatcher.build_archive, evaluation_id)
        return True

    async def _submit_benchmark_archives(self) -> None:
        """Submit one Job per claimable archive attempt; the Job takes the claim."""
        for row in await asyncio.to_thread(self._list_claimable_benchmark_archives):
            run_id = str(row["benchmark_run_id"])
            name = benchmark_archive_job_name(run_id, str(row["generation"]), int(row["attempts"]) + 1)
            try:
                await self.submitter.create_job(
                    name,
                    BenchmarkArchiveBuildJob,
                    BenchmarkArchiveBuildSpec(benchmark_run_id=run_id),
                )
            except ConflictError:
                try:
                    response = await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)
                except NotFoundError:
                    continue  # deleted since the conflict; the next pass creates it
                if response.data().status not in _ACTIVE_JOB_STATUSES:
                    # The Job for this attempt ended without claiming it, so the
                    # same name would conflict on every pass. Counting it moves
                    # the next pass to a new name and fails the archive at three.
                    await asyncio.to_thread(self._skip_benchmark_archive_attempt, row)

    def _list_claimable_benchmark_archives(self) -> list[dict[str, Any]]:
        return BenchmarkArchiveStore().list_claimable(
            claim_timeout=Dispatcher.claim_timeout,
            limit=settings.platform_jobs_phase_batch_size,
        )

    def _skip_benchmark_archive_attempt(self, row: dict[str, Any]) -> None:
        BenchmarkArchiveStore().skip_attempt(
            str(row["benchmark_run_id"]),
            generation=str(row["generation"]),
            attempts=int(row["attempts"]),
        )

    async def _cleanup_one_benchmark_archive(self) -> bool:
        return await asyncio.to_thread(self._cleanup_benchmark_archive)

    def _cleanup_benchmark_archive(self) -> bool:
        run_id = BenchmarkArchiveStore().claim_cleanup(
            interval_seconds=settings.benchmark_archive_cleanup_interval_seconds,
        )
        if run_id is None:
            return False
        self._dispatcher.cleanup_benchmark_archives(run_id)
        return True

    async def _cancel_evaluation_jobs(self) -> None:
        for row in await asyncio.to_thread(self._list_cancelled_evaluations):
            name = str(row["dispatch_job_name"])
            try:
                status = (await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)).data()
            except NotFoundError:
                await asyncio.to_thread(self._settle_cancel_teardown, row, name)
                continue
            if status.status in _PRELAUNCH_JOB_STATUSES:
                # No task has run, so no sandbox can exist. Stop the Job before
                # it pulls an image and executes an already-cancelled run.
                # Cancelling an ACTIVE Job instead would race a task that has
                # launched a sandbox but not yet persisted its handle.
                with suppress(ConflictError):
                    await self.jobs.cancel_job(workspace=settings.platform_jobs_workspace, name=name)
                continue
            if status.status in _ACTIVE_JOB_STATUSES:
                # The task observes durable cancellation and performs runtime
                # teardown. Deleting its outer Job here would bypass that path.
                continue
            await asyncio.to_thread(self._settle_cancel_teardown, row, name)

    def _claim_build(self) -> TaskBuildJob | None:
        return TaskStore().claim_next(
            worker_id=self._worker_id,
            claim_timeout=TaskBuildWorker.claim_timeout,
            max_attempts=TaskBuildWorker.max_attempts,
        )

    def _bind_build(self, job: TaskBuildJob, name: str) -> bool:
        return TaskStore().bind_platform_job(
            job.task_id,
            job.revision,
            worker_id=self._worker_id,
            job_name=name,
        )

    def _list_builds(self) -> list[dict[str, Any]]:
        return TaskStore().list_platform_jobs()

    def _heartbeat_build(self, row: dict[str, Any]) -> bool:
        return TaskStore().heartbeat(
            str(row["task_id"]),
            int(row["revision"]),
            worker_id=str(row["job_name"]),
        )

    def _fail_build(self, job: TaskBuildJob, name: str, detail: str) -> bool:
        return self._fail_build_row(
            {
                "task_id": job.task_id,
                "revision": job.revision,
                "build_attempts": job.attempt,
                "job_name": name,
            },
            detail,
        )

    def _fail_build_row(self, row: dict[str, Any], detail: str) -> bool:
        return TaskStore().retry_or_fail(
            str(row["task_id"]),
            int(row["revision"]),
            worker_id=str(row["job_name"]),
            build_error=detail,
            attempt=int(row["build_attempts"]),
            max_attempts=TaskBuildWorker.max_attempts,
            retry_delay=TaskBuildWorker.retry_delay,
        )

    def _claim_stale_evaluation(self) -> dict[str, Any] | None:
        return EvaluationStore().claim_stale_dispatch_job(
            stale_seconds=settings.dispatch_job_reconcile_stale_seconds,
            claim_timeout=TaskBuildWorker.claim_timeout,
            worker_id=self._worker_id,
        )

    def _fail_evaluation_job(self, row: dict[str, Any], detail: str) -> None:
        evaluation_id = str(row["id"])
        execution_number = int(row["current_execution"])
        EvaluationStore().record_dispatch_job_infrastructure_failure(
            evaluation_id,
            execution_number=execution_number,
            dispatch_job_name=str(row["dispatch_job_name"]),
            reconcile_worker_id=self._worker_id,
            failure_code="HelixJobInfrastructureError",
            detail=detail,
            retry_delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
        )

    def _list_cancelled_evaluations(self) -> list[dict[str, Any]]:
        return EvaluationStore().list_cancelled_platform_jobs()

    def _settle_cancel_teardown(self, row: dict[str, Any], job_name: str) -> None:
        """Terminalize teardown for a cancelled evaluation whose Job has ended."""
        store = EvaluationStore()
        if row.get("backend_handle"):
            # The Job is gone, so no task will release the sandbox. Record the
            # failure rather than leaving the evaluation pending forever at the
            # head of the teardown window, where it would also crowd out later
            # cancellations.
            store.record_cancel_teardown_failure(
                str(row["id"]),
                f"Platform evaluation job {job_name} ended before releasing its runtime",
            )
            return
        store.record_cancel_teardown_succeeded(str(row["id"]))

    def _heartbeat(self) -> None:
        with pooled_connection() as conn:
            OperationsRepository(conn).heartbeat_service("platform_jobs_controller", self._worker_id)


def _log_background_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        LOG.error("scaled-evals Platform Jobs background phase %s failed", task.get_name(), exc_info=task.exception())
