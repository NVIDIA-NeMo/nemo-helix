# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded reconciliation between scaled-evals queues and Platform Jobs."""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from typing import Any

from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.controller import NemoController
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest
from nemo_platform_plugin.sdk_provider import get_async_platform_sdk
from nemo_scaled_evals_plugin.jobs.evaluation_execution import EvaluationExecutionJob
from nemo_scaled_evals_plugin.jobs.naming import (
    evaluation_execution_job_name,
    task_image_build_job_name,
)
from nemo_scaled_evals_plugin.jobs.specs import EvaluationExecutionSpec, TaskImageBuildSpec
from nemo_scaled_evals_plugin.jobs.task_image_build import TaskImageBuildJob
from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.db import pooled_connection
from scaled_evals.api.repositories.build_repository import TaskBuildJob, TaskBuildRepository
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.repositories.ops_repository import OperationsRepository
from scaled_evals.api.settings import settings
from scaled_evals.dispatch.worker import _retry_delay_seconds

LOG = logging.getLogger(__name__)
_ACTIVE_JOB_STATUSES = {
    PlatformJobStatus.CREATED,
    PlatformJobStatus.PENDING,
    PlatformJobStatus.ACTIVE,
    PlatformJobStatus.PAUSED,
    PlatformJobStatus.PAUSING,
    PlatformJobStatus.RESUMING,
    PlatformJobStatus.CANCELLING,
}


class ScaledEvalsJobsController(NemoController):
    """Submit and reconcile scaled-evals work through Platform Jobs."""

    name = "scaled-evals-jobs"
    dependencies = ["jobs"]

    def __init__(self) -> None:
        self._jobs: AsyncJobsClient | None = None
        self._worker_id = f"scaled-evals-jobs:{socket.gethostname()}:{time.time_ns()}"
        self._healthy = True

    @property
    def jobs(self) -> AsyncJobsClient:
        if self._jobs is None:
            raise RuntimeError("scaled-evals Jobs controller has not started")
        return self._jobs

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    async def on_startup(self) -> None:
        sdk = get_async_platform_sdk(as_service="scaled-evals", internal=True)
        self._jobs = client_from_platform(sdk, AsyncJobsClient)
        if (
            settings.platform_jobs_provider == "cpu"
            and (settings.platform_build_jobs_enabled or settings.platform_evaluation_jobs_enabled)
            and not settings.platform_jobs_image
        ):
            raise RuntimeError("SCALED_EVALS_PLATFORM_JOBS_IMAGE is required when Platform Jobs are enabled")

    async def list_objects(self) -> list:
        """Return no objects because reconciliation is queue-oriented."""
        return []

    async def reconcile_one(self, obj: object) -> None:
        """Reject item reconciliation; this controller overrides reconcile()."""
        raise NotImplementedError

    async def reconcile(self) -> None:
        """Perform one bounded pass over build, evaluation, and cancellation work."""
        try:
            if settings.platform_build_jobs_enabled:
                await self._submit_one_build()
                await self._reconcile_builds()
            if settings.platform_evaluation_jobs_enabled:
                await self._submit_one_evaluation()
                await self._reconcile_one_evaluation()
                await self._cancel_evaluation_jobs()
            await asyncio.to_thread(self._heartbeat)
        except Exception:
            self._healthy = False
            LOG.exception("scaled-evals Platform Jobs reconciliation failed")
        else:
            self._healthy = True

    async def _submit_one_build(self) -> None:
        job = await asyncio.to_thread(self._claim_build)
        if job is None:
            return
        name = task_image_build_job_name(job.task_id, job.revision, job.attempt)
        bound = await asyncio.to_thread(self._bind_build, job, name)
        if not bound:
            return
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
            await self._create_job(name, TaskImageBuildJob, spec)
        except ConflictError:
            LOG.debug("Platform build job %s already exists", name)
        except Exception as exc:
            await asyncio.to_thread(self._fail_build, job, name, f"Platform Job submission failed: {exc}")
            raise

    async def _submit_one_evaluation(self) -> None:
        row = await asyncio.to_thread(self._claim_evaluation)
        if row is None:
            return
        evaluation_id = str(row["id"])
        current = await asyncio.to_thread(self._load_evaluation, evaluation_id)
        if current is None:
            return
        execution_number = int(current.get("current_execution") or 1)
        name = evaluation_execution_job_name(evaluation_id, execution_number)
        spec = EvaluationExecutionSpec(
            evaluation_id=evaluation_id,
            execution_number=execution_number,
            runtime=str(current["runtime"]),
            deadline_seconds=max(
                1,
                int(settings.dispatch_run_poll_interval_seconds * settings.dispatch_run_max_polls),
            ),
        )
        try:
            platform_job = await self._create_job(name, EvaluationExecutionJob, spec)
        except ConflictError:
            platform_job = (await self.jobs.get_job(workspace=settings.platform_jobs_workspace, name=name)).data()
        except Exception:
            await asyncio.to_thread(self._retry_evaluation_submission, evaluation_id, execution_number)
            raise
        await asyncio.to_thread(
            self._record_evaluation_job,
            evaluation_id,
            execution_number,
            name,
            platform_job.id,
        )

    async def _create_job(
        self,
        name: str,
        job_cls: type[TaskImageBuildJob] | type[EvaluationExecutionJob],
        spec: TaskImageBuildSpec | EvaluationExecutionSpec,
    ) -> Any:
        platform_spec = await job_cls.compile(
            workspace=settings.platform_jobs_workspace,
            spec=spec,
            entity_client=object(),
            job_name=name,
            async_sdk=None,
            profile=settings.platform_jobs_profile,
            options={
                "scaled_evals": {
                    "application_image": settings.platform_jobs_image,
                    "provider": settings.platform_jobs_provider,
                }
            },
        )
        request = CreatePlatformJobRequest(
            name=name,
            description=job_cls.description,
            source=f"scaled-evals.{job_cls.name}",
            spec=spec.model_dump(mode="json"),
            platform_spec=platform_spec,
        )
        return (await self.jobs.create_job(workspace=settings.platform_jobs_workspace, body=request)).data()

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
                PlatformJobStatus.ERROR,
                PlatformJobStatus.CANCELLED,
                PlatformJobStatus.COMPLETED,
            }:
                detail = f"Platform build job ended as {status.status.value} without recording a ready task revision"
                await asyncio.to_thread(self._fail_build_row, row, detail)

    async def _reconcile_one_evaluation(self) -> None:
        row = await asyncio.to_thread(self._claim_stale_evaluation)
        if row is None:
            return
        name = str(row["dispatch_job_name"])
        try:
            status = (await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)).data()
        except NotFoundError:
            await asyncio.to_thread(self._fail_evaluation_job, row, "Platform evaluation job was not found")
            return
        if status.status in _ACTIVE_JOB_STATUSES:
            await asyncio.to_thread(self._release_evaluation_reconcile_claim, row)
            return
        if status.status == PlatformJobStatus.COMPLETED:
            detail = "Platform evaluation job completed without recording a terminal evaluation status"
        else:
            detail = f"Platform evaluation job ended as {status.status.value}"
        await asyncio.to_thread(self._fail_evaluation_job, row, detail)

    async def _cancel_evaluation_jobs(self) -> None:
        for row in await asyncio.to_thread(self._list_cancelled_evaluations):
            # Once a sandbox exists, let the execution task observe cancellation
            # and run the existing teardown path before its outer Job exits.
            if row.get("backend_handle"):
                continue
            name = str(row["dispatch_job_name"])
            try:
                status = (await self.jobs.get_job_status(workspace=settings.platform_jobs_workspace, name=name)).data()
            except NotFoundError:
                await asyncio.to_thread(self._complete_cancel_teardown, row)
                continue
            if status.status in _ACTIVE_JOB_STATUSES:
                # The task observes durable cancellation and performs runtime
                # teardown. Deleting its outer Job here would bypass that path.
                continue
            await asyncio.to_thread(self._complete_cancel_teardown, row)

    def _claim_build(self) -> TaskBuildJob | None:
        with pooled_connection() as conn:
            return TaskBuildRepository(conn).claim_next(
                worker_id=self._worker_id,
                claim_timeout=TaskBuildWorker.claim_timeout,
                max_attempts=TaskBuildWorker.max_attempts,
            )

    def _bind_build(self, job: TaskBuildJob, name: str) -> bool:
        with pooled_connection() as conn:
            return TaskBuildRepository(conn).bind_platform_job(
                job.task_id,
                job.revision,
                worker_id=self._worker_id,
                job_name=name,
            )

    def _list_builds(self) -> list[dict[str, Any]]:
        with pooled_connection() as conn:
            return TaskBuildRepository(conn).list_platform_jobs()

    def _heartbeat_build(self, row: dict[str, Any]) -> bool:
        with pooled_connection() as conn:
            return TaskBuildRepository(conn).heartbeat(
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
        with pooled_connection() as conn:
            return TaskBuildRepository(conn).retry_or_fail(
                str(row["task_id"]),
                int(row["revision"]),
                worker_id=str(row["job_name"]),
                build_error=detail,
                attempt=int(row["build_attempts"]),
                max_attempts=TaskBuildWorker.max_attempts,
                retry_delay=TaskBuildWorker.retry_delay,
            )

    def _claim_evaluation(self) -> dict[str, Any] | None:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).claim_next(
                claim_timeout=TaskBuildWorker.claim_timeout,
                worker_id=self._worker_id,
                cluster_slot_limit=settings.control_plane_cluster_run_limit,
                per_user_slot_limit=settings.control_plane_per_user_run_limit,
            )

    def _load_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).load_status_runtime(evaluation_id)

    def _record_evaluation_job(
        self,
        evaluation_id: str,
        execution_number: int,
        name: str,
        uid: str,
    ) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).record_dispatch_job(
                evaluation_id,
                execution_number=execution_number,
                name=name,
                uid=uid,
            )

    def _retry_evaluation_submission(self, evaluation_id: str, execution_number: int) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).schedule_retry(
                evaluation_id,
                execution_number=execution_number,
                failure_code="PlatformJobSubmissionError",
                failure_category="infrastructure",
                delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
                expected_dispatch_owner=self._worker_id,
            )

    def _claim_stale_evaluation(self) -> dict[str, Any] | None:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).claim_stale_dispatch_job(
                stale_seconds=settings.dispatch_job_reconcile_stale_seconds,
                claim_timeout=TaskBuildWorker.claim_timeout,
                worker_id=self._worker_id,
            )

    def _release_evaluation_reconcile_claim(self, row: dict[str, Any]) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).release_dispatch_reconcile_claim(
                str(row["id"]),
                execution_number=int(row["current_execution"]),
                dispatch_job_name=str(row["dispatch_job_name"]),
                worker_id=self._worker_id,
            )

    def _fail_evaluation_job(self, row: dict[str, Any], detail: str) -> None:
        evaluation_id = str(row["id"])
        execution_number = int(row["current_execution"])
        with pooled_connection() as conn:
            EvaluationRepository(conn).record_dispatch_job_infrastructure_failure(
                evaluation_id,
                execution_number=execution_number,
                dispatch_job_name=str(row["dispatch_job_name"]),
                reconcile_worker_id=self._worker_id,
                failure_code="PlatformJobInfrastructureError",
                detail=detail,
                retry_delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
            )

    def _list_cancelled_evaluations(self) -> list[dict[str, Any]]:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).list_cancelled_platform_jobs()

    def _complete_cancel_teardown(self, row: dict[str, Any]) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).record_cancel_teardown_succeeded(str(row["id"]))

    def _heartbeat(self) -> None:
        with pooled_connection() as conn:
            OperationsRepository(conn).heartbeat_service("platform_jobs_controller", self._worker_id)
