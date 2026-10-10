# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit scaled-evals work to Platform Jobs."""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any

import anyio.from_thread
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_scaled_evals_plugin.jobs.evaluation_execution import EvaluationExecutionJob
from nemo_scaled_evals_plugin.jobs.naming import evaluation_execution_job_name
from nemo_scaled_evals_plugin.jobs.specs import EvaluationExecutionSpec
from pydantic import BaseModel
from scaled_evals.api.build.queue_worker import TaskBuildWorker
from scaled_evals.api.db import pooled_connection
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.settings import settings
from scaled_evals.dispatch.worker import _retry_delay_seconds

LOG = logging.getLogger(__name__)


class EvaluationSubmitter:
    """Claim an evaluation, create its deterministic Platform Job, and record it."""

    def __init__(self, jobs: AsyncJobsClient, worker_id: str) -> None:
        self.jobs = jobs
        self.worker_id = worker_id

    async def submit(self, evaluation_id: str | None = None) -> bool:
        """Submit the given evaluation, or the next claimable one.

        Args:
            evaluation_id: the evaluation to submit; ``None`` takes the next claimable row.

        Returns:
            bool: False when nothing was claimable.

        """
        row = await asyncio.to_thread(self._claim, evaluation_id)
        if row is None:
            return False
        claimed_id = str(row["id"])
        current = await asyncio.to_thread(self._load, claimed_id)
        if current is None:
            return True
        execution_number = int(current.get("current_execution") or 1)
        name = evaluation_execution_job_name(claimed_id, execution_number)
        spec = EvaluationExecutionSpec(
            evaluation_id=claimed_id,
            execution_number=execution_number,
            runtime=str(current["runtime"]),
            deadline_seconds=max(
                1,
                int(settings.dispatch_run_poll_interval_seconds * settings.dispatch_run_max_polls),
            ),
        )
        try:
            platform_job = await self.create_job(name, EvaluationExecutionJob, spec)
        except ConflictError:
            platform_job = (await self.jobs.get_job(workspace=settings.platform_jobs_workspace, name=name)).data()
        except Exception:
            await asyncio.to_thread(self._retry, claimed_id, execution_number)
            raise
        await asyncio.to_thread(self._record, claimed_id, execution_number, name, platform_job.id)
        return True

    async def create_job(
        self,
        name: str,
        job_cls: type[NemoJob],
        spec: BaseModel,
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
        request = CreateHelixJobRequest(
            name=name,
            description=job_cls.description,
            source=f"scaled-evals.{job_cls.name}",
            spec=spec.model_dump(mode="json"),
            platform_spec=platform_spec,
        )
        return (await self.jobs.create_job(workspace=settings.platform_jobs_workspace, body=request)).data()

    def _claim(self, evaluation_id: str | None) -> dict[str, Any] | None:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).claim_next(
                claim_timeout=TaskBuildWorker.claim_timeout,
                worker_id=self.worker_id,
                evaluation_id=evaluation_id,
            )

    def _load(self, evaluation_id: str) -> dict[str, Any] | None:
        with pooled_connection() as conn:
            return EvaluationRepository(conn).load_status_runtime(evaluation_id)

    def _record(self, evaluation_id: str, execution_number: int, name: str, uid: str) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).record_dispatch_job(
                evaluation_id,
                execution_number=execution_number,
                name=name,
                uid=uid,
            )

    def _retry(self, evaluation_id: str, execution_number: int) -> None:
        with pooled_connection() as conn:
            EvaluationRepository(conn).schedule_retry(
                evaluation_id,
                execution_number=execution_number,
                failure_code="HelixJobSubmissionError",
                failure_category="infrastructure",
                delay_seconds=_retry_delay_seconds(evaluation_id, execution_number),
                expected_dispatch_owner=self.worker_id,
            )


def submit_evaluation_now(evaluation_id: str) -> None:
    """Try to submit a just-committed evaluation from a synchronous API handler.

    Best effort: the committed row is the durable handoff, and the controller
    submits anything this misses on its next pass.

    Args:
        evaluation_id: the evaluation whose row was just committed.

    """
    try:
        jobs = AsyncJobsClient.from_client(get_async_nemo_client(as_service="scaled-evals", internal=True))
        submitter = EvaluationSubmitter(jobs, f"scaled-evals-api:{socket.gethostname()}")
        # Runs on the server's event loop, which owns the client's async HTTP transport.
        anyio.from_thread.run(submitter.submit, evaluation_id)
    except Exception:
        LOG.warning(
            "immediate Platform Job submission failed for %s; the controller will retry", evaluation_id, exc_info=True
        )
