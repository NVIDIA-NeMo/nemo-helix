# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Jobs service.

Wraps the endpoint functions from ``jobs.endpoints`` as direct methods using
the ``method()`` descriptor, following the example-plugin / Files pattern.

Usage::

    from nemo_helix_plugin.jobs.client import JobsClient
    from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

    client = JobsClient(base_url="...", workspace="default")
    resp = client.create_job(body=CreateHelixJobRequest(...))
    job = resp.data()

    for job in client.list_jobs().items():
        print(job.name)

    with client.download_job_result(job="j-1", name="out").stream() as chunks:
        for chunk in chunks:
            ...
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator, Awaitable, Iterator
from typing import Protocol

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.client.response import (
    AsyncNemoPaginatedResponse,
    NemoPaginatedResponse,
    NemoResponse,
)
from nemo_helix_plugin.client.types import CursorPagination
from nemo_helix_plugin.jobs import endpoints
from nemo_helix_plugin.jobs.schemas import (
    HelixJobLog,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import (
    JobLogsQueryParams,
    JobStatusDetailsUpdate,
)
from nemo_helix_plugin.jobs.watch_types import JobWatchEvent


class JobStatusClient(Protocol):
    def get_job_status(
        self,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        name: str,
    ) -> NemoResponse[HelixJobStatusResponse]: ...


class AsyncJobStatusClient(Protocol):
    def get_job_status(
        self,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        name: str,
    ) -> Awaitable[NemoResponse[HelixJobStatusResponse]]: ...


class JobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[HelixJobLog, CursorPagination]: ...


class AsyncJobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> Awaitable[AsyncNemoPaginatedResponse[HelixJobLog, CursorPagination]]: ...


class JobsWatchClient(JobStatusClient, JobLogsClient, Protocol):
    """Structural sync Jobs client accepted by the watcher implementation."""


class AsyncJobsWatchClient(AsyncJobStatusClient, AsyncJobLogsClient, Protocol):
    """Structural async Jobs client accepted by the watcher implementation."""


class _JobsMethods:
    # Execution profiles
    get_execution_profiles = method(endpoints.get_execution_profiles)

    # Job CRUD + lifecycle
    create_job = method(endpoints.create_job)
    list_jobs = method(endpoints.list_jobs)
    get_job_metadata = method(endpoints.get_job_metadata)
    get_job = method(endpoints.get_job)
    get_job_by_id = method(endpoints.get_job_by_id)
    get_job_by_name = method(endpoints.get_job_by_name)
    update_job = method(endpoints.update_job)
    update_job_by_id = method(endpoints.update_job_by_id)
    update_job_by_name = method(endpoints.update_job_by_name)
    delete_job = method(endpoints.delete_job)
    delete_job_by_id = method(endpoints.delete_job_by_id)
    delete_job_by_name = method(endpoints.delete_job_by_name)
    cancel_job = method(endpoints.cancel_job)
    cancel_job_by_id = method(endpoints.cancel_job_by_id)
    cancel_job_by_name = method(endpoints.cancel_job_by_name)
    pause_job = method(endpoints.pause_job)
    pause_job_by_id = method(endpoints.pause_job_by_id)
    pause_job_by_name = method(endpoints.pause_job_by_name)
    resume_job = method(endpoints.resume_job)
    resume_job_by_id = method(endpoints.resume_job_by_id)
    resume_job_by_name = method(endpoints.resume_job_by_name)
    rerun_job = method(endpoints.rerun_job)
    rerun_job_by_id = method(endpoints.rerun_job_by_id)
    rerun_job_by_name = method(endpoints.rerun_job_by_name)
    update_job_control = method(endpoints.update_job_control)
    update_job_control_by_id = method(endpoints.update_job_control_by_id)
    update_job_control_by_name = method(endpoints.update_job_control_by_name)

    # Job status
    get_job_status = method(endpoints.get_job_status)
    get_job_status_by_id = method(endpoints.get_job_status_by_id)
    get_job_status_by_name = method(endpoints.get_job_status_by_name)

    # Job attempts
    list_job_attempts = method(endpoints.list_job_attempts)
    list_job_attempts_by_id = method(endpoints.list_job_attempts_by_id)
    list_job_attempts_by_name = method(endpoints.list_job_attempts_by_name)
    get_job_attempt = method(endpoints.get_job_attempt)
    get_job_attempt_by_id = method(endpoints.get_job_attempt_by_id)
    get_job_attempt_by_name = method(endpoints.get_job_attempt_by_name)
    update_job_attempt_status_details = method(endpoints.update_job_attempt_status_details)
    update_job_attempt_status_details_by_id = method(endpoints.update_job_attempt_status_details_by_id)
    update_job_attempt_status_details_by_name = method(endpoints.update_job_attempt_status_details_by_name)

    # Job logs
    list_job_logs = method(endpoints.list_job_logs)
    list_job_logs_by_id = method(endpoints.list_job_logs_by_id)
    list_job_logs_by_name = method(endpoints.list_job_logs_by_name)

    # Job results
    create_job_result = method(endpoints.create_job_result)
    create_job_result_by_id = method(endpoints.create_job_result_by_id)
    create_job_result_by_name = method(endpoints.create_job_result_by_name)
    list_job_results = method(endpoints.list_job_results)
    list_job_results_by_id = method(endpoints.list_job_results_by_id)
    list_job_results_by_name = method(endpoints.list_job_results_by_name)
    get_job_result = method(endpoints.get_job_result)
    get_job_result_by_id = method(endpoints.get_job_result_by_id)
    get_job_result_by_name = method(endpoints.get_job_result_by_name)
    download_job_result = method(endpoints.download_job_result)
    download_job_result_by_id = method(endpoints.download_job_result_by_id)
    download_job_result_by_name = method(endpoints.download_job_result_by_name)

    # Job steps
    list_steps = method(endpoints.list_steps)
    list_steps_by_id = method(endpoints.list_steps_by_id)
    list_steps_by_name = method(endpoints.list_steps_by_name)
    get_job_step = method(endpoints.get_job_step)
    get_job_step_by_id = method(endpoints.get_job_step_by_id)
    get_job_step_by_name = method(endpoints.get_job_step_by_name)
    update_job_step_status = method(endpoints.update_job_step_status)
    update_job_step_status_by_id = method(endpoints.update_job_step_status_by_id)
    update_job_step_status_by_name = method(endpoints.update_job_step_status_by_name)
    list_attempt_steps = method(endpoints.list_attempt_steps)
    list_attempt_steps_by_id = method(endpoints.list_attempt_steps_by_id)
    list_attempt_steps_by_name = method(endpoints.list_attempt_steps_by_name)
    get_attempt_step = method(endpoints.get_attempt_step)
    get_attempt_step_by_id = method(endpoints.get_attempt_step_by_id)
    get_attempt_step_by_name = method(endpoints.get_attempt_step_by_name)
    update_attempt_step_status = method(endpoints.update_attempt_step_status)
    update_attempt_step_status_by_id = method(endpoints.update_attempt_step_status_by_id)
    update_attempt_step_status_by_name = method(endpoints.update_attempt_step_status_by_name)

    # Job tasks
    list_job_step_tasks = method(endpoints.list_job_step_tasks)
    list_job_step_tasks_by_id = method(endpoints.list_job_step_tasks_by_id)
    list_job_step_tasks_by_name = method(endpoints.list_job_step_tasks_by_name)
    get_job_step_task = method(endpoints.get_job_step_task)
    get_job_step_task_by_id = method(endpoints.get_job_step_task_by_id)
    get_job_step_task_by_name = method(endpoints.get_job_step_task_by_name)
    list_attempt_step_tasks = method(endpoints.list_attempt_step_tasks)
    list_attempt_step_tasks_by_id = method(endpoints.list_attempt_step_tasks_by_id)
    list_attempt_step_tasks_by_name = method(endpoints.list_attempt_step_tasks_by_name)
    update_attempt_step_task = method(endpoints.update_attempt_step_task)
    update_attempt_step_task_by_id = method(endpoints.update_attempt_step_task_by_id)
    update_attempt_step_task_by_name = method(endpoints.update_attempt_step_task_by_name)
    get_attempt_step_task = method(endpoints.get_attempt_step_task)
    get_attempt_step_task_by_id = method(endpoints.get_attempt_step_task_by_id)
    get_attempt_step_task_by_name = method(endpoints.get_attempt_step_task_by_name)


class JobsClient(_JobsMethods, NemoClient):
    """Sync client for the Jobs service API."""

    def list_execution_profiles(self) -> builtins.list[endpoints.ExecutionProfile]:
        return self.get_execution_profiles().data()

    def update_attempt_status_details(
        self,
        name: str,
        attempt: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        body: dict[str, object],
    ) -> None:
        return self.update_job_attempt_status_details(
            job=name,
            attempt=attempt,
            workspace=workspace,
            job_reference_kind=job_reference_kind,
            body=JobStatusDetailsUpdate.model_validate(body),
        ).data()

    def watch_job(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        poll_interval: float = 3,
        timeout: float | None = None,
        include_history: bool = True,
        include_logs: bool = True,
        attempt_id: str | None = None,
        step_id: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
    ) -> Iterator[JobWatchEvent]:
        """Watch a platform job and yield status, log, and warning events.

        Poll-based log pagination can miss delayed log entries that sort before
        the current cursor.
        """
        from nemo_helix_plugin.jobs.watch import watch_job

        return watch_job(
            self,
            name,
            workspace=workspace,
            job_reference_kind=job_reference_kind,
            poll_interval=poll_interval,
            timeout=timeout,
            include_history=include_history,
            include_logs=include_logs,
            attempt_id=attempt_id,
            step_id=step_id,
            task_id=task_id,
            limit=limit,
            page_cursor=page_cursor,
        )


class AsyncJobsClient(_JobsMethods, AsyncNemoClient):
    """Async client for the Jobs service API."""

    async def list_execution_profiles(self) -> builtins.list[endpoints.ExecutionProfile]:
        return (await self.get_execution_profiles()).data()

    async def update_attempt_status_details(
        self,
        name: str,
        attempt: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        body: dict[str, object],
    ) -> None:
        return (
            await self.update_job_attempt_status_details(
                job=name,
                attempt=attempt,
                workspace=workspace,
                job_reference_kind=job_reference_kind,
                body=JobStatusDetailsUpdate.model_validate(body),
            )
        ).data()

    def watch_job(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        poll_interval: float = 3,
        timeout: float | None = None,
        include_history: bool = True,
        include_logs: bool = True,
        attempt_id: str | None = None,
        step_id: str | None = None,
        task_id: str | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
    ) -> AsyncIterator[JobWatchEvent]:
        """Watch a platform job asynchronously and yield status, log, and warning events.

        Poll-based log pagination can miss delayed log entries that sort before
        the current cursor.
        """
        from nemo_helix_plugin.jobs.watch import async_watch_job

        return async_watch_job(
            self,
            name,
            workspace=workspace,
            job_reference_kind=job_reference_kind,
            poll_interval=poll_interval,
            timeout=timeout,
            include_history=include_history,
            include_logs=include_logs,
            attempt_id=attempt_id,
            step_id=step_id,
            task_id=task_id,
            limit=limit,
            page_cursor=page_cursor,
        )
