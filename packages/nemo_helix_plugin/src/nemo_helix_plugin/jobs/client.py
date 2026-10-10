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
        name: str,
    ) -> NemoResponse[HelixJobStatusResponse]: ...


class AsyncJobStatusClient(Protocol):
    def get_job_status(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> Awaitable[NemoResponse[HelixJobStatusResponse]]: ...


class JobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[HelixJobLog, CursorPagination]: ...


class AsyncJobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
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
    get_job = method(endpoints.get_job)
    delete_job = method(endpoints.delete_job)
    cancel_job = method(endpoints.cancel_job)
    pause_job = method(endpoints.pause_job)
    resume_job = method(endpoints.resume_job)
    rerun_job = method(endpoints.rerun_job)
    update_job_control = method(endpoints.update_job_control)

    # Job status
    get_job_status = method(endpoints.get_job_status)
    update_job_status_details = method(endpoints.update_job_status_details)

    # Job logs
    list_job_logs = method(endpoints.list_job_logs)

    # Job results
    create_job_result = method(endpoints.create_job_result)
    list_job_results = method(endpoints.list_job_results)
    get_job_result = method(endpoints.get_job_result)
    download_job_result = method(endpoints.download_job_result)

    # Job steps
    list_steps = method(endpoints.list_steps)
    get_job_step = method(endpoints.get_job_step)
    update_job_step_status = method(endpoints.update_job_step_status)

    # Job tasks
    list_job_step_tasks = method(endpoints.list_job_step_tasks)
    update_job_step_task = method(endpoints.update_job_step_task)
    get_job_step_task = method(endpoints.get_job_step_task)


class JobsClient(_JobsMethods, NemoClient):
    """Sync client for the Jobs service API."""

    def list_execution_profiles(self) -> builtins.list[endpoints.ExecutionProfile]:
        return self.get_execution_profiles().data()

    def update_status_details(
        self,
        name: str,
        *,
        workspace: str | None = None,
        body: dict[str, object],
    ) -> None:
        return self.update_job_status_details(
            name=name,
            workspace=workspace,
            body=JobStatusDetailsUpdate.model_validate(body),
        ).data()

    def watch_job(
        self,
        name: str,
        *,
        workspace: str | None = None,
        poll_interval: float = 3,
        timeout: float | None = None,
        include_history: bool = True,
        include_logs: bool = True,
        attempt_id: int | None = None,
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

    async def update_status_details(
        self,
        name: str,
        *,
        workspace: str | None = None,
        body: dict[str, object],
    ) -> None:
        return (
            await self.update_job_status_details(
                name=name,
                workspace=workspace,
                body=JobStatusDetailsUpdate.model_validate(body),
            )
        ).data()

    def watch_job(
        self,
        name: str,
        *,
        workspace: str | None = None,
        poll_interval: float = 3,
        timeout: float | None = None,
        include_history: bool = True,
        include_logs: bool = True,
        attempt_id: int | None = None,
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
