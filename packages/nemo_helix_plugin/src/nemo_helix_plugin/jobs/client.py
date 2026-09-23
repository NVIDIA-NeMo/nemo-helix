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
from functools import cached_property
from typing import Any, Protocol

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.compat import AsyncLegacyPage as AsyncLegacyPage
from nemo_helix_plugin.client.compat import AsyncLegacyPaginatedResponse as AsyncLegacyPaginatedResponse
from nemo_helix_plugin.client.compat import LegacyPageInfo as LegacyPageInfo
from nemo_helix_plugin.client.compat import LegacyPaginatedResponse as LegacyPaginatedResponse
from nemo_helix_plugin.client.compat import SyncLegacyPage as SyncLegacyPage
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.client.response import (
    AsyncNemoBinaryResponse,
    AsyncNemoPaginatedResponse,
    NemoBinaryResponse,
    NemoPaginatedResponse,
    NemoResponse,
)
from nemo_helix_plugin.client.types import CursorPagination
from nemo_helix_plugin.jobs import endpoints
from nemo_helix_plugin.jobs.schemas import (
    FileStorageType,
    HelixJobLog,
    HelixJobResultCreateRequest,
    HelixJobResultResponse,
    HelixJobStatus,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.types import (
    CreateHelixJobRequest,
    HelixJobListResultResponse,
    HelixJobListTaskResponse,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
    JobLogsQueryParams,
    JobStatusDetailsUpdate,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
)
from nemo_helix_plugin.jobs.watch_types import JobWatchEvent

FilterQueryParam = str | dict[str, Any]


def _list_jobs_query_params(
    *,
    filter: FilterQueryParam | None = None,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
) -> ListJobsQueryParams | None:
    params: ListJobsQueryParams = {}
    if filter is not None:
        params["filter"] = filter
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    return params or None


def _list_steps_query_params(
    *,
    filter: FilterQueryParam | None = None,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
) -> ListStepsQueryParams | None:
    params: ListStepsQueryParams = {}
    if filter is not None:
        params["filter"] = filter
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    return params or None


def _list_job_results_query_params(*, sort: str | None = None) -> ListJobResultsQueryParams | None:
    if sort is None:
        return None
    return {"sort": sort}


def _job_logs_query_params(
    *,
    attempt_id: int | None = None,
    limit: int | None = None,
    page_cursor: str | None = None,
    step_id: str | None = None,
    tail: int | None = None,
    task_id: str | None = None,
) -> JobLogsQueryParams | None:
    params: JobLogsQueryParams = {}
    if attempt_id is not None:
        params["attempt_id"] = attempt_id
    if limit is not None:
        params["limit"] = limit
    if page_cursor is not None:
        params["page_cursor"] = page_cursor
    if step_id is not None:
        params["step_id"] = step_id
    if tail is not None:
        params["tail"] = tail
    if task_id is not None:
        params["task_id"] = task_id
    return params or None


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

    @cached_property
    def results(self) -> "_JobsResultsCompat":
        return _JobsResultsCompat(self)

    @cached_property
    def steps(self) -> "_JobsStepsCompat":
        return _JobsStepsCompat(self)

    @cached_property
    def tasks(self) -> "_JobsTasksCompat":
        return _JobsTasksCompat(self)

    def create(
        self,
        *,
        workspace: str | None = None,
        platform_spec: object,
        source: str,
        spec: dict[str, object],
        custom_fields: dict[str, object] | None = None,
        description: str | None = None,
        name: str | None = None,
        output_location: str | None = None,
        ownership: dict[str, object] | None = None,
        project: str | None = None,
    ) -> HelixJobResponse:
        payload: dict[str, object] = {
            "platform_spec": platform_spec,
            "source": source,
            "spec": spec,
        }
        optional_fields: dict[str, object | None] = {
            "custom_fields": custom_fields,
            "description": description,
            "name": name,
            "output_location": output_location,
            "ownership": ownership,
            "project": project,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return self.create_job(
            workspace=workspace,
            body=CreateHelixJobRequest.model_validate(payload),
        ).data()

    def retrieve(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return self.get_job(name=name, workspace=workspace).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> LegacyPaginatedResponse[HelixJobResponse]:
        query_params = _list_jobs_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return LegacyPaginatedResponse(self.list_jobs(workspace=workspace, query_params=query_params))

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        return self.delete_job(name=name, workspace=workspace).data()

    def cancel(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return self.cancel_job(name=name, workspace=workspace).data()

    def pause(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return self.pause_job(name=name, workspace=workspace).data()

    def resume(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return self.resume_job(name=name, workspace=workspace).data()

    def get_status(self, name: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return self.get_job_status(name=name, workspace=workspace).data()

    def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        attempt_id: int | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
        step_id: str | None = None,
        tail: int | None = None,
        task_id: str | None = None,
    ) -> LegacyPaginatedResponse[HelixJobLog]:
        query_params = _job_logs_query_params(
            attempt_id=attempt_id,
            limit=limit,
            page_cursor=page_cursor,
            step_id=step_id,
            tail=tail,
            task_id=task_id,
        )
        return LegacyPaginatedResponse(self.list_job_logs(workspace=workspace, name=name, query_params=query_params))

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


class _JobsResultsCompat:
    def __init__(self, client: JobsClient) -> None:
        self._client = client

    def create(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        artifact_storage_type: FileStorageType,
        artifact_url: str,
    ) -> HelixJobResultResponse:
        return self._client.create_job_result(
            name=name,
            workspace=workspace,
            job=job,
            body=HelixJobResultCreateRequest(
                artifact_storage_type=artifact_storage_type,
                artifact_url=artifact_url,
            ),
        ).data()

    def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> HelixJobResultResponse:
        return self._client.get_job_result(name=name, workspace=workspace, job=job).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        sort: str | None = None,
    ) -> HelixJobListResultResponse:
        query_params = _list_job_results_query_params(sort=sort)
        return self._client.list_job_results(
            name=name,
            workspace=workspace,
            query_params=query_params,
        ).data()

    def download(self, name: str, *, workspace: str | None = None, job: str) -> NemoBinaryResponse:
        return self._client.download_job_result(name=name, workspace=workspace, job=job)


class _JobsStepsCompat:
    def __init__(self, client: JobsClient) -> None:
        self._client = client

    def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> HelixJobStepResponse:
        return self._client.get_job_step(name=name, workspace=workspace, job=job).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> LegacyPaginatedResponse[HelixJobStepWithContext]:
        query_params = _list_steps_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return LegacyPaginatedResponse(
            self._client.list_steps(name=name, workspace=workspace, query_params=query_params)
        )

    def update_status(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        status: HelixJobStatus,
        error_details: dict[str, object] | None = None,
        status_details: dict[str, object] | None = None,
    ) -> HelixJobStepResponse:
        return self._client.update_job_step_status(
            name=name,
            workspace=workspace,
            job=job,
            body=HelixJobStatusUpdateRequest(
                status=status,
                error_details=error_details,
                status_details=status_details,
            ),
        ).data()


class _JobsTasksCompat:
    def __init__(self, client: JobsClient) -> None:
        self._client = client

    def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        step: str,
    ) -> HelixJobTaskResponse:
        return self._client.get_job_step_task(name=name, workspace=workspace, job=job, step=step).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
    ) -> HelixJobListTaskResponse:
        return self._client.list_job_step_tasks(name=name, workspace=workspace, job=job).data()

    def create_or_update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        step: str,
        error_details: dict[str, object] | None = None,
        error_stack: str | None = None,
        status: HelixJobStatus = HelixJobStatus.PENDING,
        status_details: dict[str, object] | None = None,
    ) -> HelixJobTaskResponse:
        return self._client.update_job_step_task(
            name=name,
            workspace=workspace,
            job=job,
            step=step,
            body=HelixJobTaskUpdate(
                error_details=error_details,
                error_stack=error_stack,
                status=status,
                status_details=status_details,
            ),
        ).data()


class AsyncJobsClient(_JobsMethods, AsyncNemoClient):
    """Async client for the Jobs service API."""

    @cached_property
    def results(self) -> "_AsyncJobsResultsCompat":
        return _AsyncJobsResultsCompat(self)

    @cached_property
    def steps(self) -> "_AsyncJobsStepsCompat":
        return _AsyncJobsStepsCompat(self)

    @cached_property
    def tasks(self) -> "_AsyncJobsTasksCompat":
        return _AsyncJobsTasksCompat(self)

    async def create(
        self,
        *,
        workspace: str | None = None,
        platform_spec: object,
        source: str,
        spec: dict[str, object],
        custom_fields: dict[str, object] | None = None,
        description: str | None = None,
        name: str | None = None,
        output_location: str | None = None,
        ownership: dict[str, object] | None = None,
        project: str | None = None,
    ) -> HelixJobResponse:
        payload: dict[str, object] = {
            "platform_spec": platform_spec,
            "source": source,
            "spec": spec,
        }
        optional_fields: dict[str, object | None] = {
            "custom_fields": custom_fields,
            "description": description,
            "name": name,
            "output_location": output_location,
            "ownership": ownership,
            "project": project,
        }
        payload.update({key: value for key, value in optional_fields.items() if value is not None})
        return (
            await self.create_job(
                workspace=workspace,
                body=CreateHelixJobRequest.model_validate(payload),
            )
        ).data()

    async def retrieve(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return (await self.get_job(name=name, workspace=workspace)).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> AsyncLegacyPaginatedResponse[HelixJobResponse]:
        query_params = _list_jobs_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return AsyncLegacyPaginatedResponse(self.list_jobs(workspace=workspace, query_params=query_params))

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        return (await self.delete_job(name=name, workspace=workspace)).data()

    async def cancel(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return (await self.cancel_job(name=name, workspace=workspace)).data()

    async def pause(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return (await self.pause_job(name=name, workspace=workspace)).data()

    async def resume(self, name: str, *, workspace: str | None = None) -> HelixJobResponse:
        return (await self.resume_job(name=name, workspace=workspace)).data()

    async def get_status(self, name: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return (await self.get_job_status(name=name, workspace=workspace)).data()

    def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        attempt_id: int | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
        step_id: str | None = None,
        tail: int | None = None,
        task_id: str | None = None,
    ) -> AsyncLegacyPaginatedResponse[HelixJobLog]:
        query_params = _job_logs_query_params(
            attempt_id=attempt_id,
            limit=limit,
            page_cursor=page_cursor,
            step_id=step_id,
            tail=tail,
            task_id=task_id,
        )
        return AsyncLegacyPaginatedResponse(
            self.list_job_logs(workspace=workspace, name=name, query_params=query_params)
        )

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


class _AsyncJobsResultsCompat:
    def __init__(self, client: AsyncJobsClient) -> None:
        self._client = client

    async def create(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        artifact_storage_type: FileStorageType,
        artifact_url: str,
    ) -> HelixJobResultResponse:
        return (
            await self._client.create_job_result(
                name=name,
                workspace=workspace,
                job=job,
                body=HelixJobResultCreateRequest(
                    artifact_storage_type=artifact_storage_type,
                    artifact_url=artifact_url,
                ),
            )
        ).data()

    async def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> HelixJobResultResponse:
        return (await self._client.get_job_result(name=name, workspace=workspace, job=job)).data()

    async def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        sort: str | None = None,
    ) -> HelixJobListResultResponse:
        query_params = _list_job_results_query_params(sort=sort)
        return (
            await self._client.list_job_results(
                name=name,
                workspace=workspace,
                query_params=query_params,
            )
        ).data()

    async def download(self, name: str, *, workspace: str | None = None, job: str) -> AsyncNemoBinaryResponse:
        return await self._client.download_job_result(name=name, workspace=workspace, job=job)


class _AsyncJobsStepsCompat:
    def __init__(self, client: AsyncJobsClient) -> None:
        self._client = client

    async def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> HelixJobStepResponse:
        return (await self._client.get_job_step(name=name, workspace=workspace, job=job)).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> AsyncLegacyPaginatedResponse[HelixJobStepWithContext]:
        query_params = _list_steps_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return AsyncLegacyPaginatedResponse(
            self._client.list_steps(name=name, workspace=workspace, query_params=query_params)
        )

    async def update_status(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        status: HelixJobStatus,
        error_details: dict[str, object] | None = None,
        status_details: dict[str, object] | None = None,
    ) -> HelixJobStepResponse:
        return (
            await self._client.update_job_step_status(
                name=name,
                workspace=workspace,
                job=job,
                body=HelixJobStatusUpdateRequest(
                    status=status,
                    error_details=error_details,
                    status_details=status_details,
                ),
            )
        ).data()


class _AsyncJobsTasksCompat:
    def __init__(self, client: AsyncJobsClient) -> None:
        self._client = client

    async def retrieve(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        step: str,
    ) -> HelixJobTaskResponse:
        return (await self._client.get_job_step_task(name=name, workspace=workspace, job=job, step=step)).data()

    async def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
    ) -> HelixJobListTaskResponse:
        return (await self._client.list_job_step_tasks(name=name, workspace=workspace, job=job)).data()

    async def create_or_update(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
        step: str,
        error_details: dict[str, object] | None = None,
        error_stack: str | None = None,
        status: HelixJobStatus = HelixJobStatus.PENDING,
        status_details: dict[str, object] | None = None,
    ) -> HelixJobTaskResponse:
        return (
            await self._client.update_job_step_task(
                name=name,
                workspace=workspace,
                job=job,
                step=step,
                body=HelixJobTaskUpdate(
                    error_details=error_details,
                    error_stack=error_stack,
                    status=status,
                    status_details=status_details,
                ),
            )
        ).data()
