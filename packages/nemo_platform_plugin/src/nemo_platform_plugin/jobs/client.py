# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Jobs service.

Wraps the endpoint functions from ``jobs.endpoints`` as direct methods using
the ``method()`` descriptor, following the example-plugin / Files pattern.

Usage::

    from nemo_platform_plugin.jobs.client import JobsClient
    from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest

    client = JobsClient(base_url="...", workspace="default")
    resp = client.create_job(body=CreatePlatformJobRequest(...))
    job = resp.data()

    for job in client.list_jobs().items():
        print(job.name)

    with client.download_job_result(job="j-1", name="out").stream() as chunks:
        for chunk in chunks:
            ...
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar

import httpx
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.client.response import (
    AsyncNemoBinaryResponse,
    AsyncNemoPaginatedResponse,
    NemoBinaryResponse,
    NemoPaginatedResponse,
    NemoResponse,
)
from nemo_platform_plugin.client.types import CursorPagination
from nemo_platform_plugin.jobs import endpoints
from nemo_platform_plugin.jobs.schemas import (
    FileStorageType,
    PlatformJobLog,
    PlatformJobResultCreateRequest,
    PlatformJobResultResponse,
    PlatformJobStatus,
    PlatformJobStatusResponse,
)
from nemo_platform_plugin.jobs.types import (
    CreatePlatformJobRequest,
    JobLogsQueryParams,
    JobStatusDetailsUpdate,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
    PlatformJobListResultResponse,
    PlatformJobListTaskResponse,
    PlatformJobResponse,
    PlatformJobStatusUpdateRequest,
    PlatformJobStepResponse,
    PlatformJobStepWithContext,
    PlatformJobTaskResponse,
    PlatformJobTaskUpdate,
)
from nemo_platform_plugin.jobs.watch_types import JobWatchEvent
from pydantic import BaseModel

LegacyItemT = TypeVar("LegacyItemT", bound=BaseModel)
FilterQueryParam = str | dict[str, Any]


class LegacyPageInfo:
    """Small page-info object compatible with Stainless pagination examples."""

    def __init__(self, *, params: dict[str, Any]) -> None:
        self.params = params

    def __repr__(self) -> str:
        return f"{type(self).__name__}(params={self.params!r})"


class SyncLegacyPage(Generic[LegacyItemT]):
    """Small Stainless-style page adapter for the sync Jobs compatibility surface."""

    def __init__(
        self,
        *,
        data: list[LegacyItemT],
        metadata: object | None = None,
        next_page_params: dict[str, Any] | None = None,
        get_next_page: Callable[[], "SyncLegacyPage[LegacyItemT]"] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata
        self._next_page_params = next_page_params
        self._get_next_page = get_next_page

    def __iter__(self) -> Iterator[LegacyItemT]:
        return iter(self.data)

    def has_next_page(self) -> bool:
        return self._next_page_params is not None

    def next_page_info(self) -> LegacyPageInfo | None:
        if self._next_page_params is None:
            return None
        return LegacyPageInfo(params=self._next_page_params)

    def get_next_page(self) -> "SyncLegacyPage[LegacyItemT]":
        if self._get_next_page is not None:
            return self._get_next_page()
        raise RuntimeError("No next page expected; please check `.has_next_page()` before calling `.get_next_page()`.")


class AsyncLegacyPage(Generic[LegacyItemT]):
    """Small Stainless-style page adapter for the async Jobs compatibility surface."""

    def __init__(
        self,
        *,
        data: list[LegacyItemT],
        metadata: object | None = None,
        next_page_params: dict[str, Any] | None = None,
        get_next_page: Callable[[], Awaitable["AsyncLegacyPage[LegacyItemT]"]] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata
        self._next_page_params = next_page_params
        self._get_next_page = get_next_page

    def __iter__(self) -> Iterator[LegacyItemT]:
        return iter(self.data)

    def has_next_page(self) -> bool:
        return self._next_page_params is not None

    def next_page_info(self) -> LegacyPageInfo | None:
        if self._next_page_params is None:
            return None
        return LegacyPageInfo(params=self._next_page_params)

    async def get_next_page(self) -> "AsyncLegacyPage[LegacyItemT]":
        if self._get_next_page is not None:
            return await self._get_next_page()
        raise RuntimeError("No next page expected; please check `.has_next_page()` before calling `.get_next_page()`.")


class LegacyPaginatedResponse(Generic[LegacyItemT]):
    """Compatibility adapter exposing ``.data`` and item iteration."""

    def __init__(self, response: NemoPaginatedResponse[LegacyItemT, Any]) -> None:
        self._response = response
        self._first_page: SyncLegacyPage[LegacyItemT] | None = None

    def _legacy_page(self, raw: httpx.Response) -> SyncLegacyPage[LegacyItemT]:
        items, body, metadata = self._response._parse_page(raw)
        next_page = self._response._strategy.next_page(body)
        if next_page is None:
            return SyncLegacyPage(data=items, metadata=metadata)
        return SyncLegacyPage(
            data=items,
            metadata=metadata,
            next_page_params=self._response._strategy.page_query_params(next_page),
            get_next_page=lambda: self._legacy_page(self._response._fetch_page(self._response.request, next_page)),
        )

    def _page(self) -> SyncLegacyPage[LegacyItemT]:
        if self._first_page is None:
            self._first_page = self._legacy_page(self._response.http_response)
        return self._first_page

    @property
    def data(self) -> list[LegacyItemT]:
        return self._page().data

    def __iter__(self) -> Iterator[LegacyItemT]:
        return self._response.items()

    def iter_pages(self) -> Iterator[SyncLegacyPage[LegacyItemT]]:
        page = self._page()
        while True:
            yield page
            if not page.has_next_page():
                return
            page = page.get_next_page()


class AsyncLegacyPaginatedResponse(Generic[LegacyItemT]):
    """Async compatibility adapter matching Stainless async paginator basics."""

    def __init__(self, response: Awaitable[AsyncNemoPaginatedResponse[LegacyItemT, Any]]) -> None:
        self._response_awaitable = response
        self._response: AsyncNemoPaginatedResponse[LegacyItemT, Any] | None = None
        self._first_page: AsyncLegacyPage[LegacyItemT] | None = None

    async def _get_response(self) -> AsyncNemoPaginatedResponse[LegacyItemT, Any]:
        if self._response is None:
            self._response = await self._response_awaitable
        return self._response

    async def _legacy_page(
        self,
        response: AsyncNemoPaginatedResponse[LegacyItemT, Any],
        raw: httpx.Response,
    ) -> AsyncLegacyPage[LegacyItemT]:
        items, body, metadata = response._parse_page(raw)
        next_page = response._strategy.next_page(body)
        if next_page is None:
            return AsyncLegacyPage(data=items, metadata=metadata)

        async def get_next_page() -> AsyncLegacyPage[LegacyItemT]:
            raw = await response._fetch_page(response.request, next_page)
            return await self._legacy_page(response, raw)

        return AsyncLegacyPage(
            data=items,
            metadata=metadata,
            next_page_params=response._strategy.page_query_params(next_page),
            get_next_page=get_next_page,
        )

    async def _page(self) -> AsyncLegacyPage[LegacyItemT]:
        if self._first_page is not None:
            return self._first_page
        response = await self._get_response()
        self._first_page = await self._legacy_page(response, response.http_response)
        return self._first_page

    def __await__(self) -> Any:
        return self._page().__await__()

    async def __aiter__(self) -> AsyncIterator[LegacyItemT]:
        response = await self._get_response()
        async for item in response.items():
            yield item

    async def iter_pages(self) -> AsyncIterator[AsyncLegacyPage[LegacyItemT]]:
        page = await self._page()
        while True:
            yield page
            if not page.has_next_page():
                return
            page = await page.get_next_page()


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
    ) -> NemoResponse[PlatformJobStatusResponse]: ...


class AsyncJobStatusClient(Protocol):
    def get_job_status(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> Awaitable[NemoResponse[PlatformJobStatusResponse]]: ...


class JobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[PlatformJobLog, CursorPagination]: ...


class AsyncJobLogsClient(Protocol):
    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> Awaitable[AsyncNemoPaginatedResponse[PlatformJobLog, CursorPagination]]: ...


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
    ) -> PlatformJobResponse:
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
            body=CreatePlatformJobRequest.model_validate(payload),
        ).data()

    def retrieve(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return self.get_job(name=name, workspace=workspace).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> LegacyPaginatedResponse[PlatformJobResponse]:
        query_params = _list_jobs_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return LegacyPaginatedResponse(self.list_jobs(workspace=workspace, query_params=query_params))

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        return self.delete_job(name=name, workspace=workspace).data()

    def cancel(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return self.cancel_job(name=name, workspace=workspace).data()

    def pause(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return self.pause_job(name=name, workspace=workspace).data()

    def resume(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return self.resume_job(name=name, workspace=workspace).data()

    def get_status(self, name: str, *, workspace: str | None = None) -> PlatformJobStatusResponse:
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
    ) -> LegacyPaginatedResponse[PlatformJobLog]:
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
        from nemo_platform_plugin.jobs.watch import watch_job

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
    ) -> PlatformJobResultResponse:
        return self._client.create_job_result(
            name=name,
            workspace=workspace,
            job=job,
            body=PlatformJobResultCreateRequest(
                artifact_storage_type=artifact_storage_type,
                artifact_url=artifact_url,
            ),
        ).data()

    def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> PlatformJobResultResponse:
        return self._client.get_job_result(name=name, workspace=workspace, job=job).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        sort: str | None = None,
    ) -> PlatformJobListResultResponse:
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

    def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> PlatformJobStepResponse:
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
    ) -> LegacyPaginatedResponse[PlatformJobStepWithContext]:
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
        status: PlatformJobStatus,
        error_details: dict[str, object] | None = None,
        status_details: dict[str, object] | None = None,
    ) -> PlatformJobStepResponse:
        return self._client.update_job_step_status(
            name=name,
            workspace=workspace,
            job=job,
            body=PlatformJobStatusUpdateRequest(
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
    ) -> PlatformJobTaskResponse:
        return self._client.get_job_step_task(name=name, workspace=workspace, job=job, step=step).data()

    def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
    ) -> PlatformJobListTaskResponse:
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
        status: PlatformJobStatus = PlatformJobStatus.PENDING,
        status_details: dict[str, object] | None = None,
    ) -> PlatformJobTaskResponse:
        return self._client.update_job_step_task(
            name=name,
            workspace=workspace,
            job=job,
            step=step,
            body=PlatformJobTaskUpdate(
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
    ) -> PlatformJobResponse:
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
                body=CreatePlatformJobRequest.model_validate(payload),
            )
        ).data()

    async def retrieve(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return (await self.get_job(name=name, workspace=workspace)).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        filter: FilterQueryParam | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
    ) -> AsyncLegacyPaginatedResponse[PlatformJobResponse]:
        query_params = _list_jobs_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
        return AsyncLegacyPaginatedResponse(self.list_jobs(workspace=workspace, query_params=query_params))

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        return (await self.delete_job(name=name, workspace=workspace)).data()

    async def cancel(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return (await self.cancel_job(name=name, workspace=workspace)).data()

    async def pause(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return (await self.pause_job(name=name, workspace=workspace)).data()

    async def resume(self, name: str, *, workspace: str | None = None) -> PlatformJobResponse:
        return (await self.resume_job(name=name, workspace=workspace)).data()

    async def get_status(self, name: str, *, workspace: str | None = None) -> PlatformJobStatusResponse:
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
    ) -> AsyncLegacyPaginatedResponse[PlatformJobLog]:
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
        from nemo_platform_plugin.jobs.watch import async_watch_job

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
    ) -> PlatformJobResultResponse:
        return (
            await self._client.create_job_result(
                name=name,
                workspace=workspace,
                job=job,
                body=PlatformJobResultCreateRequest(
                    artifact_storage_type=artifact_storage_type,
                    artifact_url=artifact_url,
                ),
            )
        ).data()

    async def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> PlatformJobResultResponse:
        return (await self._client.get_job_result(name=name, workspace=workspace, job=job)).data()

    async def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        sort: str | None = None,
    ) -> PlatformJobListResultResponse:
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

    async def retrieve(self, name: str, *, workspace: str | None = None, job: str) -> PlatformJobStepResponse:
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
    ) -> AsyncLegacyPaginatedResponse[PlatformJobStepWithContext]:
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
        status: PlatformJobStatus,
        error_details: dict[str, object] | None = None,
        status_details: dict[str, object] | None = None,
    ) -> PlatformJobStepResponse:
        return (
            await self._client.update_job_step_status(
                name=name,
                workspace=workspace,
                job=job,
                body=PlatformJobStatusUpdateRequest(
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
    ) -> PlatformJobTaskResponse:
        return (await self._client.get_job_step_task(name=name, workspace=workspace, job=job, step=step)).data()

    async def list(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job: str,
    ) -> PlatformJobListTaskResponse:
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
        status: PlatformJobStatus = PlatformJobStatus.PENDING,
        status_details: dict[str, object] | None = None,
    ) -> PlatformJobTaskResponse:
        return (
            await self._client.update_job_step_task(
                name=name,
                workspace=workspace,
                job=job,
                step=step,
                body=PlatformJobTaskUpdate(
                    error_details=error_details,
                    error_stack=error_stack,
                    status=status,
                    status_details=status_details,
                ),
            )
        ).data()
