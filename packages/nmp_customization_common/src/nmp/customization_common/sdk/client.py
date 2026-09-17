# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parametrized typed SDK resources for Customizer backend job collections.

Automodel, Unsloth, and RL expose the same public SDK shape:
``client.customization.<backend>.jobs``. The only backend-specific value is
the route segment under ``/apis/customization/v2/workspaces/{workspace}``.

This module keeps that shared shape in one factory instead of duplicating
``resources.py`` implementations in each backend plugin. The backend plugins
remain thin shims that call :func:`make_customization_sdk` and re-export the
``<Backend>Customization`` / ``Async<Backend>Customization`` symbols imported
by the ``nemo-customizer`` SDK hub.

Route-specific create/list/retrieve calls delegate to the typed Customizer
client in this package. Operations already owned by the Jobs service should
delegate to ``nemo_platform_plugin.jobs.client.JobsClient`` instead of being
reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.client.response import AsyncNemoPaginatedResponse, NemoPaginatedResponse, NemoResponse
from nemo_platform_plugin.client.types import CursorPagination
from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobLog,
    PlatformJobStatusResponse,
)
from nemo_platform_plugin.jobs.types import JobLogsQueryParams
from nmp.customization_common.sdk import endpoints
from nmp.customization_common.sdk.types import (
    CustomizationJob,
    CustomizationJobCreateRequest,
    ListCustomizationJobsQueryParams,
)
from pydantic import BaseModel


class _CustomizationMethods:
    get_customization_health = method(endpoints.get_customization_health)
    create_customization_job = method(endpoints.create_customization_job)
    list_customization_jobs = method(endpoints.list_customization_jobs)
    get_customization_job = method(endpoints.get_customization_job)


class CustomizationClient(_CustomizationMethods, NemoClient):
    """Sync typed client for the Customizer router SDK surface."""


class AsyncCustomizationClient(_CustomizationMethods, AsyncNemoClient):
    """Async typed client for the Customizer router SDK surface."""


@dataclass(frozen=True, slots=True)
class CustomizationSDKContext:
    """Typed sync clients shared by one Customizer SDK namespace."""

    customization: CustomizationClient
    jobs: JobsClient
    workspace: str | None


@dataclass(frozen=True, slots=True)
class AsyncCustomizationSDKContext:
    """Typed async clients shared by one Customizer SDK namespace."""

    customization: AsyncCustomizationClient
    jobs: AsyncJobsClient
    workspace: str | None


def make_customization_sdk_context(client: NemoClient) -> CustomizationSDKContext:
    """Build the typed sync clients used by Customizer backend SDK resources."""
    return CustomizationSDKContext(
        customization=CustomizationClient.from_client(client),
        jobs=JobsClient.from_client(client),
        workspace=client.workspace,
    )


def make_async_customization_sdk_context(client: AsyncNemoClient) -> AsyncCustomizationSDKContext:
    """Build the typed async clients used by Customizer backend SDK resources."""
    return AsyncCustomizationSDKContext(
        customization=AsyncCustomizationClient.from_client(client),
        jobs=AsyncJobsClient.from_client(client),
        workspace=client.workspace,
    )


def _spec_payload(spec: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec, BaseModel):
        return spec.model_dump(mode="json")
    return spec


def _create_request(
    spec: BaseModel | dict[str, Any],
    *,
    name: str | None,
    description: str | None,
    project: str | None,
    profile: str | None,
    options: dict[str, Any] | None,
    ownership: dict[str, Any] | None,
    custom_fields: dict[str, Any] | None,
    output_location: str | None,
) -> CustomizationJobCreateRequest:
    body: dict[str, Any] = {"spec": _spec_payload(spec)}
    for key, value in (
        ("name", name),
        ("description", description),
        ("project", project),
        ("profile", profile),
        ("options", options),
        ("ownership", ownership),
        ("custom_fields", custom_fields),
        ("output_location", output_location),
    ):
        if value is not None:
            body[key] = value
    return CustomizationJobCreateRequest.model_validate(body)


def _list_jobs_query_params(
    *,
    page: int | None,
    page_size: int | None,
    sort: str | None,
    filter: str | None,
) -> ListCustomizationJobsQueryParams | None:
    query_params: ListCustomizationJobsQueryParams = {}
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    if filter is not None:
        query_params["filter"] = filter
    return query_params or None


def _job_logs_query_params(
    *,
    limit: int | None,
    page_cursor: str | None,
    attempt_id: int | None,
    step_id: str | None,
    tail: int | None,
    task_id: str | None,
) -> JobLogsQueryParams | None:
    query_params: JobLogsQueryParams = {}
    if limit is not None:
        query_params["limit"] = limit
    if page_cursor is not None:
        query_params["page_cursor"] = page_cursor
    if attempt_id is not None:
        query_params["attempt_id"] = attempt_id
    if step_id is not None:
        query_params["step_id"] = step_id
    if tail is not None:
        query_params["tail"] = tail
    if task_id is not None:
        query_params["task_id"] = task_id
    return query_params or None


class JobsResource:
    """Sync SDK namespace at ``client.customization.<backend>.jobs``."""

    backend: ClassVar[str]

    def __init__(self, context: CustomizationSDKContext) -> None:
        self._customization = context.customization
        self._jobs = context.jobs

    def create(
        self,
        spec: BaseModel | dict[str, Any],
        workspace: str | None = None,
        name: str | None = None,
        *,
        description: str | None = None,
        project: str | None = None,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
        ownership: dict[str, Any] | None = None,
        custom_fields: dict[str, Any] | None = None,
        output_location: str | None = None,
    ) -> NemoResponse[CustomizationJob]:
        """Submit a training job through the Customizer backend route."""
        body = _create_request(
            spec,
            name=name,
            description=description,
            project=project,
            profile=profile,
            options=options,
            ownership=ownership,
            custom_fields=custom_fields,
            output_location=output_location,
        )
        return self._customization.create_customization_job(
            workspace=workspace,
            backend=self.backend,
            body=body,
        )

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: str | None = None,
    ) -> NemoPaginatedResponse[CustomizationJob]:
        """List backend jobs through the Customizer route."""
        return self._customization.list_customization_jobs(
            workspace=workspace,
            backend=self.backend,
            query_params=_list_jobs_query_params(
                page=page,
                page_size=page_size,
                sort=sort,
                filter=filter,
            ),
        )

    def retrieve(self, name: str, *, workspace: str | None = None) -> NemoResponse[CustomizationJob]:
        """Retrieve one backend job through the Customizer route."""
        return self._customization.get_customization_job(
            workspace=workspace,
            backend=self.backend,
            name=name,
        )

    def get_job_resource(self, job_name: str, workspace: str | None = None) -> NemoResponse[CustomizationJob]:
        """Compatibility alias for ``retrieve``; returns the typed response."""
        return self.retrieve(job_name, workspace=workspace)

    def get_status(self, name: str, *, workspace: str | None = None) -> NemoResponse[PlatformJobStatusResponse]:
        """Fetch current job status from the core Jobs service."""
        return self._jobs.get_job_status(name=name, workspace=workspace)

    def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
        attempt_id: int | None = None,
        step_id: str | None = None,
        tail: int | None = None,
        task_id: str | None = None,
    ) -> NemoPaginatedResponse[PlatformJobLog, CursorPagination]:
        """Fetch job logs from the core Jobs service."""
        return self._jobs.list_job_logs(
            name=name,
            workspace=workspace,
            query_params=_job_logs_query_params(
                limit=limit,
                page_cursor=page_cursor,
                attempt_id=attempt_id,
                step_id=step_id,
                tail=tail,
                task_id=task_id,
            ),
        )


class AsyncJobsResource:
    """Async SDK namespace at ``client.customization.<backend>.jobs``."""

    backend: ClassVar[str]

    def __init__(self, context: AsyncCustomizationSDKContext) -> None:
        self._customization = context.customization
        self._jobs = context.jobs

    async def create(
        self,
        spec: BaseModel | dict[str, Any],
        workspace: str | None = None,
        name: str | None = None,
        *,
        description: str | None = None,
        project: str | None = None,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
        ownership: dict[str, Any] | None = None,
        custom_fields: dict[str, Any] | None = None,
        output_location: str | None = None,
    ) -> NemoResponse[CustomizationJob]:
        """Submit a training job through the Customizer backend route."""
        body = _create_request(
            spec,
            name=name,
            description=description,
            project=project,
            profile=profile,
            options=options,
            ownership=ownership,
            custom_fields=custom_fields,
            output_location=output_location,
        )
        return await self._customization.create_customization_job(
            workspace=workspace,
            backend=self.backend,
            body=body,
        )

    async def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: str | None = None,
    ) -> AsyncNemoPaginatedResponse[CustomizationJob]:
        """List backend jobs through the Customizer route."""
        return await self._customization.list_customization_jobs(
            workspace=workspace,
            backend=self.backend,
            query_params=_list_jobs_query_params(
                page=page,
                page_size=page_size,
                sort=sort,
                filter=filter,
            ),
        )

    async def retrieve(self, name: str, *, workspace: str | None = None) -> NemoResponse[CustomizationJob]:
        """Retrieve one backend job through the Customizer route."""
        return await self._customization.get_customization_job(
            workspace=workspace,
            backend=self.backend,
            name=name,
        )

    async def get_job_resource(self, job_name: str, workspace: str | None = None) -> NemoResponse[CustomizationJob]:
        """Compatibility alias for ``retrieve``; returns the typed response."""
        return await self.retrieve(job_name, workspace=workspace)

    async def get_status(self, name: str, *, workspace: str | None = None) -> NemoResponse[PlatformJobStatusResponse]:
        """Fetch current job status from the core Jobs service."""
        return await self._jobs.get_job_status(name=name, workspace=workspace)

    async def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        limit: int | None = None,
        page_cursor: str | None = None,
        attempt_id: int | None = None,
        step_id: str | None = None,
        tail: int | None = None,
        task_id: str | None = None,
    ) -> AsyncNemoPaginatedResponse[PlatformJobLog, CursorPagination]:
        """Fetch job logs from the core Jobs service."""
        return await self._jobs.list_job_logs(
            name=name,
            workspace=workspace,
            query_params=_job_logs_query_params(
                limit=limit,
                page_cursor=page_cursor,
                attempt_id=attempt_id,
                step_id=step_id,
                tail=tail,
                task_id=task_id,
            ),
        )


class CustomizationBackendResource:
    backend: ClassVar[str]
    jobs_resource_cls: ClassVar[type[JobsResource]]

    def __init__(self, context: CustomizationSDKContext) -> None:
        self.jobs: JobsResource
        self.jobs = self.jobs_resource_cls(context)


class AsyncCustomizationBackendResource:
    backend: ClassVar[str]
    jobs_resource_cls: ClassVar[type[AsyncJobsResource]]

    def __init__(self, context: AsyncCustomizationSDKContext) -> None:
        self.jobs: AsyncJobsResource
        self.jobs = self.jobs_resource_cls(context)


def make_customization_sdk(
    backend: str,
) -> tuple[type[CustomizationBackendResource], type[AsyncCustomizationBackendResource]]:
    """Build the sync + async ``<Backend>Customization`` SDK namespace classes."""
    title = backend.capitalize()

    sync_jobs = type(
        f"{title}JobsResource",
        (JobsResource,),
        {"backend": backend},
    )
    async_jobs = type(
        f"Async{title}JobsResource",
        (AsyncJobsResource,),
        {"backend": backend},
    )
    sync_cls = type(
        f"{title}Customization",
        (CustomizationBackendResource,),
        {"backend": backend, "jobs_resource_cls": sync_jobs},
    )
    async_cls = type(
        f"Async{title}Customization",
        (AsyncCustomizationBackendResource,),
        {"backend": backend, "jobs_resource_cls": async_jobs},
    )
    return sync_cls, async_cls
