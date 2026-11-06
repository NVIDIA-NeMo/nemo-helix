# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin SDK resources for Safe Synthesizer."""

from __future__ import annotations

from typing import cast

from nemo_helix_plugin.client.adapter import AsyncHelixClient, SyncHelixClient, client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.response import AsyncNemoPaginatedResponse, NemoPaginatedResponse
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLogPage, HelixJobStatusResponse
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import JobLogsQueryParams
from nemo_helix_plugin.sdk import NemoPluginSDKResources
from nemo_safe_synthesizer_plugin.sdk.client import AsyncSafeSynthesizerClient, SafeSynthesizerClient
from nemo_safe_synthesizer_plugin.sdk.types import (
    CreateSafeSynthesizerJobRequest,
    JsonMap,
    ListSafeSynthesizerJobsQueryParams,
    SafeSynthesizerJobResponse,
)


def _create_request(
    *,
    spec: JsonMap,
    name: str | None,
    project: str | None,
    description: str | None,
    ownership: JsonMap | None,
    custom_fields: JsonMap | None,
) -> CreateSafeSynthesizerJobRequest:
    request = CreateSafeSynthesizerJobRequest(spec=spec)
    # Only fields the caller set are sent; the body serializes with exclude_unset.
    if name is not None:
        request.name = name
    if project is not None:
        request.project = project
    if description is not None:
        request.description = description
    if ownership is not None:
        request.ownership = ownership
    if custom_fields is not None:
        request.custom_fields = custom_fields
    return request


def _list_query_params(params: dict[str, object]) -> ListSafeSynthesizerJobsQueryParams | None:
    query_params = {key: value for key, value in params.items() if value is not None}
    return cast(ListSafeSynthesizerJobsQueryParams, query_params) or None


def _log_query_params(params: dict[str, object]) -> JobLogsQueryParams | None:
    query_params = {key: value for key, value in params.items() if value is not None}
    return cast(JobLogsQueryParams, query_params) or None


class SafeSynthesizerJobsResource:
    """Sync client for Safe Synthesizer jobs, exposed as ``SafeSynthesizerResource.jobs``."""

    def __init__(self, client: NemoClient) -> None:
        self._client = client
        self._safe_synthesizer = SafeSynthesizerClient.from_client(client)
        self._jobs = JobsClient.from_client(client)

    def create(
        self,
        *,
        spec: JsonMap,
        workspace: str | None = None,
        name: str | None = None,
        project: str | None = None,
        description: str | None = None,
        ownership: JsonMap | None = None,
        custom_fields: JsonMap | None = None,
        timeout: float | None = None,
    ) -> SafeSynthesizerJobResponse:
        """Create a Safe Synthesizer platform job through the plugin route."""
        body = _create_request(
            spec=spec,
            name=name,
            project=project,
            description=description,
            ownership=ownership,
            custom_fields=custom_fields,
        )
        safe_synthesizer = self._safe_synthesizer.with_options(timeout=timeout)
        return safe_synthesizer.create_job(workspace=self._client.resolve_workspace(workspace), body=body).data()

    def list(
        self, *, workspace: str | None = None, **params: object
    ) -> NemoPaginatedResponse[SafeSynthesizerJobResponse]:
        """List Safe Synthesizer jobs."""
        return self._safe_synthesizer.list_jobs(
            workspace=self._client.resolve_workspace(workspace),
            query_params=_list_query_params(params),
        )

    def retrieve(
        self, name: str, *, workspace: str | None = None, job_reference_kind: JobReferenceKind | None = None
    ) -> SafeSynthesizerJobResponse:
        """Retrieve one Safe Synthesizer job by name."""
        return self._safe_synthesizer.get_job(
            workspace=self._client.resolve_workspace(workspace), name=name, job_reference_kind=job_reference_kind
        ).data()

    def retrieve_by_id(self, job_id: str, *, workspace: str | None = None) -> SafeSynthesizerJobResponse:
        return self.retrieve(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID)

    def retrieve_by_name(self, job_name: str, *, workspace: str | None = None) -> SafeSynthesizerJobResponse:
        return self.retrieve(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME)

    def get_status(
        self, name: str, *, workspace: str | None = None, job_reference_kind: JobReferenceKind | None = None
    ) -> HelixJobStatusResponse:
        """Retrieve Safe Synthesizer job status."""
        return self._jobs.get_job_status(name=name, workspace=workspace, job_reference_kind=job_reference_kind).data()

    def get_status_by_id(self, job_id: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return self.get_status(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID)

    def get_status_by_name(self, job_name: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return self.get_status(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME)

    def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        **params: object,
    ) -> HelixJobLogPage:
        """Retrieve paginated Safe Synthesizer job logs from the Jobs service."""
        page = self._jobs.list_job_logs(
            name=name,
            workspace=workspace,
            query_params=_log_query_params(params),
            job_reference_kind=job_reference_kind,
        ).page()
        return HelixJobLogPage(data=page.items, **page.metadata)

    def get_logs_by_id(self, job_id: str, *, workspace: str | None = None, **params: object) -> HelixJobLogPage:
        return self.get_logs(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID, **params)

    def get_logs_by_name(self, job_name: str, *, workspace: str | None = None, **params: object) -> HelixJobLogPage:
        return self.get_logs(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME, **params)


class SafeSynthesizerResource:
    """Sync client for the Safe Synthesizer plugin service."""

    def __init__(self, client: NemoClient) -> None:
        self.jobs = SafeSynthesizerJobsResource(client)


class AsyncSafeSynthesizerJobsResource:
    """Async client for Safe Synthesizer jobs, exposed as ``AsyncSafeSynthesizerResource.jobs``."""

    def __init__(self, client: AsyncNemoClient) -> None:
        self._client = client
        self._safe_synthesizer = AsyncSafeSynthesizerClient.from_client(client)
        self._jobs = AsyncJobsClient.from_client(client)

    async def create(
        self,
        *,
        spec: JsonMap,
        workspace: str | None = None,
        name: str | None = None,
        project: str | None = None,
        description: str | None = None,
        ownership: JsonMap | None = None,
        custom_fields: JsonMap | None = None,
        timeout: float | None = None,
    ) -> SafeSynthesizerJobResponse:
        """Create a Safe Synthesizer platform job through the plugin route."""
        body = _create_request(
            spec=spec,
            name=name,
            project=project,
            description=description,
            ownership=ownership,
            custom_fields=custom_fields,
        )
        safe_synthesizer = self._safe_synthesizer.with_options(timeout=timeout)
        response = await safe_synthesizer.create_job(workspace=self._client.resolve_workspace(workspace), body=body)
        return response.data()

    async def list(
        self, *, workspace: str | None = None, **params: object
    ) -> AsyncNemoPaginatedResponse[SafeSynthesizerJobResponse]:
        """List Safe Synthesizer jobs."""
        return await self._safe_synthesizer.list_jobs(
            workspace=self._client.resolve_workspace(workspace),
            query_params=_list_query_params(params),
        )

    async def retrieve(
        self, name: str, *, workspace: str | None = None, job_reference_kind: JobReferenceKind | None = None
    ) -> SafeSynthesizerJobResponse:
        """Retrieve one Safe Synthesizer job by name."""
        response = await self._safe_synthesizer.get_job(
            workspace=self._client.resolve_workspace(workspace), name=name, job_reference_kind=job_reference_kind
        )
        return response.data()

    async def retrieve_by_id(self, job_id: str, *, workspace: str | None = None) -> SafeSynthesizerJobResponse:
        return await self.retrieve(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID)

    async def retrieve_by_name(self, job_name: str, *, workspace: str | None = None) -> SafeSynthesizerJobResponse:
        return await self.retrieve(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME)

    async def get_status(
        self, name: str, *, workspace: str | None = None, job_reference_kind: JobReferenceKind | None = None
    ) -> HelixJobStatusResponse:
        """Retrieve Safe Synthesizer job status."""
        return (
            await self._jobs.get_job_status(name=name, workspace=workspace, job_reference_kind=job_reference_kind)
        ).data()

    async def get_status_by_id(self, job_id: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return await self.get_status(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID)

    async def get_status_by_name(self, job_name: str, *, workspace: str | None = None) -> HelixJobStatusResponse:
        return await self.get_status(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME)

    async def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        job_reference_kind: JobReferenceKind | None = None,
        **params: object,
    ) -> HelixJobLogPage:
        """Retrieve paginated Safe Synthesizer job logs from the Jobs service."""
        response = await self._jobs.list_job_logs(
            name=name,
            workspace=workspace,
            query_params=_log_query_params(params),
            job_reference_kind=job_reference_kind,
        )
        page = response.page()
        return HelixJobLogPage(data=page.items, **page.metadata)

    async def get_logs_by_id(self, job_id: str, *, workspace: str | None = None, **params: object) -> HelixJobLogPage:
        return await self.get_logs(job_id, workspace=workspace, job_reference_kind=JobReferenceKind.ID, **params)

    async def get_logs_by_name(
        self, job_name: str, *, workspace: str | None = None, **params: object
    ) -> HelixJobLogPage:
        return await self.get_logs(job_name, workspace=workspace, job_reference_kind=JobReferenceKind.NAME, **params)


class AsyncSafeSynthesizerResource:
    """Async client for the Safe Synthesizer plugin service."""

    def __init__(self, client: AsyncNemoClient) -> None:
        self.jobs = AsyncSafeSynthesizerJobsResource(client)


def _make_sync_resource(platform: SyncHelixClient) -> SafeSynthesizerResource:
    return SafeSynthesizerResource(client_from_platform(platform, NemoClient))


def _make_async_resource(platform: AsyncHelixClient) -> AsyncSafeSynthesizerResource:
    return AsyncSafeSynthesizerResource(client_from_platform(platform, AsyncNemoClient))


safe_synthesizer_sdk_resources = NemoPluginSDKResources(
    sync_resource=_make_sync_resource,
    async_resource=_make_async_resource,
)
