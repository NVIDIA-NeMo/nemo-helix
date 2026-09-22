# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin SDK resources for Safe Synthesizer."""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import quote

import httpx
from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLogPage
from nemo_helix_plugin.jobs.types import JobLogsQueryParams
from nemo_helix_plugin.sdk import NemoPluginSDKResources
from nemo_safe_synthesizer_plugin.sdk import http_utils


class SafeSynthesizerJobsResource:
    """Sync SDK namespace mounted as ``client.safe_synthesizer.jobs``."""

    def __init__(self, platform: NeMoHelix) -> None:
        self._platform = platform
        self._http_client = platform._client

    def create(
        self,
        *,
        spec: dict[str, Any],
        workspace: str | None = None,
        name: str | None = None,
        project: str | None = None,
        description: str | None = None,
        ownership: dict[str, object] | None = None,
        custom_fields: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Create a Safe Synthesizer platform job through the plugin route."""
        payload: dict[str, Any] = {"spec": spec}
        if name is not None:
            payload["name"] = name
        if project is not None:
            payload["project"] = project
        if description is not None:
            payload["description"] = description
        if ownership is not None:
            payload["ownership"] = ownership
        if custom_fields is not None:
            payload["custom_fields"] = custom_fields

        response = self._http_client.post(
            http_utils.url(self._platform, "/v2/workspaces/{workspace}/jobs", workspace),
            json=payload,
            headers=http_utils.platform_default_headers(self._platform),
            timeout=timeout,
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    def list(self, *, workspace: str | None = None, **params: Any) -> Any:
        """List Safe Synthesizer jobs."""
        response = self._http_client.get(
            http_utils.url(self._platform, "/v2/workspaces/{workspace}/jobs", workspace),
            params={key: value for key, value in params.items() if value is not None},
            headers=http_utils.platform_default_headers(self._platform),
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    def retrieve(self, name: str, *, workspace: str | None = None) -> Any:
        """Retrieve one Safe Synthesizer job by name."""
        response = self._http_client.get(
            http_utils.url(
                self._platform,
                f"/v2/workspaces/{{workspace}}/jobs/{quote(name, safe='')}",
                workspace,
            ),
            headers=http_utils.platform_default_headers(self._platform),
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    def get_status(self, name: str, *, workspace: str | None = None) -> Any:
        """Retrieve Safe Synthesizer job status."""
        return client_from_platform(self._platform, JobsClient).get_job_status(name=name, workspace=workspace).data()

    def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        **params: Any,
    ) -> Any:
        """Retrieve paginated Safe Synthesizer job logs from the Jobs service."""
        query_params = {key: value for key, value in params.items() if value is not None}
        page = (
            client_from_platform(self._platform, JobsClient)
            .list_job_logs(
                name=name,
                workspace=workspace,
                query_params=cast(JobLogsQueryParams, query_params) or None,
            )
            .page()
        )
        return HelixJobLogPage(data=page.items, **page.metadata)


class SafeSynthesizerResource:
    """Sync SDK namespace mounted as ``client.safe_synthesizer``."""

    def __init__(self, platform: NeMoHelix) -> None:
        self._platform = platform
        self.jobs = SafeSynthesizerJobsResource(platform)


class AsyncSafeSynthesizerJobsResource:
    """Async SDK namespace mounted as ``client.safe_synthesizer.jobs``."""

    def __init__(self, platform: AsyncNeMoHelix) -> None:
        self._platform = platform
        self._http_client = platform._client

    async def create(
        self,
        *,
        spec: dict[str, Any],
        workspace: str | None = None,
        name: str | None = None,
        project: str | None = None,
        description: str | None = None,
        ownership: dict[str, object] | None = None,
        custom_fields: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Create a Safe Synthesizer platform job through the plugin route."""
        payload: dict[str, Any] = {"spec": spec}
        if name is not None:
            payload["name"] = name
        if project is not None:
            payload["project"] = project
        if description is not None:
            payload["description"] = description
        if ownership is not None:
            payload["ownership"] = ownership
        if custom_fields is not None:
            payload["custom_fields"] = custom_fields

        response = await self._http_client.post(
            http_utils.url(self._platform, "/v2/workspaces/{workspace}/jobs", workspace),
            json=payload,
            headers=http_utils.platform_default_headers(self._platform),
            timeout=timeout,
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    async def list(self, *, workspace: str | None = None, **params: Any) -> Any:
        """List Safe Synthesizer jobs."""
        response = await self._http_client.get(
            http_utils.url(self._platform, "/v2/workspaces/{workspace}/jobs", workspace),
            params={key: value for key, value in params.items() if value is not None},
            headers=http_utils.platform_default_headers(self._platform),
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    async def retrieve(self, name: str, *, workspace: str | None = None) -> Any:
        """Retrieve one Safe Synthesizer job by name."""
        response = await self._http_client.get(
            http_utils.url(
                self._platform,
                f"/v2/workspaces/{{workspace}}/jobs/{quote(name, safe='')}",
                workspace,
            ),
            headers=http_utils.platform_default_headers(self._platform),
        )
        _raise_for_status(response)
        return _object_from_mapping(response.json())

    async def get_status(self, name: str, *, workspace: str | None = None) -> Any:
        """Retrieve Safe Synthesizer job status."""
        jobs = client_from_platform(self._platform, AsyncJobsClient)
        return (await jobs.get_job_status(name=name, workspace=workspace)).data()

    async def get_logs(
        self,
        name: str,
        *,
        workspace: str | None = None,
        **params: Any,
    ) -> Any:
        """Retrieve paginated Safe Synthesizer job logs from the Jobs service."""
        query_params = {key: value for key, value in params.items() if value is not None}
        response = await client_from_platform(self._platform, AsyncJobsClient).list_job_logs(
            name=name,
            workspace=workspace,
            query_params=cast(JobLogsQueryParams, query_params) or None,
        )
        page = response.page()
        return HelixJobLogPage(data=page.items, **page.metadata)


class AsyncSafeSynthesizerResource:
    """Async SDK namespace mounted as ``client.safe_synthesizer``."""

    def __init__(self, platform: AsyncNeMoHelix) -> None:
        self._platform = platform
        self.jobs = AsyncSafeSynthesizerJobsResource(platform)


def _object_from_mapping(value: Any) -> Any:
    """Convert JSON objects into attribute-accessible objects recursively."""
    if isinstance(value, dict):
        return _SDKObject({str(key): _object_from_mapping(child) for key, child in value.items()})
    if isinstance(value, list):
        return [_object_from_mapping(child) for child in value]
    return value


def _raise_for_status(response: httpx.Response) -> None:
    """Raise HTTP errors with FastAPI detail text included."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        detail = _response_detail(response)
        if detail:
            message = f"{e}. Response detail: {detail}"
            raise httpx.HTTPStatusError(message, request=e.request, response=e.response) from e
        raise


def _response_detail(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text or None
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body) if body else None


class _SDKObject(dict[str, Any]):
    """Small dict wrapper with attribute access for plugin route responses."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


safe_synthesizer_sdk_resources = NemoPluginSDKResources(
    sync_resource=SafeSynthesizerResource,
    async_resource=AsyncSafeSynthesizerResource,
)
