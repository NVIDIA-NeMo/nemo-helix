# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Agents service.

Wraps the endpoint functions from ``agents.endpoints`` as direct methods
using the ``method()`` descriptor, following the files/models pattern.
"""

from __future__ import annotations

from functools import cached_property

from nemo_platform_plugin.agents import endpoints
from nemo_platform_plugin.agents.types import CreateExecuteJobRequest, JsonObject
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method


class _AgentsMethods:
    get_agent = method(endpoints.get_agent)
    list_agents = method(endpoints.list_agents)
    create_agent = method(endpoints.create_agent)
    delete_agent = method(endpoints.delete_agent)
    get_deployment = method(endpoints.get_deployment)
    list_deployments = method(endpoints.list_deployments)
    create_deployment = method(endpoints.create_deployment)
    delete_deployment = method(endpoints.delete_deployment)
    get_deployment_logs = method(endpoints.get_deployment_logs)
    stream_deployment_logs = method(endpoints.stream_deployment_logs)
    create_session = method(endpoints.create_session)
    list_sessions = method(endpoints.list_sessions)
    get_session = method(endpoints.get_session)
    close_session = method(endpoints.close_session)
    delete_session = method(endpoints.delete_session)
    create_environment_spec = method(endpoints.create_environment_spec)
    list_environment_specs = method(endpoints.list_environment_specs)
    get_environment_spec = method(endpoints.get_environment_spec)
    delete_environment_spec = method(endpoints.delete_environment_spec)
    create_environment = method(endpoints.create_environment)
    list_environments = method(endpoints.list_environments)
    get_environment = method(endpoints.get_environment)
    delete_environment = method(endpoints.delete_environment)
    create_compute_spec = method(endpoints.create_compute_spec)
    list_compute_specs = method(endpoints.list_compute_specs)
    get_compute_spec = method(endpoints.get_compute_spec)
    delete_compute_spec = method(endpoints.delete_compute_spec)
    invoke_agent = method(endpoints.invoke_agent)
    invoke_deployment = method(endpoints.invoke_deployment)
    create_execute_job = method(endpoints.create_execute_job)
    get_execute_job = method(endpoints.get_execute_job)
    list_execute_job_results = method(endpoints.list_execute_job_results)
    create_agent_job = method(endpoints.create_agent_job)
    list_agent_jobs = method(endpoints.list_agent_jobs)
    get_agent_job = method(endpoints.get_agent_job)
    delete_agent_job = method(endpoints.delete_agent_job)
    cancel_agent_job = method(endpoints.cancel_agent_job)
    list_agent_job_logs = method(endpoints.list_agent_job_logs)
    list_agent_job_results = method(endpoints.list_agent_job_results)
    get_agent_job_status = method(endpoints.get_agent_job_status)
    get_agent_job_result = method(endpoints.get_agent_job_result)
    download_agent_job_result = method(endpoints.download_agent_job_result)


def _execute_job_request(
    *,
    spec: JsonObject,
    name: str | None,
    description: str | None,
) -> CreateExecuteJobRequest:
    payload: dict[str, JsonObject | str] = {"spec": spec}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    return CreateExecuteJobRequest.model_validate(payload)


class _ExecuteJobsCompat:
    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    def create(
        self,
        *,
        spec: JsonObject,
        name: str | None = None,
        description: str | None = None,
        workspace: str | None = None,
    ) -> JsonObject:
        return self._client.create_execute_job(
            workspace=workspace,
            body=_execute_job_request(spec=spec, name=name, description=description),
        ).data()

    def get(self, name: str, *, workspace: str | None = None) -> JsonObject:
        return self._client.get_execute_job(name=name, workspace=workspace).data()

    def list_results(self, name: str, *, workspace: str | None = None) -> JsonObject:
        return self._client.list_execute_job_results(name=name, workspace=workspace).data()


class _AsyncExecuteJobsCompat:
    def __init__(self, client: AsyncAgentsClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        spec: JsonObject,
        name: str | None = None,
        description: str | None = None,
        workspace: str | None = None,
    ) -> JsonObject:
        response = await self._client.create_execute_job(
            workspace=workspace,
            body=_execute_job_request(spec=spec, name=name, description=description),
        )
        return response.data()

    async def get(self, name: str, *, workspace: str | None = None) -> JsonObject:
        return (await self._client.get_execute_job(name=name, workspace=workspace)).data()

    async def list_results(self, name: str, *, workspace: str | None = None) -> JsonObject:
        return (await self._client.list_execute_job_results(name=name, workspace=workspace)).data()


class _JobsCompat:
    def __init__(self, client: AgentsClient) -> None:
        self._client = client

    @cached_property
    def execute(self) -> _ExecuteJobsCompat:
        return _ExecuteJobsCompat(self._client)


class _AsyncJobsCompat:
    def __init__(self, client: AsyncAgentsClient) -> None:
        self._client = client

    @cached_property
    def execute(self) -> _AsyncExecuteJobsCompat:
        return _AsyncExecuteJobsCompat(self._client)


class AgentsClient(_AgentsMethods, NemoClient):
    """Sync client for the Agents service API."""

    @property
    def jobs(self) -> _JobsCompat:
        return _JobsCompat(self)


class AsyncAgentsClient(_AgentsMethods, AsyncNemoClient):
    """Async client for the Agents service API."""

    @property
    def jobs(self) -> _AsyncJobsCompat:
        return _AsyncJobsCompat(self)
