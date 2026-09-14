# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Agents service."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.agents.types import (
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentJob,
    AgentJobCollection,
    AgentJobLog,
    AgentJobLogsQueryParams,
    AgentJobRequest,
    AgentJobResult,
    AgentJobResultListResponse,
    AgentJobStatusResponse,
    AgentSession,
    CreateAgentDeploymentRequest,
    CreateAgentRequest,
    CreateComputeSpecRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentSpecRequest,
    CreateExecuteJobRequest,
    CreateSessionRequest,
    DeploymentLogsQueryParams,
    DeploymentLogsResponse,
    InvokeAgentRequest,
    JsonMap,
    JsonObject,
    ListAgentJobResultsQueryParams,
    ListAgentJobsQueryParams,
    ListAgentsQueryParams,
    ListDeploymentsQueryParams,
    ListEnvironmentResourcesQueryParams,
    ListSessionsQueryParams,
    LogLine,
)
from nemo_platform_plugin.client.endpoint import delete, get, post
from nemo_platform_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest, Stream

_PREFIX = "/apis/agents/v2/workspaces/{workspace}"
_AGENTS = f"{_PREFIX}/agents"
_DEPLOYMENTS = f"{_PREFIX}/deployments"
_SESSIONS = f"{_PREFIX}/sessions"
_ENVIRONMENT_SPECS = f"{_PREFIX}/environment-specs"
_ENVIRONMENTS = f"{_PREFIX}/environments"
_COMPUTE_SPECS = f"{_PREFIX}/compute-specs"
_EXECUTE_JOBS = f"{_PREFIX}/jobs/execute"
_JOBS = f"{_PREFIX}/jobs/{{collection}}"


# ---------------------------------------------------------------------------
# Agent CRUD
# ---------------------------------------------------------------------------


@get(f"{_AGENTS}/{{name}}")
@abstractmethod
def get_agent(*, workspace: str | None = None, name: str) -> Agent: ...


@get(_AGENTS)
@abstractmethod
def list_agents(
    *, workspace: str | None = None, query_params: ListAgentsQueryParams | None = None
) -> Paginated[Agent]: ...


def _get_agent_on_conflict(body: CreateAgentRequest, workspace: str | None) -> PreparedRequest[Agent]:
    """Build the retrieve request replayed when ``create_agent(exist_ok=True)`` 409s."""
    return get_agent(name=body.name, workspace=workspace)


@post(_AGENTS, get_on_conflict=_get_agent_on_conflict)
@abstractmethod
def create_agent(*, workspace: str | None = None, body: CreateAgentRequest, exist_ok: bool = False) -> Agent: ...


@delete(f"{_AGENTS}/{{name}}")
@abstractmethod
def delete_agent(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Agent deployment CRUD + logs
# ---------------------------------------------------------------------------


@get(f"{_DEPLOYMENTS}/{{name}}")
@abstractmethod
def get_deployment(*, workspace: str | None = None, name: str) -> AgentDeployment: ...


@get(_DEPLOYMENTS)
@abstractmethod
def list_deployments(
    *, workspace: str | None = None, query_params: ListDeploymentsQueryParams | None = None
) -> Paginated[AgentDeployment]: ...


@post(_DEPLOYMENTS)
@abstractmethod
def create_deployment(*, workspace: str | None = None, body: CreateAgentDeploymentRequest) -> AgentDeployment: ...


@delete(f"{_DEPLOYMENTS}/{{name}}")
@abstractmethod
def delete_deployment(*, workspace: str | None = None, name: str) -> None: ...


@get(f"{_DEPLOYMENTS}/{{name}}/logs")
@abstractmethod
def get_deployment_logs(
    *, workspace: str | None = None, name: str, query_params: DeploymentLogsQueryParams | None = None
) -> DeploymentLogsResponse: ...


@get(f"{_DEPLOYMENTS}/{{name}}/logs/stream")
@abstractmethod
def stream_deployment_logs(*, workspace: str | None = None, name: str) -> Stream[LogLine]: ...


# ---------------------------------------------------------------------------
# Agent sessions
# ---------------------------------------------------------------------------


@post(_SESSIONS)
@abstractmethod
def create_session(*, workspace: str | None = None, body: CreateSessionRequest) -> AgentSession: ...


@get(_SESSIONS)
@abstractmethod
def list_sessions(
    *, workspace: str | None = None, query_params: ListSessionsQueryParams | None = None
) -> Paginated[AgentSession]: ...


@get(f"{_SESSIONS}/{{name}}")
@abstractmethod
def get_session(*, workspace: str | None = None, name: str) -> AgentSession: ...


@post(f"{_SESSIONS}/{{name}}/close")
@abstractmethod
def close_session(*, workspace: str | None = None, name: str) -> AgentSession: ...


@delete(f"{_SESSIONS}/{{name}}")
@abstractmethod
def delete_session(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Environment, environment-spec, and compute-spec CRUD
# ---------------------------------------------------------------------------


@post(_ENVIRONMENT_SPECS)
@abstractmethod
def create_environment_spec(
    *, workspace: str | None = None, body: CreateEnvironmentSpecRequest
) -> AgentEnvironmentSpec: ...


@get(_ENVIRONMENT_SPECS)
@abstractmethod
def list_environment_specs(
    *, workspace: str | None = None, query_params: ListEnvironmentResourcesQueryParams | None = None
) -> Paginated[AgentEnvironmentSpec]: ...


@get(f"{_ENVIRONMENT_SPECS}/{{name}}")
@abstractmethod
def get_environment_spec(*, workspace: str | None = None, name: str) -> AgentEnvironmentSpec: ...


@delete(f"{_ENVIRONMENT_SPECS}/{{name}}")
@abstractmethod
def delete_environment_spec(*, workspace: str | None = None, name: str) -> None: ...


@post(_ENVIRONMENTS)
@abstractmethod
def create_environment(*, workspace: str | None = None, body: CreateEnvironmentRequest) -> AgentEnvironment: ...


@get(_ENVIRONMENTS)
@abstractmethod
def list_environments(
    *, workspace: str | None = None, query_params: ListEnvironmentResourcesQueryParams | None = None
) -> Paginated[AgentEnvironment]: ...


@get(f"{_ENVIRONMENTS}/{{name}}")
@abstractmethod
def get_environment(*, workspace: str | None = None, name: str) -> AgentEnvironment: ...


@delete(f"{_ENVIRONMENTS}/{{name}}")
@abstractmethod
def delete_environment(*, workspace: str | None = None, name: str) -> None: ...


@post(_COMPUTE_SPECS)
@abstractmethod
def create_compute_spec(*, workspace: str | None = None, body: CreateComputeSpecRequest) -> AgentComputeSpec: ...


@get(_COMPUTE_SPECS)
@abstractmethod
def list_compute_specs(
    *, workspace: str | None = None, query_params: ListEnvironmentResourcesQueryParams | None = None
) -> Paginated[AgentComputeSpec]: ...


@get(f"{_COMPUTE_SPECS}/{{name}}")
@abstractmethod
def get_compute_spec(*, workspace: str | None = None, name: str) -> AgentComputeSpec: ...


@delete(f"{_COMPUTE_SPECS}/{{name}}")
@abstractmethod
def delete_compute_spec(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Gateway invocation
# ---------------------------------------------------------------------------


@post(f"{_AGENTS}/{{name}}/-/v1/chat/completions")
@abstractmethod
def invoke_agent(*, workspace: str | None = None, name: str, body: InvokeAgentRequest) -> JsonMap: ...


@post(f"{_DEPLOYMENTS}/{{name}}/-/v1/chat/completions")
@abstractmethod
def invoke_deployment(*, workspace: str | None = None, name: str, body: InvokeAgentRequest) -> JsonMap: ...


# ---------------------------------------------------------------------------
# agents.execute jobs
# ---------------------------------------------------------------------------


@post(_EXECUTE_JOBS)
@abstractmethod
def create_execute_job(*, workspace: str | None = None, body: CreateExecuteJobRequest) -> JsonObject: ...


@get(f"{_EXECUTE_JOBS}/{{name}}")
@abstractmethod
def get_execute_job(*, workspace: str | None = None, name: str) -> JsonObject: ...


@get(f"{_EXECUTE_JOBS}/{{name}}/results")
@abstractmethod
def list_execute_job_results(*, workspace: str | None = None, name: str) -> JsonObject: ...


# ---------------------------------------------------------------------------
# Agents job collections
# ---------------------------------------------------------------------------


@post(_JOBS)
@abstractmethod
def create_agent_job(
    *, workspace: str | None = None, collection: AgentJobCollection, body: AgentJobRequest
) -> AgentJob: ...


@get(_JOBS)
@abstractmethod
def list_agent_jobs(
    *,
    workspace: str | None = None,
    collection: AgentJobCollection,
    query_params: ListAgentJobsQueryParams | None = None,
) -> Paginated[AgentJob]: ...


@get(f"{_JOBS}/{{name}}")
@abstractmethod
def get_agent_job(*, workspace: str | None = None, collection: AgentJobCollection, name: str) -> AgentJob: ...


@delete(f"{_JOBS}/{{name}}")
@abstractmethod
def delete_agent_job(*, workspace: str | None = None, collection: AgentJobCollection, name: str) -> None: ...


@post(f"{_JOBS}/{{name}}/cancel")
@abstractmethod
def cancel_agent_job(*, workspace: str | None = None, collection: AgentJobCollection, name: str) -> AgentJob: ...


@get(f"{_JOBS}/{{name}}/logs")
@abstractmethod
def list_agent_job_logs(
    *,
    workspace: str | None = None,
    collection: AgentJobCollection,
    name: str,
    query_params: AgentJobLogsQueryParams | None = None,
) -> Paginated[AgentJobLog, CursorPagination]: ...


@get(f"{_JOBS}/{{name}}/results")
@abstractmethod
def list_agent_job_results(
    *,
    workspace: str | None = None,
    collection: AgentJobCollection,
    name: str,
    query_params: ListAgentJobResultsQueryParams | None = None,
) -> AgentJobResultListResponse: ...


@get(f"{_JOBS}/{{name}}/status")
@abstractmethod
def get_agent_job_status(
    *, workspace: str | None = None, collection: AgentJobCollection, name: str
) -> AgentJobStatusResponse: ...


@get(f"{_JOBS}/{{job}}/results/{{name}}")
@abstractmethod
def get_agent_job_result(
    *, workspace: str | None = None, collection: AgentJobCollection, job: str, name: str
) -> AgentJobResult: ...


@get(f"{_JOBS}/{{job}}/results/{{name}}/download")
@abstractmethod
def download_agent_job_result(
    *, workspace: str | None = None, collection: AgentJobCollection, job: str, name: str
) -> BinaryContent: ...
