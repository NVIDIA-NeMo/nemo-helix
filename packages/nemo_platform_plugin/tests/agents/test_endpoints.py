# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agents service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_args, get_origin

import pytest
from nemo_platform_plugin.agents import endpoints
from nemo_platform_plugin.agents.types import (
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentJob,
    AgentJobCollection,
    AgentJobLog,
    AgentJobRequest,
    AgentJobResultListResponse,
    AgentJobStatusResponse,
    AgentSession,
    CreateAgentRequest,
    CreateComputeSpecRequest,
    CreateDeploymentRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentSpecRequest,
    CreateExecuteJobRequest,
    CreateSessionRequest,
    DeploymentLogsResponse,
    InvokeAgentRequest,
    JsonObject,
    LogLine,
)
from nemo_platform_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest, Stream
from pydantic import JsonValue

_PREFIX = "/apis/agents/v2/workspaces/{workspace}"
_COLLECTIONS: tuple[AgentJobCollection, ...] = (
    "analyze",
    "evaluate",
    "evaluate-suite",
    "execute",
    "optimize",
    "optimize-skills",
    "package",
)


def _json_body(prepared: PreparedRequest) -> dict[str, JsonValue]:
    assert isinstance(prepared.content, bytes)
    body = json.loads(prepared.content)
    assert isinstance(body, dict)
    return body


def test_create_agent() -> None:
    body = CreateAgentRequest(name="calc", config={"workflow": {}})
    prepared = endpoints.create_agent(workspace="default", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == f"{_PREFIX}/agents"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content_type == "application/json"
    assert prepared.response_type is Agent
    assert _json_body(prepared) == {"name": "calc", "config": {"workflow": {}}}
    assert prepared.on_conflict_get is not None
    assert prepared.on_conflict_get.path_params == {"workspace": "default", "name": "calc"}


def test_list_and_get_agents() -> None:
    list_prepared = endpoints.list_agents(workspace="default", query_params={"page": 2, "sort": "-created_at"})
    get_prepared = endpoints.get_agent(workspace="default", name="calc")

    assert get_origin(list_prepared.response_type) is Paginated
    assert list_prepared.query_params == {"page": 2, "sort": "-created_at"}
    assert get_prepared.path_params == {"workspace": "default", "name": "calc"}
    assert get_prepared.response_type is Agent


def test_deployment_crud_and_logs() -> None:
    body = CreateDeploymentRequest(agent="calc", name="calc-prod", deployment_mode="k8s", image="repo/calc:1")
    create_prepared = endpoints.create_deployment(workspace="default", body=body)
    list_prepared = endpoints.list_deployments(workspace="default")
    get_prepared = endpoints.get_deployment(workspace="default", name="calc-prod")
    logs_prepared = endpoints.get_deployment_logs(workspace="default", name="calc-prod", query_params={"tail": 100})
    stream_prepared = endpoints.stream_deployment_logs(workspace="default", name="calc-prod")

    assert create_prepared.path_template == f"{_PREFIX}/deployments"
    assert _json_body(create_prepared) == {
        "agent": "calc",
        "name": "calc-prod",
        "deployment_mode": "k8s",
        "image": "repo/calc:1",
    }
    assert get_origin(list_prepared.response_type) is Paginated
    assert get_prepared.response_type is AgentDeployment
    assert logs_prepared.query_params == {"tail": 100}
    assert logs_prepared.response_type is DeploymentLogsResponse
    assert get_origin(stream_prepared.response_type) is Stream
    assert get_args(stream_prepared.response_type) == (LogLine,)


def test_sessions() -> None:
    create_prepared = endpoints.create_session(
        workspace="default",
        body=CreateSessionRequest(deployment_id="dep-id", name="chat-one"),
    )
    list_prepared = endpoints.list_sessions(workspace="default", query_params={"filter": {"deployment_id": "dep-id"}})
    close_prepared = endpoints.close_session(workspace="default", name="chat-one")

    assert create_prepared.path_template == f"{_PREFIX}/sessions"
    assert create_prepared.response_type is AgentSession
    assert _json_body(create_prepared) == {"deployment_id": "dep-id", "name": "chat-one"}
    assert get_origin(list_prepared.response_type) is Paginated
    assert list_prepared.query_params == {"filter": {"deployment_id": "dep-id"}}
    assert close_prepared.method == "POST"
    assert close_prepared.path_template == f"{_PREFIX}/sessions/{{name}}/close"


def test_environment_resources() -> None:
    env_spec_create = endpoints.create_environment_spec(
        workspace="default",
        body=CreateEnvironmentSpecRequest(name="env-spec", env={"LOG_LEVEL": "debug"}),
    )
    env_create = endpoints.create_environment(
        workspace="default",
        body=CreateEnvironmentRequest(name="env", environment_spec="default/env-spec"),
    )
    compute_create = endpoints.create_compute_spec(
        workspace="default",
        body=CreateComputeSpecRequest(name="small"),
    )

    assert env_spec_create.path_template == f"{_PREFIX}/environment-specs"
    assert env_spec_create.response_type is AgentEnvironmentSpec
    assert get_origin(endpoints.list_environment_specs(workspace="default").response_type) is Paginated
    assert endpoints.get_environment_spec(workspace="default", name="env-spec").response_type is AgentEnvironmentSpec
    assert endpoints.delete_environment_spec(workspace="default", name="env-spec").response_type is None
    assert env_create.response_type is AgentEnvironment
    assert get_origin(endpoints.list_environments(workspace="default").response_type) is Paginated
    assert endpoints.get_environment(workspace="default", name="env").response_type is AgentEnvironment
    assert endpoints.delete_environment(workspace="default", name="env").response_type is None
    assert compute_create.response_type is AgentComputeSpec
    assert get_origin(endpoints.list_compute_specs(workspace="default").response_type) is Paginated
    assert endpoints.get_compute_spec(workspace="default", name="small").response_type is AgentComputeSpec
    assert endpoints.delete_compute_spec(workspace="default", name="small").response_type is None


def test_gateway_invocation() -> None:
    body = InvokeAgentRequest(messages=[{"role": "user", "content": "hello"}], stream=False)
    agent_prepared = endpoints.invoke_agent(workspace="default", name="calc", body=body)
    deployment_prepared = endpoints.invoke_deployment(workspace="default", name="calc-prod", body=body)

    assert agent_prepared.path_template == f"{_PREFIX}/agents/{{name}}/-/v1/chat/completions"
    assert deployment_prepared.path_template == f"{_PREFIX}/deployments/{{name}}/-/v1/chat/completions"
    assert _json_body(agent_prepared) == {"messages": [{"role": "user", "content": "hello"}], "stream": False}


def test_execute_job_compat_endpoints() -> None:
    create_prepared = endpoints.create_execute_job(
        workspace="default",
        body=CreateExecuteJobRequest(name="job-1", spec={"agent": "calc", "input": "2+2"}),
    )
    get_prepared = endpoints.get_execute_job(workspace="default", name="job-1")
    results_prepared = endpoints.list_execute_job_results(workspace="default", name="job-1")

    assert create_prepared.path_template == f"{_PREFIX}/jobs/execute"
    assert create_prepared.response_type == JsonObject
    assert _json_body(create_prepared) == {"name": "job-1", "spec": {"agent": "calc", "input": "2+2"}}
    assert get_prepared.path_template == f"{_PREFIX}/jobs/execute/{{name}}"
    assert get_prepared.response_type == JsonObject
    assert results_prepared.path_template == f"{_PREFIX}/jobs/execute/{{name}}/results"
    assert results_prepared.response_type == JsonObject


@pytest.mark.parametrize("collection", _COLLECTIONS)
def test_agent_job_collection_crud(collection: AgentJobCollection) -> None:
    create_prepared = endpoints.create_agent_job(
        workspace="default",
        collection=collection,
        body=AgentJobRequest(spec={"agent": "calc", "input": "2+2"}),
    )
    list_prepared = endpoints.list_agent_jobs(
        workspace="default",
        collection=collection,
        query_params={"page": 2, "filter": {"status": "completed"}},
    )
    get_prepared = endpoints.get_agent_job(workspace="default", collection=collection, name="job-1")
    cancel_prepared = endpoints.cancel_agent_job(workspace="default", collection=collection, name="job-1")

    assert create_prepared.path_template == f"{_PREFIX}/jobs/{{collection}}"
    assert create_prepared.path_params == {"workspace": "default", "collection": collection}
    assert create_prepared.response_type is AgentJob
    assert get_origin(list_prepared.response_type) is Paginated
    assert list_prepared.query_params == {"page": 2, "filter": {"status": "completed"}}
    assert get_prepared.response_type is AgentJob
    assert cancel_prepared.path_template == f"{_PREFIX}/jobs/{{collection}}/{{name}}/cancel"
    assert cancel_prepared.response_type is AgentJob
    assert endpoints.delete_agent_job(workspace="default", collection=collection, name="job-1").response_type is None


def test_agent_job_logs_results_status_and_download() -> None:
    logs_prepared = endpoints.list_agent_job_logs(
        workspace="default",
        collection="execute",
        name="job-1",
        query_params={"limit": 50, "page_cursor": "abc", "tail": 500},
    )
    results_prepared = endpoints.list_agent_job_results(workspace="default", collection="execute", name="job-1")
    status_prepared = endpoints.get_agent_job_status(workspace="default", collection="execute", name="job-1")
    result_prepared = endpoints.get_agent_job_result(
        workspace="default",
        collection="execute",
        job="job-1",
        name="result",
    )
    download_prepared = endpoints.download_agent_job_result(
        workspace="default",
        collection="execute",
        job="job-1",
        name="result",
    )

    assert get_args(logs_prepared.response_type) == (AgentJobLog, CursorPagination)
    assert logs_prepared.query_params == {"limit": 50, "page_cursor": "abc", "tail": 500}
    assert results_prepared.response_type is AgentJobResultListResponse
    assert status_prepared.response_type is AgentJobStatusResponse
    assert result_prepared.path_template == f"{_PREFIX}/jobs/{{collection}}/{{job}}/results/{{name}}"
    assert download_prepared.response_type is BinaryContent
