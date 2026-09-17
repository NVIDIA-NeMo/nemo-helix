# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentsClient transport tests via mocked httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest
from nemo_platform_plugin.agents.client import AgentsClient, AsyncAgentsClient
from nemo_platform_plugin.agents.types import (
    AgentJobRequest,
    CreateAgentRequest,
    CreateDeploymentRequest,
    CreateSessionRequest,
    InvokeAgentRequest,
)
from nemo_platform_plugin.client.errors import BadRequestError, ConflictError, NotFoundError, UnprocessableEntityError
from nemo_platform_plugin.jobs.schemas import FileStorageType
from pydantic import JsonValue

BASE = "http://test:8000"


def _page(item: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "data": [item],
        "pagination": {
            "page": 1,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": 1,
            "total_results": 1,
        },
    }


def _read_json(request: httpx.Request) -> dict[str, JsonValue]:
    body = json.loads(request.content)
    assert isinstance(body, dict)
    return body


def test_create_agent_serializes_body_and_unwraps() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/agents"
        assert _read_json(request) == {"name": "calc", "config": {"workflow": {}}}
        return httpx.Response(201, json={"name": "calc", "workspace": "default", "config": {"workflow": {}}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    agent = client.create_agent(body=CreateAgentRequest(name="calc", config={"workflow": {}})).data()

    assert agent.name == "calc"
    assert agent.config == {"workflow": {}}
    assert len(seen) == 1


def test_create_agent_exist_ok_replays_get_on_conflict() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(409, json={"detail": "already exists"})
        assert request.method == "GET"
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/agents/calc"
        return httpx.Response(200, json={"name": "calc", "workspace": "default", "config": {"workflow": {}}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    agent = client.create_agent(body=CreateAgentRequest(name="calc", config={"workflow": {}}), exist_ok=True).data()

    assert agent.name == "calc"
    assert [request.method for request in seen] == ["POST", "GET"]


def test_list_agents_paginates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/agents?page_size=1"
        return httpx.Response(200, json=_page({"name": "calc", "workspace": "default", "config": {}}))

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    names = [agent.name for agent in client.list_agents(query_params={"page_size": 1}).items()]

    assert names == ["calc"]


def test_deployment_session_and_invoke_routes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/deployments"):
            assert _read_json(request) == {"agent": "calc", "name": "calc-prod"}
            return httpx.Response(201, json={"name": "calc-prod", "workspace": "default", "agent": "calc"})
        if request.url.path.endswith("/sessions"):
            assert _read_json(request) == {"deployment_id": "dep-id"}
            return httpx.Response(
                201,
                json={"name": "session-one", "workspace": "default", "deployment_id": "dep-id"},
            )
        assert request.url.path.endswith("/deployments/calc-prod/-/v1/chat/completions")
        assert request.headers["X-Nemo-Session-Id"] == "session-one"
        return httpx.Response(200, json={"id": "completion-id"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    deployment = client.create_deployment(body=CreateDeploymentRequest(agent="calc", name="calc-prod")).data()
    session = client.create_session(body=CreateSessionRequest(deployment_id="dep-id")).data()
    completion = client.with_headers({"X-Nemo-Session-Id": session.name}).invoke_deployment(
        name=deployment.name,
        body=InvokeAgentRequest(messages=[{"role": "user", "content": "hello"}]),
    )

    assert completion.data() == {"id": "completion-id"}
    assert [request.method for request in seen] == ["POST", "POST", "POST"]


def test_stream_deployment_logs_parses_sse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/deployments/calc-prod/logs/stream"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'id: 7\ndata: {"timestamp":"2026-01-01T00:00:00","message":"ready"}\n\n',
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    with client.stream_deployment_logs(name="calc-prod").stream() as lines:
        parsed = list(lines)

    assert len(parsed) == 1
    assert parsed[0].message == "ready"


def test_download_agent_job_result_reads_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/jobs/execute/job-1/results/result/download"
        return httpx.Response(200, content=b'{"ok": true}\n')

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    result = client.download_agent_job_result(collection="execute", job="job-1", name="result").read()

    assert result == b'{"ok": true}\n'


def test_agent_job_methods() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/jobs/execute"
            assert _read_json(request) == {"spec": {"agent": "calc", "input": "2+2"}}
            return httpx.Response(
                201,
                json={"name": "job-1", "workspace": "default", "spec": {"agent": "calc"}, "status": "created"},
            )
        if request.url.path.endswith("/results"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "result",
                            "job": "job-1",
                            "workspace": "default",
                            "artifact_url": "fileset://result",
                            "artifact_storage_type": FileStorageType.FILESET.value,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "job-1",
                "name": "job-1",
                "status": "created",
                "status_details": {},
                "error_details": None,
                "steps": [],
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    job = client.create_agent_job(
        collection="execute",
        body=AgentJobRequest(spec={"agent": "calc", "input": "2+2"}),
    ).data()
    results = client.list_agent_job_results(collection="execute", name=job.name).data()
    status = client.get_agent_job_status(collection="execute", name=job.name).data()

    assert job.name == "job-1"
    assert results.data[0].name == "result"
    assert status.name == "job-1"
    assert [request.method for request in seen] == ["POST", "GET", "GET"]


def test_execute_job_compat_methods_return_json_maps() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/jobs/execute"
            assert _read_json(request) == {"name": "job-1", "spec": {"agent": "calc", "input": "2+2"}}
            return httpx.Response(
                201,
                json={"name": "job-1", "workspace": "default", "spec": {"agent": "calc"}, "status": "created"},
            )
        if request.url.path.endswith("/results"):
            return httpx.Response(200, json={"data": [{"name": "result"}]})
        return httpx.Response(200, json={"name": "job-1", "workspace": "default", "spec": {"agent": "calc"}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    job = client.jobs.execute.create(name="job-1", spec={"agent": "calc", "input": "2+2"})
    retrieved = client.jobs.execute.get("job-1")
    results = client.jobs.execute.list_results("job-1")

    assert job["name"] == "job-1"
    assert retrieved["name"] == "job-1"
    assert results["data"] == [{"name": "result"}]
    assert [request.method for request in seen] == ["POST", "GET", "GET"]


def test_execute_job_compat_uses_cloned_client_after_preclone_access() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/cloned/jobs/execute/job-1"
        assert request.headers["X-Clone"] == "yes"
        return httpx.Response(200, json={"name": "job-1", "workspace": "cloned", "spec": {}})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)
    _ = client.jobs

    cloned = client.with_workspace("cloned").with_headers({"X-Clone": "yes"})
    retrieved = cloned.jobs.execute.get("job-1")

    assert retrieved["workspace"] == "cloned"


@pytest.mark.asyncio
async def test_async_execute_job_compat_uses_cloned_client_after_preclone_access() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/cloned/jobs/execute/job-1"
        return httpx.Response(200, json={"name": "job-1", "workspace": "cloned", "spec": {}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAgentsClient(base_url=BASE, workspace="default", http_client=http_client)
    _ = client.jobs

    cloned = client.with_workspace("cloned")
    retrieved = await cloned.jobs.execute.get("job-1")

    assert retrieved["workspace"] == "cloned"
    await http_client.aclose()


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(400, BadRequestError), (404, NotFoundError), (409, ConflictError), (422, UnprocessableEntityError)],
)
def test_typed_http_errors(status_code: int, error_type: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": f"bad {request.url.path}"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    with pytest.raises(error_type):
        client.get_agent(name="missing").data()


@pytest.mark.asyncio
async def test_async_create_session() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url == f"{BASE}/apis/agents/v2/workspaces/default/sessions"
        return httpx.Response(201, json={"name": "session-one", "deployment_id": "dep-id"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAgentsClient(base_url=BASE, workspace="default", http_client=http_client)

    session = (await client.create_session(body=CreateSessionRequest(deployment_id="dep-id"))).data()

    assert session.name == "session-one"
    assert len(seen) == 1
    await http_client.aclose()
