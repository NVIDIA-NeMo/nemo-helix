# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AgentHardenerClient / AsyncAgentHardenerClient via mocked httpx transport."""

from __future__ import annotations

import json

import httpx
import pytest
from nemo_platform_plugin.agent_hardener.client import AgentHardenerClient, AsyncAgentHardenerClient
from nemo_platform_plugin.agent_hardener.types import (
    EventIn,
    ManifestInit,
    SynthBenignJobRequest,
    SynthBenignSpec,
    WarGameJobRequest,
    WarGameSpec,
)

BASE = "http://test:8000"


def _manifest_json(name: str = "manifest-1") -> dict[str, str | int | list[str] | dict[str, str]]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": "default",
        "agent": "default/agent",
        "source_type": "agent",
        "port": 8000,
        "secrets": [],
        "egress": [],
        "env": {},
        "warnings": [],
        "benign_suite": [],
        "benign_interview": [],
        "defenders": [],
        "attack_intensity": "standard",
        "rounds": 1,
    }


def _run_json(name: str) -> dict[str, str | int]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": "default",
        "agent": "default/agent",
        "job_id": f"job-{name}",
        "port": 8000,
        "manifest": "/tmp/agent-hardener.yaml",
        "manifest_id": "manifest-1",
        "status": "completed",
        "returncode": 0,
        "summary": "ok",
    }


def _war_game_job_json(name: str = "job-1") -> dict[str, str | dict[str, str]]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": "default",
        "spec": {"manifest_id": "manifest-1"},
        "status": "created",
        "status_details": {},
        "error_details": {},
    }


def _synth_job_json(name: str = "synth-1") -> dict[str, str | dict[str, str]]:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": "default",
        "spec": {"manifest_id": "manifest-1"},
        "status": "created",
        "status_details": {},
        "error_details": {},
    }


def test_create_manifest_serializes_body_and_unwraps() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.url == f"{BASE}/apis/agent-hardener/v2/workspaces/default/manifests"
        assert json.loads(request.content) == {"name": "manifest-1", "agent": "default/agent"}
        return httpx.Response(201, json=_manifest_json())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)

    manifest = client.create_manifest(body=ManifestInit(name="manifest-1", agent="default/agent")).data()

    assert manifest.name == "manifest-1"
    assert manifest.id == "id-manifest-1"
    assert len(seen) == 1


def test_list_runs_paginates_and_preserves_query_params() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            return httpx.Response(
                200,
                json={
                    "data": [_run_json("run-1")],
                    "pagination": {
                        "page": 1,
                        "page_size": 1,
                        "current_page_size": 1,
                        "total_pages": 2,
                        "total_results": 2,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [_run_json("run-2")],
                "pagination": {
                    "page": 2,
                    "page_size": 1,
                    "current_page_size": 1,
                    "total_pages": 2,
                    "total_results": 2,
                },
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)

    names = [run.name for run in client.list_runs(query_params={"page_size": 1, "filter[status]": "completed"}).items()]

    assert names == ["run-1", "run-2"]
    assert seen[0].params["page_size"] == "1"
    assert seen[0].params["filter[status]"] == "completed"
    assert seen[1].params["page"] == "2"
    assert seen[1].params["filter[status]"] == "completed"


def test_submit_war_game_job_uses_jobs_route() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url == f"{BASE}/apis/agent-hardener/v2/workspaces/default/jobs"
        assert json.loads(request.content) == {"name": "job-1", "spec": {"manifest_id": "manifest-1"}}
        return httpx.Response(201, json=_war_game_job_json())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)
    body = WarGameJobRequest(name="job-1", spec=WarGameSpec(manifest_id="manifest-1"))

    job = client.create_war_game_job(body=body).data()

    assert job.name == "job-1"
    assert job.spec.manifest_id == "manifest-1"
    assert len(seen) == 1


def test_download_synth_benign_job_result_reads_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert (
            request.url
            == f"{BASE}/apis/agent-hardener/v2/workspaces/default/synth-benign/jobs/synth-1/results/suite/download"
        )
        return httpx.Response(200, content=b"tool,payload,label\n")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)

    assert client.download_synth_benign_job_result(job="synth-1", name="suite").read() == b"tool,payload,label\n"


@pytest.mark.asyncio
async def test_async_create_synth_benign_job() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.url == f"{BASE}/apis/agent-hardener/v2/workspaces/default/synth-benign/jobs"
        assert json.loads(request.content) == {"name": "synth-1", "spec": {"manifest_id": "manifest-1"}}
        return httpx.Response(201, json=_synth_job_json())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)
    body = SynthBenignJobRequest(name="synth-1", spec=SynthBenignSpec(manifest_id="manifest-1"))

    job = (await client.create_synth_benign_job(body=body)).data()

    assert job.name == "synth-1"
    assert job.spec.manifest_id == "manifest-1"
    assert len(seen) == 1
    await http_client.aclose()


@pytest.mark.asyncio
async def test_async_event_routes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            assert request.url == f"{BASE}/apis/agent-hardener/v2/workspaces/default/runs/run-1/events"
            assert json.loads(request.content) == {"event": "started"}
            return httpx.Response(204)
        assert request.url == f"{BASE}/apis/agent-hardener/v2/workspaces/default/runs/run-1/events?after=7"
        return httpx.Response(200, json={"events": [{"id": 8, "event": "started", "payload": {}}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAgentHardenerClient(base_url=BASE, workspace="default", http_client=http_client)

    assert (await client.ingest_event(name="run-1", body=EventIn(event="started"))).data() is None
    events = (await client.get_events(name="run-1", query_params={"after": 7})).data()

    assert events.events[0]["event"] == "started"
    assert [request.method for request in seen] == ["POST", "GET"]
    await http_client.aclose()
