# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real ClickHouse persistence through the manifest gateway and an external HTTP app.

The adapter has a real loopback listener. Its typed Intake HTTP transport bridges
into the fixture's real Intake application (auth-disabled test configuration),
not a span-store mock. Gateway authorization itself has separate OPA tests.
"""

from __future__ import annotations

import asyncio
import importlib
import socket
import threading
import time
from collections.abc import Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import uvicorn
from anyio.to_thread import run_sync
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nhx.platform_runner.community_plugins import installation
from nhx.platform_runner.community_plugins.gateway import ExternalPluginsService

KEY = "local-test-adapter-key-not-a-secret"
PUBLIC = "/apis/plugins/phoenix/v2/workspaces/default"


@contextmanager
def external_server(factory: Callable[[str], FastAPI]) -> Generator[str]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(factory(base), log_level="error", ws="none"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started, "External adapter did not start"
            yield base
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "External adapter did not stop"


def test_registered_external_adapter_round_trips_real_clickhouse(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The independent app is deliberately not a platform workspace dependency.
    source = Path(__file__).resolve().parents[5] / "plugins/phoenix-traces/backend/src"
    monkeypatch.syspath_prepend(str(source))
    adapter = importlib.import_module("nemo_phoenix_traces.app")
    intake_calls: list[httpx.Request] = []
    secret_calls: list[httpx.Request] = []

    async def intake_transport(request: httpx.Request) -> httpx.Response:
        intake_calls.append(request)
        assert request.headers["authorization"] == "Bearer dedicated-intake-test-token"
        assert "cookie" not in request.headers
        response = await run_sync(
            lambda: client.request(
                request.method,
                request.url.raw_path.decode("ascii"),
                headers=dict(request.headers),
                content=request.content,
            )
        )
        return httpx.Response(response.status_code, headers=dict(response.headers), content=response.content)

    def secret_transport(request: httpx.Request) -> httpx.Response:
        secret_calls.append(request)
        assert request.method == "GET"
        assert request.url.path.endswith("/secrets/phoenix-traces-adapter-key/access")
        assert request.headers["authorization"] == "Bearer human-platform-test-token"
        return httpx.Response(
            200,
            json={"name": "phoenix-traces-adapter-key", "workspace": request.url.path.split("/")[5], "value": KEY},
        )

    intake_base = AsyncNemoClient(
        base_url="http://127.0.0.1:8080",
        default_headers={"Authorization": "Bearer dedicated-intake-test-token"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(intake_transport)),
        owns_http_client=True,
    )
    platform_client = AsyncNemoClient(
        base_url="http://127.0.0.1:8080",
        default_headers={"Authorization": "Bearer human-platform-test-token"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(secret_transport)),
        owns_http_client=True,
    )
    service = ExternalPluginsService(tmp_path / "plugins" / "installed.json")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.on_startup()
        try:
            yield
        finally:
            await service.on_shutdown()

    platform = FastAPI(lifespan=lifespan)
    platform.dependency_overrides[get_nemo_client] = lambda: platform_client
    # This test exercises registration/consent/persistence; the real admin and
    # caller-kind guards are exercised in the gateway/installation/OPA suites.
    platform.dependency_overrides[installation.require_plugin_admin] = lambda: "test-human-admin"
    platform.include_router(service.get_routers()[0].router, prefix="/apis/plugins")

    def create_external(base: str) -> FastAPI:
        settings = adapter.Settings(adapter_key=KEY, public_url=base, bundle_dir=tmp_path / "no-bundle")
        return adapter.create_app(settings, intake_client=AsyncIntakeClient.from_client(intake_base))

    try:
        with external_server(create_external) as base, TestClient(platform) as browser:
            source_manifest = {"url": f"{base}/plugin.yaml"}
            preview = browser.post("/apis/plugins/-/preview", json=source_manifest)
            assert preview.status_code == 200, preview.text
            installed = browser.post(
                "/apis/plugins/-/installed",
                json={**source_manifest, "manifest_sha256": preview.json()["manifest_sha256"], "consent": True},
            )
            assert installed.status_code == 201, installed.text
            assert service.store.load().plugins[0].name == "phoenix"
            start = datetime.now(timezone.utc) - timedelta(minutes=1)
            spans = [
                {
                    "name": "persisted agent",
                    "context": {"trace_id": "persisted-trace", "span_id": "root"},
                    "span_kind": "AGENT",
                    "start_time": start.isoformat(),
                    "end_time": (start + timedelta(seconds=2)).isoformat(),
                    "status_code": "OK",
                    "attributes": {"input.value": {"question": "hello 雪"}},
                },
                {
                    "name": "persisted model",
                    "context": {"trace_id": "persisted-trace", "span_id": "child"},
                    "parent_id": "root",
                    "span_kind": "LLM",
                    "start_time": (start + timedelta(seconds=1)).isoformat(),
                    "end_time": None,
                    "status_code": "ERROR",
                    "status_message": "provider error",
                    "attributes": {
                        "llm.model_name": "stored-model",
                        "llm.token_count.prompt": 7,
                        "llm.token_count.completion": 5,
                        "llm.token_count.total": 12,
                        "provider.raw": {"nested": [1, True, None, "雪"]},
                        "gen_ai.project": "cannot-override-selected-project",
                    },
                },
            ]
            project = "real project / 雪"
            headers = {"Authorization": "Bearer human-platform-test-token", "Cookie": "session=private"}
            stored = browser.post(
                f"{PUBLIC}/spans", params={"project": project}, json={"spans": spans}, headers=headers
            )
            assert stored.status_code == 201, stored.text
            assert any(call.url.path.endswith("/ingest/spans") for call in intake_calls)
            projects = browser.get(f"{PUBLIC}/projects", headers=headers)
            assert projects.status_code == 200, projects.text
            assert projects.json()["data"] == [{"name": project}]
            first = browser.get(f"{PUBLIC}/spans", params={"project": project, "limit": 1}, headers=headers)
            assert first.status_code == 200, (first.text, client.get("/apis/intake/v2/workspaces/default/spans").text)
            root = first.json()["data"][0]
            assert root["context"] == {"trace_id": "persisted-trace", "span_id": "root"}
            assert root["attributes"]["input.value"] == {"question": "hello 雪"}
            second = browser.get(
                f"{PUBLIC}/spans",
                params={"project": project, "limit": 1, "cursor": first.json()["next_cursor"]},
                headers=headers,
            )
            assert second.status_code == 200, second.text
            child = second.json()["data"][0]
            assert child["parent_id"] == "root"
            assert child["end_time"] is None
            assert child["status_code"] == "ERROR"
            assert child["status_message"] == "provider error"
            assert child["attributes"]["llm.model_name"] == "stored-model"
            assert child["attributes"]["llm.token_count.total"] == 12
            assert child["attributes"]["provider.raw"] == {"nested": [1, True, None, "雪"]}
            assert second.json()["next_cursor"] is None
            # Even if an operator accidentally copies the same adapter secret
            # into another workspace, the adapter cannot use its Intake token there.
            denied = browser.get(PUBLIC.replace("/default", "/other") + "/projects", headers=headers)
            assert denied.status_code == 403, denied.text
            assert all("/workspaces/default/" in call.url.path for call in intake_calls)
            assert len(secret_calls) >= 5

        # A newly constructed app has no span cache/state to recover. The same
        # trace is still read from Intake/ClickHouse after the first app stops.
        with external_server(create_external) as restarted:
            response = httpx.get(
                f"{restarted}/v1/workspaces/default/spans",
                params={"project": project},
                headers={"Authorization": f"Bearer {KEY}"},
                trust_env=False,
            )
            assert response.status_code == 200, response.text
            assert {span["context"]["span_id"] for span in response.json()["data"]} == {"root", "child"}
    finally:
        asyncio.run(intake_base.aclose())
        asyncio.run(platform_client.aclose())


def test_project_groups_and_unassigned_filter_are_persisted_and_workspace_scoped(client: TestClient):
    created = client.post("/apis/entities/v2/workspaces", json={"name": "other"})
    assert created.status_code == 201, created.text
    now = datetime.now(timezone.utc).isoformat()
    for workspace, project, span_id in (
        ("default", "", "unassigned"),
        ("default", "demo", "named"),
        ("other", "other", "hidden"),
    ):
        body = {
            "source": "project-group-test",
            "spans": [
                {"span_id": span_id, "trace_id": span_id, "started_at": now, "attributes": {"project.name": project}}
            ],
        }
        result = client.post(f"/apis/intake/v2/workspaces/{workspace}/ingest/spans", json=body)
        assert result.status_code == 201, result.text
    groups = client.get("/apis/intake/v2/workspaces/default/spans/groups", params={"by": "project"})
    assert groups.status_code == 200, groups.text
    assert {item["group"]["project"] for item in groups.json()["data"]} == {"", "demo"}
    unassigned = client.get("/apis/intake/v2/workspaces/default/spans", params={"filter[project]": ""})
    assert unassigned.status_code == 200, unassigned.text
    assert [span["span_id"] for span in unassigned.json()["data"]] == ["unassigned"]
