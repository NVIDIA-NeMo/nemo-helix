# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only AgentHardenerRun route handlers.

Uses FastAPI's TestClient with dependency_overrides to mock the entity client. The router is
mounted at the same prefix the platform mounts in production, so the URLs match what Studio hits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.api.v2 import runs as runs_router_module
from nemo_agent_hardener_plugin.entities import AgentHardenerRun
from nemo_platform_plugin.entity_client import (
    NemoEntityNotFoundError,
    NemoPaginationInfo,
    get_entity_client,
)

NOW = datetime.now(timezone.utc)
PREFIX = "/apis/agent-hardener/v2/workspaces/{workspace}"


def _make_run(name: str = "run-1", workspace: str = "default", **fields) -> AgentHardenerRun:
    fields.setdefault("agent", "clockbot")
    fields.setdefault("status", "completed")
    run = AgentHardenerRun(name=name, workspace=workspace, **fields)
    run._id = f"agent-hardener-run-{name}-id"
    run._created_at = NOW
    run._updated_at = NOW
    return run


def _list_response(items):
    resp = MagicMock()
    resp.data = items
    resp.pagination = NemoPaginationInfo(
        page=1, page_size=20, current_page_size=len(items), total_pages=1, total_results=len(items)
    )
    return resp


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_entity_client: AsyncMock) -> TestClient:
    app = FastAPI()
    app.include_router(runs_router_module.router, prefix=PREFIX)
    app.dependency_overrides[get_entity_client] = lambda: mock_entity_client
    return TestClient(app, raise_server_exceptions=False)


class TestListRuns:
    def test_returns_envelope(self, client, mock_entity_client) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([_make_run("run-1")]))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert [r["name"] for r in body["data"]] == ["run-1"]
        assert body["pagination"]["total_results"] == 1
        assert body["sort"] == "-created_at"

    def test_agent_filter_is_passed_through(self, client, mock_entity_client) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs?filter[agent]=clockbot")

        assert resp.status_code == 200, resp.text
        call = mock_entity_client.list.await_args
        assert call is not None
        assert call.kwargs["filter_obj"] == {"agent": "clockbot"}

    def test_unknown_filter_key_returns_422(self, client, mock_entity_client) -> None:
        mock_entity_client.list = AsyncMock(return_value=_list_response([]))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs?filter[bogus]=x")

        assert resp.status_code == 422, resp.text

    def test_store_error_returns_500(self, client, mock_entity_client) -> None:
        mock_entity_client.list = AsyncMock(side_effect=RuntimeError("boom"))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs")

        assert resp.status_code == 500


class TestGetRun:
    def test_returns_run(self, client, mock_entity_client) -> None:
        mock_entity_client.get = AsyncMock(return_value=_make_run("run-1", returncode=0))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs/run-1")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == "run-1"
        assert body["agent"] == "clockbot"
        assert body["id"] == "agent-hardener-run-run-1-id"

    def test_missing_run_returns_404(self, client, mock_entity_client) -> None:
        mock_entity_client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))

        resp = client.get("/apis/agent-hardener/v2/workspaces/default/runs/ghost")

        assert resp.status_code == 404


class TestDeleteRun:
    def test_deletes_run(self, client, mock_entity_client) -> None:
        mock_entity_client.delete = AsyncMock(return_value=None)

        resp = client.delete("/apis/agent-hardener/v2/workspaces/default/runs/run-1")

        assert resp.status_code == 204, resp.text
        call = mock_entity_client.delete.await_args
        assert call is not None
        assert call.kwargs["name"] == "run-1"

    def test_missing_run_returns_404(self, client, mock_entity_client) -> None:
        mock_entity_client.delete = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))

        resp = client.delete("/apis/agent-hardener/v2/workspaces/default/runs/ghost")

        assert resp.status_code == 404
