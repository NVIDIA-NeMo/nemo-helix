# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancellation read-repair does not depend on worker shutdown hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agent_hardener_plugin.api.v2 import runs
from nemo_agent_hardener_plugin.entities import AgentHardenerRun
from nemo_helix_plugin.entity_client import NemoEntityConflictError, get_entity_client


@pytest.fixture
def jobs_client(monkeypatch):
    client = AsyncMock()
    client.get_job_status.return_value = MagicMock()
    client.get_job_status.return_value.data.return_value = SimpleNamespace(status="cancelled")
    context = MagicMock()
    context.__aenter__ = AsyncMock()
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(runs, "get_async_nemo_client", MagicMock(return_value=context))
    monkeypatch.setattr(runs.AsyncJobsClient, "from_client", MagicMock(return_value=client))
    return client


@pytest.mark.parametrize("endpoint", ["detail", "list"])
def test_cancelled_run_is_persisted(endpoint, jobs_client):
    # A precreated record and an executing worker's record have the same active
    # state here. This tests read-repair, not the worker scheduling lifecycle.
    run = AgentHardenerRun(name="run", workspace="default", status="running", job_id="job")
    entity_client = AsyncMock()
    entity_client.get.return_value = run
    entity_client.list.return_value = SimpleNamespace(data=[run], pagination=None)
    entity_client.update.side_effect = lambda updated: updated
    app = FastAPI()
    app.include_router(runs.router, prefix="/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    path = "/workspaces/default/runs" + ("/run" if endpoint == "detail" else "")
    response = TestClient(app).get(path)
    assert response.status_code == 200
    result = response.json() if endpoint == "detail" else response.json()["data"][0]
    assert result["status"] == "failed"
    assert result["error_message"] == "Run job cancelled."
    entity_client.update.assert_awaited_once()
    jobs_client.get_job_status.assert_awaited_once_with(name="job", workspace="default")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_preserves_terminal_outcomes(status, jobs_client):
    run = AgentHardenerRun(name="run", workspace="default", status=status, job_id="job")
    entity_client = AsyncMock()
    assert await runs._reconcile_run(run, entity_client) is run
    jobs_client.get_job_status.assert_not_awaited()
    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_job_id_is_not_reconciled(jobs_client):
    run = AgentHardenerRun(name="run", workspace="default", status="running")
    entity_client = AsyncMock()
    assert await runs._reconcile_run(run, entity_client) is run
    jobs_client.get_job_status.assert_not_awaited()
    entity_client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobs_outage_keeps_run_readable(jobs_client):
    jobs_client.get_job_status.side_effect = RuntimeError("Jobs unavailable")
    run = AgentHardenerRun(name="run", workspace="default", status="running", job_id="job")
    entity_client = AsyncMock()
    assert await runs._reconcile_run(run, entity_client) is run
    entity_client.update.assert_not_awaited()


def test_running_filter_uses_selection_snapshot_then_converges(jobs_client):
    stored = AgentHardenerRun(name="run", workspace="default", status="running", job_id="job")
    entity_client = AsyncMock()

    async def select(*args, **kwargs):
        assert kwargs["filter_obj"] == {"status": "running"}
        return SimpleNamespace(data=[stored] if stored.status == "running" else [], pagination=None)

    async def persist(updated):
        nonlocal stored
        stored = updated
        return stored

    entity_client.list.side_effect = select
    entity_client.update.side_effect = persist
    app = FastAPI()
    app.include_router(runs.router, prefix="/workspaces/{workspace}")
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    client = TestClient(app)
    path = "/workspaces/default/runs?filter[status]=running"
    first = client.get(path)
    assert first.status_code == 200
    assert first.json()["data"][0]["status"] == "failed"
    assert stored.status == "failed"
    assert client.get(path).json()["data"] == []


@pytest.mark.asyncio
async def test_concurrent_completion_is_preserved(jobs_client):
    run = AgentHardenerRun(name="run", workspace="default", status="running", job_id="job")
    completed = run.model_copy(update={"status": "completed"})
    entity_client = AsyncMock()
    entity_client.update.side_effect = NemoEntityConflictError("changed")
    entity_client.get.return_value = completed
    assert await runs._reconcile_run(run, entity_client) is completed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed"])
async def test_non_cancelled_jobs_are_not_rewritten(status, jobs_client):
    jobs_client.get_job_status.return_value.data.return_value.status = status
    run = AgentHardenerRun(name="run", workspace="default", status="running", job_id="job")
    entity_client = AsyncMock()
    assert await runs._reconcile_run(run, entity_client) is run
    entity_client.update.assert_not_awaited()
