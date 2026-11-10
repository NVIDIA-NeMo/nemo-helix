# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for evals service dependency wiring."""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Depends, FastAPI
from nemo_evals.api import dependencies
from nemo_evals.api.dependencies import get_taskset_run_service, get_taskset_service
from nemo_evals.api.service.taskset_run_service import TasksetRunService, normalize_run_caller
from nemo_evals.entities import TasksetEntity
from nemo_evals.taskset_runs.identifiers import derive_run_id
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.dependencies import get_effective_principal_id, get_request_authorizer
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import Entity


async def test_taskset_service_adapts_request_authorizer(entity_store) -> None:
    """Task membership checks use the caller-scoped HTTP authorizer."""
    authorize = AsyncMock()
    service = get_taskset_service(entity_store, AsyncMock(), authorize)

    await service.authorize_task_read("source workspace", "task/name")

    authorize.assert_awaited_once_with("GET", "/apis/evals/v2/workspaces/source%20workspace/tasks/task%2Fname")


async def test_taskset_run_dependency_closes_factory_client(monkeypatch):
    clients = []
    close_calls = []
    factory = dependencies.get_async_nemo_client

    def provide(**kwargs):
        assert kwargs == {"as_service": "evals", "internal": True, "on_behalf_of": "user:alice"}
        client = factory(**kwargs)
        clients.append(client)
        close = AsyncMock(wraps=client.close)
        close_calls.append(close)
        monkeypatch.setattr(client, "close", close)
        return client

    monkeypatch.setattr(dependencies, "get_async_nemo_client", provide)
    app = FastAPI()
    app.dependency_overrides[get_effective_principal_id] = lambda: "user:alice"
    app.dependency_overrides[get_request_authorizer] = lambda: AsyncMock()

    @app.get("/probe")
    async def probe(service: TasksetRunService = Depends(get_taskset_run_service)):
        assert not service._async_client._http.is_closed
        assert isinstance(service._store.entity_client, EntityClient)
        assert isinstance(service._store.entity_client._client, AsyncEntitiesClient)
        assert service._store.entity_client._client.retry is not None
        assert 409 not in service._store.entity_client._client.retry.retryable_status_codes
        return {"alive": True}

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/probe")).json() == {"alive": True}
        assert len(clients) == 1
        close_calls[0].assert_awaited_once()
    finally:
        for client in clients:
            await client._http.aclose()


@pytest.mark.parametrize("caller", ["user:alice", "service:runner", ""])
async def test_taskset_run_delegation_matches_stamped_creator(monkeypatch, caller):
    sent = []

    def create(request):
        sent.append(request)
        body = json.loads(request.content)
        effective_id = request.headers.get("X-NHX-Principal-On-Behalf-Of") or request.headers["X-NHX-Principal-Id"]
        now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        entity = Entity(
            entity_type="taskset",
            id="entity-1",
            name=body["name"],
            workspace="default",
            data=body["data"],
            created_by=effective_id,
            created_at=now,
            updated_at=now,
            db_version=0,
        )
        return httpx.Response(201, json=entity.model_dump(mode="json"))

    factory = dependencies.get_async_nemo_client
    async with httpx.MockTransport(create) as transport:

        def provide(**kwargs):
            client = factory(**kwargs)
            monkeypatch.setattr(client._http, "_transport_for_url", lambda url: transport)
            return client

        monkeypatch.setattr(dependencies, "get_async_nemo_client", provide)
        dependency = get_taskset_run_service(caller, AsyncMock())
        service = await anext(dependency)
        try:
            created = await service._entities.create(TasksetEntity(name="creator-probe", workspace="default"))
            expected = normalize_run_caller(caller)
            assert created.created_by == expected
            assert created.created_by is not None
            assert sent[0].headers["X-NHX-Principal-On-Behalf-Of"] == expected
            assert derive_run_id("default", created.created_by, "key") == derive_run_id("default", expected, "key")
        finally:
            await dependency.aclose()
            await service._async_client._http.aclose()


async def test_taskset_run_dependency_delegates_the_callers_full_principal(monkeypatch):
    principal = AuthContext(
        principal_id="user:bob",
        principal_email="bob@ac9.test",
        principal_groups=["ac9-users"],
        principal_account_id="a1",
    )
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(404, json={"detail": "absent"})

    factory = dependencies.get_async_nemo_client
    async with httpx.MockTransport(answer) as transport:

        def provide(**kwargs):
            client = factory(**kwargs)
            monkeypatch.setattr(client._http, "_transport_for_url", lambda url: transport)
            return client

        monkeypatch.setattr(dependencies, "get_async_nemo_client", provide)
        monkeypatch.setattr(dependencies, "current_auth_context", lambda: principal)
        monkeypatch.setattr(dependencies, "current_request_principal", principal.to_principal)
        dependency = get_taskset_run_service("user:bob", AsyncMock())
        service = await anext(dependency)
        try:
            assert service._auth_context == principal  # stored on each new run for the controller
            assert await service._store.find("default", "run-x") is None
            headers = sent[0].headers
            assert headers["X-NHX-Principal-On-Behalf-Of"] == "user:bob"
            assert headers["X-NHX-Principal-On-Behalf-Of-Email"] == "bob@ac9.test"
            assert headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "ac9-users"
        finally:
            await dependency.aclose()
            await service._async_client._http.aclose()


async def test_taskset_run_dependency_keeps_the_callers_access_key_id(monkeypatch):
    """A limited service-account key keeps its workspace limit on the calls evals makes for it."""
    jti = "ak_" + "b" * 32
    # The platform's request principal, as current_request_principal returns it, carrying the key ID.
    request_principal: Any = AuthContext(principal_id="service-account:ci").to_principal()
    caller = request_principal.model_copy(update={"access_key_jti": jti})
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(404, json={"detail": "absent"})

    factory = dependencies.get_async_nemo_client
    async with httpx.MockTransport(answer) as transport:

        def provide(**kwargs):
            client = factory(**kwargs)
            monkeypatch.setattr(client._http, "_transport_for_url", lambda url: transport)
            return client

        monkeypatch.setattr(dependencies, "get_async_nemo_client", provide)
        monkeypatch.setattr(dependencies, "current_auth_context", lambda: AuthContext.from_principal(caller))
        monkeypatch.setattr(dependencies, "current_request_principal", lambda: caller)
        dependency = get_taskset_run_service("service-account:ci", AsyncMock())
        service = await anext(dependency)
        try:
            assert await service._store.find("default", "run-x") is None
            assert sent[0].headers["X-NHX-Principal-On-Behalf-Of"] == "service-account:ci"
            assert sent[0].headers["X-NHX-Subject-Access-Key-Jti"] == jti
            assert service._auth_context is not None
            assert "jti" not in service._auth_context.model_dump_json()  # the stored context keeps no key ID
        finally:
            await dependency.aclose()
            await service._async_client._http.aclose()


async def test_delegated_service_client_keeps_the_callers_access_key_id(monkeypatch):
    """Service-only reads (judge secrets, models) made for a limited key keep its workspace limit."""
    jti = "ak_" + "b" * 32
    request_principal: Any = AuthContext(principal_id="service-account:ci").to_principal()
    caller = request_principal.model_copy(update={"access_key_jti": jti})
    calls: list[dict[str, Any]] = []
    service_client = object()

    @asynccontextmanager
    async def fake_get_async_nemo_client(**kwargs: Any):
        calls.append(kwargs)
        yield service_client

    monkeypatch.setattr(dependencies, "current_request_principal", lambda: caller)
    monkeypatch.setattr(dependencies, "get_async_nemo_client", fake_get_async_nemo_client)

    async with dependencies.delegated_service_client(AsyncMock()) as client:
        assert client is service_client

    assert calls[0]["as_service"] == "evals"
    assert calls[0]["on_behalf_of"].access_key_jti == jti
