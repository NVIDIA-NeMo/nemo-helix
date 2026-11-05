# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creator snapshot policy through Entity Store's request dependencies."""

import json
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from nemo_helix_plugin.auth.decisions import DecisionEvaluationInput
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.middleware import AuthorizationMiddleware
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.server import app
from nhx.core.entities.app.repository import (
    SQLAlchemyEntityRepository,
    SQLAlchemyWorkspaceRepository,
    get_async_session_maker,
)
from nhx.core.entities.service import EntitiesService
from nhx.platform_runner import server
from nhx.testing.policy import PolicyHarness
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

pytestmark = pytest.mark.asyncio


def _client(principal: Principal, *, auth_enabled: bool) -> AsyncClient:
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=auth_enabled))
    wrapper = FastAPI()

    @wrapper.middleware("http")
    async def bind_auth(request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = auth_client_context.set(auth_client)
        try:
            return await call_next(request)
        finally:
            auth_client_context.reset(token)

    wrapper.mount("/apis/entities", app)
    return AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test")


@pytest_asyncio.fixture
async def policy_store(
    session_maker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[SQLAlchemyEntityRepository]:
    policy = PolicyHarness()
    policy.document["authz"]["roles"]["SnapshotReader"] = {"permissions": ["entities.read"]}
    policy.document["authz"]["principals"]["user@example.com"] = {
        "workspaces": {"policy-one": ["Editor", "SnapshotReader"]}
    }
    policy.document["authz"]["principals"]["snapshot-admin"] = {"workspaces": {"system": ["HelixAdmin"]}}
    policy.install(policy.document)
    original_evaluate = AuthClient._evaluate_decision

    async def evaluate(self: AuthClient, document: DecisionEvaluationInput):
        if not self.auth_enabled:
            return await original_evaluate(self, document)
        return policy.decide(document)

    monkeypatch.setattr(AuthClient, "_evaluate_decision", evaluate)

    async def check(self: AuthClient, pdp_endpoint: str, auth_input: dict, result_key: str, check_name: str) -> bool:
        return policy.evaluate(pdp_endpoint, auth_input).get(result_key) is True

    monkeypatch.setattr(AuthClient, "_pdp_check", check)
    old_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_async_session_maker] = lambda: session_maker
    workspace_repo = SQLAlchemyWorkspaceRepository(session_maker)
    entity_repo = SQLAlchemyEntityRepository(session_maker)
    await workspace_repo.create_workspace(name="policy-one")
    await workspace_repo.create_workspace(name="policy-two")
    await entity_repo.create_entity(
        workspace="policy-one",
        entity_type="role_binding",
        name="policy-user-admin",
        data={
            "principal": "user@example.com",
            "workspace": "policy-one",
            "role": "Admin",
            "granted_by": "service:entities",
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "revoked_at": None,
        },
    )
    try:
        yield entity_repo
    finally:
        app.dependency_overrides = old_overrides


async def test_snapshot_visibility_and_preservation(policy_store: SQLAlchemyEntityRepository) -> None:
    path = "/apis/entities/v2/workspaces/policy-one/entities/platform_job"
    service_principal = Principal(id="service:jobs")
    user_principal = Principal(id="user@example.com", email="user@example.com")
    delegated_principal = Principal(id="service:jobs", on_behalf_of="user@example.com")
    original = {"id": "user@example.com", "groups": ["team-a"]}
    replacement = {"id": "other@example.com", "groups": ["team-b"]}

    async with AsyncExitStack() as stack:
        service = await stack.enter_async_context(_client(service_principal, auth_enabled=True))
        user = await stack.enter_async_context(_client(user_principal, auth_enabled=True))
        delegated = await stack.enter_async_context(_client(delegated_principal, auth_enabled=True))
        admin = await stack.enter_async_context(_client(Principal(id="snapshot-admin"), auth_enabled=True))

        created = await service.post(path, json={"name": "protected", "data": {"value": 1, "_auth_context": original}})
        assert created.status_code == 201, created.text
        assert created.json()["data"]["_auth_context"] == original
        entity_id = created.json()["id"]

        for reader in (user, delegated, admin):
            by_name = await reader.get(f"{path}/protected")
            by_id = await reader.get(f"/apis/entities/v2/entities/{entity_id}")
            listed = await reader.get(path)
            cross_workspace = await reader.get("/apis/entities/v2/workspaces/-/entities/platform_job")
            # This fixture omits route middleware. The resolved-ID guard still
            # enforces the internal-service/admin read contract itself.
            if reader is user:
                assert by_id.status_code == 403
                visible = (by_name, listed, cross_workspace)
            else:
                visible = (by_name, by_id, listed, cross_workspace)
            for response in visible:
                assert response.status_code == 200, response.text
                row = response.json()["data"][0] if isinstance(response.json()["data"], list) else response.json()
                assert "_auth_context" not in row["data"]

        ignored = await user.put(path + "/protected", json={"data": {"value": 2, "_auth_context": replacement}})
        assert ignored.status_code == 200, ignored.text
        assert ignored.json()["data"] == {"value": 2}
        stored = await policy_store.get_entity_by_id(entity_id=entity_id)
        assert stored is not None and stored.data["_auth_context"] == original

        for new_data in ({"value": 3}, {"value": 4, "_auth_context": None}):
            preserved = await service.put(path + "/protected", json={"data": new_data})
            assert preserved.status_code == 200, preserved.text
            assert preserved.json()["data"]["_auth_context"] == original

        delegated_write = await delegated.put(
            path + "/protected", json={"data": {"value": 5, "_auth_context": replacement}}
        )
        assert delegated_write.status_code == 200, delegated_write.text
        assert "_auth_context" not in delegated_write.json()["data"]
        direct_read = await service.get(path + "/protected")
        assert direct_read.json()["data"]["_auth_context"] == original

        submitted = await delegated.post(
            path, json={"name": "delegated-submit", "data": {"_auth_context": original, "value": 6}}
        )
        assert submitted.status_code == 201, submitted.text
        assert "_auth_context" not in submitted.json()["data"]
        controller_read = await service.get(path + "/delegated-submit")
        assert "_auth_context" not in controller_read.json()["data"]

        forged = await user.post(path, json={"name": "forged", "data": {"_auth_context": replacement, "value": 1}})
        assert forged.status_code == 201, forged.text
        assert forged.json()["data"] == {"value": 1}
        forged_stored = await policy_store.get_entity_by_id(entity_id=forged.json()["id"])
        assert forged_stored is not None and forged_stored.data == {"value": 1}


async def test_snapshot_query_paths_are_rejected(policy_store: SQLAlchemyEntityRepository) -> None:
    path = "/apis/entities/v2/workspaces/policy-one/entities/model"
    user_principal = Principal(id="user@example.com", email="user@example.com")
    service_principal = Principal(id="service:jobs")
    forbidden_queries = [
        {"filter": json.dumps({"$or": [{"name": "x"}, {"data._auth_context.groups": "team-a"}]})},
        {"filter": 'data._auth_context.id:"user@example.com"'},
        {"filter[data._auth_context.id][$eq]": "user@example.com"},
        {"filter": json.dumps({"adapters": {"_auth_context.id": "user@example.com"}})},
        {"sort": "-data._auth_context.id"},
        {"count_by": "data._auth_context"},
        {"filter": json.dumps({"data": {"$eq": {"_auth_context": {"id": "user@example.com"}}}})},
    ]
    async with AsyncExitStack() as stack:
        user = await stack.enter_async_context(_client(user_principal, auth_enabled=True))
        service = await stack.enter_async_context(_client(service_principal, auth_enabled=True))
        for query in forbidden_queries:
            response = await user.get(path, params=query)
            assert response.status_code == 400, (query, response.text)
            assert "not queryable" in response.text, (query, response.text)
        cross_workspace = await user.get(
            "/apis/entities/v2/workspaces/-/entities/model", params={"sort": "data._auth_context"}
        )
        assert cross_workspace.status_code == 400, cross_workspace.text
        assert "not queryable" in cross_workspace.text
        allowed = await service.get(path, params={"sort": "data._auth_context.id"})
        assert allowed.status_code == 200, allowed.text


async def test_auth_off_keeps_existing_behavior(policy_store: SQLAlchemyEntityRepository) -> None:
    path = "/apis/entities/v2/workspaces/policy-one/entities/platform_job"
    async with _client(Principal(), auth_enabled=False) as client:
        created = await client.post(path, json={"name": "auth-off", "data": {"_auth_context": {"id": "untrusted"}}})
        assert created.status_code == 201, created.text
        assert created.json()["data"]["_auth_context"] == {"id": "untrusted"}
        updated = await client.put(path + "/auth-off", json={"data": {"_auth_context": None}})
        assert updated.status_code == 200, updated.text
        assert updated.json()["data"]["_auth_context"] is None


async def test_platform_auth_middleware_applies_entity_field_policy(
    policy_store: SQLAlchemyEntityRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    auth_config = AuthConfig(enabled=True)
    platform_config = SimpleNamespace(seed_on_startup=False, redirect_root_to_studio=False, services="")
    monkeypatch.setattr(server, "get_platform_config", lambda: platform_config)
    monkeypatch.setattr(server, "get_auth_config", lambda: auth_config)
    monkeypatch.setattr("nhx.common.auth.middleware.get_auth_config", lambda: auth_config)

    async def allow_request(self, auth_client, request, scopes, *, anonymous_denial_is_401):
        return None

    monkeypatch.setattr(AuthorizationMiddleware, "_call_pdp", allow_request)
    platform_app = server.create_app(services=[EntitiesService()])
    platform_app.dependency_overrides[get_async_session_maker] = lambda: policy_store.session_maker
    path = "/apis/entities/v2/workspaces/policy-one/entities/platform_job"
    service_headers = {"X-NHX-Principal-Id": "service:jobs"}
    user_headers = {"X-NHX-Principal-Id": "user@example.com"}

    async with AsyncClient(transport=ASGITransport(app=platform_app), base_url="http://test") as client:
        created = await client.post(
            path,
            headers=service_headers,
            json={"name": "platform-policy", "data": {"_auth_context": {"groups": ["team-a"]}}},
        )
        assert created.status_code == 201, created.text
        user_read = await client.get(path + "/platform-policy", headers=user_headers)
        direct_read = await client.get(path + "/platform-policy", headers=service_headers)

    assert user_read.status_code == direct_read.status_code == 200
    assert "_auth_context" not in user_read.json()["data"]
    assert direct_read.json()["data"]["_auth_context"] == {"groups": ["team-a"]}


async def test_evals_creator_filters_use_submitter_with_legacy_fallback(policy_store) -> None:
    from datetime import timedelta

    from nemo_evals.api.taskset_run_schemas import to_status
    from nemo_evals.taskset_runs.entities import TasksetRunEntity
    from nemo_evals.taskset_runs.store import TasksetRunStore
    from nemo_helix_plugin.client.client import AsyncNemoClient
    from nemo_helix_plugin.entities import EntityClient
    from nemo_helix_plugin.entities.client import AsyncEntitiesClient
    from nemo_helix_plugin.files.client import AsyncFilesClient

    now = datetime.now(timezone.utc)
    async with _client(Principal(id="service:evals"), auth_enabled=True) as http:
        async with AsyncNemoClient(base_url="http://test", http_client=http) as client:
            entities = AsyncEntitiesClient.from_client(client)
            service_entities = EntityClient(entities)
            for name, creator, legacy_writer in [
                ("new-user", "user@example.com", None),
                ("new-other", "other@example.com", None),
                ("legacy-user", "", "user@example.com"),
            ]:
                run = TasksetRunEntity(
                    name=name,
                    workspace="policy-one",
                    creator=creator,
                    plan_ref="policy-one/plan#plan.json",
                    plan_digest="a" * 64,
                    request_digest="digest",
                    taskset_ref="policy-one/suite#1",
                    accepted_at=now,
                    timeout_seconds=3600,
                    execution_deadline_at=now + timedelta(hours=1),
                    finalization_deadline_at=now + timedelta(hours=2),
                )
                if legacy_writer:
                    await policy_store.create_entity(
                        workspace="policy-one",
                        entity_type="taskset_run",
                        name=name,
                        data=run._get_data_fields(),
                        created_by=legacy_writer,
                    )
                else:
                    saved = await service_entities.create(run)
                    assert saved.created_by == "service:evals"
                    assert to_status(saved).created_by == creator
            store = TasksetRunStore(
                entities,
                AsyncFilesClient.from_client(client),
                service_entities=service_entities,
            )
            mine = await store.list_runs("policy-one", caller="user@example.com")
            assert {run.name for run in mine.data} == {"new-user", "legacy-user"}
            assert mine.pagination.total_results == 2
            assert {run.name for run in (await store.list_runs("policy-one", caller="service:evals")).data} == set()
