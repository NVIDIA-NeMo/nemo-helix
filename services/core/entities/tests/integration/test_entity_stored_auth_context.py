# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: stored creator auth context (``data._auth_context``) is readable only by service principals
and writable only by services."""

import json
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from nhx.core.entities.api.v2.entities.endpoints import STORED_AUTH_CONTEXT_KEY
from nhx.core.entities.api.v2.utils import ROLE_BINDING_ENTITY_TYPE
from nhx.core.entities.app.repository import EntityRepositoryInterface, WorkspaceRepositoryInterface
from nhx.core.entities.entities import Entity

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TEST_PRINCIPAL = "test-user@example.com"
_WS = "ws-stored-auth-context"
_ENTITY_TYPE = "taskset_run"
_NAME = "run-1"
_AUTH_CONTEXT = {
    "principal_id": "creator@example.com",
    "principal_email": "creator@example.com",
    "principal_groups": ["eval-admins"],
    "principal_authz_aliases": ["creator-alias"],
}


async def _seed_run(repos) -> Entity:
    """Create the workspace, an Editor binding for the test user, and a run entity carrying a stored auth context."""
    workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
    entity_repo: EntityRepositoryInterface = repos["entity"]
    await workspace_repo.create_workspace(name=_WS, description="stored auth context")
    await entity_repo.create_entity(
        workspace=_WS,
        entity_type=ROLE_BINDING_ENTITY_TYPE,
        name="rb-stored-auth-context",
        data={
            "principal": TEST_PRINCIPAL,
            "workspace": _WS,
            "role": "Editor",
            "granted_by": TEST_PRINCIPAL,
            "granted_at": datetime.now(timezone.utc).isoformat(),
            "revoked_at": None,
        },
    )
    return await entity_repo.create_entity(
        workspace=_WS,
        entity_type=_ENTITY_TYPE,
        name=_NAME,
        data={"status": "running", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT},
        created_by="creator@example.com",
    )


async def _read_all(client: AsyncClient, entity_id: str) -> list[dict]:
    """Return the run's ``data`` as seen through every read endpoint."""
    by_name = await client.get(f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}")
    by_id = await client.get(f"/apis/entities/v2/entities/{entity_id}")
    listed = await client.get(f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}")
    listed_all = await client.get(f"/apis/entities/v2/workspaces/-/entities/{_ENTITY_TYPE}")
    for response in (by_name, by_id, listed, listed_all):
        assert response.status_code == 200, response.text
    return [
        by_name.json()["data"],
        by_id.json()["data"],
        *(entity["data"] for entity in listed.json()["data"]),
        *(entity["data"] for entity in listed_all.json()["data"]),
    ]


async def test_user_reads_omit_stored_auth_context(client_with_auth: AsyncClient, repos) -> None:
    run = await _seed_run(repos)

    datas = await _read_all(client_with_auth, run.id)

    assert len(datas) == 4
    assert all(data == {"status": "running"} for data in datas)


async def test_auth_disabled_user_reads_omit_stored_auth_context(client: AsyncClient, repos) -> None:
    run = await _seed_run(repos)

    datas = await _read_all(client, run.id)

    assert all(STORED_AUTH_CONTEXT_KEY not in data for data in datas)


async def test_delegated_service_reads_omit_stored_auth_context(
    client_with_auth_service_on_behalf_of: AsyncClient, repos
) -> None:
    """A service acting on behalf of a user reads as that user."""
    run = await _seed_run(repos)

    datas = await _read_all(client_with_auth_service_on_behalf_of, run.id)

    assert all(STORED_AUTH_CONTEXT_KEY not in data for data in datas)


async def test_service_principal_reads_keep_stored_auth_context(
    client_with_auth_service_principal: AsyncClient, repos
) -> None:
    """Controllers (direct service principals) still hydrate the creator auth context."""
    run = await _seed_run(repos)

    datas = await _read_all(client_with_auth_service_principal, run.id)

    assert len(datas) == 4
    assert all(data[STORED_AUTH_CONTEXT_KEY] == _AUTH_CONTEXT for data in datas)


_FORGED = {"principal_id": "attacker@example.com", "principal_groups": ["platform-admins"]}


async def _stored_data(repos, name: str) -> dict:
    stored = await repos["entity"].get_entity_by_name(workspace=_WS, entity_type=_ENTITY_TYPE, name=name)
    assert stored is not None
    return stored.data


async def test_user_create_drops_forged_auth_context(client_with_auth: AsyncClient, repos) -> None:
    await _seed_run(repos)

    created = await client_with_auth.post(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}",
        json={"name": "run-2", "data": {"status": "pending", STORED_AUTH_CONTEXT_KEY: _FORGED}},
    )

    assert created.status_code == 201, created.text
    assert created.json()["data"] == {"status": "pending"}
    assert await _stored_data(repos, "run-2") == {"status": "pending"}


@pytest.mark.parametrize("sent", [_FORGED, None], ids=["forged", "sdk-redacted-none"])
async def test_user_update_keeps_stored_auth_context(client_with_auth: AsyncClient, repos, sent) -> None:
    """Users can neither replace the creator auth context nor wipe it with a redacted read-modify-write."""
    await _seed_run(repos)

    updated = await client_with_auth.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}",
        json={"data": {"status": "cancelling", STORED_AUTH_CONTEXT_KEY: sent}},
    )

    assert updated.status_code == 200, updated.text
    assert updated.json()["data"] == {"status": "cancelling"}
    assert await _stored_data(repos, _NAME) == {"status": "cancelling", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}


async def test_auth_disabled_user_update_keeps_stored_auth_context(client: AsyncClient, repos) -> None:
    await _seed_run(repos)

    updated = await client.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}",
        json={"data": {"status": "cancelling", STORED_AUTH_CONTEXT_KEY: _FORGED}},
    )

    assert updated.status_code == 200, updated.text
    assert (await _stored_data(repos, _NAME))[STORED_AUTH_CONTEXT_KEY] == _AUTH_CONTEXT


async def test_delegated_service_create_stores_auth_context(
    client_with_auth_service_on_behalf_of: AsyncClient, repos
) -> None:
    """Plugin APIs create entities while delegating for the user; the creator auth context they set is kept."""
    await _seed_run(repos)

    created = await client_with_auth_service_on_behalf_of.post(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}",
        json={"name": "run-2", "data": {"status": "pending", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}},
    )

    assert created.status_code == 201, created.text
    assert STORED_AUTH_CONTEXT_KEY not in created.json()["data"]
    assert (await _stored_data(repos, "run-2"))[STORED_AUTH_CONTEXT_KEY] == _AUTH_CONTEXT


async def test_service_principal_update_replaces_auth_context(
    client_with_auth_service_principal: AsyncClient, repos
) -> None:
    await _seed_run(repos)
    replacement = {**_AUTH_CONTEXT, "principal_groups": ["eval-admins", "new-group"]}

    updated = await client_with_auth_service_principal.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}",
        json={"data": {"status": "running", STORED_AUTH_CONTEXT_KEY: replacement}},
    )

    assert updated.status_code == 200, updated.text
    assert updated.json()["data"][STORED_AUTH_CONTEXT_KEY] == replacement
    assert (await _stored_data(repos, _NAME))[STORED_AUTH_CONTEXT_KEY] == replacement


@pytest.mark.parametrize("sent", [{}, {STORED_AUTH_CONTEXT_KEY: None}], ids=["absent", "sdk-redacted-none"])
async def test_delegated_service_update_without_auth_context_keeps_it(
    client_with_auth_service_on_behalf_of: AsyncClient, repos, sent
) -> None:
    """A service acting for a user reads the context redacted, so writing that back must not wipe it."""
    await _seed_run(repos)

    updated = await client_with_auth_service_on_behalf_of.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}",
        json={"data": {"status": "cancelling", **sent}},
    )

    assert updated.status_code == 200, updated.text
    assert await _stored_data(repos, _NAME) == {"status": "cancelling", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}


async def test_delegated_service_update_replaces_auth_context(
    client_with_auth_service_on_behalf_of: AsyncClient, repos
) -> None:
    """Rerun-style flows re-apply the caller's context while delegating for them."""
    await _seed_run(repos)
    caller = {**_AUTH_CONTEXT, "principal_id": TEST_PRINCIPAL, "principal_email": TEST_PRINCIPAL}

    updated = await client_with_auth_service_on_behalf_of.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/{_NAME}",
        json={"data": {"status": "running", STORED_AUTH_CONTEXT_KEY: caller}},
    )

    assert updated.status_code == 200, updated.text
    assert (await _stored_data(repos, _NAME))[STORED_AUTH_CONTEXT_KEY] == caller


_PROBES = {
    "filter-field": {"filter": json.dumps({"data._auth_context.principal_id": "creator@example.com"})},
    "filter-has-key": {"filter": json.dumps({"data": {"$hasKey": STORED_AUTH_CONTEXT_KEY}})},
    "filter-root-eq": {
        "filter": json.dumps({"data": {"$eq": {"status": "running", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}}})
    },
    "filter-root-in": {"filter": json.dumps({"data": {"$in": [{STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}]}})},
    "filter-nested": {"filter": json.dumps({"$or": [{"data.status": "running"}, {"data._auth_context.x": 1}]})},
    "sort": {"sort": "-data._auth_context.principal_id"},
    "count_by": {"count_by": "data._auth_context"},
}


@pytest.mark.parametrize("params", _PROBES.values(), ids=_PROBES.keys())
async def test_user_list_cannot_query_stored_auth_context(client_with_auth: AsyncClient, repos, params) -> None:
    """Match counts, group counts and ordering would otherwise reveal the creator auth context."""
    await _seed_run(repos)

    listed = await client_with_auth.get(f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}", params=params)

    assert listed.status_code == 400, listed.text
    assert STORED_AUTH_CONTEXT_KEY in listed.json()["detail"]


async def test_service_principal_list_can_filter_on_stored_auth_context(
    client_with_auth_service_principal: AsyncClient, repos
) -> None:
    await _seed_run(repos)

    listed = await client_with_auth_service_principal.get(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}", params=_PROBES["filter-field"]
    )

    assert listed.status_code == 200, listed.text
    assert [entity["name"] for entity in listed.json()["data"]] == [_NAME]


async def test_user_list_can_filter_other_fields_for_the_key_text(client_with_auth: AsyncClient, repos) -> None:
    """Only field paths are guarded; a value that happens to spell the key is an ordinary filter."""
    await _seed_run(repos)

    listed = await client_with_auth.get(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}",
        params={"filter": json.dumps({"data.status": STORED_AUTH_CONTEXT_KEY})},
    )

    assert listed.status_code == 200, listed.text
    assert listed.json()["data"] == []
