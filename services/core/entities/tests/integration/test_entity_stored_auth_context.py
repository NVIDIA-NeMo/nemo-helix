# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: stored creator auth context (``data._auth_context``) is readable only by service principals."""

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


async def test_user_write_responses_omit_stored_auth_context(client_with_auth: AsyncClient, repos) -> None:
    """Create and update echo stored data; the auth context is still persisted for controllers."""
    await _seed_run(repos)
    data = {"status": "pending", STORED_AUTH_CONTEXT_KEY: _AUTH_CONTEXT}

    created = await client_with_auth.post(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}", json={"name": "run-2", "data": data}
    )
    updated = await client_with_auth.put(
        f"/apis/entities/v2/workspaces/{_WS}/entities/{_ENTITY_TYPE}/run-2", json={"data": data}
    )

    assert created.status_code == 201, created.text
    assert updated.status_code == 200, updated.text
    assert created.json()["data"] == {"status": "pending"}
    assert updated.json()["data"] == {"status": "pending"}
    stored = await repos["entity"].get_entity_by_name(workspace=_WS, entity_type=_ENTITY_TYPE, name="run-2")
    assert stored is not None
    assert stored.data[STORED_AUTH_CONTEXT_KEY] == _AUTH_CONTEXT
