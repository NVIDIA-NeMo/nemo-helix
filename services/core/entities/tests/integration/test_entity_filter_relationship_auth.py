# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: relationship child workspace scoping with auth enabled vs disabled."""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from nhx.core.entities.api.v2.utils import ROLE_BINDING_ENTITY_TYPE
from nhx.core.entities.app.repository import (
    EntityRepositoryInterface,
    WorkspaceRepositoryInterface,
)

# Must match the principal in integration ``conftest``'s auth client
TEST_USER_EMAIL = "test-user@example.com"

_DEFAULT_WS = "def-ws-relauth"
_OTHER_WS = "oth-ws-relauth"

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_WS_SEED: tuple[tuple[str, str], ...] = (
    (_DEFAULT_WS, "Model lives here; rel-auth integration test"),
    (_OTHER_WS, "Second workspace; rel-auth integration test"),
)


async def _seed_workspaces(
    workspace_repo: WorkspaceRepositoryInterface,
    entity_repo: EntityRepositoryInterface,
    member_of: set[str] | None = None,
) -> None:
    """Create the two test workspaces. For each name in *member_of*, add a role_binding (Editor)."""
    for name, desc in _WS_SEED:
        if await workspace_repo.get_workspace_by_name(name=name) is None:
            await workspace_repo.create_workspace(name=name, description=desc)
    if not member_of:
        return
    granted = datetime.now(timezone.utc).isoformat()
    for ws in member_of:
        await entity_repo.create_entity(
            workspace=ws,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            name=f"rb-int-{ws}",
            data={
                "principal": TEST_USER_EMAIL,
                "workspace": ws,
                "role": "Editor",
                "granted_by": TEST_USER_EMAIL,
                "granted_at": granted,
                "revoked_at": None,
            },
        )


class TestRelationshipFilterWithAuth:
    """When auth is enabled, EXISTS for adapters only counts children in accessible workspaces."""

    async def test_with_auth_on_adapter_in_other_ws_does_not_satisfy_exists(
        self, client_with_auth: AsyncClient, repos
    ) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        mresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-auth", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        # API create with child in the other workspace now requires access to that workspace; seed via repo
        # so we still exercise the relationship filter (EXISTS) with a graph the user may not be able to create.
        await er.create_entity(
            workspace=_OTHER_WS,
            entity_type="adapter",
            name="a-remote",
            data={},
            parent=model_id,
            created_by=TEST_USER_EMAIL,
        )

        fresp = await client_with_auth.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-auth" not in [e["name"] for e in fresp.json()["data"]]

    async def test_with_auth_on_user_can_read_other_ws_cross_ws_adapter_included(
        self, client_with_auth: AsyncClient, repos
    ) -> None:
        """Once the user has a role in *both* workspaces, the remote adapter matches EXISTS.

        Same graph as the excluded case: model in default, adapter in other, but
        `get_accessible_workspaces` includes the other workspace so
        `relationship_child_workspaces` no longer filters out the child.
        """
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS, _OTHER_WS})

        mresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-both", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        aresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-remote-now-readable", "parent": model_id, "data": {}},
        )
        assert aresp.status_code == 201, aresp.text

        fresp = await client_with_auth.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-both" in [e["name"] for e in fresp.json()["data"]]

    async def test_with_auth_on_adapter_in_accessible_ws_matches(self, client_with_auth: AsyncClient, repos) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        mresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-ok", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        for name in ("a-in-def-1", "a-in-def-2"):
            r = await client_with_auth.post(
                f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/adapter",
                json={"name": name, "parent": model_id, "data": {"finetuning_type": "LoRA"}},
            )
            assert r.status_code == 201, r.text

        fresp = await client_with_auth.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"finetuning_type":"LoRA"}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert [e["name"] for e in fresp.json()["data"]] == ["m-ok"]

    async def test_with_auth_off_cross_ws_adapter_still_matches(self, client: AsyncClient, repos) -> None:
        """Unauthenticated mode does not add relationship_child_workspaces; behavior unchanged."""
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er)
        mresp = await client.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-legacy", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        aresp = await client.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-remote-legacy", "parent": model_id, "data": {}},
        )
        assert aresp.status_code == 201, aresp.text

        fresp = await client.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-legacy" in [e["name"] for e in fresp.json()["data"]]


class TestRelationshipFilterWithServiceOnBehalfOf:
    """Inbound calls with service:principal + on-behalf-of (e.g. Models -> Entities) use the user's scope.

    Unlike a bare service principal, delegated scope must not bypass ``require_accessible_workspaces`` or
    relationship EXISTS filtering; behavior should match a direct end-user call with the same email.
    """

    async def test_on_behalf_of_excludes_remote_adapter_in_exists(
        self, client_with_auth_service_on_behalf_of: AsyncClient, repos
    ) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        mresp = await client_with_auth_service_on_behalf_of.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-obo-exists", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        await er.create_entity(
            workspace=_OTHER_WS,
            entity_type="adapter",
            name="a-remote-obo",
            data={},
            parent=model_id,
            created_by=TEST_USER_EMAIL,
        )

        fresp = await client_with_auth_service_on_behalf_of.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-obo-exists" not in [e["name"] for e in fresp.json()["data"]]

    async def test_on_behalf_of_create_in_other_workspace_rejected(
        self, client_with_auth_service_on_behalf_of: AsyncClient, repos
    ) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        mresp = await client_with_auth_service_on_behalf_of.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-obo-403", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        aresp = await client_with_auth_service_on_behalf_of.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-denied", "parent": model_id, "data": {}},
        )
        assert aresp.status_code == 403, aresp.text

    async def test_on_behalf_of_matches_both_workspaces_like_direct_user(
        self, client_with_auth_service_on_behalf_of: AsyncClient, repos
    ) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS, _OTHER_WS})
        mresp = await client_with_auth_service_on_behalf_of.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-obo-both", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        aresp = await client_with_auth_service_on_behalf_of.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-remote-obo-ok", "parent": model_id, "data": {}},
        )
        assert aresp.status_code == 201, aresp.text

        fresp = await client_with_auth_service_on_behalf_of.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-obo-both" in [e["name"] for e in fresp.json()["data"]]


class TestRelationshipFilterWithServicePrincipal:
    """Service principals have all-workspace data scope; relationship EXISTS sees adapters in any workspace."""

    async def test_service_principal_includes_adapter_in_other_workspace_in_exists(
        self, client_with_auth_service_principal: AsyncClient, repos
    ) -> None:
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        # No role bindings: user-scoped access would 403/omit cross-ws (see
        # TestRelationshipFilterWithAuth.test_with_auth_on_adapter_in_other_ws_does_not_satisfy_exists).
        await _seed_workspaces(wr, er, member_of=set())

        mresp = await client_with_auth_service_principal.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-sp", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        # Child only in the other workspace without membership there — only OK for all-workspace scope
        await er.create_entity(
            workspace=_OTHER_WS,
            entity_type="adapter",
            name="a-remote-sp",
            data={},
            parent=model_id,
            created_by=TEST_USER_EMAIL,
        )

        fresp = await client_with_auth_service_principal.get(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            params={"filter": '{"adapters":{"$exists":true}}'},
        )
        assert fresp.status_code == 200, fresp.text
        assert "m-sp" in [e["name"] for e in fresp.json()["data"]]
