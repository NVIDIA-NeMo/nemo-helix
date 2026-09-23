# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: create entity must allow access to path workspace and parent workspace (when set)."""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from nhx.core.entities.api.v2.utils import ROLE_BINDING_ENTITY_TYPE
from nhx.core.entities.app.repository import (
    EntityRepositoryInterface,
    WorkspaceRepositoryInterface,
)

TEST_USER_EMAIL = "test-user@example.com"

_DEFAULT_WS = "def-ws-createparent"
_OTHER_WS = "oth-ws-createparent"

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_WS_SEED: tuple[tuple[str, str], ...] = (
    (_DEFAULT_WS, "default ws create-parent auth"),
    (_OTHER_WS, "other ws create-parent auth"),
)


async def _seed_workspaces(
    workspace_repo: WorkspaceRepositoryInterface,
    entity_repo: EntityRepositoryInterface,
    member_of: set[str] | None = None,
) -> None:
    """Create the two test workspaces, then optionally one Editor role_binding per name in *member_of*."""
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
            name=f"rb-createpar-{ws}",
            data={
                "principal": TEST_USER_EMAIL,
                "workspace": ws,
                "role": "Editor",
                "granted_by": TEST_USER_EMAIL,
                "granted_at": granted,
                "revoked_at": None,
            },
        )


class TestCreateEntityParentWorkspaceAuth:
    async def test_403_cannot_create_in_path_workspace_without_binding(
        self, client_with_auth: AsyncClient, repos
    ) -> None:
        """Workspaces exist but the user has no role binding; POST in that workspace returns 403."""
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        await _seed_workspaces(wr, repos["entity"], member_of=set())  # workspaces only, no role bindings
        r = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-no-rb", "data": {}},
        )
        assert r.status_code == 403, r.text

    async def test_403_cannot_set_parent_in_foreign_workspace(self, client_with_auth: AsyncClient, repos) -> None:
        """User has the path workspace (default) but not the parent's workspace -> 403."""
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        # Parent lives in _OTHER_WS; caller cannot read/create there via API
        other_parent = await er.create_entity(
            workspace=_OTHER_WS,
            entity_type="model",
            name="m-in-other",
            data={},
            parent=None,
            created_by=TEST_USER_EMAIL,
        )
        r = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/adapter",
            json={"name": "a1", "parent": other_parent.id, "data": {"finetuning_type": "LoRA"}},
        )
        assert r.status_code == 422, r.text

    async def test_403_cannot_create_in_other_ws_when_only_default_bound(
        self, client_with_auth: AsyncClient, repos
    ) -> None:
        """User is bound to the default workspace only; creating a child in the other workspace returns 403."""
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        er: EntityRepositoryInterface = repos["entity"]
        await _seed_workspaces(wr, er, member_of={_DEFAULT_WS})
        mresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-need-both", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        r = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-need-oth", "parent": model_id, "data": {}},
        )
        assert r.status_code == 403, r.text

    async def test_201_create_cross_ws_when_user_has_both(self, client_with_auth: AsyncClient, repos) -> None:
        """With Editor in both workspaces, a cross-workspace child linked to a model in the default path returns 201."""
        wr: WorkspaceRepositoryInterface = repos["workspace"]
        await _seed_workspaces(wr, repos["entity"], member_of={_DEFAULT_WS, _OTHER_WS})
        mresp = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_DEFAULT_WS}/entities/model",
            json={"name": "m-both", "data": {}},
        )
        assert mresp.status_code == 201, mresp.text
        model_id = mresp.json()["id"]
        r = await client_with_auth.post(
            f"/apis/entities/v2/workspaces/{_OTHER_WS}/entities/adapter",
            json={"name": "a-both", "parent": model_id, "data": {"finetuning_type": "LoRA"}},
        )
        assert r.status_code == 201, r.text
        assert r.json()["parent"] == model_id
        assert r.json()["workspace"] == _OTHER_WS
