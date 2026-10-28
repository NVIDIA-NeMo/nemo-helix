# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Named-workspace authorization failures use consistent, actionable errors."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "suffix", "action", "payload"),
    [
        ("POST", "", "create", {"name": "new-model", "data": {}}),
        ("GET", "", "list", None),
        ("GET", "/test-model", "read", None),
        ("PUT", "/test-model", "update", {"data": {"changed": True}}),
        ("DELETE", "/test-model", "delete", None),
    ],
    ids=["create", "list", "read", "update", "delete"],
)
async def test_workspace_denial_is_forbidden_and_action_specific(
    client_with_auth: AsyncClient, repos, method: str, suffix: str, action: str, payload: dict | None
) -> None:
    for workspace in ("allowed-workspace", "denied-workspace"):
        await repos["workspace"].create_workspace(name=workspace, description="Authorization regression test")
    await repos["entity"].create_entity(
        workspace="allowed-workspace",
        entity_type="role_binding",
        name="workspace-admin",
        data={"principal": "test-user@example.com", "workspace": "allowed-workspace", "role": "Admin"},
    )
    await repos["entity"].create_entity(workspace="denied-workspace", entity_type="model", name="test-model", data={})

    response = await client_with_auth.request(
        method, f"/apis/entities/v2/workspaces/denied-workspace/entities/model{suffix}", json=payload
    )

    assert response.status_code == 403, response.text
    assert response.json() == {"detail": f"Not allowed to {action} entities in workspace 'denied-workspace'"}
    # Denied writes must neither mutate the existing entity nor create a new one.
    existing = await repos["entity"].get_entity_by_name(
        workspace="denied-workspace", entity_type="model", name="test-model"
    )
    assert existing is not None
    assert existing.data == {}
    assert (
        await repos["entity"].get_entity_by_name(workspace="denied-workspace", entity_type="model", name="new-model")
        is None
    )
