# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity reads recover when another replica grants workspace access."""

import pytest
from httpx import AsyncClient
from nhx.common.config import Configuration
from nhx.core.entities.api.v2.utils import clear_principal_bindings_cache
from nhx.core.entities.config import EntitiesConfig


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "/test-model"], ids=["list", "get"])
async def test_entity_read_refreshes_stale_workspace_access(client_with_auth: AsyncClient, repos, suffix: str) -> None:
    Configuration.set_override(EntitiesConfig(principal_bindings_cache_enabled=True))
    await clear_principal_bindings_cache()
    workspace = "new-workspace"
    await repos["workspace"].create_workspace(name=workspace, description="Cache regression test")
    await repos["entity"].create_entity(workspace=workspace, entity_type="model", name="test-model", data={})
    url = f"/apis/entities/v2/workspaces/{workspace}/entities/model{suffix}"

    try:
        denied = await client_with_auth.get(url)
        assert denied.status_code == 422

        # Writing through the repository models a grant committed by another replica.
        await repos["entity"].create_entity(
            workspace=workspace,
            entity_type="role_binding",
            name="workspace-admin",
            data={"principal": "test-user@example.com", "workspace": workspace, "role": "Admin"},
        )

        response = await client_with_auth.get(url)
        assert response.status_code == 200, response.text
        models = response.json()["data"] if not suffix else [response.json()]
        assert [model["name"] for model in models] == ["test-model"]
    finally:
        await clear_principal_bindings_cache()
