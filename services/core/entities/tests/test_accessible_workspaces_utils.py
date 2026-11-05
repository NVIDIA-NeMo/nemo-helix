# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python enforces the typed PDP scope without calculating role grants."""

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.auth.decisions import CollectionTarget, WorkspaceDecision
from nhx.common.auth import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.v2.utils import (
    add_workspace_filtering,
    get_accessible_workspaces,
    require_workspace_access,
    workspace_decision_context,
    workspaces_from_decision,
)


def decision(access: dict) -> WorkspaceDecision:
    return WorkspaceDecision.model_validate(
        {
            "protocol_version": 2,
            "operation": "entities.workspaces.visible",
            "target": CollectionTarget(workspace="-", resource_type="model").model_dump(),
            "principal_context_digest": "bound",
            "policy_revision": "revision",
            "reason": "policy_decision",
            "decision_type": "workspaces",
            "access": access,
        }
    )


@pytest.mark.parametrize("workspaces", [[], ["team"], ["team", "other"]])
def test_restricted_scope_preserves_empty_and_finite_sets(workspaces: list[str]) -> None:
    assert workspaces_from_decision(decision({"kind": "restricted", "workspaces": workspaces})) == set(workspaces)


def test_only_explicit_unrestricted_scope_skips_filtering() -> None:
    assert workspaces_from_decision(decision({"kind": "unrestricted"})) is None
    with pytest.raises(HTTPException):
        workspaces_from_decision(decision({"kind": "denied"}))
    assert add_workspace_filtering(set(), None) is not None
    assert add_workspace_filtering(None, None) is None


async def test_request_scope_is_reused_and_named_workspace_enforced() -> None:
    auth_client = AuthClient(principal=Principal(id="user"), config=AuthConfig(enabled=True))
    token = auth_client_context.set(auth_client)
    scope_token = workspace_decision_context.set(decision({"kind": "restricted", "workspaces": ["team"]}))
    try:
        assert await get_accessible_workspaces() == {"team"}
        await require_workspace_access("team")
        with pytest.raises(HTTPException) as exc:
            await require_workspace_access("other")
        assert exc.value.status_code == 403
    finally:
        workspace_decision_context.reset(scope_token)
        auth_client_context.reset(token)
