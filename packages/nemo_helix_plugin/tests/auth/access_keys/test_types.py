# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyWorkspaceGrant,
)


def test_access_key_create_request_omits_unset_expiry_from_json() -> None:
    request = AccessKeyCreateRequest(name="default-expiry")

    assert request.expires_in_seconds is None
    assert "expires_in_seconds" not in request.model_fields_set
    assert request.model_dump_json(exclude_unset=True) == '{"name":"default-expiry"}'


def test_access_key_create_request_preserves_explicit_null_expiry_in_json() -> None:
    request = AccessKeyCreateRequest(name="unlimited", expires_in_seconds=None)

    assert request.expires_in_seconds is None
    assert "expires_in_seconds" in request.model_fields_set
    assert request.model_dump_json(exclude_unset=True) == '{"name":"unlimited","expires_in_seconds":null}'


def test_access_key_create_request_rejects_normalized_duplicate_workspace_grants() -> None:
    with pytest.raises(ValueError, match="duplicate workspace names: team-a"):
        AccessKeyCreateRequest(
            workspaces=[
                AccessKeyWorkspaceGrant(workspace=" team-a ", roles=["Viewer"]),
                AccessKeyWorkspaceGrant(workspace="team-a", roles=["Editor"]),
            ]
        )


def test_access_key_create_request_normalizes_the_workspace_grants() -> None:
    request = AccessKeyCreateRequest(
        service_account_id="otel",
        workspaces=[
            AccessKeyWorkspaceGrant(workspace=" team-a ", roles=[" Viewer "]),
            AccessKeyWorkspaceGrant(workspace="team-b", roles=["Editor"]),
        ],
    )

    assert [(grant.workspace, grant.roles) for grant in request.workspaces or []] == [
        ("team-a", ["Viewer"]),
        ("team-b", ["Editor"]),
    ]


def test_access_key_workspace_grant_defaults_to_editor() -> None:
    assert AccessKeyWorkspaceGrant.model_validate({"workspace": "team-a"}).roles == ["Editor"]


@pytest.mark.parametrize("roles", [[], [" "]])
def test_access_key_workspace_grant_rejects_blank_roles(roles) -> None:
    with pytest.raises(ValueError, match="roles, if provided, must contain at least one non-empty role"):
        AccessKeyWorkspaceGrant.model_validate({"workspace": "team-a", "roles": roles})
