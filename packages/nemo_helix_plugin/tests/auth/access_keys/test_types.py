# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.auth.access_keys.types import (
    AccessKeyBoundWorkspace,
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


def test_access_key_create_request_requires_service_account_for_bound_workspaces() -> None:
    with pytest.raises(ValueError, match="bound_workspaces requires service_account_id"):
        AccessKeyCreateRequest(bound_workspaces=[AccessKeyBoundWorkspace(workspace="team-a", roles=["Editor"])])


def test_access_key_create_request_normalizes_the_bound_workspaces() -> None:
    request = AccessKeyCreateRequest(
        service_account_id="otel",
        bound_workspaces=[
            AccessKeyBoundWorkspace(workspace=" team-a ", roles=[" Viewer "]),
            AccessKeyBoundWorkspace(workspace="team-b", roles=["Editor"]),
        ],
    )

    assert [(bound.workspace, bound.roles) for bound in request.bound_workspaces or []] == [
        ("team-a", ["Viewer"]),
        ("team-b", ["Editor"]),
    ]


def test_access_key_create_request_rejects_duplicate_bound_workspaces() -> None:
    with pytest.raises(ValueError, match="bound_workspaces must not contain duplicate"):
        AccessKeyCreateRequest(
            service_account_id="otel",
            bound_workspaces=[
                AccessKeyBoundWorkspace(workspace="team-a", roles=["Viewer"]),
                AccessKeyBoundWorkspace(workspace=" team-a", roles=["Editor"]),
            ],
        )


def test_access_key_create_request_rejects_an_empty_bound_workspace_list() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        AccessKeyCreateRequest(service_account_id="otel", bound_workspaces=[])


@pytest.mark.parametrize("roles", [None, [], [" "]])
def test_access_key_bound_workspace_requires_explicit_roles(roles) -> None:
    payload = {"workspace": "team-a"} if roles is None else {"workspace": "team-a", "roles": roles}
    with pytest.raises(ValueError):
        AccessKeyBoundWorkspace.model_validate(payload)


def test_access_key_create_request_rejects_workspaces_alongside_bound_workspaces() -> None:
    with pytest.raises(ValueError, match="workspaces cannot be combined with bound_workspaces"):
        AccessKeyCreateRequest(
            service_account_id="otel",
            bound_workspaces=[AccessKeyBoundWorkspace(workspace="team-a", roles=["Viewer"])],
            workspaces=[AccessKeyWorkspaceGrant(workspace="team-b")],
        )
