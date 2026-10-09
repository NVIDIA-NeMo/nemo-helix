# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest, AccessKeyWorkspaceGrant


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


def test_access_key_create_request_requires_service_account_for_a_bound_workspace() -> None:
    with pytest.raises(ValueError, match="workspace requires service_account_id"):
        AccessKeyCreateRequest(workspace="team-a")


def test_access_key_create_request_normalizes_the_bound_workspace() -> None:
    request = AccessKeyCreateRequest(
        service_account_id="team-a/otel",
        workspace=" team-a ",
        workspaces=[AccessKeyWorkspaceGrant(workspace="team-a", roles=["Viewer"])],
    )

    assert request.workspace == "team-a"


def test_access_key_create_request_rejects_grants_outside_the_bound_workspace() -> None:
    with pytest.raises(ValueError, match="only grant access to the workspace the key is bound to"):
        AccessKeyCreateRequest(
            service_account_id="team-a/otel",
            workspace="team-a",
            workspaces=[AccessKeyWorkspaceGrant(workspace="team-b")],
        )
