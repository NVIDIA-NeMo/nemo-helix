# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the authorization flow was completed successfully.

Uses iam.role_bindings to check the full history of role changes,
not just the final state. Each role grant/revocation is tracked as
a separate binding with granted_at and revoked_at timestamps.
"""

import os

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.iam.client import IAMClient
from nemo_platform_plugin.workspaces.client import WorkspacesClient


def _get_client() -> NeMoPlatform:
    nmp_base_url = os.environ.get("NMP_BASE_URL", "http://localhost:8080")
    return NeMoPlatform(base_url=nmp_base_url)


def test_workspace_exists() -> None:
    """Test that the harbor-auth-test workspace was created."""
    client = _get_client()
    response = client_from_platform(client, WorkspacesClient).list_workspaces()
    workspace_names = [ws.name for ws in response.items()]

    assert "harbor-auth-test" in workspace_names, (
        f"Workspace 'harbor-auth-test' was not created! Found workspaces: {workspace_names}"
    )


def test_current_members() -> None:
    """Test that the current member list matches expected final state."""
    client = _get_client()
    response = (
        client_from_platform(client, WorkspacesClient).list_workspace_members(workspace="harbor-auth-test").data()
    )
    members = {m.principal: m.roles for m in response.data}

    # viewer@test.com should now be Editor (was promoted from Viewer)
    assert "viewer@test.com" in members, f"viewer@test.com not found! Members: {list(members.keys())}"
    assert "Editor" in members["viewer@test.com"], (
        f"viewer@test.com should be Editor, got: {members['viewer@test.com']}"
    )

    # admin@test.com should be Admin
    assert "admin@test.com" in members, f"admin@test.com not found! Members: {list(members.keys())}"
    assert "Admin" in members["admin@test.com"], f"admin@test.com should be Admin, got: {members['admin@test.com']}"

    # editor@test.com should have been removed
    assert "editor@test.com" not in members, (
        f"editor@test.com should be removed but has roles: {members.get('editor@test.com')}"
    )


def test_role_binding_history() -> None:
    """Use iam.role_bindings to verify role transitions actually happened.

    The role_bindings API retains revoked bindings with a revoked_at timestamp,
    giving us an audit trail of every grant and revocation.
    """
    client = _get_client()
    response = client_from_platform(client, IAMClient).list_role_bindings(query_params={"page_size": 100})

    # Filter to our workspace
    bindings = [b for b in response.items() if b.workspace == "harbor-auth-test"]

    # Build a lookup: {principal: [(role, is_revoked), ...]}
    history: dict[str, list[tuple[str, bool]]] = {}
    for b in bindings:
        is_revoked = bool(b.revoked_at)
        history.setdefault(b.principal, []).append((b.role, is_revoked))

    # viewer@test.com should have a revoked Viewer binding (original role)
    # AND an active Editor binding (after promotion)
    assert "viewer@test.com" in history, (
        f"No role bindings found for viewer@test.com. Principals: {list(history.keys())}"
    )
    viewer_bindings = history["viewer@test.com"]
    viewer_roles = {role for role, _ in viewer_bindings}
    viewer_revoked = [(role, revoked) for role, revoked in viewer_bindings]

    assert "Viewer" in viewer_roles, (
        f"viewer@test.com should have had a Viewer binding (proves original role). Found bindings: {viewer_revoked}"
    )
    assert "Editor" in viewer_roles, (
        f"viewer@test.com should have an Editor binding (proves promotion). Found bindings: {viewer_revoked}"
    )

    # The Viewer binding should be revoked (proves the role was changed, not just added)
    viewer_viewer_bindings = [(r, rev) for r, rev in viewer_bindings if r == "Viewer"]
    assert any(rev for _, rev in viewer_viewer_bindings), (
        f"viewer@test.com's Viewer binding should be revoked (proves promotion happened). "
        f"Viewer bindings: {viewer_viewer_bindings}"
    )

    # The Editor binding should be active
    viewer_editor_bindings = [(r, rev) for r, rev in viewer_bindings if r == "Editor"]
    assert any(not rev for _, rev in viewer_editor_bindings), (
        f"viewer@test.com should have an active Editor binding. Editor bindings: {viewer_editor_bindings}"
    )

    # editor@test.com should have a revoked Editor binding (proves they existed then were removed)
    assert "editor@test.com" in history, (
        f"No role bindings found for editor@test.com (should have revoked binding as proof of removal). "
        f"Principals: {list(history.keys())}"
    )
    editor_bindings = history["editor@test.com"]
    assert all(revoked for _, revoked in editor_bindings), (
        f"All bindings for editor@test.com should be revoked. Found: {editor_bindings}"
    )

    # admin@test.com should have an active Admin binding
    assert "admin@test.com" in history, f"No role bindings found for admin@test.com. Principals: {list(history.keys())}"
    admin_bindings = history["admin@test.com"]
    assert any(role == "Admin" and not revoked for role, revoked in admin_bindings), (
        f"admin@test.com should have an active Admin binding. Found: {admin_bindings}"
    )

    print("Role binding history verified: all transitions confirmed via audit trail")
