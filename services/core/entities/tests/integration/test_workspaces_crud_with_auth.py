# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace CRUD operations with authorization enabled.

These tests verify:
- Authorization behavior (401 without auth, proper access with auth)
- Role-based access control (Admin, Viewer permissions)
- Workspace visibility (users only see workspaces they have access to)
- Role binding creation on workspace creation

Uses the create_test_client pattern for fast in-memory testing.
"""

import uuid
from contextlib import contextmanager
from time import monotonic, sleep
from typing import Generator

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import ConflictError, PermissionDeniedError
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.entities.types import EntityCreateInput
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
)
from nhx.core.entities.service import EntitiesService
from nhx.testing import TEST_USER_EMAIL, create_test_client, short_unique_name


@contextmanager
def as_user(sdk: NeMoHelix, email: str) -> Generator[None, None, None]:
    """Context manager to temporarily set the SDK's auth headers for a specific user."""
    old_headers = dict(sdk._client.headers)
    sdk._client.headers["X-NHX-Principal-Id"] = email
    try:
        yield
    finally:
        sdk._client.headers.clear()
        sdk._client.headers.update(old_headers)


@contextmanager
def as_user_with_id_and_email(sdk: NeMoHelix, principal_id: str, email: str) -> Generator[None, None, None]:
    """Like production OIDC: subject/object id in Principal-Id, email in Principal-Email."""
    old_headers = dict(sdk._client.headers)
    sdk._client.headers["X-NHX-Principal-Id"] = principal_id
    sdk._client.headers["X-NHX-Principal-Email"] = email
    try:
        yield
    finally:
        sdk._client.headers.clear()
        sdk._client.headers.update(old_headers)


@contextmanager
def as_user_id_only(sdk: NeMoHelix, principal_id: str) -> Generator[None, None, None]:
    """Principal-Id set without X-NHX-Principal-Email (e.g. token without email claim)."""
    old_headers = dict(sdk._client.headers)
    sdk._client.headers["X-NHX-Principal-Id"] = principal_id
    sdk._client.headers.pop("X-NHX-Principal-Email", None)
    try:
        yield
    finally:
        sdk._client.headers.clear()
        sdk._client.headers.update(old_headers)


@contextmanager
def as_user_with_id_and_groups(sdk: NeMoHelix, principal_id: str, groups: list[str]) -> Generator[None, None, None]:
    """Subject id with group memberships (comma-separated X-NHX-Principal-Groups)."""
    old_headers = dict(sdk._client.headers)
    sdk._client.headers["X-NHX-Principal-Id"] = principal_id
    sdk._client.headers["X-NHX-Principal-Groups"] = ",".join(groups)
    sdk._client.headers.pop("X-NHX-Principal-Email", None)
    try:
        yield
    finally:
        sdk._client.headers.clear()
        sdk._client.headers.update(old_headers)


@contextmanager
def as_service(sdk: NeMoHelix, service_name: str) -> Generator[None, None, None]:
    """Context manager to temporarily set the SDK's auth headers for a service principal."""
    old_headers = dict(sdk._client.headers)
    sdk._client.headers["X-NHX-Principal-Id"] = f"service:{service_name}"
    try:
        yield
    finally:
        sdk._client.headers.clear()
        sdk._client.headers.update(old_headers)


def restrict_workspace_creation_to_named_users(sdk: NeMoHelix) -> None:
    """Revoke the seeded wildcard WorkspaceCreator binding while preserving Viewer."""
    with as_service(sdk, "auth"):
        client_from_platform(sdk, WorkspacesClient).update_workspace_member(
            principal_id="*",
            workspace="system",
            body=UpdateWorkspaceMemberRequest(roles=["Viewer"]),
        ).data()


def wait_for_workspace_create_authz(
    sdk: NeMoHelix, principal_id: str, expected: bool, timeout_s: float = 5.0
) -> None:
    """Poll the authz allow endpoint until workspace creation reaches the expected decision."""
    deadline = monotonic() + timeout_s
    payload = {
        "input": {
            "principal_id": principal_id,
            "method": "POST",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["entities:write", "platform:write"],
        }
    }

    with as_service(sdk, "auth"):
        while monotonic() < deadline:
            response = sdk._client.post("/apis/auth/v2/authz/allow", json=payload)
            assert response.status_code == 200
            allowed = response.json()["result"]["allowed"]
            if allowed is expected:
                return
            sleep(0.1)

    raise AssertionError(f"workspace-create authz for {principal_id} did not become {expected} within {timeout_s}s")


@pytest.fixture(scope="module")
def sdk() -> Generator[NeMoHelix, None, None]:
    """SDK client with EntitiesService (auth enabled)."""
    with create_test_client(
        EntitiesService,
        auth_enabled=True,
        workspaces=[],  # Don't auto-create workspaces - we're testing workspace CRUD
        projects=[],  # Skip project creation
    ) as sdk:
        yield sdk


@pytest.mark.integration
class TestWorkspaceCRUDWithAuth:
    """Test workspace CRUD operations with authorization enabled."""

    def test_default_workspaces_created_on_startup(self, sdk: NeMoHelix):
        """Test that 'default' and 'system' workspaces are created automatically on startup."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        # These workspaces are created by EntitiesService.startup() and are public
        with as_user(sdk, TEST_USER_EMAIL):
            default_ws = workspaces.get_workspace(name="default").data()
            assert default_ws.name == "default"
            assert default_ws.description == "General-purpose workspace (all users have write access)"

            system_ws = workspaces.get_workspace(name="system").data()
            assert system_ws.name == "system"
            assert system_ws.description == "Platform-provided resources (read-only for users)"

            # Verify regular user can see public workspaces when listing
            result = workspaces.list_workspaces()
            workspace_names = [ws.name for ws in result.items()]
            assert "default" in workspace_names, "Regular user should see 'default' workspace in list"
            assert "system" in workspace_names, "Regular user should see 'system' workspace in list"

    def test_create_workspace_without_auth_fails(self, sdk: NeMoHelix):
        """Test that creating a workspace without auth headers returns 401."""
        workspace_name = short_unique_name("noauth")

        # Use raw client to test without auth headers
        response = sdk._client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Should fail"},
        )

        assert response.status_code == 401

    def test_create_workspace_with_auth(self, sdk: NeMoHelix):
        """Test creating a workspace with proper auth headers."""
        workspace_name = short_unique_name("auth-ws")

        with as_user(sdk, TEST_USER_EMAIL):
            workspace = (
                client_from_platform(sdk, WorkspacesClient)
                .create_workspace(body=CreateWorkspaceRequest(name=workspace_name, description="Auth test workspace"))
                .data()
            )

        assert workspace.name == workspace_name
        assert workspace.description == "Auth test workspace"
        assert workspace.created_by == TEST_USER_EMAIL
        assert workspace.updated_by == TEST_USER_EMAIL

    def test_workspace_create_remains_open_by_default(self, sdk: NeMoHelix):
        """Default seeding keeps workspace creation open to authenticated users."""
        creator_email = f"creator-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("default-open")

        with as_user(sdk, creator_email):
            created = (
                client_from_platform(sdk, WorkspacesClient)
                .create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
                .data()
            )

        assert created.name == workspace_name
        assert created.created_by == creator_email

    def test_workspace_create_can_be_restricted_by_rebinding_system_role(self, sdk: NeMoHelix):
        """Operators can rebind system workspace creation access to specific principals."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        creator_email = f"creator-{uuid.uuid4().hex[:8]}@example.com"
        denied_email = f"denied-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("restricted")
        seeded_wildcard_roles = ["Viewer", "WorkspaceCreator"]

        try:
            restrict_workspace_creation_to_named_users(sdk)
            wait_for_workspace_create_authz(sdk, denied_email, expected=False)

            with as_user(sdk, denied_email):
                with pytest.raises(PermissionDeniedError):
                    workspaces.create_workspace(
                        body=CreateWorkspaceRequest(name=short_unique_name("restricted-denied"))
                    ).data()

            with as_service(sdk, "auth"):
                workspaces.create_workspace_member(
                    workspace="system",
                    body=CreateWorkspaceMemberRequest(principal=creator_email, roles=["Viewer", "WorkspaceCreator"]),
                ).data()

            wait_for_workspace_create_authz(sdk, creator_email, expected=True)

            with as_service(sdk, "auth"):
                members = workspaces.list_workspace_members(workspace="system").data()

            wildcard_member = next((m for m in members.data if m.principal == "*"), None)
            creator_member = next((m for m in members.data if m.principal == creator_email), None)

            assert wildcard_member is not None
            assert set(wildcard_member.roles) == {"Viewer"}
            assert creator_member is not None
            assert set(creator_member.roles) == {"Viewer", "WorkspaceCreator"}

            wait_for_workspace_create_authz(sdk, denied_email, expected=False)

            with as_user(sdk, denied_email):
                with pytest.raises(PermissionDeniedError):
                    workspaces.create_workspace(
                        body=CreateWorkspaceRequest(name=short_unique_name("restricted-still-denied"))
                    ).data()

            with as_user(sdk, creator_email):
                created = workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
                assert created.name == workspace_name
        finally:
            with as_service(sdk, "auth"):
                workspaces.update_workspace_member(
                    principal_id="*",
                    workspace="system",
                    body=UpdateWorkspaceMemberRequest(roles=seeded_wildcard_roles),
                ).data()

    def test_creator_gets_admin_role(self, sdk: NeMoHelix):
        """Test that workspace creator automatically gets Admin role."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("admin-ws")
        creator_email = f"creator-{uuid.uuid4().hex[:8]}@example.com"

        with as_user(sdk, creator_email):
            # Create workspace
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # Verify creator can access the workspace (has Admin role)
            workspace = workspaces.get_workspace(name=workspace_name).data()
            assert workspace.name == workspace_name
            assert workspace.created_by == creator_email
            assert workspace.updated_by == creator_email

            # Verify creator is listed as a member with Admin role
            members = workspaces.list_workspace_members(workspace=workspace_name).data()
            creator_member = next((m for m in members.data if m.principal == creator_email), None)
            assert creator_member is not None, "Creator should be listed as a member"
            assert "Admin" in creator_member.roles, "Creator should have Admin role"

    def test_creator_admin_binding_prefers_email_when_id_differs(self, sdk: NeMoHelix):
        """Admin role binding principal is email when sub/oid differs from email (IdP-style headers)."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("email-bind")
        principal_id = str(uuid.uuid4())
        creator_email = f"creator-{uuid.uuid4().hex[:8]}@example.com"

        with as_user_with_id_and_email(sdk, principal_id, creator_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            workspace = workspaces.get_workspace(name=workspace_name).data()
            assert workspace.created_by == principal_id

            members = workspaces.list_workspace_members(workspace=workspace_name).data()
        creator_member = next((m for m in members.data if "Admin" in m.roles), None)
        assert creator_member is not None
        assert creator_member.principal == creator_email
        assert creator_member.principal != principal_id
        assert creator_member.granted_by == principal_id

    def test_creator_admin_binding_uses_id_when_email_header_absent(self, sdk: NeMoHelix):
        """Admin role binding falls back to principal id when X-NHX-Principal-Email is not set."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("id-bind")
        principal_id = str(uuid.uuid4())

        with as_user_id_only(sdk, principal_id):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            members = workspaces.list_workspace_members(workspace=workspace_name).data()

        creator_member = next((m for m in members.data if "Admin" in m.roles), None)
        assert creator_member is not None
        assert creator_member.principal == principal_id

    def test_member_added_by_email_lists_workspace_when_subject_is_uuid(self, sdk: NeMoHelix):
        """Email-keyed role bindings must count for list_workspaces when JWT id is oid/sub, not email."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("invite-email")
        owner_email = f"owner-{uuid.uuid4().hex[:8]}@example.com"
        member_id = str(uuid.uuid4())
        member_email = f"member-{uuid.uuid4().hex[:8]}@example.com"

        with as_user(sdk, owner_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            workspaces.create_workspace_member(
                workspace=workspace_name,
                body=CreateWorkspaceMemberRequest(principal=member_email, roles=["Editor"]),
            ).data()

        with as_user_with_id_and_email(sdk, member_id, member_email):
            result = workspaces.list_workspaces()
            names = [ws.name for ws in result.items()]

        assert workspace_name in names

    def test_member_added_by_group_lists_workspace_when_user_in_group(self, sdk: NeMoHelix):
        """Group-keyed role bindings must count for list_workspaces when the user carries that group."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("invite-group")
        owner_email = f"owner-{uuid.uuid4().hex[:8]}@example.com"
        group_principal = f"team-{uuid.uuid4().hex[:12]}"
        member_user_id = str(uuid.uuid4())

        with as_user(sdk, owner_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            workspaces.create_workspace_member(
                workspace=workspace_name,
                body=CreateWorkspaceMemberRequest(principal=group_principal, roles=["Editor"]),
            ).data()

        with as_user_with_id_and_groups(sdk, member_user_id, [group_principal]):
            result = workspaces.list_workspaces()
            names = [ws.name for ws in result.items()]

        assert workspace_name in names

    def test_list_workspaces_only_shows_accessible(self, sdk: NeMoHelix):
        """Test that listing workspaces only shows workspaces the user can access."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        user1_email = f"user1-{uuid.uuid4().hex[:8]}@example.com"
        user2_email = f"user2-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("priv-ws")

        # User1 creates a workspace
        with as_user(sdk, user1_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # User1 should see the workspace
            result = workspaces.list_workspaces()
            user1_workspaces = [ws.name for ws in result.items()]
            assert workspace_name in user1_workspaces

        # User2 should NOT see the workspace (no role binding)
        with as_user(sdk, user2_email):
            result = workspaces.list_workspaces()
            user2_workspaces = [ws.name for ws in result.items()]
            assert workspace_name not in user2_workspaces

    def test_user_without_role_cannot_access_workspace(self, sdk: NeMoHelix):
        """Test that a user without a role cannot access a workspace."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        owner_email = f"owner-{uuid.uuid4().hex[:8]}@example.com"
        other_email = f"other-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("no-access")

        # Owner creates workspace
        with as_user(sdk, owner_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        # Other user tries to access - should fail
        with as_user(sdk, other_email):
            with pytest.raises(PermissionDeniedError):
                workspaces.get_workspace(name=workspace_name).data()

    def test_admin_can_update_workspace(self, sdk: NeMoHelix):
        """Test that an Admin can update their workspace."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        admin_email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("upd-auth")

        with as_user(sdk, admin_email):
            # Create workspace (admin gets Admin role automatically)
            created = workspaces.create_workspace(
                body=CreateWorkspaceRequest(name=workspace_name, description="Original")
            ).data()
            assert created.created_by == admin_email

            # Update workspace
            updated = workspaces.update_workspace(
                name=workspace_name, body=UpdateWorkspaceRequest(description="Updated by admin")
            ).data()
            assert updated.description == "Updated by admin"
            assert updated.created_by == admin_email
            assert updated.updated_by == admin_email

    def test_updated_by_changes_when_different_user_updates(self, sdk: NeMoHelix):
        """Test that updated_by reflects the user who made the update, not the creator."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        creator_email = f"creator-{uuid.uuid4().hex[:8]}@example.com"
        updater_email = f"updater-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("upd-by")

        # Creator creates workspace
        with as_user(sdk, creator_email):
            created = workspaces.create_workspace(
                body=CreateWorkspaceRequest(name=workspace_name, description="Original")
            ).data()
            assert created.created_by == creator_email
            assert created.updated_by == creator_email

            # Add updater as Admin so they can update
            workspaces.create_workspace_member(
                workspace=workspace_name,
                body=CreateWorkspaceMemberRequest(principal=updater_email, roles=["Admin"]),
            ).data()

        # Different user updates the workspace
        with as_user(sdk, updater_email):
            updated = workspaces.update_workspace(
                name=workspace_name, body=UpdateWorkspaceRequest(description="Updated by different user")
            ).data()
            assert updated.description == "Updated by different user"
            # created_by should remain the original creator
            assert updated.created_by == creator_email
            # updated_by should be the user who made the update
            assert updated.updated_by == updater_email

    def test_viewer_cannot_update_workspace(self, sdk: NeMoHelix):
        """Test that a Viewer cannot update a workspace."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        owner_email = f"owner-{uuid.uuid4().hex[:8]}@example.com"
        viewer_email = f"viewer-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("view-only")

        # Owner creates workspace and adds viewer
        with as_user(sdk, owner_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()
            workspaces.create_workspace_member(
                workspace=workspace_name,
                body=CreateWorkspaceMemberRequest(principal=viewer_email, roles=["Viewer"]),
            ).data()

        # Viewer can read but cannot update
        with as_user(sdk, viewer_email):
            workspace = workspaces.get_workspace(name=workspace_name).data()
            assert workspace.name == workspace_name

            with pytest.raises(PermissionDeniedError):
                workspaces.update_workspace(
                    name=workspace_name, body=UpdateWorkspaceRequest(description="Updated by viewer")
                ).data()

    def test_delete_workspace_deletes_role_bindings(self, sdk: NeMoHelix):
        """Test that workspace deletion automatically deletes role bindings.

        Role bindings are system-managed and should not block workspace deletion.
        """
        workspaces = client_from_platform(sdk, WorkspacesClient)
        admin_email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("del-ok")

        with as_user(sdk, admin_email):
            # Create workspace (creates Admin role binding automatically)
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # Verify role binding exists
            members = workspaces.list_workspace_members(workspace=workspace_name).data()
            assert len(members.data) > 0

            # Delete workspace - should succeed, role bindings are deleted automatically
            workspaces.delete_workspace(name=workspace_name).data()

        # Verify workspace is deleted by checking it's not in the list
        # (after deletion, user no longer has access so we check via list)
        with as_user(sdk, admin_email):
            ws_list = workspaces.list_workspaces()
            workspace_names = [ws.name for ws in ws_list.items()]
            assert workspace_name not in workspace_names

    def test_cannot_remove_last_admin_via_member_delete(self, sdk: NeMoHelix):
        """Test that removing the last Admin via member delete fails."""
        workspaces = client_from_platform(sdk, WorkspacesClient)

        admin_email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("last-admin")

        with as_user(sdk, admin_email):
            # Create workspace (admin is the only Admin)
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # Try to remove self (the last admin) - should fail
            with pytest.raises(ConflictError) as exc_info:
                workspaces.delete_workspace_member(workspace=workspace_name, principal_id=admin_email).data()

            assert "last admin" in str(exc_info.value).lower()

    def test_cannot_remove_last_admin_via_role_update(self, sdk: NeMoHelix):
        """Test that removing the Admin role via update when they're the last Admin fails."""
        workspaces = client_from_platform(sdk, WorkspacesClient)

        admin_email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("upd-admin")

        with as_user(sdk, admin_email):
            # Create workspace (admin is the only Admin)
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # Try to change own role from Admin to Viewer - should fail
            with pytest.raises(ConflictError) as exc_info:
                workspaces.update_workspace_member(
                    workspace=workspace_name,
                    principal_id=admin_email,
                    body=UpdateWorkspaceMemberRequest(roles=["Viewer"]),
                ).data()

            assert "last admin" in str(exc_info.value).lower()

    def test_can_remove_admin_when_another_admin_exists(self, sdk: NeMoHelix):
        """Test that removing an Admin succeeds when another Admin exists."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        admin1_email = f"admin1-{uuid.uuid4().hex[:8]}@example.com"
        admin2_email = f"admin2-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("two-admins")

        with as_user(sdk, admin1_email):
            # Create workspace (admin1 is the Admin)
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

            # Add admin2 as another Admin
            workspaces.create_workspace_member(
                workspace=workspace_name,
                body=CreateWorkspaceMemberRequest(principal=admin2_email, roles=["Admin"]),
            ).data()

            # Now admin1 can remove themselves since admin2 is also an Admin
            workspaces.delete_workspace_member(workspace=workspace_name, principal_id=admin1_email).data()

        # Verify admin1 is no longer a member (check as admin2 since admin1 lost access)
        with as_user(sdk, admin2_email):
            members = workspaces.list_workspace_members(workspace=workspace_name).data()
            member_principals = [m.principal for m in members.data]
            assert admin1_email not in member_principals
            assert admin2_email in member_principals

    def test_delete_workspace_with_entities_marks_for_deletion(self, sdk: NeMoHelix):
        """Test that deleting a workspace with entities marks it for async deletion.

        With cascade delete, workspaces are marked for deletion and become inaccessible.
        An async cleanup controller handles entity deletion.

        Note: After deletion, the user may get 403 (role bindings deleted) or 404
        (workspace marked for deletion). Both indicate the workspace is inaccessible.
        """
        workspaces = client_from_platform(sdk, WorkspacesClient)
        from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError

        admin_email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
        workspace_name = short_unique_name("has-ent")

        with as_user(sdk, admin_email):
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        # Create entity as service principal (generic entities API requires service credentials)
        with as_service(sdk, "entities"):
            client_from_platform(sdk, EntitiesClient).create_entity(
                workspace=workspace_name,
                entity_type="test-entity-type",
                body=EntityCreateInput(
                    name="test-entity",
                    data={"key": "value"},
                ),
            ).data()

        with as_user(sdk, admin_email):
            workspaces.delete_workspace(name=workspace_name).data()

            # Verify workspace is inaccessible (403 or 404)
            # 403: Role bindings deleted, user has no access
            # 404: Workspace marked for deletion
            with pytest.raises((NotFoundError, PermissionDeniedError)):
                workspaces.get_workspace(name=workspace_name).data()
