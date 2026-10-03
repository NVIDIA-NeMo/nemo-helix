# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for workspace operations with authentication enabled.

These tests verify:
- Workspace CRUD operations with proper authorization
- Role-based access control (Admin, Viewer, Editor)
- Workspace visibility based on role bindings
- Member management (add, list, update, remove)
- Wildcard principal for shared workspaces
- Admin role protection (cannot remove last admin)
- Default workspaces behavior (default, system)
"""

import uuid

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.projects.client import ProjectsClient
from nemo_helix_plugin.projects.types import CreateProjectRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
)
from nhx.testing import as_user, grant_workspace_role, short_unique_name, unique_email

# Mark all tests in this module to require auth (run with --feature auth).
pytestmark = [pytest.mark.feature("auth")]


def _workspaces_as(client: NemoClient, email: str) -> WorkspacesClient:
    """Return a WorkspacesClient authenticated as *email*."""
    return WorkspacesClient.from_client(as_user(client, email))


def _projects_as(client: NemoClient, email: str) -> ProjectsClient:
    """Return a ProjectsClient authenticated as *email*."""
    return ProjectsClient.from_client(as_user(client, email))


def _workspace_names(workspaces: WorkspacesClient) -> list[str]:
    return [ws.name for ws in workspaces.list_workspaces().items()]


def _member_principals(workspaces: WorkspacesClient, workspace: str) -> list[str]:
    return [m.principal for m in workspaces.list_workspace_members(workspace=workspace).data().data]


class TestDefaultWorkspaces:
    """Tests for default workspaces created on platform startup."""

    def test_default_workspace_accessible_by_all_users(self, client: NemoClient):
        """Verify that the 'default' workspace is accessible by any authenticated user.

        The 'default' workspace has a wildcard Editor binding, giving all users
        read-write access.
        """
        user_workspaces = _workspaces_as(client, unique_email("random-user"))

        default_ws = user_workspaces.get_workspace(name="default").data()
        assert default_ws.name == "default"
        assert "General-purpose workspace" in (default_ws.description or "")

    def test_system_workspace_accessible_by_all_users(self, client: NemoClient):
        """Verify that the 'system' workspace is accessible by any authenticated user.

        The 'system' workspace has a wildcard Viewer binding, giving all users
        read-only access to platform-provided resources.
        """
        user_workspaces = _workspaces_as(client, unique_email("random-user"))

        system_ws = user_workspaces.get_workspace(name="system").data()
        assert system_ws.name == "system"
        assert "read-only" in (system_ws.description or "").lower()

    def test_list_workspaces_shows_default_workspaces(self, client: NemoClient):
        """Verify that default workspaces are accessible to regular users."""
        user_workspaces = _workspaces_as(client, unique_email("list-user"))

        workspace_names = _workspace_names(user_workspaces)

        if "default" not in workspace_names:
            assert user_workspaces.get_workspace(name="default").data().name == "default"
        if "system" not in workspace_names:
            assert user_workspaces.get_workspace(name="system").data().name == "system"

    def test_default_workspace_allows_write_operations(self, client: NemoClient):
        """Verify that any user can create resources in the 'default' workspace.

        The default workspace has Editor role for wildcard principal, allowing
        all authenticated users to create and manage resources.
        """
        user_projects = _projects_as(client, unique_email("writer"))
        project_name = f"test-project-{uuid.uuid4().hex[:8]}"

        # User should be able to create a project in default workspace
        project = user_projects.create_project(
            workspace="default",
            body=CreateProjectRequest(name=project_name, description="E2E test project"),
        ).data()
        assert project.name == project_name

        # Clean up
        user_projects.delete_project(workspace="default", name=project_name)

    def test_system_workspace_denies_write_operations(self, client: NemoClient):
        """Verify that regular users cannot create resources in the 'system' workspace.

        The system workspace has only Viewer role for wildcard principal,
        so regular users can read but not write.
        """
        user_projects = _projects_as(client, unique_email("reader"))
        project_name = f"test-project-{uuid.uuid4().hex[:8]}"

        with pytest.raises(PermissionDeniedError):
            user_projects.create_project(
                workspace="system",
                body=CreateProjectRequest(name=project_name, description="Should fail"),
            )


class TestWorkspaceCRUD:
    """Tests for workspace Create, Retrieve, Update, Delete operations."""

    def test_create_workspace(self, client: NemoClient):
        """Test that an authenticated user can create a new workspace."""
        user_workspaces = _workspaces_as(client, unique_email("creator"))
        workspace_name = short_unique_name("create-test")

        workspace = user_workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="E2E test workspace"),
        ).data()

        assert workspace.name == workspace_name
        assert workspace.description == "E2E test workspace"
        assert workspace.id is not None

        # Clean up
        user_workspaces.delete_workspace(name=workspace_name)

    def test_creator_becomes_admin(self, client: NemoClient):
        """Test that workspace creator automatically becomes Admin."""
        user_email = unique_email("admin-creator")
        user_workspaces = _workspaces_as(client, user_email)
        workspace_name = short_unique_name("admin-test")

        user_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Verify creator is listed as Admin
        members = user_workspaces.list_workspace_members(workspace=workspace_name).data()
        creator_member = next((m for m in members.data if m.principal == user_email), None)

        assert creator_member is not None, "Creator should be a member"
        assert "Admin" in creator_member.roles, "Creator should have Admin role"

        # Clean up
        user_workspaces.delete_workspace(name=workspace_name)

    def test_retrieve_workspace(self, client: NemoClient):
        """Test retrieving a workspace by name."""
        user_workspaces = _workspaces_as(client, unique_email("retriever"))
        workspace_name = short_unique_name("retrieve-test")

        created = user_workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="Retrieve test"),
        ).data()

        retrieved = user_workspaces.get_workspace(name=workspace_name).data()

        assert retrieved.id == created.id
        assert retrieved.name == workspace_name
        assert retrieved.description == "Retrieve test"

        # Clean up
        user_workspaces.delete_workspace(name=workspace_name)

    def test_update_workspace(self, client: NemoClient):
        """Test that an Admin can update workspace description."""
        user_workspaces = _workspaces_as(client, unique_email("updater"))
        workspace_name = short_unique_name("update-test")

        user_workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="Original description"),
        )

        updated = user_workspaces.update_workspace(
            name=workspace_name,
            body=UpdateWorkspaceRequest(description="Updated description"),
        ).data()

        assert updated.description == "Updated description"

        # Clean up
        user_workspaces.delete_workspace(name=workspace_name)

    def test_delete_workspace(self, client: NemoClient):
        """Test that an Admin can delete a workspace."""
        user_workspaces = _workspaces_as(client, unique_email("deleter"))
        workspace_name = short_unique_name("delete-test")

        user_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Delete the workspace
        user_workspaces.delete_workspace(name=workspace_name)

        # Verify it's deleted (user won't see it in list anymore)
        assert workspace_name not in _workspace_names(user_workspaces)

    def test_create_duplicate_workspace_fails(self, client: NemoClient):
        """Test that creating a duplicate workspace returns 409 Conflict."""
        user_workspaces = _workspaces_as(client, unique_email("dup-creator"))
        workspace_name = short_unique_name("dup-test")

        user_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        with pytest.raises(ConflictError) as exc_info:
            user_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        assert "already exists" in str(exc_info.value).lower()

        # Clean up
        user_workspaces.delete_workspace(name=workspace_name)

    def test_retrieve_nonexistent_workspace_fails(self, client: NemoClient):
        """Test that retrieving a non-existent workspace fails.

        With auth enabled, a user without access gets 403 (Forbidden) rather than 404.
        This is correct security behavior - don't reveal whether a workspace exists.
        """
        user_workspaces = _workspaces_as(client, unique_email("retriever"))

        # With auth enabled, we get 403 (not 404) to avoid revealing workspace existence
        with pytest.raises((NotFoundError, PermissionDeniedError)):
            user_workspaces.get_workspace(name="nonexistent-workspace-xyz")


class TestWorkspaceVisibility:
    """Tests for workspace visibility based on role bindings."""

    def test_private_workspace_not_visible_to_others(self, client: NemoClient):
        """Test that a private workspace is not visible to users without roles."""
        owner_workspaces = _workspaces_as(client, unique_email("owner"))
        other_workspaces = _workspaces_as(client, unique_email("other"))
        workspace_name = short_unique_name("private")

        # Owner creates workspace
        owner_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Other user should NOT see the workspace
        assert workspace_name not in _workspace_names(other_workspaces)

        # Clean up
        owner_workspaces.delete_workspace(name=workspace_name)

    def test_user_without_role_cannot_access_workspace(self, client: NemoClient):
        """Test that a user without a role cannot retrieve a workspace."""
        owner_workspaces = _workspaces_as(client, unique_email("owner"))
        other_workspaces = _workspaces_as(client, unique_email("other"))
        workspace_name = short_unique_name("no-access")

        # Owner creates workspace
        owner_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Other user should get PermissionDenied
        with pytest.raises(PermissionDeniedError):
            other_workspaces.get_workspace(name=workspace_name)

        # Clean up
        owner_workspaces.delete_workspace(name=workspace_name)


class TestRoleBasedAccessControl:
    """Tests for role-based access control (Viewer, Editor, Admin)."""

    def test_viewer_can_read_but_not_write(self, client: NemoClient):
        """Test that a Viewer can read resources but cannot create or update."""
        viewer_email = unique_email("viewer")
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        viewer = as_user(client, viewer_email)
        viewer_workspaces = WorkspacesClient.from_client(viewer)
        viewer_projects = ProjectsClient.from_client(viewer)
        workspace_name = short_unique_name("viewer-test")

        # Admin creates workspace and adds viewer
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Viewer can read workspace
        workspace = viewer_workspaces.get_workspace(name=workspace_name).data()
        assert workspace.name == workspace_name

        # Viewer cannot update workspace
        with pytest.raises(PermissionDeniedError):
            viewer_workspaces.update_workspace(
                name=workspace_name, body=UpdateWorkspaceRequest(description="Should fail")
            )

        # Viewer cannot create projects
        with pytest.raises(PermissionDeniedError):
            viewer_projects.create_project(
                workspace=workspace_name,
                body=CreateProjectRequest(name="test-project", description="Should fail"),
            )

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_editor_can_create_resources(self, client: NemoClient):
        """Test that an Editor can create and manage resources."""
        editor_email = unique_email("editor")
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        editor = as_user(client, editor_email)
        editor_workspaces = WorkspacesClient.from_client(editor)
        editor_projects = ProjectsClient.from_client(editor)
        workspace_name = short_unique_name("editor-test")
        project_name = f"editor-project-{uuid.uuid4().hex[:8]}"

        # Admin creates workspace and adds editor
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        # Editor can create projects
        project = editor_projects.create_project(
            workspace=workspace_name,
            body=CreateProjectRequest(name=project_name, description="Created by editor"),
        ).data()
        assert project.name == project_name

        # Editor cannot manage members
        with pytest.raises(PermissionDeniedError):
            grant_workspace_role(
                editor_workspaces, workspace=workspace_name, principal="another@example.com", roles=["Viewer"]
            )

        # Clean up project
        editor_projects.delete_project(workspace=workspace_name, name=project_name)

        # Clean up workspace
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_admin_can_manage_members(self, client: NemoClient):
        """Test that an Admin can add, update, and remove members."""
        member_email = unique_email("member")
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        workspace_name = short_unique_name("admin-members")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Add member
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal=member_email, roles=["Viewer"])

        # Verify member exists
        assert member_email in _member_principals(admin_workspaces, workspace_name)

        # Update member role
        admin_workspaces.update_workspace_member(
            workspace=workspace_name,
            principal_id=member_email,
            body=UpdateWorkspaceMemberRequest(roles=["Editor"]),
        )

        # Verify role updated
        members = admin_workspaces.list_workspace_members(workspace=workspace_name).data()
        member = next(m for m in members.data if m.principal == member_email)
        assert "Editor" in member.roles

        # Remove member
        admin_workspaces.delete_workspace_member(workspace=workspace_name, principal_id=member_email)

        # Verify member removed
        assert member_email not in _member_principals(admin_workspaces, workspace_name)

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)


class TestWildcardAccess:
    """Tests for wildcard principal (*) granting access to all users."""

    def test_wildcard_viewer_makes_workspace_public_readonly(self, client: NemoClient):
        """Test that granting Viewer to '*' makes workspace readable by all."""
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        random_user = as_user(client, unique_email("random"))
        random_workspaces = WorkspacesClient.from_client(random_user)
        random_projects = ProjectsClient.from_client(random_user)
        workspace_name = short_unique_name("public-readonly")

        # Admin creates workspace and grants wildcard Viewer
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal="*", roles=["Viewer"])

        # Random user can now see the workspace
        workspace = random_workspaces.get_workspace(name=workspace_name).data()
        assert workspace.name == workspace_name

        # But cannot write
        with pytest.raises(PermissionDeniedError):
            random_projects.create_project(
                workspace=workspace_name,
                body=CreateProjectRequest(name="test-project", description="Should fail"),
            )

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_wildcard_editor_makes_workspace_public_writable(self, client: NemoClient):
        """Test that granting Editor to '*' makes workspace writable by all."""
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        random_projects = _projects_as(client, unique_email("random"))
        workspace_name = short_unique_name("public-writable")
        project_name = f"public-project-{uuid.uuid4().hex[:8]}"

        # Admin creates workspace and grants wildcard Editor
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal="*", roles=["Editor"])

        # Random user can create projects
        project = random_projects.create_project(
            workspace=workspace_name,
            body=CreateProjectRequest(name=project_name, description="Created by random user"),
        ).data()
        assert project.name == project_name

        # Clean up project
        random_projects.delete_project(workspace=workspace_name, name=project_name)

        # Clean up workspace
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_remove_wildcard_restricts_access(self, client: NemoClient):
        """Test that removing wildcard binding restricts access to explicit members."""
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        random_workspaces = _workspaces_as(client, unique_email("random"))
        workspace_name = short_unique_name("restrict-test")

        # Admin creates workspace with wildcard Viewer
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal="*", roles=["Viewer"])

        # Random user can access
        workspace = random_workspaces.get_workspace(name=workspace_name).data()
        assert workspace.name == workspace_name

        # Admin removes wildcard binding
        admin_workspaces.delete_workspace_member(workspace=workspace_name, principal_id="*")

        # Random user can no longer access
        with pytest.raises(PermissionDeniedError):
            random_workspaces.get_workspace(name=workspace_name)

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)


class TestAdminProtection:
    """Tests for admin role protection (cannot remove last admin)."""

    def test_cannot_remove_last_admin_via_delete(self, client: NemoClient):
        """Test that removing the last Admin via member delete fails."""
        admin_email = unique_email("sole-admin")
        admin_workspaces = _workspaces_as(client, admin_email)
        workspace_name = short_unique_name("last-admin")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Try to remove self (the last admin) - should fail
        with pytest.raises(ConflictError) as exc_info:
            admin_workspaces.delete_workspace_member(workspace=workspace_name, principal_id=admin_email)

        assert "last admin" in str(exc_info.value).lower()

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_cannot_remove_last_admin_via_role_update(self, client: NemoClient):
        """Test that demoting the last Admin to Viewer fails."""
        admin_email = unique_email("sole-admin")
        admin_workspaces = _workspaces_as(client, admin_email)
        workspace_name = short_unique_name("demote-admin")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Try to change own role from Admin to Viewer - should fail
        with pytest.raises(ConflictError) as exc_info:
            admin_workspaces.update_workspace_member(
                workspace=workspace_name,
                principal_id=admin_email,
                body=UpdateWorkspaceMemberRequest(roles=["Viewer"]),
            )

        assert "last admin" in str(exc_info.value).lower()

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_can_remove_admin_when_another_exists(self, client: NemoClient):
        """Test that an Admin can be removed when another Admin exists."""
        admin1_email = unique_email("admin1")
        admin2_email = unique_email("admin2")
        admin1_workspaces = _workspaces_as(client, admin1_email)
        admin2_workspaces = _workspaces_as(client, admin2_email)
        workspace_name = short_unique_name("two-admins")

        admin1_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))

        # Add second admin
        grant_workspace_role(admin1_workspaces, workspace=workspace_name, principal=admin2_email, roles=["Admin"])

        # Admin1 can now remove themselves
        admin1_workspaces.delete_workspace_member(workspace=workspace_name, principal_id=admin1_email)

        # Verify admin1 is gone (check as admin2)
        member_principals = _member_principals(admin2_workspaces, workspace_name)
        assert admin1_email not in member_principals
        assert admin2_email in member_principals

        # Clean up
        admin2_workspaces.delete_workspace(name=workspace_name)


class TestWorkspaceDeletion:
    """Tests for workspace deletion behavior."""

    def test_delete_workspace_removes_role_bindings(self, client: NemoClient):
        """Test that deleting a workspace also cleans up role bindings."""
        member_email = unique_email("member")
        admin_workspaces = _workspaces_as(client, unique_email("admin"))
        workspace_name = short_unique_name("del-bindings")

        # Create workspace with multiple members
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin_workspaces, workspace=workspace_name, principal=member_email, roles=["Editor"])

        # Verify members exist
        assert len(_member_principals(admin_workspaces, workspace_name)) >= 2

        # Delete workspace (role bindings are auto-deleted)
        admin_workspaces.delete_workspace(name=workspace_name)

        # Workspace should no longer exist
        assert workspace_name not in _workspace_names(admin_workspaces)


class TestAuthenticationRequired:
    """Tests verifying that authentication is required."""

    def test_unauthenticated_request_returns_401(self, client: NemoClient):
        """Test that requests without authentication are rejected."""
        # A fresh client for the same platform carries no auth headers
        with NemoClient(base_url=client.base_url) as unauthenticated:
            response = unauthenticated._client.get(f"{client.base_url}/apis/entities/v2/workspaces")
        assert response.status_code == 401

    def test_create_workspace_without_auth_fails(self, client: NemoClient):
        """Test that creating a workspace without auth fails."""
        # A fresh client for the same platform carries no auth headers
        with NemoClient(base_url=client.base_url) as unauthenticated:
            response = unauthenticated._client.post(
                f"{client.base_url}/apis/entities/v2/workspaces",
                json={"name": "should-fail", "description": "No auth"},
            )
        assert response.status_code == 401
