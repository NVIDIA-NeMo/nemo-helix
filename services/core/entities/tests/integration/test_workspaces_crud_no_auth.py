# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace CRUD operations without authorization.

These tests verify:
- Workspace CRUD operations (create, retrieve, list, update, delete)
- Validation (duplicate names, non-existent workspaces, invalid input)

Uses the create_test_client pattern for fast in-memory testing.
"""

from typing import Generator

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceRequest,
    ListWorkspacesQueryParams,
    UpdateWorkspaceRequest,
)
from nhx.core.entities.service import EntitiesService
from nhx.testing import create_test_client, short_unique_name


@pytest.fixture(scope="module")
def sdk() -> Generator[NeMoHelix, None, None]:
    """SDK client with EntitiesService (auth disabled)."""
    with create_test_client(
        EntitiesService,
        workspaces=[],  # Don't auto-create workspaces - we're testing workspace CRUD
        projects=[],  # Skip project creation
    ) as sdk:
        yield sdk


@pytest.mark.integration
class TestWorkspaceCRUD:
    """Test workspace CRUD operations without authorization."""

    def test_default_workspaces_created_on_startup(self, sdk: NeMoHelix):
        """Test that 'default' and 'system' workspaces are created automatically on startup."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        # These workspaces are created by EntitiesService.startup()
        default_ws = workspaces.get_workspace(name="default").data()
        assert default_ws.name == "default"
        assert default_ws.description == "General-purpose workspace (all users have write access)"

        system_ws = workspaces.get_workspace(name="system").data()
        assert system_ws.name == "system"
        assert system_ws.description == "Platform-provided resources (read-only for users)"

    def test_create_workspace(self, sdk: NeMoHelix):
        """Test creating a new workspace."""
        workspace_name = short_unique_name("test-ws")

        workspace = (
            client_from_platform(sdk, WorkspacesClient)
            .create_workspace(
                body=CreateWorkspaceRequest(name=workspace_name, description="Test workspace for integration tests")
            )
            .data()
        )

        assert workspace.name == workspace_name
        assert workspace.description == "Test workspace for integration tests"
        assert workspace.id is not None
        assert workspace.created_at is not None
        assert workspace.updated_at is not None
        # Without auth, created_by/updated_by are empty string
        assert workspace.created_by == ""
        assert workspace.updated_by == ""

    def test_create_duplicate_workspace_fails(self, sdk: NeMoHelix):
        """Test that creating a duplicate workspace returns 409."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("dup-ws")

        # Create first workspace
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        # Try to create duplicate
        from nemo_helix_plugin.client.errors import ConflictError

        with pytest.raises(ConflictError) as exc_info:
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        assert "already exists" in str(exc_info.value)

    def test_retrieve_workspace(self, sdk: NeMoHelix):
        """Test retrieving a workspace by name."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("get-ws")
        created = workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="Retrieve test")
        ).data()

        retrieved = workspaces.get_workspace(name=workspace_name).data()

        assert retrieved.id == created.id
        assert retrieved.name == workspace_name
        assert retrieved.description == "Retrieve test"

    def test_retrieve_nonexistent_workspace_fails(self, sdk: NeMoHelix):
        """Test that retrieving a non-existent workspace returns 404."""
        from nemo_helix_plugin.client.errors import NotFoundError

        with pytest.raises(NotFoundError):
            client_from_platform(sdk, WorkspacesClient).get_workspace(name="nonexistent-workspace").data()

    def test_list_workspaces(self, sdk: NeMoHelix):
        """Test listing workspaces."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("list-ws")
        created = workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        result = workspaces.list_workspaces()

        workspace_ids = [ws.id for ws in result.items()]
        assert created.id in workspace_ids

    def test_list_workspaces_with_pagination(self, sdk: NeMoHelix):
        """Test pagination when listing workspaces."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        # Create a few workspaces
        for i in range(3):
            name = short_unique_name(f"page-{i}")
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()

        # List with pagination
        result = workspaces.list_workspaces(query_params=ListWorkspacesQueryParams(page=1, page_size=2))
        page = result.page()

        assert page.metadata is not None
        assert page.metadata["page"] == 1
        assert page.metadata["page_size"] == 2

    def test_update_workspace(self, sdk: NeMoHelix):
        """Test updating a workspace description."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("upd-ws")
        workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="Original description")
        ).data()

        updated = workspaces.update_workspace(
            name=workspace_name, body=UpdateWorkspaceRequest(description="Updated description")
        ).data()

        assert updated.name == workspace_name
        assert updated.description == "Updated description"
        # Without auth, created_by/updated_by are empty string
        assert updated.created_by == ""
        assert updated.updated_by == ""

    def test_update_nonexistent_workspace_fails(self, sdk: NeMoHelix):
        """Test that updating a non-existent workspace returns 404."""
        from nemo_helix_plugin.client.errors import NotFoundError

        with pytest.raises(NotFoundError):
            client_from_platform(sdk, WorkspacesClient).update_workspace(
                name="nonexistent-workspace", body=UpdateWorkspaceRequest(description="Should fail")
            ).data()

    def test_delete_workspace(self, sdk: NeMoHelix):
        """Test deleting a workspace."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("del-ws")
        workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name)).data()

        # Delete the workspace
        workspaces.delete_workspace(name=workspace_name).data()

        # Verify it's deleted
        from nemo_helix_plugin.client.errors import NotFoundError

        with pytest.raises(NotFoundError):
            workspaces.get_workspace(name=workspace_name).data()

    def test_delete_nonexistent_workspace_fails(self, sdk: NeMoHelix):
        """Test that deleting a non-existent workspace returns 404."""
        from nemo_helix_plugin.client.errors import NotFoundError

        with pytest.raises(NotFoundError):
            client_from_platform(sdk, WorkspacesClient).delete_workspace(name="nonexistent-workspace").data()

    def test_workspace_crud_lifecycle(self, sdk: NeMoHelix):
        """Test full CRUD lifecycle for workspaces."""
        workspaces = client_from_platform(sdk, WorkspacesClient)
        workspace_name = short_unique_name("crud-ws")

        # CREATE
        created = workspaces.create_workspace(
            body=CreateWorkspaceRequest(name=workspace_name, description="CRUD lifecycle test")
        ).data()
        assert created.name == workspace_name
        assert created.id is not None

        # READ
        retrieved = workspaces.get_workspace(name=workspace_name).data()
        assert retrieved.id == created.id

        # UPDATE
        updated = workspaces.update_workspace(
            name=workspace_name, body=UpdateWorkspaceRequest(description="Updated in lifecycle test")
        ).data()
        assert updated.description == "Updated in lifecycle test"

        # DELETE
        workspaces.delete_workspace(name=workspace_name).data()

        # Verify deleted
        from nemo_helix_plugin.client.errors import NotFoundError

        with pytest.raises(NotFoundError):
            workspaces.get_workspace(name=workspace_name).data()


@pytest.mark.integration
class TestWorkspaceValidation:
    """Test workspace input validation."""

    def test_create_workspace_invalid_name(self, sdk: NeMoHelix):
        """Test that creating a workspace with invalid name returns 422."""
        # Names with spaces are invalid
        response = sdk._client.post(
            "/apis/entities/v2/workspaces",
            json={"name": "invalid name with spaces"},
        )
        assert response.status_code == 422

    def test_create_workspace_empty_name(self, sdk: NeMoHelix):
        """Test that creating a workspace with empty name returns 422."""
        response = sdk._client.post(
            "/apis/entities/v2/workspaces",
            json={"name": ""},
        )
        assert response.status_code == 422
