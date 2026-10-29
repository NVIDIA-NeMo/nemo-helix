# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace member management endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceMemberCRUD:
    """Test CRUD operations for workspace members."""

    async def test_list_empty_members(self, client: AsyncClient, ctx):
        """Test listing members of a workspace with no members."""
        response = await client.get("/apis/entities/v2/workspaces/default/members")

        assert response.status_code == 200
        result = response.json()
        assert "data" in result
        assert isinstance(result["data"], list)
        # Initially empty (no role bindings)
        assert len(result["data"]) == 0

    async def test_add_member_with_default_role(self, client: AsyncClient, ctx):
        """Test adding a member with default role (Editor)."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "user@example.com"},
        )

        assert response.status_code == 201
        result = response.json()
        assert result["principal"] == "user@example.com"
        assert result["roles"] == ["Editor"]
        assert "granted_at" in result

    async def test_add_member_with_multiple_roles(self, client: AsyncClient, ctx):
        """Test adding a member with multiple roles."""
        response = await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={
                "principal": "admin@example.com",
                "roles": ["Viewer", "Editor", "Admin"],
            },
        )

        assert response.status_code == 201
        result = response.json()
        assert result["principal"] == "admin@example.com"
        assert set(result["roles"]) == {"Viewer", "Editor", "Admin"}

    async def test_list_members_after_adding(self, client: AsyncClient, ctx):
        """Test listing members after adding some."""
        # Add a member
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "list-test@example.com", "roles": ["Editor"]},
        )

        response = await client.get("/apis/entities/v2/workspaces/default/members")

        assert response.status_code == 200
        result = response.json()
        assert len(result["data"]) >= 1

        # Find our member
        member = next((m for m in result["data"] if m["principal"] == "list-test@example.com"), None)
        assert member is not None
        assert "Editor" in member["roles"]

    async def test_update_member_roles_add_role(self, client: AsyncClient, ctx):
        """Test adding a role to an existing member."""
        # Add member with one role
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "update-add@example.com", "roles": ["Viewer"]},
        )

        # Update to add Editor role
        response = await client.put(
            "/apis/entities/v2/workspaces/default/members/update-add@example.com?wait_role_propagation=false",
            json={"roles": ["Viewer", "Editor"]},
        )

        assert response.status_code == 200
        result = response.json()
        assert result["principal"] == "update-add@example.com"
        assert set(result["roles"]) == {"Viewer", "Editor"}

    async def test_update_member_roles_remove_role(self, client: AsyncClient, ctx):
        """Test removing a role from an existing member."""
        # Add member with multiple roles
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "update-remove@example.com", "roles": ["Viewer", "Editor"]},
        )

        # Update to remove Editor role
        response = await client.put(
            "/apis/entities/v2/workspaces/default/members/update-remove@example.com?wait_role_propagation=false",
            json={"roles": ["Viewer"]},
        )

        assert response.status_code == 200
        result = response.json()
        assert result["principal"] == "update-remove@example.com"
        assert result["roles"] == ["Viewer"]

    async def test_update_member_roles_replace_all(self, client: AsyncClient, ctx):
        """Test replacing all roles for a member."""
        # Add member with one role
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "update-replace@example.com", "roles": ["Viewer"]},
        )

        # Replace with completely different roles
        response = await client.put(
            "/apis/entities/v2/workspaces/default/members/update-replace@example.com?wait_role_propagation=false",
            json={"roles": ["Admin"]},
        )

        assert response.status_code == 200
        result = response.json()
        assert result["principal"] == "update-replace@example.com"
        assert result["roles"] == ["Admin"]

    async def test_remove_member(self, client: AsyncClient, ctx):
        """Test removing a member from a workspace."""
        # Add member
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "remove-me@example.com", "roles": ["Editor"]},
        )

        # Remove member
        response = await client.delete(
            "/apis/entities/v2/workspaces/default/members/remove-me@example.com?wait_role_propagation=false"
        )

        assert response.status_code == 200
        result = response.json()
        assert result["id"] == "remove-me@example.com"

        # Verify member is no longer listed
        list_response = await client.get("/apis/entities/v2/workspaces/default/members")
        members = list_response.json()["data"]
        assert not any(m["principal"] == "remove-me@example.com" for m in members)

    async def test_add_existing_member_is_idempotent(self, client: AsyncClient, ctx):
        """Test that adding an existing member with same role is idempotent."""
        # Add member first time
        response1 = await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "idempotent@example.com", "roles": ["Editor"]},
        )
        assert response1.status_code == 201

        # Add same member again with same role
        response2 = await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "idempotent@example.com", "roles": ["Editor"]},
        )
        assert response2.status_code == 201

        # Verify only one member entry
        list_response = await client.get("/apis/entities/v2/workspaces/default/members")
        members = [m for m in list_response.json()["data"] if m["principal"] == "idempotent@example.com"]
        assert len(members) == 1
        assert members[0]["roles"] == ["Editor"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceMemberErrors:
    """Test error cases for workspace member operations."""

    async def test_list_members_nonexistent_workspace(self, client: AsyncClient):
        """Test listing members of a non-existent workspace returns 404."""
        response = await client.get("/apis/entities/v2/workspaces/nonexistent-workspace/members")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_add_member_nonexistent_workspace(self, client: AsyncClient):
        """Test adding a member to a non-existent workspace returns 404."""
        response = await client.post(
            "/apis/entities/v2/workspaces/nonexistent-workspace/members?wait_role_propagation=false",
            json={"principal": "user@example.com", "roles": ["Editor"]},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_update_member_nonexistent_workspace(self, client: AsyncClient):
        """Test updating a member in a non-existent workspace returns 404."""
        response = await client.put(
            "/apis/entities/v2/workspaces/nonexistent-workspace/members/user@example.com?wait_role_propagation=false",
            json={"roles": ["Editor"]},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_remove_member_nonexistent_workspace(self, client: AsyncClient):
        """Test removing a member from a non-existent workspace returns 404."""
        response = await client.delete(
            "/apis/entities/v2/workspaces/nonexistent-workspace/members/user@example.com?wait_role_propagation=false"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_remove_nonexistent_member(self, client: AsyncClient, ctx):
        """Test removing a non-existent member returns 404."""
        response = await client.delete(
            "/apis/entities/v2/workspaces/default/members/nonexistent-user@example.com?wait_role_propagation=false"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceMemberMultipleWorkspaces:
    """Test member operations across multiple workspaces."""

    async def test_member_in_multiple_workspaces(self, client: AsyncClient, ctx):
        """Test that a member can be in multiple workspaces with different roles."""
        # Create a second workspace
        await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": "workspace-two", "description": "Second workspace"},
        )

        # Add same user to both workspaces with different roles
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "multi-ws@example.com", "roles": ["Viewer"]},
        )
        await client.post(
            "/apis/entities/v2/workspaces/workspace-two/members?wait_role_propagation=false",
            json={"principal": "multi-ws@example.com", "roles": ["Admin"]},
        )

        # Verify roles in first workspace
        response1 = await client.get("/apis/entities/v2/workspaces/default/members")
        members1 = [m for m in response1.json()["data"] if m["principal"] == "multi-ws@example.com"]
        assert len(members1) == 1
        assert "Viewer" in members1[0]["roles"]

        # Verify roles in second workspace
        response2 = await client.get("/apis/entities/v2/workspaces/workspace-two/members")
        members2 = [m for m in response2.json()["data"] if m["principal"] == "multi-ws@example.com"]
        assert len(members2) == 1
        assert "Admin" in members2[0]["roles"]

    async def test_remove_member_from_one_workspace_doesnt_affect_other(self, client: AsyncClient, ctx):
        """Test that removing a member from one workspace doesn't affect other workspaces."""
        # Create a second workspace
        await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": "workspace-three", "description": "Third workspace"},
        )

        # Add same user to both workspaces
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "cross-ws@example.com", "roles": ["Editor"]},
        )
        await client.post(
            "/apis/entities/v2/workspaces/workspace-three/members?wait_role_propagation=false",
            json={"principal": "cross-ws@example.com", "roles": ["Editor"]},
        )

        # Remove from first workspace
        await client.delete(
            "/apis/entities/v2/workspaces/default/members/cross-ws@example.com?wait_role_propagation=false"
        )

        # Verify removed from first workspace
        response1 = await client.get("/apis/entities/v2/workspaces/default/members")
        members1 = [m for m in response1.json()["data"] if m["principal"] == "cross-ws@example.com"]
        assert len(members1) == 0

        # Verify still in second workspace
        response2 = await client.get("/apis/entities/v2/workspaces/workspace-three/members")
        members2 = [m for m in response2.json()["data"] if m["principal"] == "cross-ws@example.com"]
        assert len(members2) == 1


@pytest.mark.integration
@pytest.mark.asyncio
class TestBoundWorkspaceGuardWiring:
    """Where the bound-workspace check runs: on granting roles, never on taking them away."""

    @pytest.fixture
    def checked(self, monkeypatch):
        """Record every principal the bound-workspace check is asked about."""
        from nhx.core.entities.api.v2.workspaces import endpoints

        principals: list[str] = []

        async def record(principal, workspace, **_):
            principals.append(principal)

        monkeypatch.setattr(endpoints, "_require_bound_workspace_admin", record)
        return principals

    async def test_adding_a_member_with_roles_is_checked(self, client: AsyncClient, ctx, checked):
        response = await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "service-account:otel", "roles": ["Editor"]},
        )

        assert response.status_code == 201
        assert checked == ["service-account:otel"]

    async def test_updating_a_member_is_checked_only_while_roles_are_added(self, client: AsyncClient, ctx, checked):
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "service-account:otel", "roles": ["Editor", "Viewer"]},
        )
        checked.clear()

        narrowed = await client.put(
            "/apis/entities/v2/workspaces/default/members/service-account:otel?wait_role_propagation=false",
            json={"roles": ["Viewer"]},
        )
        assert narrowed.status_code == 200
        assert checked == []

        widened = await client.put(
            "/apis/entities/v2/workspaces/default/members/service-account:otel?wait_role_propagation=false",
            json={"roles": ["Viewer", "Editor"]},
        )
        assert widened.status_code == 200
        assert checked == ["service-account:otel"]

    async def test_removing_a_member_is_not_checked(self, client: AsyncClient, ctx, checked):
        await client.post(
            "/apis/entities/v2/workspaces/default/members?wait_role_propagation=false",
            json={"principal": "service-account:otel", "roles": ["Editor"]},
        )
        checked.clear()

        response = await client.delete(
            "/apis/entities/v2/workspaces/default/members/service-account:otel?wait_role_propagation=false"
        )

        assert response.status_code == 200
        assert checked == []

    async def test_a_missing_workspace_is_a_404_before_any_check(self, client: AsyncClient, ctx, checked):
        response = await client.post(
            "/apis/entities/v2/workspaces/nope/members?wait_role_propagation=false",
            json={"principal": "service-account:otel", "roles": ["Editor"]},
        )

        assert response.status_code == 404
        assert checked == []
