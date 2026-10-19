# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for workspace deletion behavior in entity API."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient
from nhx.common.api.filter import ComparisonOperation, FilterOperator
from nhx.common.auth import get_auth_client
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.core.entities.api.server import app
from nhx.core.entities.app.repository import WorkspaceRepositoryInterface
from nhx.core.entities.entities import WorkspaceDeletionStage


def _create_service_principal_auth_client() -> AuthClient:
    """Create a mock AuthClient for service principal testing."""
    principal = Principal(
        id="service:entities",
        email=None,
        groups=[],
        on_behalf_of=None,
    )
    auth_client = MagicMock(spec=AuthClient)
    auth_client.principal = principal
    auth_client.auth_enabled = False
    auth_client.wait_role = AsyncMock(return_value=True)
    auth_client.has_permissions = AsyncMock(return_value=True)
    auth_client.on_behalf_of_has_permissions = AsyncMock(return_value=True)
    return auth_client


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceDeletionBehavior:
    """Test entity API behavior when workspaces are being deleted."""

    async def test_user_cannot_create_entity_in_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that users get 404 when creating entities in a workspace being deleted."""
        workspace_name = "test-deleting-ws"

        # Create a workspace
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.PENDING,
        )

        # Try to create an entity - should get 404
        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_cannot_get_entity_in_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that users get 404 when getting entities from a workspace being deleted."""
        workspace_name = "test-deleting-ws2"

        # Create a workspace and an entity
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.DELETING,
        )

        # Try to get the entity - should get 404
        response = await client.get(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_cannot_update_entity_in_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that users get 404 when updating entities in a workspace being deleted."""
        workspace_name = "test-deleting-ws3"

        # Create a workspace and an entity
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )

        # Try to update the entity - should get 404
        response = await client.put(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config",
            json={"data": {"target_id": "llama-3-8b"}},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_cannot_delete_entity_in_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that users get 404 when deleting entities from a workspace being deleted."""
        workspace_name = "test-deleting-ws4"

        # Create a workspace and an entity
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.PENDING,
        )

        # Try to delete the entity - should get 404
        response = await client.delete(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config"
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_user_cannot_list_entities_in_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that users get 404 when listing entities in a workspace being deleted."""
        workspace_name = "test-deleting-ws5"

        # Create a workspace and an entity
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.DELETING,
        )

        # Try to list entities - should get 404
        response = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_service_principal_can_access_deleting_workspace(self, client: AsyncClient, ctx, repos):
        """Test that service principals can access entities in workspaces being deleted."""
        workspace_name = "test-deleting-ws6"

        # Create a workspace and an entity
        response = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Test workspace"},
        )
        assert response.status_code == 201

        response = await client.post(
            f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
            json={"name": "test-config", "data": {"target_id": "llama-2-7b"}},
        )
        assert response.status_code == 201

        # Mark workspace for deletion
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.DELETING,
        )

        # Override auth client with service principal
        app.dependency_overrides[get_auth_client] = _create_service_principal_auth_client

        try:
            # Service principal should be able to list entities
            response = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config")
            assert response.status_code == 200
            entities = response.json()
            assert entities["pagination"]["total_results"] == 1

            # Service principal should be able to get entity
            response = await client.get(
                f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config"
            )
            assert response.status_code == 200
            assert response.json()["name"] == "test-config"

            # Service principal should be able to update entity
            response = await client.put(
                f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config",
                json={"data": {"target_id": "llama-3-8b"}},
            )
            assert response.status_code == 200
            assert response.json()["data"]["target_id"] == "llama-3-8b"

            # Service principal should be able to delete entity
            response = await client.delete(
                f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config/test-config"
            )
            assert response.status_code == 200

            # Service principal should be able to create entity
            response = await client.post(
                f"/apis/entities/v2/workspaces/{workspace_name}/entities/customization_config",
                json={"name": "new-config", "data": {"target_id": "llama-2-7b"}},
            )
            assert response.status_code == 201
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    async def test_mark_failed_advances_updated_at(self, client: AsyncClient, repos):
        workspace_name = "mark-failed-bumps-updated-at"
        created = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "updated_at contract"},
        )
        assert created.status_code == 201

        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        before = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert before is not None

        marked = await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.FAILED,
        )
        assert marked is True

        after = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert after is not None
        assert after.updated_at > before.updated_at


def _workspace_names_and_total(list_response) -> tuple[set[str], int]:
    body = list_response.json()
    names = {ws["name"] for ws in body["data"]}
    return names, body["pagination"]["total_results"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceListExcludesDeleting:
    """List must not return workspaces the single-entity GET path already treats as gone."""

    async def test_list_excludes_pending_workspace_after_delete(self, client: AsyncClient):
        workspace_name = "list-hide-pending"
        created = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Pending deletion list test"},
        )
        assert created.status_code == 201

        listed_before = await client.get("/apis/entities/v2/workspaces", params={"page_size": 100})
        assert listed_before.status_code == 200
        names_before, total_before = _workspace_names_and_total(listed_before)
        assert workspace_name in names_before

        deleted = await client.delete(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert deleted.status_code == 200

        listed_after = await client.get("/apis/entities/v2/workspaces", params={"page_size": 100})
        assert listed_after.status_code == 200
        names_after, total_after = _workspace_names_and_total(listed_after)
        assert workspace_name not in names_after
        assert total_after == total_before - 1

        get_response = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert get_response.status_code == 404

    async def test_list_still_shows_live_sibling_after_delete(self, client: AsyncClient):
        keep_name = "list-keep-sibling"
        delete_name = "list-delete-sibling"
        for name in (keep_name, delete_name):
            created = await client.post(
                "/apis/entities/v2/workspaces",
                json={"name": name, "description": "Sibling list test"},
            )
            assert created.status_code == 201

        deleted = await client.delete(f"/apis/entities/v2/workspaces/{delete_name}")
        assert deleted.status_code == 200

        listed = await client.get("/apis/entities/v2/workspaces", params={"page_size": 100})
        assert listed.status_code == 200
        names, _ = _workspace_names_and_total(listed)
        assert keep_name in names
        assert delete_name not in names

    async def test_cleanup_filter_still_finds_pending_workspace(self, client: AsyncClient, repos):
        workspace_name = "cleanup-filter-pending"
        created = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": "Cleanup query contract"},
        )
        assert created.status_code == 201

        deleted = await client.delete(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert deleted.status_code == 200

        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        pending, _ = await workspace_repo.list_workspaces(
            filter_op=ComparisonOperation(
                operator=FilterOperator.EQ,
                field="deletion_stage",
                value=WorkspaceDeletionStage.PENDING,
            ),
            page_size=1,
        )
        assert any(ws.name == workspace_name for ws in pending)

    @pytest.mark.parametrize(
        "stage",
        [WorkspaceDeletionStage.DELETING, WorkspaceDeletionStage.FAILED],
    )
    async def test_list_excludes_deleting_and_failed_workspaces(
        self,
        client: AsyncClient,
        repos,
        stage: WorkspaceDeletionStage,
    ):
        workspace_name = f"list-hide-{stage.value}"
        created = await client.post(
            "/apis/entities/v2/workspaces",
            json={"name": workspace_name, "description": f"{stage.value} list test"},
        )
        assert created.status_code == 201

        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        marked = await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=stage,
        )
        assert marked is True

        listed = await client.get("/apis/entities/v2/workspaces", params={"page_size": 100})
        assert listed.status_code == 200
        names, _ = _workspace_names_and_total(listed)
        assert workspace_name not in names

        get_response = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert get_response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
class TestCreateReportsInProgressDeletion:
    """Create must not describe a hidden cleanup row as a workspace that already exists."""

    async def test_failed_cleanup_conflict_includes_the_stored_reason(self, client: AsyncClient, repos):
        workspace_name = "create-while-cleanup-failed"
        created = await client.post("/apis/entities/v2/workspaces", json={"name": workspace_name})
        assert created.status_code == 201

        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.PENDING,
        )
        pending = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert pending is not None
        claimed = await workspace_repo.claim_workspace_for_deletion(
            name=workspace_name,
            expected_stage=WorkspaceDeletionStage.PENDING,
            expected_updated_at=pending.updated_at,
        )
        assert claimed is not None
        reason = "shared-weights: HTTP 409: Cannot delete fileset because 1 model entity reference(s) still use it"
        assert await workspace_repo.fail_workspace_deletion_if_claimed(
            name=workspace_name,
            expected_updated_at=claimed.updated_at,
            deletion_error=reason,
        )

        conflict = await client.post("/apis/entities/v2/workspaces", json={"name": workspace_name})
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == f"Workspace '{workspace_name}' is still being deleted ({reason})"
        assert "already exists" not in conflict.json()["detail"]

        hidden = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert hidden.status_code == 404

    @pytest.mark.parametrize(
        "stage",
        [WorkspaceDeletionStage.PENDING, WorkspaceDeletionStage.DELETING],
    )
    async def test_in_progress_cleanup_without_an_error_is_still_not_already_exists(
        self,
        client: AsyncClient,
        repos,
        stage: WorkspaceDeletionStage,
    ):
        workspace_name = f"create-while-{stage.value}"
        created = await client.post("/apis/entities/v2/workspaces", json={"name": workspace_name})
        assert created.status_code == 201
        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        assert await workspace_repo.mark_workspace_for_deletion(name=workspace_name, deletion_stage=stage)

        conflict = await client.post("/apis/entities/v2/workspaces", json={"name": workspace_name})
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == f"Workspace '{workspace_name}' is still being deleted"

    async def test_live_duplicate_still_says_already_exists(self, client: AsyncClient):
        created = await client.post("/apis/entities/v2/workspaces", json={"name": "create-live-duplicate"})
        assert created.status_code == 201
        conflict = await client.post("/apis/entities/v2/workspaces", json={"name": "create-live-duplicate"})
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == "Workspace 'create-live-duplicate' already exists"


def _status_auth_client(*, principal_id: str, auth_enabled: bool, has_role: bool) -> MagicMock:
    principal = Principal(id=principal_id, email=principal_id, groups=[], on_behalf_of=None)
    auth_client = MagicMock(spec=AuthClient)
    auth_client.principal = principal
    auth_client.auth_enabled = auth_enabled
    auth_client.has_role = AsyncMock(return_value=has_role)
    return auth_client


@pytest.mark.integration
@pytest.mark.asyncio
class TestWorkspaceDeletionStatus:
    """Deletion status reads the stored cleanup row and does not move its lease."""

    async def test_status_returns_the_stored_error_without_touching_the_lease(self, client: AsyncClient, repos):
        workspace_name = "deletion-status-failed"
        created = await client.post("/apis/entities/v2/workspaces", json={"name": workspace_name})
        assert created.status_code == 201
        created_by = created.json()["created_by"]
        reason = "shared-weights: HTTP 409: Cannot delete fileset because 1 model entity reference(s) still use it"

        missing = await client.get("/apis/entities/v2/workspace-deletions/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Workspace 'does-not-exist' not found"

        live = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert live.status_code == 404
        assert live.json()["detail"] == f"Workspace '{workspace_name}' not found"

        workspace_repo: WorkspaceRepositoryInterface = repos["workspace"]
        assert await workspace_repo.mark_workspace_for_deletion(
            name=workspace_name,
            deletion_stage=WorkspaceDeletionStage.PENDING,
        )
        pending = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert pending.status_code == 200
        assert pending.json()["deletion_stage"] == "pending"
        assert pending.json()["deletion_error"] is None

        current = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert current is not None
        claimed = await workspace_repo.claim_workspace_for_deletion(
            name=workspace_name,
            expected_stage=WorkspaceDeletionStage.PENDING,
            expected_updated_at=current.updated_at,
        )
        assert claimed is not None
        assert await workspace_repo.fail_workspace_deletion_if_claimed(
            name=workspace_name,
            expected_updated_at=claimed.updated_at,
            deletion_error=reason,
        )

        before = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert before is not None
        failed = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert failed.status_code == 200
        body = failed.json()
        assert body["name"] == workspace_name
        assert body["deletion_stage"] == "failed"
        assert body["deletion_error"] == reason
        assert "description" not in body
        after = await workspace_repo.get_workspace_by_name(name=workspace_name)
        assert after is not None
        assert after.updated_at == before.updated_at
        assert after._deletion_stage == WorkspaceDeletionStage.FAILED
        assert after._deletion_error == reason

        hidden = await client.get(f"/apis/entities/v2/workspaces/{workspace_name}")
        assert hidden.status_code == 404

        app.dependency_overrides[get_auth_client] = lambda: _status_auth_client(
            principal_id="other@example.com",
            auth_enabled=True,
            has_role=False,
        )
        denied = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert denied.status_code == 404
        assert denied.json()["detail"] == f"Workspace '{workspace_name}' not found"

        app.dependency_overrides[get_auth_client] = lambda: _status_auth_client(
            principal_id=created_by,
            auth_enabled=True,
            has_role=False,
        )
        allowed = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert allowed.status_code == 200
        assert allowed.json()["deletion_error"] == reason

        app.dependency_overrides[get_auth_client] = lambda: _status_auth_client(
            principal_id="admin@example.com",
            auth_enabled=True,
            has_role=True,
        )
        admin = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert admin.status_code == 200
        assert admin.json()["deletion_stage"] == "failed"

        app.dependency_overrides[get_auth_client] = lambda: _status_auth_client(
            principal_id="service:entities",
            auth_enabled=True,
            has_role=False,
        )
        service = await client.get(f"/apis/entities/v2/workspace-deletions/{workspace_name}")
        assert service.status_code == 200
        assert service.json()["deletion_error"] == reason
