# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth service seeding logic."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.common.entities import SYSTEM_WORKSPACE, EntityConflictError, EntityNotFoundError
from nhx.core.auth.app.seeding import (
    DEFAULT_WORKSPACE_ROLE,
    PLATFORM_ADMIN_ROLE,
    SYSTEM_WORKSPACE_ROLE,
    WILDCARD_PRINCIPAL,
    WORKSPACE_CREATOR_ROLE,
    _generate_binding_name,
    run_seeding,
    seed_default_workspace_editor,
    seed_platform_admin,
    seed_system_workspace_viewer,
    seed_workspace_creator,
)
from nhx.core.auth.entities import RoleBindingEntity


class TestGenerateBindingName:
    """Tests for binding name generation."""

    def test_basic_email(self):
        """Test binding name generation with a basic email."""
        name = _generate_binding_name("user@example.com", "system", "HelixAdmin")
        assert name == "user-example-com-system-helixadmin"

    def test_complex_email(self):
        """Test binding name generation with a complex email."""
        name = _generate_binding_name("admin.user@sub.domain.nvidia.com", "default", "Admin")
        assert name == "admin-user-sub-domain-nvidia-com-default-admin"

    def test_wildcard_principal(self):
        """Test binding name generation with wildcard principal."""
        name = _generate_binding_name("*", "default", "Editor")
        assert name == "wildcard-default-editor"

    def test_wildcard_principal_system_workspace(self):
        """Test binding name generation with wildcard principal for system workspace."""
        name = _generate_binding_name("*", "system", "Viewer")
        assert name == "wildcard-system-viewer"


class TestSeedHelixAdmin:
    """Tests for platform admin seeding."""

    @pytest.fixture
    def mock_entity_client(self):
        """Create a mock entity client with get raising NotFound (new binding case)."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        client.create = AsyncMock()
        return client

    @pytest.fixture
    def mock_config_with_admin(self):
        """Mock config with admin_email set."""
        config = MagicMock()
        config.admin_email = "admin@nvidia.com"
        return config

    @pytest.fixture
    def mock_config_no_admin(self):
        """Mock config without admin_email."""
        config = MagicMock()
        config.admin_email = None
        return config

    @pytest.mark.asyncio
    async def test_seed_creates_role_binding(self, mock_entity_client, mock_config_with_admin):
        """Test that seeding creates the role binding entity when it doesn't exist."""
        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await seed_platform_admin(mock_entity_client)

        assert result is True
        mock_entity_client.get.assert_called_once()  # Check-first
        mock_entity_client.create.assert_called_once()

        # Verify the created entity
        created_entity = mock_entity_client.create.call_args[0][0]
        assert isinstance(created_entity, RoleBindingEntity)
        assert created_entity.workspace == SYSTEM_WORKSPACE
        assert created_entity.principal == "admin@nvidia.com"
        assert created_entity.role == PLATFORM_ADMIN_ROLE
        assert created_entity.granted_by == "system"
        assert created_entity.revoked_at is None

    @pytest.mark.asyncio
    async def test_seed_handles_existing_binding_via_get(self, mock_config_with_admin):
        """Test that seeding returns True when binding already exists (found via get)."""
        mock_entity_client = MagicMock()
        # Return existing active binding
        existing_binding = MagicMock()
        existing_binding.revoked_at = None
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await seed_platform_admin(mock_entity_client)

        assert result is True
        mock_entity_client.get.assert_called_once()
        mock_entity_client.create.assert_not_called()  # Should not create if exists

    @pytest.mark.asyncio
    async def test_seed_handles_concurrent_creation(self, mock_entity_client, mock_config_with_admin):
        """Test that seeding handles concurrent creation by another instance."""
        # get returns NotFound, but create raises Conflict (another instance created it)
        mock_entity_client.create.side_effect = EntityConflictError("Already exists")

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await seed_platform_admin(mock_entity_client)

        assert result is True  # Should return True - binding exists

    @pytest.mark.asyncio
    async def test_seed_returns_false_for_revoked_binding(self, mock_config_with_admin):
        """Test that seeding returns False when binding exists but is revoked."""
        mock_entity_client = MagicMock()
        # Return revoked binding
        existing_binding = MagicMock()
        existing_binding.revoked_at = datetime.now(timezone.utc)
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await seed_platform_admin(mock_entity_client)

        assert result is False  # Revoked binding shouldn't count as success

    @pytest.mark.asyncio
    async def test_seed_skips_when_no_admin_email(self, mock_entity_client, mock_config_no_admin):
        """Test that seeding is skipped when no admin_email is configured."""
        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_no_admin):
            result = await seed_platform_admin(mock_entity_client)

        assert result is False
        mock_entity_client.get.assert_not_called()
        mock_entity_client.create.assert_not_called()


class TestSeedDefaultWorkspaceEditor:
    """Tests for default workspace Editor seeding."""

    @pytest.fixture
    def mock_entity_client(self):
        """Create a mock entity client with get raising NotFound (new binding case)."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        client.create = AsyncMock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Mock config with default_workspace set."""
        config = MagicMock()
        config.default_workspace = "default"
        return config

    @pytest.mark.asyncio
    async def test_seed_creates_wildcard_editor_binding(self, mock_entity_client, mock_config):
        """Test that seeding creates the wildcard Editor binding."""
        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config):
            result = await seed_default_workspace_editor(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_called_once()

        # Verify the created entity
        created_entity = mock_entity_client.create.call_args[0][0]
        assert isinstance(created_entity, RoleBindingEntity)
        assert created_entity.workspace == "default"
        assert created_entity.principal == WILDCARD_PRINCIPAL
        assert created_entity.role == DEFAULT_WORKSPACE_ROLE
        assert created_entity.granted_by == "system"

    @pytest.mark.asyncio
    async def test_seed_handles_existing_binding(self, mock_config):
        """Test that seeding returns True when binding already exists."""
        mock_entity_client = MagicMock()
        existing_binding = MagicMock()
        existing_binding.revoked_at = None
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config):
            result = await seed_default_workspace_editor(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_uses_configured_workspace(self, mock_entity_client):
        """Test that seeding uses the configured default_workspace."""
        mock_config = MagicMock()
        mock_config.default_workspace = "my-custom-workspace"

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config):
            result = await seed_default_workspace_editor(mock_entity_client)

        assert result is True
        created_entity = mock_entity_client.create.call_args[0][0]
        assert created_entity.workspace == "my-custom-workspace"

    @pytest.mark.asyncio
    async def test_seed_treats_revoked_binding_as_intentional_override(self, mock_config):
        """Test that a revoked wildcard binding is not recreated or treated as failure."""
        mock_entity_client = MagicMock()
        existing_binding = MagicMock()
        existing_binding.revoked_at = datetime.now(timezone.utc)
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config):
            result = await seed_default_workspace_editor(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()


class TestSeedSystemWorkspaceViewer:
    """Tests for system workspace Viewer seeding."""

    @pytest.fixture
    def mock_entity_client(self):
        """Create a mock entity client with get raising NotFound (new binding case)."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        client.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_seed_creates_wildcard_viewer_binding(self, mock_entity_client):
        """Test that seeding creates the wildcard Viewer binding for system workspace."""
        result = await seed_system_workspace_viewer(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_called_once()

        # Verify the created entity
        created_entity = mock_entity_client.create.call_args[0][0]
        assert isinstance(created_entity, RoleBindingEntity)
        assert created_entity.workspace == SYSTEM_WORKSPACE
        assert created_entity.principal == WILDCARD_PRINCIPAL
        assert created_entity.role == SYSTEM_WORKSPACE_ROLE
        assert created_entity.granted_by == "system"

    @pytest.mark.asyncio
    async def test_seed_handles_existing_binding(self):
        """Test that seeding returns True when binding already exists."""
        mock_entity_client = MagicMock()
        existing_binding = MagicMock()
        existing_binding.revoked_at = None
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        result = await seed_system_workspace_viewer(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_handles_concurrent_creation(self, mock_entity_client):
        """Test that seeding handles concurrent creation by another instance."""
        mock_entity_client.create.side_effect = EntityConflictError("Already exists")

        result = await seed_system_workspace_viewer(mock_entity_client)

        assert result is True


class TestSeedWorkspaceCreator:
    """Tests for system workspace WorkspaceCreator seeding."""

    @pytest.fixture
    def mock_entity_client(self):
        """Create a mock entity client with get raising NotFound (new binding case)."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        client.create = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_seed_creates_wildcard_workspace_creator_binding(self, mock_entity_client):
        """Test that seeding creates the wildcard WorkspaceCreator binding."""
        result = await seed_workspace_creator(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_called_once()

        created_entity = mock_entity_client.create.call_args[0][0]
        assert isinstance(created_entity, RoleBindingEntity)
        assert created_entity.workspace == SYSTEM_WORKSPACE
        assert created_entity.principal == WILDCARD_PRINCIPAL
        assert created_entity.role == WORKSPACE_CREATOR_ROLE
        assert created_entity.name == "wildcard-system-workspacecreator"
        assert created_entity.granted_by == "system"

    @pytest.mark.asyncio
    async def test_seed_handles_existing_binding(self):
        """Test that seeding returns True when binding already exists."""
        mock_entity_client = MagicMock()
        existing_binding = MagicMock()
        existing_binding.revoked_at = None
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        result = await seed_workspace_creator(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_treats_revoked_binding_as_intentional_override(self):
        """Test that a revoked WorkspaceCreator wildcard binding is preserved."""
        mock_entity_client = MagicMock()
        existing_binding = MagicMock()
        existing_binding.revoked_at = datetime.now(timezone.utc)
        mock_entity_client.get = AsyncMock(return_value=existing_binding)
        mock_entity_client.create = AsyncMock()

        result = await seed_workspace_creator(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()


class TestRunSeeding:
    """Tests for the main seeding entry point."""

    @pytest.fixture
    def mock_entity_client(self):
        """Create a mock entity client."""
        client = MagicMock()
        client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        client.create = AsyncMock()
        return client

    @pytest.fixture
    def mock_config_with_admin(self):
        """Mock config with admin_email and default_workspace set."""
        config = MagicMock()
        config.admin_email = "admin@test.com"
        config.default_workspace = "default"
        return config

    @pytest.fixture
    def mock_config_no_admin(self):
        """Mock config without admin_email."""
        config = MagicMock()
        config.admin_email = None
        config.default_workspace = "default"
        return config

    @pytest.mark.asyncio
    async def test_run_seeding_returns_true_on_success(self, mock_entity_client, mock_config_with_admin):
        """Test that run_seeding returns True when all seeding succeeds."""
        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await run_seeding(mock_entity_client)

        assert result is True
        # Should create: platform admin + default workspace editor + system workspace viewer + workspace creator
        assert mock_entity_client.create.call_count == 4

    @pytest.mark.asyncio
    async def test_run_seeding_seeds_wildcard_bindings_without_admin(self, mock_entity_client, mock_config_no_admin):
        """Test that run_seeding seeds wildcard bindings even without admin_email."""
        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_no_admin):
            result = await run_seeding(mock_entity_client)

        assert result is True
        # Should create: default workspace editor + system workspace viewer + workspace creator (no platform admin)
        assert mock_entity_client.create.call_count == 3

    @pytest.mark.asyncio
    async def test_run_seeding_returns_false_on_platform_admin_failure(self, mock_config_with_admin):
        """Test that run_seeding returns False when platform admin seeding fails."""
        mock_entity_client = MagicMock()
        mock_entity_client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        mock_entity_client.create = AsyncMock(side_effect=Exception("Database error"))

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_with_admin):
            result = await run_seeding(mock_entity_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_run_seeding_returns_false_on_wildcard_failure(self, mock_config_no_admin):
        """Test that run_seeding returns False when wildcard seeding fails."""
        mock_entity_client = MagicMock()
        mock_entity_client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))
        mock_entity_client.create = AsyncMock(side_effect=Exception("Database error"))

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_no_admin):
            result = await run_seeding(mock_entity_client)

        assert result is False

    @pytest.mark.asyncio
    async def test_run_seeding_succeeds_when_workspace_creator_binding_was_revoked(self, mock_config_no_admin):
        """Test that a revoked wildcard binding is treated as an operator override."""
        mock_entity_client = MagicMock()
        wildcard_editor = MagicMock()
        wildcard_editor.revoked_at = None
        wildcard_viewer = MagicMock()
        wildcard_viewer.revoked_at = None
        wildcard_creator = MagicMock()
        wildcard_creator.revoked_at = datetime.now(timezone.utc)
        mock_entity_client.get = AsyncMock(side_effect=[wildcard_editor, wildcard_viewer, wildcard_creator])
        mock_entity_client.create = AsyncMock()

        with patch("nhx.core.auth.app.seeding.get_service_config", return_value=mock_config_no_admin):
            result = await run_seeding(mock_entity_client)

        assert result is True
        mock_entity_client.create.assert_not_called()
