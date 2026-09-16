# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth service seeding logic.

This module handles bootstrapping authorization data such as:

- Platform admin role binding (from NHX_AUTH_ADMIN_EMAIL)
- Wildcard principal (*) bindings for default workspace access

Idempotency:
    All seeding operations are idempotent and safe to run concurrently across
    multiple instances. The pattern used is "create-and-handle-conflict":

    1. Attempt to create the entity
    2. If EntityConflictError (already exists), treat as success
    3. If other error, log warning and continue

    This approach is preferred over "check-then-create" because it avoids
    race conditions where multiple instances might simultaneously see the
    entity as missing and attempt to create it.
"""

import logging
from datetime import datetime, timezone

from nhx.common.config import get_service_config
from nhx.common.entities import SYSTEM_WORKSPACE, EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.entities import RoleBindingEntity

logger = logging.getLogger(__name__)

# Constants
PLATFORM_ADMIN_ROLE = "PlatformAdmin"
WILDCARD_PRINCIPAL = "*"
DEFAULT_WORKSPACE_ROLE = "Editor"
SYSTEM_WORKSPACE_ROLE = "Viewer"
WORKSPACE_CREATOR_ROLE = "WorkspaceCreator"


def _generate_binding_name(principal: str, workspace: str, role: str) -> str:
    """Generate a deterministic name for a role binding.

    Uses a consistent format to ensure the same binding always has the same name,
    which is critical for idempotent seeding across multiple instances.

    Format: {sanitized_principal}-{workspace}-{role}
    """
    # Handle wildcard principal specially
    if principal == WILDCARD_PRINCIPAL:
        sanitized = "wildcard"
    else:
        # Sanitize principal (replace @ and . with -)
        sanitized = principal.replace("@", "-").replace(".", "-")
    return f"{sanitized}-{workspace}-{role}".lower()


async def seed_platform_admin(entity_client: EntityClient) -> bool:
    """Seed the platform admin role binding if configured.

    Creates a PlatformAdmin role binding in the system workspace for the
    configured admin email (NHX_AUTH_ADMIN_EMAIL).

    This operation is idempotent:

    - If binding doesn't exist: creates it
    - If binding already exists: logs and returns success
    - Safe to run concurrently on multiple instances

    Args:
        entity_client: EntityClient for creating the role binding

    Returns:
        True if binding exists (created or already present), False if not configured
        or if creation failed.
    """
    config = get_service_config(AuthServiceConfig)
    admin_email = config.admin_email

    if not admin_email:
        logger.debug("No admin_email configured, skipping platform admin seeding")
        return False

    binding_name = _generate_binding_name(admin_email, SYSTEM_WORKSPACE, PLATFORM_ADMIN_ROLE)

    # Check if binding already exists (fast path for restarts)
    try:
        existing = await entity_client.get(
            RoleBindingEntity,
            name=binding_name,
            workspace=SYSTEM_WORKSPACE,
        )
        # Binding exists - verify it's not revoked
        if existing.revoked_at is None:
            logger.debug(
                "Platform admin role binding already exists",
                extra={
                    "principal": admin_email,
                    "workspace": SYSTEM_WORKSPACE,
                    "role": PLATFORM_ADMIN_ROLE,
                },
            )
            return True
        else:
            # Binding was revoked - this is unexpected for bootstrap admin
            logger.warning(
                "Platform admin role binding exists but is revoked",
                extra={
                    "principal": admin_email,
                    "workspace": SYSTEM_WORKSPACE,
                    "role": PLATFORM_ADMIN_ROLE,
                    "revoked_at": existing.revoked_at.isoformat() if existing.revoked_at else None,
                },
            )
            return False
    except EntityNotFoundError:
        # Binding doesn't exist - create it
        pass

    # Create the role binding entity
    granted_at = datetime.now(timezone.utc)
    binding = RoleBindingEntity(
        name=binding_name,
        workspace=SYSTEM_WORKSPACE,
        principal=admin_email,
        role=PLATFORM_ADMIN_ROLE,
        granted_by="system",  # Bootstrap grant
        granted_at=granted_at,
        revoked_at=None,
    )

    try:
        await entity_client.create(binding)
        logger.info(
            "Created platform admin role binding",
            extra={
                "principal": admin_email,
                "workspace": SYSTEM_WORKSPACE,
                "role": PLATFORM_ADMIN_ROLE,
            },
        )
        return True
    except EntityConflictError:
        # Another instance created it concurrently - this is fine
        logger.debug(
            "Platform admin role binding created by another instance",
            extra={
                "principal": admin_email,
                "workspace": SYSTEM_WORKSPACE,
                "role": PLATFORM_ADMIN_ROLE,
            },
        )
        return True
    except Exception as e:
        # Unexpected error - log but don't fail startup
        logger.warning(
            "Failed to seed platform admin role binding: %s",
            e,
            extra={
                "principal": admin_email,
                "workspace": SYSTEM_WORKSPACE,
                "role": PLATFORM_ADMIN_ROLE,
            },
        )
        return False


async def _seed_wildcard_binding(
    entity_client: EntityClient,
    workspace: str,
    role: str,
    description: str,
) -> bool:
    """Helper to seed a wildcard principal binding.

    Creates a role binding for the wildcard principal "*" which grants
    access to all authenticated users.

    Args:
        entity_client: EntityClient for creating the role binding
        workspace: The workspace to grant access to
        role: The role to grant (e.g., "Editor", "Viewer")
        description: Human-readable description for logging

    Returns:
        True if binding exists, was intentionally revoked, or was created successfully.
        False only if creation fails unexpectedly.
    """
    binding_name = _generate_binding_name(WILDCARD_PRINCIPAL, workspace, role)

    # Check if binding already exists (fast path for restarts)
    try:
        existing = await entity_client.get(
            RoleBindingEntity,
            name=binding_name,
            workspace=workspace,
        )
        if existing.revoked_at is None:
            logger.debug(
                f"Wildcard {description} binding already exists",
                extra={"workspace": workspace, "role": role},
            )
            return True
        else:
            logger.info(
                f"Wildcard {description} binding is revoked; preserving operator override",
                extra={"workspace": workspace, "role": role},
            )
            return True
    except EntityNotFoundError:
        pass

    # Create the binding
    binding = RoleBindingEntity(
        name=binding_name,
        workspace=workspace,
        principal=WILDCARD_PRINCIPAL,
        role=role,
        granted_by="system",
        granted_at=datetime.now(timezone.utc),
        revoked_at=None,
    )

    try:
        await entity_client.create(binding)
        logger.info(
            f"Created wildcard {description} binding",
            extra={"workspace": workspace, "role": role},
        )
        return True
    except EntityConflictError:
        logger.debug(
            f"Wildcard {description} binding created by another instance",
            extra={"workspace": workspace, "role": role},
        )
        return True
    except Exception as e:
        logger.warning(
            f"Failed to seed wildcard {description} binding: {e}",
            extra={"workspace": workspace, "role": role},
        )
        return False


async def seed_default_workspace_editor(entity_client: EntityClient) -> bool:
    """Seed Editor role binding for wildcard principal on default workspace.

    This gives all authenticated users Editor access to the default workspace,
    enabling them to create and manage resources without explicit role assignment.

    Args:
        entity_client: EntityClient for creating the role binding

    Returns:
        True if binding exists (created or already present), False if creation failed.
    """
    config = get_service_config(AuthServiceConfig)
    workspace = config.default_workspace

    return await _seed_wildcard_binding(
        entity_client,
        workspace=workspace,
        role=DEFAULT_WORKSPACE_ROLE,
        description="default workspace Editor",
    )


async def seed_system_workspace_viewer(entity_client: EntityClient) -> bool:
    """Seed Viewer role binding for wildcard principal on system workspace.

    This gives all authenticated users read-only access to the system workspace,
    enabling them to view platform-level resources and configurations.

    Args:
        entity_client: EntityClient for creating the role binding

    Returns:
        True if binding exists (created or already present), False if creation failed.
    """
    return await _seed_wildcard_binding(
        entity_client,
        workspace=SYSTEM_WORKSPACE,
        role=SYSTEM_WORKSPACE_ROLE,
        description="system workspace Viewer",
    )


async def seed_workspace_creator(entity_client: EntityClient) -> bool:
    """Seed WorkspaceCreator role binding for wildcard principal on system workspace.

    This preserves the current product behavior: any authenticated user can create
    workspaces by default. Operators can later revoke or replace this binding to
    restrict creation to specific users or groups.
    """
    return await _seed_wildcard_binding(
        entity_client,
        workspace=SYSTEM_WORKSPACE,
        role=WORKSPACE_CREATOR_ROLE,
        description="system workspace WorkspaceCreator",
    )


async def run_seeding(entity_client: EntityClient) -> bool:
    """Run all seeding operations.

    This is the main entry point for auth service seeding. All operations
    are idempotent and safe to run concurrently across multiple instances.

    Args:
        entity_client: EntityClient for creating entities

    Returns:
        True if all seeding operations succeeded, False if any required seeding failed.
    """
    logger.debug("Running auth service seeding...")

    # Seed platform admin (required if admin_email is configured)
    config = get_service_config(AuthServiceConfig)
    if config.admin_email:
        if not await seed_platform_admin(entity_client):
            logger.error("Failed to seed platform admin - this is required when admin_email is configured")
            return False

    # Seed wildcard bindings for default access
    if not await seed_default_workspace_editor(entity_client):
        logger.error("Failed to seed default workspace Editor for wildcard principal")
        return False

    if not await seed_system_workspace_viewer(entity_client):
        logger.error("Failed to seed system workspace Viewer for wildcard principal")
        return False

    if not await seed_workspace_creator(entity_client):
        logger.error("Failed to seed system workspace WorkspaceCreator for wildcard principal")
        return False

    logger.debug("Auth service seeding complete")
    return True
