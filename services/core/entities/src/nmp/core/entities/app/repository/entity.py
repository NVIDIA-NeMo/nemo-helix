# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Entity operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from nmp.common.api.filter import FilterOperation
from nmp.core.entities.entities import Entity
from sqlalchemy.ext.asyncio import AsyncSession


class EntityRepositoryInterface(ABC):
    """Abstract interface for Entity repository operations.

    The entity repository is schema-agnostic - it treats entity data as opaque.
    """

    @abstractmethod
    async def create_entity(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        data: Dict[str, Any],
        parent: str | None = None,
        project: str | None = None,
        created_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Entity:
        """Create a new entity.

        Args:
            workspace: Workspace name/identifier
            entity_type: Entity type (e.g., 'customization_config')
            name: Entity name (must be unique within workspace+entity_type+parent)
            data: Entity-specific data (JSON)
            parent: Optional parent entity ID for nested entities
            project: Optional project name for the entity
            session: Optional database session

        Returns:
            Created entity with ID and timestamps
        """
        pass

    @abstractmethod
    async def get_entity_by_id(
        self,
        *,
        entity_id: str,
        session: AsyncSession | None = None,
    ) -> Optional[Entity]:
        """Get an entity by id."""
        pass

    @abstractmethod
    async def get_entity_by_name(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        parent: Optional[str] = None,
        session: AsyncSession | None = None,
    ) -> Optional[Entity]:
        """Get an entity by workspace, type, and name, optionally scoped to a parent.

        Args:
            workspace: Workspace name/identifier
            entity_type: Entity type
            name: Entity name
            parent: Optional parent entity ID (None for root entities)

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_entities(
        self,
        *,
        workspace: str,
        entity_type: str | None = None,
        page: int = 1,
        page_size: int = 100,
        sort: Optional[str] = None,
        filter_op: Optional[FilterOperation] = None,
        relationship_child_workspaces: Optional[set[str]] = None,
        session: AsyncSession | None = None,
    ) -> tuple[List[Entity], int]:
        """List entities with optional filtering.

        Args:
            workspace: Workspace to filter by, or "*" to query all workspaces
            entity_type: Entity type to list, or None to list all types
            page: Page number (1-indexed)
            page_size: Number of items per page
            sort: Sort field (prefix with - for descending)
            filter_op: Filter operation for filtering (can include workspace filter)
            relationship_child_workspaces: If set, relationship/EXISTS filters only consider
                child entities in these workspaces. If None, child workspace is not restricted
                (full access / same as parent scope when auth is off).
            session: Optional database session

        Returns:
            Tuple of (entities, total_count)
        """
        pass

    @abstractmethod
    async def count_entities_by(
        self,
        *,
        workspace: str,
        entity_type: str,
        group_by: str,
        filter_op: FilterOperation | None = None,
        relationship_child_workspaces: set[str] | None = None,
        session: AsyncSession | None = None,
    ) -> dict[str, int]:
        """Count filtered entities grouped by a direct string data field."""
        pass

    @abstractmethod
    async def distinct_child_workspaces(
        self,
        *,
        parent_id: str,
        session: AsyncSession | None = None,
    ) -> set[str]:
        """Return every distinct workspace holding a child of *parent_id*.

        Exact regardless of child count, unlike paging the children themselves: callers
        need the set of affected workspaces, not the rows.
        """
        pass

    @abstractmethod
    async def update_entity(
        self,
        *,
        entity_id: str,
        data: Dict[str, Any],
        name: str | None = None,
        project: str | None = None,
        updated_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Entity:
        """Update an entity by ID, optionally changing its name."""
        pass

    @abstractmethod
    async def update_entity_by_name(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        data: Dict[str, Any],
        new_name: str | None = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        updated_by: str | None = None,
        expected_db_version: Optional[int] = None,
        session: AsyncSession | None = None,
    ) -> Entity:
        """Update an entity by name, optionally changing its name.

        Args:
            workspace: Workspace name/identifier
            entity_type: Entity type
            name: Current entity name
            data: New entity data
            new_name: Optional new name for the entity
            parent: Optional parent entity ID (None for root entities)
            project: Optional project name for the entity
            updated_by: Optional principal ID for the updater
            refuse_children_outside: When set to a workspace name, refuse the delete with
                ForeignChildEntitiesError if the entity has children in any other workspace.
                Checked inside the delete transaction with the row locked, so it cannot be
                raced by a child created after the check.
            expected_db_version: Optional expected database version for optimistic locking. If provided,
                update will fail if current version doesn't match.

        Returns:
            Updated entity

        Raises:
            ValueError: Entity not found or version mismatch
        """
        pass

    @abstractmethod
    async def delete_entity(
        self,
        *,
        entity_id: str,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete an entity by ID."""
        pass

    @abstractmethod
    async def delete_entity_by_name(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        parent: Optional[str] = None,
        expected_db_version: Optional[int] = None,
        refuse_children_outside: str | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete an entity by name.

        Args:
            workspace: Workspace name/identifier
            entity_type: Entity type
            name: Entity name
            parent: Optional parent entity ID (None for root entities)
            expected_db_version: Optional expected database version for optimistic locking. If provided,
                delete only succeeds when the stored entity still has this version.

        Returns:
            Number of deleted entities (0 or 1)
        """
        pass
