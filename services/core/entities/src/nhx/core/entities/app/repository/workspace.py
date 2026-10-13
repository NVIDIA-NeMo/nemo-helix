# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Workspace operations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from nhx.common.api.filter import FilterOperation
from nhx.core.entities.entities import Workspace
from sqlalchemy.ext.asyncio import AsyncSession


class WorkspaceRepositoryInterface(ABC):
    """Abstract interface for Workspace repository operations."""

    @abstractmethod
    async def create_workspace(
        self,
        *,
        name: str,
        description: str | None = None,
        created_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Workspace:
        """Create a new workspace with auto-generated ID."""
        pass

    @abstractmethod
    async def get_workspace_by_name(
        self,
        *,
        name: str,
        session: AsyncSession | None = None,
    ) -> Optional[Workspace]:
        """Get a workspace by name."""
        pass

    @abstractmethod
    async def list_workspaces(
        self,
        *,
        session: AsyncSession | None = None,
        page: int = 1,
        page_size: int = 100,
        sort: Optional[str] = None,
        filter_op: Optional[FilterOperation] = None,
    ) -> tuple[List[Workspace], int]:
        """List workspaces with optional filtering.

        Args:
            session: Optional database session
            page: Page number (1-indexed)
            page_size: Number of items per page
            sort: Sort field (prefix with - for descending)
            filter_op: Filter operation for filtering (can include name filter for access control)

        Returns:
            Tuple of (workspaces, total_count)
        """
        pass

    @abstractmethod
    async def update_workspace(
        self,
        *,
        name: str,
        description: str | None = None,
        updated_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Optional[Workspace]:
        """Update a workspace by name."""
        pass

    @abstractmethod
    async def mark_workspace_for_deletion(
        self,
        *,
        name: str,
        deletion_stage: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Mark a workspace for deletion by setting its deletion_stage.

        Args:
            name: Workspace name
            deletion_stage: The deletion stage to set (e.g., 'pending', 'deleting', 'failed')
            session: Optional database session

        Returns:
            True if updated, False if not found/no changes made
        """
        pass

    @abstractmethod
    async def delete_workspace(
        self,
        *,
        name: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Delete a workspace by name."""
        pass
