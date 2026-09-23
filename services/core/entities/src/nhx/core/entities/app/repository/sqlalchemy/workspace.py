# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy implementation of Workspace repository."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from nhx.common.api.filter import FilterOperation
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBWorkspace
from nhx.core.entities.app.repository.workspace import WorkspaceRepositoryInterface
from nhx.core.entities.entities import Workspace
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


class SQLAlchemyWorkspaceRepository(WorkspaceRepositoryInterface):
    """SQLAlchemy implementation of Workspace repository."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker

    @asynccontextmanager
    async def _get_session(self, session: AsyncSession | None) -> AsyncIterator[AsyncSession]:
        """Get or create a database session."""
        if session is not None:
            yield session
        else:
            async with self.session_maker() as new_session:
                yield new_session

    async def create_workspace(
        self,
        *,
        name: str,
        description: str | None = None,
        created_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Workspace:
        """Create a new workspace with auto-generated ID."""
        async with self._get_session(session) as sess:
            # Generate a compound ID (workspace-{base58_uuid})
            workspace_id = generate_entity_id("workspace")
            db_workspace = DBWorkspace(
                id=workspace_id,
                name=name,
                description=description,
                created_by=created_by,
            )
            sess.add(db_workspace)
            await sess.commit()
            await sess.refresh(db_workspace)

            return db_workspace.to_pydantic()

    async def get_workspace_by_name(
        self,
        *,
        name: str,
        session: AsyncSession | None = None,
    ) -> Optional[Workspace]:
        """Get a workspace by name."""
        async with self._get_session(session) as sess:
            result = await sess.execute(select(DBWorkspace).where(DBWorkspace.name == name))
            db_workspace = result.scalar_one_or_none()

            if db_workspace is None:
                return None

            return db_workspace.to_pydantic()

    async def list_workspaces(
        self,
        *,
        session: AsyncSession | None = None,
        page: int = 1,
        page_size: int = 100,
        sort: Optional[str] = None,
        filter_op: Optional[FilterOperation] = None,
    ) -> tuple[List[Workspace], int]:
        """List workspaces with optional filtering."""
        async with self._get_session(session) as sess:
            query = select(DBWorkspace)

            if filter_op is not None:
                filter_repo = SQLAlchemyFilterRepository(DBWorkspace)
                query = query.where(filter_op.apply(filter_repo))

            count_query = select(func.count()).select_from(query.subquery())
            total_result = await sess.execute(count_query)
            total = total_result.scalar() or 0

            if sort is not None:
                reverse = sort.startswith("-")
                sort_key = sort.lstrip("-")
                query = query.order_by(
                    getattr(DBWorkspace, sort_key).desc() if reverse else getattr(DBWorkspace, sort_key).asc()
                )
            else:
                query = query.order_by(DBWorkspace.created_at.desc())

            query = query.offset((page - 1) * page_size).limit(page_size)

            result = await sess.execute(query)
            db_workspaces = result.scalars().all()
            workspaces = [ws.to_pydantic() for ws in db_workspaces]

            return workspaces, total

    async def update_workspace(
        self,
        *,
        name: str,
        description: str | None = None,
        updated_by: str | None = None,
        session: AsyncSession | None = None,
    ) -> Optional[Workspace]:
        """Update a workspace by name."""
        async with self._get_session(session) as sess:
            result = await sess.execute(select(DBWorkspace).where(DBWorkspace.name == name))
            db_workspace = result.scalar_one_or_none()

            if db_workspace is None:
                return None

            if description is not None:
                db_workspace.description = description

            if updated_by is not None:
                db_workspace.updated_by = updated_by

            await sess.commit()
            await sess.refresh(db_workspace)

            return db_workspace.to_pydantic()

    async def mark_workspace_for_deletion(
        self,
        *,
        name: str,
        deletion_stage: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Mark a workspace for deletion by setting its deletion_stage."""
        async with self._get_session(session) as sess:
            result = await sess.execute(
                update(DBWorkspace)
                .where(
                    (DBWorkspace.name == name)
                    & (DBWorkspace.deletion_stage.is_(None) | (DBWorkspace.deletion_stage != deletion_stage))
                )
                .values(deletion_stage=deletion_stage)
                .returning(DBWorkspace)
            )
            db_workspace = result.scalar_one_or_none()
            if db_workspace is None:
                return False
            await sess.commit()

            return True

    async def delete_workspace(
        self,
        *,
        name: str,
        session: AsyncSession | None = None,
    ) -> bool:
        """Delete a workspace by name."""
        async with self._get_session(session) as sess:
            result = await sess.execute(select(DBWorkspace).where(DBWorkspace.name == name))
            db_workspace = result.scalar_one_or_none()

            if db_workspace is None:
                return False

            await sess.delete(db_workspace)
            await sess.commit()

            return True
