# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy implementation of Entity repository."""

from asyncio import Lock
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from nhx.common.api.filter import FilterOperation
from nhx.common.entities import ALL_WORKSPACES
from nhx.core.entities.app.repository.entity import EntityRepositoryInterface
from nhx.core.entities.app.repository.exceptions import (
    EntityNotFoundError,
    EntityVersionConflictError,
    ForeignChildEntitiesError,
)
from nhx.core.entities.app.repository.sqlalchemy.filter import SQLAlchemyFilterRepository
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity
from nhx.core.entities.entities import Entity
from nhx.core.entities.utils.identifiers import generate_entity_id
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.exc import StaleDataError

MAX_GROUP_COUNT_ROWS = 1000


class SQLAlchemyEntityRepository(EntityRepositoryInterface):
    """SQLAlchemy implementation of Entity repository."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker
        self._write_lock = Lock()

    def _is_sqlite(self, session: AsyncSession) -> bool:
        bind = session.get_bind()
        return bind.dialect.name == "sqlite"

    @asynccontextmanager
    async def _get_session(
        self, session: AsyncSession | None, *, for_write: bool = False
    ) -> AsyncIterator[AsyncSession]:
        """Get or create a database session."""
        if session is not None:
            if for_write and self._is_sqlite(session):
                async with self._write_lock:
                    yield session
            else:
                yield session
        else:
            async with self.session_maker() as new_session:
                if for_write and self._is_sqlite(new_session):
                    async with self._write_lock:
                        yield new_session
                else:
                    yield new_session

    def _get_sort_key(self, sort: str) -> Any:
        reverse = sort.startswith("-")
        sort_key = sort.lstrip("-")

        if sort_key.startswith("data."):
            path = sort_key.split(".")[1:]
            result = DBEntity.data
            for key in path:
                result = result[key]
            return result.desc() if reverse else result.asc()

        if hasattr(DBEntity, sort_key):
            return getattr(DBEntity, sort_key).desc() if reverse else getattr(DBEntity, sort_key).asc()

        raise ValueError(f"Invalid sort field {sort_key}")

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
        """Create a new entity."""
        async with self._get_session(session, for_write=True) as sess:
            entity_id = generate_entity_id(entity_type)

            db_entity = DBEntity(
                id=entity_id,
                workspace=workspace,
                entity_type=entity_type,
                name=name,
                parent=parent,
                project=project,
                created_by=created_by,
                data=data,
            )

            sess.add(db_entity)
            await sess.commit()
            await sess.refresh(db_entity)

            return db_entity.to_pydantic()

    async def get_entity_by_id(
        self,
        *,
        entity_id: str,
        session: AsyncSession | None = None,
    ) -> Optional[Entity]:
        """Get an entity by id."""
        async with self._get_session(session) as sess:
            result = await sess.execute(select(DBEntity).where(DBEntity.id == entity_id))
            db_entity = result.scalar_one_or_none()
            if db_entity is None:
                return None

            return db_entity.to_pydantic()

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

        With the unique constraint on (workspace, entity_type, parent, name),
        this will return exactly one entity or None.
        """
        async with self._get_session(session) as sess:
            query = select(DBEntity).where(
                DBEntity.workspace == workspace,
                DBEntity.entity_type == entity_type,
                DBEntity.name == name,
            )
            if parent is None:
                query = query.where(DBEntity.parent.is_(None))
            else:
                query = query.where(DBEntity.parent == parent)

            result = await sess.execute(query)
            db_entity = result.scalar_one_or_none()

            if db_entity is None:
                return None

            return db_entity.to_pydantic()

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
        """List entities with filtering."""
        async with self._get_session(session) as sess:
            # Start with base query
            query = select(DBEntity)

            # Apply entity type filter if specified
            if entity_type is not None:
                query = query.where(DBEntity.entity_type == entity_type)

            # Apply workspace filtering if specified
            if workspace != ALL_WORKSPACES:
                query = query.where(DBEntity.workspace == workspace)
            # If workspace is ALL_WORKSPACES, query all workspaces (access control via filter)

            if filter_op is not None:
                filter_repo = SQLAlchemyFilterRepository(
                    DBEntity, relationship_child_workspaces=relationship_child_workspaces
                )
                query = query.where(filter_op.apply(filter_repo))

            count_query = select(func.count()).select_from(query.subquery())
            total_result = await sess.execute(count_query)
            total = total_result.scalar() or 0

            if sort is not None:
                query = query.order_by(self._get_sort_key(sort))
            else:
                query = query.order_by(DBEntity.created_at.desc())

            query = query.offset((page - 1) * page_size).limit(page_size)

            result = await sess.execute(query)
            db_entities = result.scalars().all()

            entities = [e.to_pydantic() for e in db_entities]

            return entities, total

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
        async with self._get_session(session) as sess:
            parts = group_by.split(".")
            if len(parts) != 2 or parts[0] != "data" or not parts[1].isidentifier():
                raise ValueError(f"Field '{group_by}' is not a direct string data field")

            field = parts[1]
            raw_group_column = DBEntity.data[field]
            group_column = raw_group_column.as_string()
            if self._is_sqlite(sess):
                json_type = func.json_type(DBEntity.data, f"$.{field}")
                string_type = "text"
            else:
                json_type = func.json_typeof(raw_group_column)
                string_type = "string"

            filter_repo = SQLAlchemyFilterRepository(
                DBEntity, relationship_child_workspaces=relationship_child_workspaces
            )
            query = select(group_column, func.count()).select_from(DBEntity).where(DBEntity.entity_type == entity_type)

            if workspace != ALL_WORKSPACES:
                query = query.where(DBEntity.workspace == workspace)

            if filter_op is not None:
                query = query.where(filter_op.apply(filter_repo))

            query = query.where(json_type == string_type)

            rows = (await sess.execute(query.group_by(group_column).limit(MAX_GROUP_COUNT_ROWS + 1))).all()
            if len(rows) > MAX_GROUP_COUNT_ROWS:
                raise ValueError(
                    f"Grouped count has more than {MAX_GROUP_COUNT_ROWS} distinct values; "
                    "narrow the filter or choose a lower-cardinality field."
                )
            return {str(key): int(count) for key, count in rows}

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
        async with self._get_session(session, for_write=True) as sess:
            result = await sess.execute(select(DBEntity).where(DBEntity.id == entity_id))
            existing_entity = result.scalar_one_or_none()
            if existing_entity is None:
                raise EntityNotFoundError(f"Entity with ID '{entity_id}' not found")

            existing_entity.data = data
            if name is not None:
                existing_entity.name = name
            if project is not None:
                existing_entity.project = project
            if updated_by is not None:
                existing_entity.updated_by = updated_by
            try:
                await sess.commit()
            except StaleDataError as err:
                raise EntityVersionConflictError(
                    f"Entity with ID '{entity_id}' was modified by another request. Please refetch and retry."
                ) from err
            await sess.refresh(existing_entity)

            return existing_entity.to_pydantic()

    async def update_entity_by_name(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        data: Dict[str, Any],
        new_name: str | None = None,
        parent: str | None = None,
        project: str | None = None,
        updated_by: str | None = None,
        expected_db_version: int | None = None,
        session: AsyncSession | None = None,
    ) -> Entity:
        """Update an entity by name, optionally changing its name."""
        async with self._get_session(session, for_write=True) as sess:
            query = select(DBEntity).where(
                DBEntity.workspace == workspace,
                DBEntity.entity_type == entity_type,
                DBEntity.name == name,
            )
            if parent is None:
                query = query.where(DBEntity.parent.is_(None))
            else:
                query = query.where(DBEntity.parent == parent)

            result = await sess.execute(query)
            existing_entity = result.scalar_one_or_none()
            if existing_entity is None:
                raise EntityNotFoundError(
                    f"Entity '{name}' of type '{entity_type}' not found in workspace '{workspace}'"
                )

            # Check expected_db_version atomically in the same transaction
            if expected_db_version is not None and existing_entity.db_version != expected_db_version:
                raise EntityVersionConflictError(
                    f"Entity '{name}' of type '{entity_type}' in workspace '{workspace}' was modified by another request. "
                    f"Expected version {expected_db_version}, but current version is {existing_entity.db_version}. Please refetch and retry."
                )

            existing_entity.data = data
            if new_name is not None:
                existing_entity.name = new_name
            if project is not None:
                existing_entity.project = project
            if updated_by is not None:
                existing_entity.updated_by = updated_by
            try:
                await sess.commit()
            except StaleDataError as err:
                raise EntityVersionConflictError(
                    f"Entity '{name}' of type '{entity_type}' in workspace '{workspace}' was modified by another request. Please refetch and retry."
                ) from err
            await sess.refresh(existing_entity)

            return existing_entity.to_pydantic()

    async def delete_entity(
        self,
        *,
        entity_id: str,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete an entity by ID."""
        async with self._get_session(session, for_write=True) as sess:
            result = await sess.execute(select(DBEntity).where(DBEntity.id == entity_id))
            existing_entity = result.scalar_one_or_none()
            if existing_entity is None:
                return 0

            await sess.delete(existing_entity)
            await sess.commit()
            return 1

    async def delete_entity_by_name(
        self,
        *,
        workspace: str,
        entity_type: str,
        name: str,
        parent: Optional[str] = None,
        expected_db_version: int | None = None,
        refuse_children_outside: str | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """Delete an entity by name, refusing with ForeignChildEntitiesError per *refuse_children_outside*."""
        async with self._get_session(session, for_write=True) as sess:
            query = select(DBEntity).where(
                DBEntity.workspace == workspace,
                DBEntity.entity_type == entity_type,
                DBEntity.name == name,
            )
            if parent is None:
                query = query.where(DBEntity.parent.is_(None))
            else:
                query = query.where(DBEntity.parent == parent)

            if refuse_children_outside is not None:
                # Child inserts take FOR KEY SHARE on the parent, so this blocks them until the delete commits.
                query = query.with_for_update()

            result = await sess.execute(query)
            existing_entity = result.scalar_one_or_none()
            if existing_entity is None:
                return 0

            if refuse_children_outside is not None:
                child_workspaces_query = select(DBEntity.workspace).where(DBEntity.parent == existing_entity.id)
                child_workspaces = (await sess.execute(child_workspaces_query.distinct())).scalars().all()
                foreign = {w for w in child_workspaces if w and w != refuse_children_outside}
                if foreign:
                    raise ForeignChildEntitiesError(foreign)

            if expected_db_version is not None and existing_entity.db_version != expected_db_version:
                raise EntityVersionConflictError(
                    f"Entity '{name}' of type '{entity_type}' in workspace '{workspace}' was modified by another request. "
                    f"Expected version {expected_db_version}, but current version is {existing_entity.db_version}. Please refetch and retry."
                )

            try:
                await sess.delete(existing_entity)
                await sess.commit()
            except StaleDataError as err:
                await sess.rollback()
                raise EntityVersionConflictError(
                    f"Entity '{name}' of type '{entity_type}' in workspace '{workspace}' was modified by another request. Please refetch and retry."
                ) from err
            return 1
