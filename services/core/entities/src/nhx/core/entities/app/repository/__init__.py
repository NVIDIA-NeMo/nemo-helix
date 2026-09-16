# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository layer for entities service.

Provides abstract interfaces and implementations for database operations.
"""

from nhx.core.entities.app.database import create_async_engine_for_entities
from nhx.core.entities.app.repository.entity import EntityRepositoryInterface
from nhx.core.entities.app.repository.sqlalchemy.entity import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.sqlalchemy.workspace import SQLAlchemyWorkspaceRepository
from nhx.core.entities.app.repository.workspace import WorkspaceRepositoryInterface
from nhx.core.entities.config import EntitiesConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

_async_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


async def initialize_async_engine(config: EntitiesConfig) -> None:
    """Initialize the async engine with the given config.

    This should be called once during service startup.

    Args:
        config: Entities service configuration
    """
    global _async_engine, _async_session_maker

    if _async_engine is not None:
        return  # Already initialized

    _async_engine = create_async_engine_for_entities(config)
    _async_session_maker = async_sessionmaker(bind=_async_engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get async session maker for entities database.

    Note: initialize_async_engine() must be called first during service startup.

    Returns:
        The initialized async session maker

    Raises:
        RuntimeError: If called before initialize_async_engine()
    """
    global _async_session_maker
    if _async_session_maker is None:
        raise RuntimeError(
            "Async session maker not initialized. Call initialize_async_engine(config) during service startup."
        )
    return _async_session_maker


async def ping_database() -> bool:
    """Run a trivial query to verify database connectivity.

    Returns:
        True if the database is reachable, False otherwise.
    """
    if _async_session_maker is None:
        return False
    try:
        async with _async_session_maker() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def dispose_async_engine() -> None:
    """Dispose the async engine, closing all connections.

    Should be called during service shutdown to properly close database connections.
    """
    global _async_engine, _async_session_maker
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_maker = None


def dep_workspace_repository(session_maker: async_sessionmaker[AsyncSession]) -> WorkspaceRepositoryInterface:
    """Dependency function for Workspace repository."""
    return SQLAlchemyWorkspaceRepository(session_maker)


def dep_entity_repository(session_maker: async_sessionmaker[AsyncSession]) -> EntityRepositoryInterface:
    """Dependency function for Entity repository."""
    return SQLAlchemyEntityRepository(session_maker)


__all__ = [
    "WorkspaceRepositoryInterface",
    "EntityRepositoryInterface",
    "SQLAlchemyWorkspaceRepository",
    "SQLAlchemyEntityRepository",
    "dep_workspace_repository",
    "dep_entity_repository",
    "dispose_async_engine",
    "get_async_session_maker",
    "initialize_async_engine",
    "ping_database",
]
