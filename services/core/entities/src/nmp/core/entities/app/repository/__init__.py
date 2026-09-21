# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository layer for entities service.

Provides abstract interfaces and implementations for database operations.
"""

import asyncio
from typing import TYPE_CHECKING

from nmp.core.entities.app.database import create_async_engine_for_entities
from nmp.core.entities.app.repository.account_identity import (
    AccountIdentityConflictError,
    AccountIdentityRecord,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
)
from nmp.core.entities.app.repository.entity import EntityRepositoryInterface
from nmp.core.entities.app.repository.sqlalchemy.entity import SQLAlchemyEntityRepository
from nmp.core.entities.app.repository.sqlalchemy.workspace import SQLAlchemyWorkspaceRepository
from nmp.core.entities.app.repository.workspace import WorkspaceRepositoryInterface
from nmp.core.entities.config import EntitiesConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

# One engine/session-maker pair per event loop.
#
# asyncpg pools bind their internal asyncio primitives (queues, waiters) to the
# event loop that first uses them.  The service runs at least two loops in one
# process: the uvicorn lifespan loop (service.py) and the controller worker
# loop (controllers/main.py, daemon thread).  A process-global singleton lets
# whichever loop initializes first capture the pool; when the pool is saturated
# the other loop parks on a waiter bound to the foreign loop and every acquire
# fails with "RuntimeError: <Queue> is bound to a different event loop"
# (intermittent: an unsaturated pool never parks, so light load hides the bug).
# Keying the singleton by running loop gives each loop its own pool while
# keeping the once-per-loop initialization contract.  See NVBug 6588975 /
# GitHub issue #1230.
_engines_by_loop: dict["AbstractEventLoop", AsyncEngine] = {}
_session_makers_by_loop: dict["AbstractEventLoop", async_sessionmaker[AsyncSession]] = {}


async def initialize_async_engine(config: EntitiesConfig) -> None:
    """Initialize the async engine for the current event loop.

    This should be called once during startup of every component that runs its
    own event loop (service lifespan, controller worker thread).  Calling it
    again on the same loop is a no-op; calling it from a different loop creates
    an independent engine bound to that loop.

    Args:
        config: Entities service configuration
    """
    loop = asyncio.get_running_loop()

    if loop in _engines_by_loop:
        return  # Already initialized on this loop

    engine = create_async_engine_for_entities(config)
    _engines_by_loop[loop] = engine
    _session_makers_by_loop[loop] = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )


async def get_async_engine() -> AsyncEngine:
    """Get the async engine bound to the current event loop.

    Returns:
        The initialized async engine for this loop

    Raises:
        RuntimeError: If called before initialize_async_engine() on this loop
    """
    loop = asyncio.get_running_loop()
    engine = _engines_by_loop.get(loop)
    if engine is None:
        raise RuntimeError(
            "Async engine not initialized on this event loop. "
            "Call initialize_async_engine(config) during startup of the component that owns this loop."
        )
    return engine


async def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get the async session maker bound to the current event loop.

    Note: initialize_async_engine() must be called first during startup of the
    component that owns the current event loop.

    Returns:
        The initialized async session maker for this loop

    Raises:
        RuntimeError: If called before initialize_async_engine() on this loop
    """
    loop = asyncio.get_running_loop()
    session_maker = _session_makers_by_loop.get(loop)
    if session_maker is None:
        raise RuntimeError(
            "Async session maker not initialized on this event loop. "
            "Call initialize_async_engine(config) during startup of the component that owns this loop."
        )
    return session_maker


async def ping_database() -> bool:
    """Run a trivial query to verify database connectivity.

    Returns:
        True if the database is reachable, False otherwise.
    """
    loop = asyncio.get_running_loop()
    session_maker = _session_makers_by_loop.get(loop)
    if session_maker is None:
        return False
    try:
        async with session_maker() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def dispose_async_engine() -> None:
    """Dispose the current loop's async engine, closing all connections.

    Should be called during shutdown of the component that owns the current
    event loop.  Engines owned by other loops are left untouched so a
    controller shutdown cannot yank connections out from under the service
    loop (and vice versa).
    """
    loop = asyncio.get_running_loop()
    engine = _engines_by_loop.pop(loop, None)
    _session_makers_by_loop.pop(loop, None)
    if engine is not None:
        await engine.dispose()


def dep_workspace_repository(session_maker: async_sessionmaker[AsyncSession]) -> WorkspaceRepositoryInterface:
    """Dependency function for Workspace repository."""
    return SQLAlchemyWorkspaceRepository(session_maker)


def dep_entity_repository(session_maker: async_sessionmaker[AsyncSession]) -> EntityRepositoryInterface:
    """Dependency function for Entity repository."""
    return SQLAlchemyEntityRepository(session_maker)


__all__ = [
    "WorkspaceRepositoryInterface",
    "AccountIdentityConflictError",
    "AccountIdentityRecord",
    "AccountIdentityStore",
    "AccountIdentityUnavailableError",
    "EntityRepositoryInterface",
    "SQLAlchemyWorkspaceRepository",
    "SQLAlchemyEntityRepository",
    "dep_workspace_repository",
    "dep_entity_repository",
    "dispose_async_engine",
    "get_async_engine",
    "get_async_session_maker",
    "initialize_async_engine",
    "ping_database",
]
