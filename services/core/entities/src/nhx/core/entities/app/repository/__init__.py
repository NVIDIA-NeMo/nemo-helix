# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository layer for entities service.

Provides abstract interfaces and implementations for database operations.
"""

import asyncio
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from nhx.core.entities.app.database import create_async_engine_for_entities
from nhx.core.entities.app.repository.account_credential import (
    AccountCredentialConflictError,
    AccountCredentialRecord,
    AccountCredentialStatus,
    AccountCredentialStore,
    AccountCredentialType,
    credential_lookup_hash,
)
from nhx.core.entities.app.repository.account_identity import (
    AccountIdentityConflictError,
    AccountIdentityRecord,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
)
from nhx.core.entities.app.repository.account_preference import (
    PREFERENCE_MAX_BYTES,
    PREFERENCE_MAX_DOCUMENTS,
    AccountPreferenceStore,
    PreferenceLimitError,
    PreferenceRecord,
    PreferenceTooLargeError,
    PreferenceVersionConflictError,
    merge_patch,
)
from nhx.core.entities.app.repository.account_profile import (
    AccountProfileNotFoundError,
    AccountProfileRecord,
    AccountProfileStore,
    IdentityClaims,
    ResolvedPrincipal,
)
from nhx.core.entities.app.repository.entity import EntityRepositoryInterface
from nhx.core.entities.app.repository.policy import NO_OP_ENTITY_FIELD_POLICY, EntityFieldPolicy
from nhx.core.entities.app.repository.sqlalchemy.entity import SQLAlchemyEntityRepository
from nhx.core.entities.app.repository.sqlalchemy.workspace import SQLAlchemyWorkspaceRepository
from nhx.core.entities.app.repository.workspace import WorkspaceRepositoryInterface
from nhx.core.entities.config import EntitiesConfig
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


@dataclass
class _LoopEngineBinding:
    """The async engine and session maker owned by one event loop."""

    engine: AsyncEngine
    session_maker: async_sessionmaker[AsyncSession]


# One engine per event loop: SQLAlchemy's async pool binds to the loop that first uses it.
# Every loop that initializes an engine must call dispose_async_engine() before it ends.
_async_engines: WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopEngineBinding] = WeakKeyDictionary()


def _binding_for_current_loop() -> _LoopEngineBinding | None:
    """Return the engine binding the running event loop owns, if any."""
    return _async_engines.get(asyncio.get_running_loop())


async def initialize_async_engine(config: EntitiesConfig) -> None:
    """Initialize the async engine for the current event loop with the given config.

    Call once per event loop that queries the entity store.

    Args:
        config: Entities service configuration
    """
    loop = asyncio.get_running_loop()
    binding = _async_engines.get(loop)
    if binding is not None:
        if binding.engine.url != make_url(config.get_async_sqlalchemy_url()):
            raise RuntimeError(
                "The entities engine is already initialized for a different database on this event loop. "
                "Dispose this loop's engine before initializing another."
            )
        return  # Already initialized on this loop

    engine = create_async_engine_for_entities(config)
    _async_engines[loop] = _LoopEngineBinding(
        engine=engine,
        session_maker=async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False),
    )


async def get_async_engine() -> AsyncEngine:
    """Get the async engine for the current event loop.

    Raises:
        RuntimeError: If called before initialize_async_engine() on this event loop
    """
    binding = _binding_for_current_loop()
    if binding is None:
        raise RuntimeError(
            "Async engine not initialized on this event loop. Call initialize_async_engine(config) during service "
            "startup."
        )
    return binding.engine


async def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get async session maker for the entities database on the current event loop.

    Note: initialize_async_engine() must have been called on this event loop first.

    Returns:
        The initialized async session maker

    Raises:
        RuntimeError: If called before initialize_async_engine()
    """
    binding = _binding_for_current_loop()
    if binding is None:
        raise RuntimeError(
            "Async session maker not initialized. Call initialize_async_engine(config) during service startup."
        )
    return binding.session_maker


async def ping_database() -> bool:
    """Run a trivial query to verify database connectivity.

    Returns:
        True if the database is reachable, False otherwise.
    """
    binding = _binding_for_current_loop()
    if binding is None:
        return False
    try:
        async with binding.session_maker() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def dispose_async_engine() -> None:
    """Dispose the current event loop's async engine, closing its connections."""
    binding = _async_engines.pop(asyncio.get_running_loop(), None)
    if binding is not None:
        await binding.engine.dispose()


def dep_workspace_repository(session_maker: async_sessionmaker[AsyncSession]) -> WorkspaceRepositoryInterface:
    """Dependency function for Workspace repository."""
    return SQLAlchemyWorkspaceRepository(session_maker)


def dep_entity_repository(
    session_maker: async_sessionmaker[AsyncSession],
    field_policy: EntityFieldPolicy = NO_OP_ENTITY_FIELD_POLICY,
) -> EntityRepositoryInterface:
    """Dependency function for Entity repository."""
    return SQLAlchemyEntityRepository(session_maker, field_policy)


__all__ = [
    "WorkspaceRepositoryInterface",
    "AccountIdentityConflictError",
    "AccountIdentityRecord",
    "AccountIdentityStore",
    "AccountIdentityUnavailableError",
    "AccountCredentialConflictError",
    "AccountCredentialRecord",
    "AccountCredentialStatus",
    "AccountCredentialStore",
    "AccountCredentialType",
    "AccountPreferenceStore",
    "AccountProfileNotFoundError",
    "AccountProfileRecord",
    "AccountProfileStore",
    "IdentityClaims",
    "PREFERENCE_MAX_BYTES",
    "PREFERENCE_MAX_DOCUMENTS",
    "PreferenceLimitError",
    "PreferenceRecord",
    "PreferenceTooLargeError",
    "PreferenceVersionConflictError",
    "ResolvedPrincipal",
    "merge_patch",
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
    "credential_lookup_hash",
]
