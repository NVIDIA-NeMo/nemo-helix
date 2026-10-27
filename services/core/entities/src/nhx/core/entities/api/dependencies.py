# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency injection for entities API."""

from typing import Annotated

from fastapi import Depends
from nemo_helix_plugin.auth.access import AuthContextAccessRule
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import get_auth_client
from nhx.core.entities.app.repository import (
    EntityRepositoryInterface,
    WorkspaceRepositoryInterface,
    dep_entity_repository,
    dep_workspace_repository,
    get_async_session_maker,
)
from nhx.core.entities.app.repository.field_policy import ProtectedEntityField
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

AUTH_CONTEXT_FIELD = ProtectedEntityField("_auth_context")


def dep_entity_repository_with_session(
    session_maker: async_sessionmaker[AsyncSession] = Depends(get_async_session_maker),
    auth_client: AuthClient = Depends(get_auth_client),
) -> EntityRepositoryInterface:
    """Dependency function for Entity repository with injected session maker."""
    access = AuthContextAccessRule.from_auth_client(auth_client)
    policy = AUTH_CONTEXT_FIELD.bind(
        policy_enabled=access.policy_enabled, can_read=access.can_read, can_write=access.can_write
    )
    return dep_entity_repository(session_maker, policy)


def dep_workspace_repository_with_session(
    session_maker: async_sessionmaker[AsyncSession] = Depends(get_async_session_maker),
) -> WorkspaceRepositoryInterface:
    """Dependency function for Workspace repository with injected session maker."""
    return dep_workspace_repository(session_maker)


AsyncSessionMaker = Annotated[async_sessionmaker, Depends(get_async_session_maker)]
WorkspaceRepository = Annotated[WorkspaceRepositoryInterface, Depends(dep_workspace_repository_with_session)]
EntityRepository = Annotated[EntityRepositoryInterface, Depends(dep_entity_repository_with_session)]


AuthClientDep = Annotated[AuthClient, Depends(get_auth_client)]


__all__ = [
    "AsyncSessionMaker",
    "WorkspaceRepository",
    "EntityRepository",
    "AuthClientDep",
    "dep_entity_repository_with_session",
    "dep_workspace_repository_with_session",
]
