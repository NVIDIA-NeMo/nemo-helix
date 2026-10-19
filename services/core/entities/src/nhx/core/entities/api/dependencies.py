# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency injection for entities API."""

from typing import Annotated

from fastapi import Depends
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import get_auth_client
from nhx.core.entities.app.repository import (
    EntityRepositoryInterface,
    WorkspaceRepositoryInterface,
    dep_entity_repository,
    dep_workspace_repository,
    get_async_session_maker,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def dep_entity_repository_with_session(
    session_maker: async_sessionmaker[AsyncSession] = Depends(get_async_session_maker),
) -> EntityRepositoryInterface:
    """Dependency function for Entity repository with injected session maker."""
    return dep_entity_repository(session_maker)


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
