# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency injection for entities API."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from nemo_helix_plugin.auth.access import AuthContextAccessRule
from nemo_helix_plugin.auth.decisions import CollectionTarget, DecisionRequest, FieldTarget, WorkspaceDecision
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import get_auth_client
from nhx.core.entities.api.v2.utils import workspace_decision_context
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


async def dep_entity_repository_with_session(
    request: Request,
    session_maker: async_sessionmaker[AsyncSession] = Depends(get_async_session_maker),
    auth_client: AuthClient = Depends(get_auth_client),
) -> AsyncIterator[EntityRepositoryInterface]:
    """Dependency function for Entity repository with injected session maker."""
    access = await AuthContextAccessRule.from_auth_client(
        auth_client,
        target=FieldTarget(
            workspace=str(request.path_params.get("workspace", "-")),
            resource_type=str(request.path_params.get("entity_type", "entity")),
        ),
    )
    policy = AUTH_CONTEXT_FIELD.bind(
        policy_enabled=access.policy_enabled, can_read=access.can_read, can_write=access.can_write
    )
    scope = await auth_client.decide(
        DecisionRequest(
            operation="entities.workspaces.visible",
            target=CollectionTarget(workspace="-", resource_type=str(request.path_params.get("entity_type", "entity"))),
        )
    )
    if not isinstance(scope, WorkspaceDecision):
        raise ValueError("Repository scope requires a typed workspace decision")
    token = workspace_decision_context.set(scope)
    try:
        yield dep_entity_repository(session_maker, policy)
    finally:
        workspace_decision_context.reset(token)


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
