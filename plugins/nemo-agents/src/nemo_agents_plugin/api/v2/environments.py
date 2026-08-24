# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD routes for AgentEnvironment, AgentEnvironmentSpec, and AgentComputeSpec.

Mounted under ``/apis/agents/v2/workspaces/{workspace}`` at:
- ``/environments`` (AgentEnvironment)
- ``/environment-specs`` (AgentEnvironmentSpec)
- ``/compute-specs`` (AgentComputeSpec)

Each collection is a thin CRUD surface over the generic entity client, matching
the Agent/Deployment route conventions (mandatory ``@path_rule`` authz, generic
NemoEntitiesClient, 404/409 mapping).
"""

from __future__ import annotations

import logging
from typing import TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agents_plugin.api.v2._perms import ComputeSpecPerms, EnvironmentPerms, EnvironmentSpecPerms, SandboxSpecPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.entities import AgentComputeSpec, AgentEnvironment, AgentEnvironmentSpec, AgentSandboxSpec
from nemo_agents_plugin.schema import (
    ComputeSpecFilter,
    ComputeSpecPage,
    CreateComputeSpecRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentSpecRequest,
    CreateSandboxSpecRequest,
    EnvironmentFilter,
    EnvironmentPage,
    EnvironmentSpecFilter,
    EnvironmentSpecPage,
    SandboxSpecFilter,
    SandboxSpecPage,
)
from nemo_platform_plugin.api.filters import make_filter_obj_dep
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity import NemoEntity
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.schema import NemoFilter, NemoListResponse, PaginationData

logger = logging.getLogger(__name__)

# Generics for the shared CRUD helpers below: one entity type, its list-response
# page type, and its query filter type.
EntityT = TypeVar("EntityT", bound=NemoEntity)
PageT = TypeVar("PageT", bound=NemoListResponse)
FilterT = TypeVar("FilterT", bound=NemoFilter)

router = APIRouter()

_environment_filter_dep = make_filter_obj_dep(EnvironmentFilter)
_environment_spec_filter_dep = make_filter_obj_dep(EnvironmentSpecFilter)
_compute_spec_filter_dep = make_filter_obj_dep(ComputeSpecFilter)
_sandbox_spec_filter_dep = make_filter_obj_dep(SandboxSpecFilter)


# ---------------------------------------------------------------------------
# AgentEnvironment
# ---------------------------------------------------------------------------


@router.post("/environments", response_model=AgentEnvironment, status_code=201, tags=["Agent Environments"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentPerms.CREATE])
async def create_environment(
    workspace: str,
    body: CreateEnvironmentRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentEnvironment:
    """Create a new AgentEnvironment."""
    environment = AgentEnvironment(
        name=body.name,
        workspace=workspace,
        description=body.description,
        environment_spec=body.environment_spec,
        sandbox_spec=body.sandbox_spec,
        compute_spec=body.compute_spec,
    )
    return await _create_entity(entity_client, environment, kind="environment", name=body.name, workspace=workspace)


@router.get("/environments", response_model=EnvironmentPage, tags=["Agent Environments"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentPerms.LIST])
async def list_environments(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: EnvironmentFilter = Depends(_environment_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> EnvironmentPage:
    """List AgentEnvironments in the workspace."""
    return await _list_entities(
        entity_client,
        AgentEnvironment,
        EnvironmentPage,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter=filter,
        kind="environments",
    )


@router.get("/environments/{name}", response_model=AgentEnvironment, tags=["Agent Environments"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentPerms.READ])
async def get_environment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentEnvironment:
    """Get an AgentEnvironment by name."""
    return await _get_entity(entity_client, AgentEnvironment, name=name, workspace=workspace, kind="environment")


@router.delete("/environments/{name}", status_code=204, tags=["Agent Environments"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentPerms.DELETE])
async def delete_environment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an AgentEnvironment by name."""
    await _delete_entity(entity_client, AgentEnvironment, name=name, workspace=workspace, kind="environment")


# ---------------------------------------------------------------------------
# AgentEnvironmentSpec
# ---------------------------------------------------------------------------


@router.post(
    "/environment-specs", response_model=AgentEnvironmentSpec, status_code=201, tags=["Agent Environment Specs"]
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentSpecPerms.CREATE])
async def create_environment_spec(
    workspace: str,
    body: CreateEnvironmentSpecRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentEnvironmentSpec:
    """Create a new AgentEnvironmentSpec."""
    spec = AgentEnvironmentSpec(**body.model_dump(), workspace=workspace)
    return await _create_entity(entity_client, spec, kind="environment spec", name=body.name, workspace=workspace)


@router.get("/environment-specs", response_model=EnvironmentSpecPage, tags=["Agent Environment Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentSpecPerms.LIST])
async def list_environment_specs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: EnvironmentSpecFilter = Depends(_environment_spec_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> EnvironmentSpecPage:
    """List AgentEnvironmentSpecs in the workspace."""
    return await _list_entities(
        entity_client,
        AgentEnvironmentSpec,
        EnvironmentSpecPage,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter=filter,
        kind="environment specs",
    )


@router.get("/environment-specs/{name}", response_model=AgentEnvironmentSpec, tags=["Agent Environment Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentSpecPerms.READ])
async def get_environment_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentEnvironmentSpec:
    """Get an AgentEnvironmentSpec by name."""
    return await _get_entity(
        entity_client, AgentEnvironmentSpec, name=name, workspace=workspace, kind="environment spec"
    )


@router.delete("/environment-specs/{name}", status_code=204, tags=["Agent Environment Specs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EnvironmentSpecPerms.DELETE])
async def delete_environment_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an AgentEnvironmentSpec by name."""
    await _delete_entity(entity_client, AgentEnvironmentSpec, name=name, workspace=workspace, kind="environment spec")


# ---------------------------------------------------------------------------
# AgentComputeSpec
# ---------------------------------------------------------------------------


@router.post("/compute-specs", response_model=AgentComputeSpec, status_code=201, tags=["Agent Compute Specs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ComputeSpecPerms.CREATE])
async def create_compute_spec(
    workspace: str,
    body: CreateComputeSpecRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentComputeSpec:
    """Create a new AgentComputeSpec."""
    spec = AgentComputeSpec(**body.model_dump(), workspace=workspace)
    return await _create_entity(entity_client, spec, kind="compute spec", name=body.name, workspace=workspace)


@router.get("/compute-specs", response_model=ComputeSpecPage, tags=["Agent Compute Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ComputeSpecPerms.LIST])
async def list_compute_specs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: ComputeSpecFilter = Depends(_compute_spec_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ComputeSpecPage:
    """List AgentComputeSpecs in the workspace."""
    return await _list_entities(
        entity_client,
        AgentComputeSpec,
        ComputeSpecPage,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter=filter,
        kind="compute specs",
    )


@router.get("/compute-specs/{name}", response_model=AgentComputeSpec, tags=["Agent Compute Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ComputeSpecPerms.READ])
async def get_compute_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentComputeSpec:
    """Get an AgentComputeSpec by name."""
    return await _get_entity(entity_client, AgentComputeSpec, name=name, workspace=workspace, kind="compute spec")


@router.delete("/compute-specs/{name}", status_code=204, tags=["Agent Compute Specs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ComputeSpecPerms.DELETE])
async def delete_compute_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an AgentComputeSpec by name."""
    await _delete_entity(entity_client, AgentComputeSpec, name=name, workspace=workspace, kind="compute spec")


# ---------------------------------------------------------------------------
# AgentSandboxSpec
# ---------------------------------------------------------------------------


@router.post("/sandbox-specs", response_model=AgentSandboxSpec, status_code=201, tags=["Agent Sandbox Specs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SandboxSpecPerms.CREATE])
async def create_sandbox_spec(
    workspace: str,
    body: CreateSandboxSpecRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentSandboxSpec:
    """Create a new AgentSandboxSpec."""
    spec = AgentSandboxSpec(**body.model_dump(), workspace=workspace)
    return await _create_entity(entity_client, spec, kind="sandbox spec", name=body.name, workspace=workspace)


@router.get("/sandbox-specs", response_model=SandboxSpecPage, tags=["Agent Sandbox Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SandboxSpecPerms.LIST])
async def list_sandbox_specs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: SandboxSpecFilter = Depends(_sandbox_spec_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> SandboxSpecPage:
    """List AgentSandboxSpecs in the workspace."""
    return await _list_entities(
        entity_client,
        AgentSandboxSpec,
        SandboxSpecPage,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter=filter,
        kind="sandbox specs",
    )


@router.get("/sandbox-specs/{name}", response_model=AgentSandboxSpec, tags=["Agent Sandbox Specs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SandboxSpecPerms.READ])
async def get_sandbox_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentSandboxSpec:
    """Get an AgentSandboxSpec by name."""
    return await _get_entity(entity_client, AgentSandboxSpec, name=name, workspace=workspace, kind="sandbox spec")


@router.delete("/sandbox-specs/{name}", status_code=204, tags=["Agent Sandbox Specs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SandboxSpecPerms.DELETE])
async def delete_sandbox_spec(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an AgentSandboxSpec by name."""
    await _delete_entity(entity_client, AgentSandboxSpec, name=name, workspace=workspace, kind="sandbox spec")


# ---------------------------------------------------------------------------
# Shared CRUD helpers
# ---------------------------------------------------------------------------


async def _create_entity(
    entity_client: NemoEntitiesClient,
    entity: EntityT,
    *,
    kind: str,
    name: str,
    workspace: str,
) -> EntityT:
    try:
        return await entity_client.create(entity)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Agent {kind} '{name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create agent %s '%s'", kind, name)
        raise HTTPException(status_code=500, detail=f"Failed to create agent {kind}.") from exc


async def _list_entities(
    entity_client: NemoEntitiesClient,
    entity_type: type[EntityT],
    page_type: type[PageT],
    *,
    workspace: str,
    page: int,
    page_size: int,
    sort: str,
    filter: FilterT,
    kind: str,
) -> PageT:
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            entity_type,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list agent %s in workspace '%s'", kind, workspace)
        raise HTTPException(status_code=500, detail=f"Failed to list agent {kind}.") from exc

    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return page_type(data=result.data, pagination=pagination, sort=sort, filter=filter)


async def _get_entity(
    entity_client: NemoEntitiesClient,
    entity_type: type[EntityT],
    *,
    name: str,
    workspace: str,
    kind: str,
) -> EntityT:
    try:
        return await entity_client.get(entity_type, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {kind} '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get agent %s '%s'", kind, name)
        raise HTTPException(status_code=500, detail=f"Failed to get agent {kind}.") from exc


async def _delete_entity(
    entity_client: NemoEntitiesClient,
    entity_type: type[EntityT],
    *,
    name: str,
    workspace: str,
    kind: str,
) -> None:
    try:
        await entity_client.delete(entity_type, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {kind} '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Agent {kind} '{name}' was modified by another request in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete agent %s '%s'", kind, name)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent {kind}.") from exc
