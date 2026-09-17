# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeploymentConfig CRUD routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_deployments_plugin.api.v2._perms import DeploymentConfigPerms
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
from nemo_deployments_plugin.authz import scope
from nemo_deployments_plugin.entities import Container, DeploymentConfig
from nemo_deployments_plugin.references import deployment_names_using_config
from nemo_deployments_plugin.schema import (
    CreateDeploymentConfigRequest,
    DeploymentConfigFilter,
    DeploymentConfigPage,
    RequestContainer,
)
from nemo_platform_plugin.api.filters import make_filter_obj_dep
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)
from nemo_platform_plugin.schema import PaginationData

logger = logging.getLogger(__name__)

router = APIRouter()

_config_filter_dep = make_filter_obj_dep(DeploymentConfigFilter)


def _container_from_request(container: RequestContainer) -> Container:
    """Convert a public request container into the persisted entity shape."""
    data = container.model_dump(by_alias=True, exclude={"env"}, exclude_none=True)
    return Container(**data, env=[item.to_entity() for item in container.env])


@router.post("/deployment-configs", response_model=DeploymentConfig, status_code=201, tags=["Deployment Configs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentConfigPerms.CREATE])
async def create_deployment_config(
    workspace: str,
    body: CreateDeploymentConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentConfig:
    payload = body.model_dump(exclude={"name", "containers", "init_containers"}, exclude_none=True)
    config = DeploymentConfig(
        name=body.name,
        workspace=workspace,
        containers=[_container_from_request(item) for item in body.containers],
        initContainers=[_container_from_request(item) for item in body.init_containers],
        **payload,
    )
    try:
        return await entity_client.create(config)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"DeploymentConfig '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except NemoEntityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/deployment-configs", response_model=DeploymentConfigPage, tags=["Deployment Configs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentConfigPerms.LIST])
async def list_deployment_configs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: DeploymentConfigFilter = Depends(_config_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentConfigPage:
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    result = await entity_client.list(
        DeploymentConfig,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter_obj=filter_dict or None,
    )
    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return DeploymentConfigPage(data=result.data, pagination=pagination, sort=sort, filter=filter)


@router.get("/deployment-configs/{name}", response_model=DeploymentConfig, tags=["Deployment Configs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentConfigPerms.READ])
async def get_deployment_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentConfig:
    try:
        return await entity_client.get(DeploymentConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"DeploymentConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc


@router.delete("/deployment-configs/{name}", status_code=204, tags=["Deployment Configs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentConfigPerms.DELETE])
async def delete_deployment_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    # Best-effort referential check before delete. The version check below protects the
    # config row itself, but cannot make this relationship check atomic against a
    # concurrent deployment create.
    referencing = await deployment_names_using_config(
        entity_client,
        workspace=workspace,
        config_name=name,
    )
    if referencing:
        joined = ", ".join(referencing)
        raise HTTPException(
            status_code=409,
            detail=(
                f"DeploymentConfig '{name}' is referenced by deployment(s): {joined}. Delete those deployments first."
            ),
        )

    try:
        await entity_client.delete(DeploymentConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"DeploymentConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"DeploymentConfig '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the config and try again."
            ),
        ) from exc
