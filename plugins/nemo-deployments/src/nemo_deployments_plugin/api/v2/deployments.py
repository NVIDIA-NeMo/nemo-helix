# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment CRUD routes."""

from __future__ import annotations

import logging
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from nemo_deployments_plugin.api.v2._perms import DeploymentPerms
from nemo_deployments_plugin.api.v2.dependencies import get_entity_client
from nemo_deployments_plugin.authz import scope
from nemo_deployments_plugin.entities import Deployment, DeploymentConfig, DeploymentStatus
from nemo_deployments_plugin.reconciler.entity_client import list_all_pages
from nemo_deployments_plugin.schema import CreateDeploymentRequest, DeploymentFilter, DeploymentPage
from nemo_deployments_plugin.validation import (
    PrerequisiteCycleError,
    build_existing_prerequisite_map,
    deployment_graph_key,
    detect_prerequisite_cycle,
    prerequisite_names,
)
from nemo_platform_plugin.api.filters import make_filter_obj_dep
from nemo_platform_plugin.auth import current_auth_context
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    NemoEntityValidationError,
)
from nemo_platform_plugin.filter_ops import ComparisonOperation, FilterOperator
from nemo_platform_plugin.schema import PaginationData

logger = logging.getLogger(__name__)

router = APIRouter()

_deployment_filter_dep = make_filter_obj_dep(DeploymentFilter)

_VALID_DEPLOYMENT_STATUSES: frozenset[str] = frozenset(
    {"PENDING", "STARTING", "READY", "SUCCEEDED", "FAILED", "LOST", "DELETING"}
)


def _parse_status_in(status_in: str | None) -> list[DeploymentStatus]:
    if not status_in:
        return []
    values = [part.strip().upper() for part in status_in.split(",") if part.strip()]
    invalid = [value for value in values if value not in _VALID_DEPLOYMENT_STATUSES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid deployment status values: {', '.join(invalid)}",
        )
    return cast(list[DeploymentStatus], values)


def _parse_deployment_config_ref(ref: str, default_workspace: str) -> tuple[str, str]:
    """Return (config_workspace, config_name) from a bare name or workspace/name ref."""
    if "/" in ref:
        config_workspace, config_name = ref.split("/", 1)
        if not config_workspace or not config_name:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid deployment_config ref '{ref}'; expected 'name' or 'workspace/name'.",
            )
        return config_workspace, config_name
    return default_workspace, ref


@router.post("/deployments", response_model=Deployment, status_code=201, tags=["Deployments"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.CREATE])
async def create_deployment(
    workspace: str,
    body: CreateDeploymentRequest,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Deployment:
    config_workspace, config_name = _parse_deployment_config_ref(body.deployment_config, workspace)
    try:
        await entity_client.get(DeploymentConfig, name=config_name, workspace=config_workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(f"DeploymentConfig '{config_name}' not found in workspace '{config_workspace}'."),
        ) from exc

    prereq_names = prerequisite_names(body.prerequisites, workspace)
    try:
        existing_deployments = await list_all_pages(entity_client, Deployment, workspace=workspace)
        existing_map = build_existing_prerequisite_map(existing_deployments)
        detect_prerequisite_cycle(
            deployment_name=deployment_graph_key(workspace, body.name),
            prerequisites=prereq_names,
            existing=existing_map,
        )
    except PrerequisiteCycleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    deployment = Deployment(
        name=body.name,
        workspace=workspace,
        deployment_config=config_name,
        desired_state=body.desired_state,
        executor=body.executor,
        prerequisites=body.prerequisites,
        status="PENDING",
    ).with_auth_context(current_auth_context())
    try:
        return await entity_client.create(deployment)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Deployment '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except NemoEntityValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/deployments", response_model=DeploymentPage, tags=["Deployments"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.LIST])
async def list_deployments(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    status_in: str | None = Query(
        default=None,
        description="Comma-separated deployment statuses for bulk reconciler queries.",
    ),
    filter: DeploymentFilter = Depends(_deployment_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> DeploymentPage:
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    statuses = _parse_status_in(status_in) if status_in else []
    filter_operation = None
    if statuses:
        filter_operation = ComparisonOperation(
            operator=FilterOperator.IN,
            field="data.status",
            value=statuses,
        )
    result = await entity_client.list(
        Deployment,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter_obj=filter_dict or None,
        filter_operation=filter_operation,
    )
    pagination = PaginationData.model_validate(result.pagination.model_dump()) if result.pagination else None
    return DeploymentPage(data=result.data, pagination=pagination, sort=sort, filter=filter)


@router.get("/deployments/{name}", response_model=Deployment, tags=["Deployments"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.READ])
async def get_deployment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> Deployment:
    try:
        return await entity_client.get(Deployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc


@router.delete("/deployments/{name}", status_code=204, tags=["Deployments"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[DeploymentPerms.DELETE])
async def delete_deployment(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    try:
        deployment = await entity_client.get(Deployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Deployment '{name}' not found in workspace '{workspace}'.",
        ) from exc

    deployment.status = "DELETING"
    try:
        await entity_client.update(deployment)
    except NemoEntityNotFoundError:
        logger.info("Deployment already deleted before status update")
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Deployment '{name}' is being modified concurrently.",
        ) from exc
