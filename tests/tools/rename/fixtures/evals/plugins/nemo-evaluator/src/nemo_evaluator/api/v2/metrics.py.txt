# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD routes for stored metrics under /apis/evaluator/v2/workspaces/{workspace}/metrics."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from nemo_evaluator.api.dependencies import get_metric_service
from nemo_evaluator.api.schemas import (
    Metric,
    MetricFilter,
    MetricInline,
    MetricSort,
)
from nemo_evaluator.api.service.metric_service import MetricService
from nemo_evaluator.authz import scope
from nemo_evaluator.entities import MAX_NAME_LENGTH, NAME_PATTERN
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entities import EntityValidationError
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import Page

logger = logging.getLogger(__name__)


class MetricPerms(PermissionSet, namespace="evaluator.metrics"):
    """Permissions for the stored-metrics CRUD collection."""

    CREATE = perm("Create a stored metric")
    LIST = perm("List stored metrics")
    READ = perm("Read a stored metric")
    DELETE = perm("Delete a stored metric")


router = APIRouter()


@router.get(
    "/metrics",
    summary="List Metrics By Workspace",
    response_description="Return stored metrics for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[Metric],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=MetricFilter,
        filter_description="Filter metrics by workspace, name, metric_type, description, created_at, and updated_at.",
    ),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[MetricPerms.LIST])
async def list_metrics(
    workspace: str,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    sort: MetricSort = Query(
        default=MetricSort.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    include_derived: bool = Query(
        default=False,
        description="Include derived (task-internal) metrics, which are hidden from the listing by default.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(MetricFilter)),
    service: MetricService = Depends(get_metric_service),
) -> Page[Metric]:
    """List stored metrics for a specific workspace."""
    # Discard any workspace override in the filter — always scope to the path workspace.
    parsed_filter.remove("workspace")
    try:
        return await service.list_metrics(
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
            include_derived=include_derived,
        )
    except Exception:
        logger.exception(f"Failed to list metrics for workspace {sanitize_for_log(workspace)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/metrics/{name}",
    summary="Create Metric",
    response_description="Store a new metric",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Metric already exists"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[MetricPerms.CREATE])
async def create_metric(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    metric: MetricInline,
    project: str | None = Query(default=None, description="Optional project to associate with the metric."),
    service: MetricService = Depends(get_metric_service),
) -> Metric:
    """Store a new metric, addressed by workspace/name."""
    safe_workspace = sanitize_for_log(workspace)
    safe_name = sanitize_for_log(name)
    logger.info(f"Creating metric: {safe_workspace}/{safe_name}")
    try:
        return await service.create_metric(name, metric, workspace=workspace, project=project)
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during metric creation: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Metric already exists: {safe_workspace}/{safe_name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Metric with workspace '{workspace}' and name '{name}' already exists",
            )
        logger.warning(f"Metric creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metric data")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create metric")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/metrics/{name}",
    summary="Get Metric",
    response_description="Return stored metric details",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Metric not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[MetricPerms.READ])
async def get_metric(
    workspace: str,
    name: str,
    service: MetricService = Depends(get_metric_service),
) -> Metric:
    """Get a stored metric by workspace and name."""
    logger.debug(f"Getting metric: {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
    try:
        metric = await service.get_metric(workspace, name)
        if not metric:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metric not found: {workspace}/{name}",
            )
        return metric
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Failed to get metric {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.delete(
    "/metrics/{name}",
    summary="Delete Metric",
    response_description="Delete a stored metric",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Metric not found"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[MetricPerms.DELETE])
async def delete_metric(
    workspace: str,
    name: str,
    service: MetricService = Depends(get_metric_service),
):
    """Delete a stored metric by workspace and name."""
    logger.info(f"Deleting metric: {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
    try:
        deleted = await service.delete_metric(workspace, name)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metric not found: {workspace}/{name}",
            )
        return None
    except HTTPException:
        raise
    except NemoEntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception(f"Failed to delete metric {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
