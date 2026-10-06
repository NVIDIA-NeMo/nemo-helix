# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanTarget CRUD routes.

Mounted by the plugin service at ``/apis/garak-plugin/v2/workspaces/{workspace}/targets``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from garak_plugin.api.v2._filters import make_filter_dep
from garak_plugin.api.v2._perms import ScanTargetPerms
from garak_plugin.api.v2.schemas import CreateScanTargetRequest, TargetFilter, UpdateScanTargetRequest
from garak_plugin.authz import scope
from garak_plugin.entities import ScanTarget
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

router = APIRouter()

_target_filter_dep = make_filter_dep(TargetFilter)


@router.post("/targets", response_model=ScanTarget, status_code=201, tags=["Garak Plugin Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanTargetPerms.CREATE],
)
async def create_target(
    workspace: str,
    body: CreateScanTargetRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanTarget:
    """Create a new scan target."""
    target = ScanTarget(
        name=body.name,
        workspace=workspace,
        description=body.description,
        type=body.type,
        model=body.model,
        options=body.options,
    )
    try:
        return await entity_client.create(target)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"ScanTarget '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create scan target '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to create scan target.") from exc


@router.get(
    "/targets",
    tags=["Garak Plugin Targets"],
    openapi_extra=generate_openapi_extra_params(filter_schema=TargetFilter),
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanTargetPerms.LIST],
)
async def list_targets(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    parsed_filter: TargetFilter = Depends(_target_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List scan targets in the workspace with pagination and filter support."""
    filter_dict = parsed_filter if isinstance(parsed_filter, dict) else parsed_filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            ScanTarget,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list scan targets in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list scan targets.") from exc
    return {
        "data": [t.model_dump(mode="json") for t in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": parsed_filter or None,
    }


@router.get("/targets/{name}", response_model=ScanTarget, tags=["Garak Plugin Targets"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanTargetPerms.READ],
)
async def get_target(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanTarget:
    """Get a scan target by name."""
    try:
        return await entity_client.get(ScanTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get scan target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get scan target.") from exc


@router.put("/targets/{name}", response_model=ScanTarget, tags=["Garak Plugin Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanTargetPerms.UPDATE],
)
async def update_target(
    workspace: str,
    name: str,
    body: UpdateScanTargetRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanTarget:
    """Replace a scan target's contents."""
    try:
        existing = await entity_client.get(ScanTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc

    existing.description = body.description
    existing.type = body.type
    existing.model = body.model
    existing.options = body.options

    try:
        return await entity_client.update(existing)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        logger.info(
            "Conflict updating scan target '%s' in workspace '%s'",
            sanitize_for_log(name),
            sanitize_for_log(workspace),
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"ScanTarget '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the target and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update scan target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to update scan target.") from exc


@router.delete("/targets/{name}", status_code=204, tags=["Garak Plugin Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanTargetPerms.DELETE],
)
async def delete_target(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a scan target by name."""
    try:
        await entity_client.delete(ScanTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"ScanTarget '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the target and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete scan target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to delete scan target.") from exc
