# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuditTarget CRUD routes.

Mounted by the plugin service at ``/apis/auditor/v2/workspaces/{workspace}/targets``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_auditor.api.v2._filters import make_filter_dep
from nemo_auditor.api.v2._perms import AuditTargetPerms
from nemo_auditor.api.v2.schemas import CreateAuditTargetRequest, TargetFilter, UpdateAuditTargetRequest
from nemo_auditor.authz import scope
from nemo_auditor.entities import AuditTarget
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


@router.post("/targets", response_model=AuditTarget, status_code=201, tags=["Auditor Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditTargetPerms.CREATE],
)
async def create_target(
    workspace: str,
    body: CreateAuditTargetRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditTarget:
    """Create a new audit target."""
    target = AuditTarget(
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
            detail=f"AuditTarget '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create audit target '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to create audit target.") from exc


@router.get(
    "/targets",
    tags=["Auditor Targets"],
    openapi_extra=generate_openapi_extra_params(filter_schema=TargetFilter),
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditTargetPerms.LIST],
)
async def list_targets(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    parsed_filter: TargetFilter = Depends(_target_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List audit targets in the workspace with pagination and filter support."""
    filter_dict = parsed_filter if isinstance(parsed_filter, dict) else parsed_filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            AuditTarget,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list audit targets in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list audit targets.") from exc
    return {
        "data": [t.model_dump(mode="json") for t in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": parsed_filter or None,
    }


@router.get("/targets/{name}", response_model=AuditTarget, tags=["Auditor Targets"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditTargetPerms.READ],
)
async def get_target(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditTarget:
    """Get an audit target by name."""
    try:
        return await entity_client.get(AuditTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get audit target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get audit target.") from exc


@router.put("/targets/{name}", response_model=AuditTarget, tags=["Auditor Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditTargetPerms.UPDATE],
)
async def update_target(
    workspace: str,
    name: str,
    body: UpdateAuditTargetRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditTarget:
    """Replace an audit target's contents."""
    try:
        existing = await entity_client.get(AuditTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditTarget '{name}' not found in workspace '{workspace}'.",
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
            detail=f"AuditTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        logger.info(
            "Conflict updating audit target '%s' in workspace '%s'",
            sanitize_for_log(name),
            sanitize_for_log(workspace),
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"AuditTarget '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the target and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update audit target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to update audit target.") from exc


@router.delete("/targets/{name}", status_code=204, tags=["Auditor Targets"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditTargetPerms.DELETE],
)
async def delete_target(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an audit target by name."""
    try:
        await entity_client.delete(AuditTarget, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditTarget '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"AuditTarget '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the target and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete audit target '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to delete audit target.") from exc
