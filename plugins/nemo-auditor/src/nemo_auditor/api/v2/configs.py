# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuditConfig CRUD routes.

Mounted by the plugin service at ``/apis/auditor/v2/workspaces/{workspace}/configs``.
Request bodies are pydantic-validated before any persistence call, so the
plugin enforces schema correctness even though the underlying entity store
treats the ``data`` payload as opaque.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_auditor.api.v2._filters import make_filter_dep
from nemo_auditor.api.v2._perms import AuditConfigPerms
from nemo_auditor.api.v2.schemas import ConfigFilter, CreateAuditConfigRequest, UpdateAuditConfigRequest
from nemo_auditor.authz import scope
from nemo_auditor.entities import AuditConfig
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

_config_filter_dep = make_filter_dep(ConfigFilter)


@router.post("/configs", response_model=AuditConfig, status_code=201, tags=["Auditor Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditConfigPerms.CREATE],
)
async def create_config(
    workspace: str,
    body: CreateAuditConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditConfig:
    """Create a new audit config."""
    config = AuditConfig(
        name=body.name,
        workspace=workspace,
        description=body.description,
        system=body.system,
        run=body.run,
        plugins=body.plugins,
        reporting=body.reporting,
    )
    try:
        return await entity_client.create(config)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"AuditConfig '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create audit config '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to create audit config.") from exc


@router.get(
    "/configs",
    tags=["Auditor Configs"],
    openapi_extra=generate_openapi_extra_params(filter_schema=ConfigFilter),
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditConfigPerms.LIST],
)
async def list_configs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: ConfigFilter = Depends(_config_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List audit configs in the workspace with pagination and filter support."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            AuditConfig,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list audit configs in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list audit configs.") from exc
    return {
        "data": [c.model_dump(mode="json") for c in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": filter or None,
    }


@router.get("/configs/{name}", response_model=AuditConfig, tags=["Auditor Configs"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditConfigPerms.READ],
)
async def get_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditConfig:
    """Get an audit config by name."""
    try:
        return await entity_client.get(AuditConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get audit config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get audit config.") from exc


@router.put("/configs/{name}", response_model=AuditConfig, tags=["Auditor Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditConfigPerms.UPDATE],
)
async def update_config(
    workspace: str,
    name: str,
    body: UpdateAuditConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AuditConfig:
    """Replace an audit config's contents."""
    try:
        existing = await entity_client.get(AuditConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc

    existing.description = body.description
    existing.system = body.system
    existing.run = body.run
    existing.plugins = body.plugins
    existing.reporting = body.reporting

    try:
        return await entity_client.update(existing)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        logger.info(
            "Conflict updating audit config '%s' in workspace '%s'",
            sanitize_for_log(name),
            sanitize_for_log(workspace),
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"AuditConfig '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the config and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update audit config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to update audit config.") from exc


@router.delete("/configs/{name}", status_code=204, tags=["Auditor Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[AuditConfigPerms.DELETE],
)
async def delete_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete an audit config by name."""
    try:
        await entity_client.delete(AuditConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AuditConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"AuditConfig '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the config and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete audit config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to delete audit config.") from exc
