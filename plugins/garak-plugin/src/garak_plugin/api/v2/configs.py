# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanConfig CRUD routes.

Mounted by the plugin service at ``/apis/garak-plugin/v2/workspaces/{workspace}/configs``.
Request bodies are pydantic-validated before any persistence call, so the
plugin enforces schema correctness even though the underlying entity store
treats the ``data`` payload as opaque.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from garak_plugin.api.v2._filters import make_filter_dep
from garak_plugin.api.v2._perms import ScanConfigPerms
from garak_plugin.api.v2.schemas import ConfigFilter, CreateScanConfigRequest, UpdateScanConfigRequest
from garak_plugin.authz import scope
from garak_plugin.entities import ScanConfig
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


@router.post("/configs", response_model=ScanConfig, status_code=201, tags=["Garak Plugin Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanConfigPerms.CREATE],
)
async def create_config(
    workspace: str,
    body: CreateScanConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanConfig:
    """Create a new scan config."""
    config = ScanConfig(
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
            detail=f"ScanConfig '{body.name}' already exists in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to create scan config '%s'", body.name)
        raise HTTPException(status_code=500, detail="Failed to create scan config.") from exc


@router.get(
    "/configs",
    tags=["Garak Plugin Configs"],
    openapi_extra=generate_openapi_extra_params(filter_schema=ConfigFilter),
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanConfigPerms.LIST],
)
async def list_configs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: ConfigFilter = Depends(_config_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List scan configs in the workspace with pagination and filter support."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            ScanConfig,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list scan configs in workspace '%s'", workspace)
        raise HTTPException(status_code=500, detail="Failed to list scan configs.") from exc
    return {
        "data": [c.model_dump(mode="json") for c in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": filter or None,
    }


@router.get("/configs/{name}", response_model=ScanConfig, tags=["Garak Plugin Configs"])
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanConfigPerms.READ],
)
async def get_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanConfig:
    """Get a scan config by name."""
    try:
        return await entity_client.get(ScanConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get scan config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get scan config.") from exc


@router.put("/configs/{name}", response_model=ScanConfig, tags=["Garak Plugin Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanConfigPerms.UPDATE],
)
async def update_config(
    workspace: str,
    name: str,
    body: UpdateScanConfigRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ScanConfig:
    """Replace a scan config's contents."""
    try:
        existing = await entity_client.get(ScanConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanConfig '{name}' not found in workspace '{workspace}'.",
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
            detail=f"ScanConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        logger.info(
            "Conflict updating scan config '%s' in workspace '%s'",
            sanitize_for_log(name),
            sanitize_for_log(workspace),
            exc_info=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"ScanConfig '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the config and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to update scan config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to update scan config.") from exc


@router.delete("/configs/{name}", status_code=204, tags=["Garak Plugin Configs"])
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[ScanConfigPerms.DELETE],
)
async def delete_config(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a scan config by name."""
    try:
        await entity_client.delete(ScanConfig, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"ScanConfig '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"ScanConfig '{name}' was modified by another request in workspace '{workspace}'. "
                "Refresh the config and try again."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete scan config '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to delete scan config.") from exc
