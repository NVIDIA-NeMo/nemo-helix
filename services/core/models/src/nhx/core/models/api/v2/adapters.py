# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level Adapters API (adapter entities scoped by workspace)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.dependencies import get_adapter_entity_service, get_files_client
from nhx.core.models.api.permissions import check_fileset_access
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_entity_service import FilesetValidationError
from nhx.core.models.schemas import (
    Adapter,
    AdapterEntityFilter,
    CreateAdapterRequest,
    CreateModelAdapterRequest,
    UpdateAdapterRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/adapters",
    summary="Create Adapter",
    response_description="Create a new adapter for a model",
    status_code=status.HTTP_201_CREATED,
    response_model=Adapter,
)
async def create_adapter(
    workspace: str,
    adapter_create: CreateAdapterRequest,
    service: AdapterEntityService = Depends(get_adapter_entity_service),
    files: AsyncFilesClient = Depends(get_files_client),
) -> Adapter:
    """Create an adapter under a base model specified by the "model" field in the body."""
    logger.info(f"Creating adapter entity: {workspace} for model {adapter_create.model!r}")
    try:
        await check_fileset_access(files, adapter_create.fileset, workspace)
        body = CreateModelAdapterRequest.model_validate(
            adapter_create.model_dump(exclude={"model"}, exclude_unset=True)
        )
        created_adapter = await service.create_adapter(workspace, body, base_model=adapter_create.model)
        if created_adapter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Base model for adapter (model={adapter_create.model!r}) not found",
            )
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during adapter creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except PermissionError as e:
        logger.warning(f"Permission denied during adapter creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as err:
        if "already exists" in str(err).lower():
            logger.warning(
                f"Adapter already exists: {workspace}/{adapter_create.name} (model {adapter_create.model!r}) - {err}"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(err),
            )
        logger.warning(f"Adapter create validation error: {err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create adapter - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create adapter")

    return created_adapter


@router.get(
    "/v2/workspaces/{workspace}/adapters",
    summary="List Adapters",
    response_description="List adapters in the workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[Adapter],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=AdapterEntityFilter,
        filter_description=(
            "Filter adapters by name, model (parent model ref string, stored on the adapter), "
            "description, fileset, finetuning_type, enabled, created_at, and updated_at."
        ),
    ),
)
async def list_adapters(
    workspace: str,
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=100, description="Page size.", ge=1, le=1000),
    sort: str = Query(
        default="created_at",
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(AdapterEntityFilter)),
    service: AdapterEntityService = Depends(get_adapter_entity_service),
) -> Page[Adapter]:
    try:
        return await service.list_adapters(
            adapter_workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            parsed_filter=parsed_filter,
        )
    except Exception:
        logger.exception("Failed to list adapter entities")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list adapter entities")


@router.get(
    "/v2/workspaces/{workspace}/adapters/{name}",
    summary="Get Adapter",
    response_description="Get one adapter by name. Parent model is taken from the adapter's stored parent (entity `parent` field).",
    status_code=status.HTTP_200_OK,
    response_model=Adapter,
)
async def get_adapter(
    workspace: str,
    name: str,
    service: AdapterEntityService = Depends(get_adapter_entity_service),
) -> Adapter:
    try:
        a = await service.get_adapter(workspace, name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if a is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter {name!r} not found in workspace {workspace!r}",
        )
    return a


@router.delete(
    "/v2/workspaces/{workspace}/adapters/{name}",
    summary="Delete Adapter",
    response_description="Delete adapter by name. The entity store delete uses the adapter's stored parent (entity `parent` field).",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_adapter(
    workspace: str,
    name: str,
    service: AdapterEntityService = Depends(get_adapter_entity_service),
) -> None:
    try:
        deleted = await service.delete_adapter_in_workspace(workspace, name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    if deleted == -2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter {name!r} not found in workspace {workspace!r}",
        )
    return None


@router.patch(
    "/v2/workspaces/{workspace}/adapters/{name}",
    summary="Update Adapter",
    response_description="Update adapter metadata. Updates are applied to the row identified by the adapter's stored parent (entity `parent` field).",
    status_code=status.HTTP_200_OK,
    response_model=Adapter,
)
async def update_adapter(
    workspace: str,
    name: str,
    adapter_update: UpdateAdapterRequest,
    service: AdapterEntityService = Depends(get_adapter_entity_service),
    files: AsyncFilesClient = Depends(get_files_client),
) -> Adapter:
    try:
        if adapter_update.fileset:
            await check_fileset_access(files, adapter_update.fileset, workspace)
        updated = await service.update_adapter_in_workspace(workspace, name, adapter_update)
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during adapter update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except PermissionError as e:
        logger.warning(f"Permission denied during adapter update: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except FilesetValidationError as e:
        logger.warning(f"Fileset validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        logger.warning(f"Adapter update: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update adapter - {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update adapter")

    if updated == -2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter {name!r} not found in workspace {workspace!r}",
        )
    return updated
