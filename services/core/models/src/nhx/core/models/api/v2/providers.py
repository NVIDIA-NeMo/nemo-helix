# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_helix import AsyncNeMoHelix
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, AuthContext, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.common.service.dependencies import get_sdk_client
from nhx.core.models.api.dependencies import get_model_provider_service
from nhx.core.models.api.permissions import check_deployment_access, check_secret_access
from nhx.core.models.api.service.model_provider_service import ModelProviderService, ModelProviderValidationError
from nhx.core.models.schemas import (
    CreateModelProviderRequest,
    DeleteModelProviderRequest,
    GetModelProviderRequest,
    ModelProvider,
    ModelProviderFilter,
    ModelProviderSort,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/providers",
    summary="List ModelProviders By Workspace",
    response_description="Return model providers for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[ModelProvider],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ModelProviderFilter,
        filter_description=(
            "Filter model providers by workspace, project, status, model_deployment_id, "
            "name, description, host_url, created_at, and updated_at."
        ),
    ),
)
async def list_providers(
    workspace: str,
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=100, description="Page size."),
    sort: ModelProviderSort = Query(
        default=ModelProviderSort.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(ModelProviderFilter)),
    service: ModelProviderService = Depends(get_model_provider_service),
) -> Page[ModelProvider]:
    """
    List model providers for a specific workspace.
    """
    # Extract workspace — inject from path param if not in filter
    filter_workspace = parsed_filter.remove("workspace") or workspace
    try:
        result = await service.list_model_providers(
            workspace=filter_workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
        return result
    except Exception as e:
        logger.exception(f"Failed to list model providers for workspace {workspace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/v2/workspaces/{workspace}/providers",
    summary="Create ModelProvider",
    response_description="Create a new model provider",
    status_code=status.HTTP_201_CREATED,
)
async def create_provider(
    workspace: str,
    request: CreateModelProviderRequest,
    service: ModelProviderService = Depends(get_model_provider_service),
    auth_client: AuthClient = Depends(get_auth_client),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> ModelProvider:
    """
    Create a new model provider.
    """
    logger.info(f"Creating model provider: {workspace}/{request.name}")
    auth_context = AuthContext.from_principal(auth_client.principal)

    try:
        if request.api_key_secret_name:
            await check_secret_access(nhx_sdk, request.api_key_secret_name, workspace)
        if request.model_deployment_id:
            await check_deployment_access(auth_client, request.model_deployment_id, workspace)
        provider = await service.create_model_provider(request, workspace, auth_context=auth_context)
        return provider
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during model provider creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except ModelProviderValidationError as e:
        logger.warning(f"Model provider validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        logger.warning(f"Permission denied during model provider creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Model provider already exists: {workspace}/{request.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model provider with workspace '{workspace}' and name '{request.name}' already exists",
            )
        logger.warning(f"Model provider creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create model provider")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put(
    "/v2/workspaces/{workspace}/providers/{name}",
    summary="Upsert ModelProvider",
    response_description="Create or update a model provider",
    status_code=status.HTTP_200_OK,
)
async def upsert_provider(
    workspace: str,
    name: str,
    request: UpsertModelProviderRequest,
    service: ModelProviderService = Depends(get_model_provider_service),
    auth_client: AuthClient = Depends(get_auth_client),
    nhx_sdk: AsyncNeMoHelix = Depends(get_sdk_client),
) -> ModelProvider:
    """
    Create or update a model provider.
    """
    logger.debug(f"Upserting model provider: {workspace}/{name}")
    auth_context = AuthContext.from_principal(auth_client.principal)

    try:
        if request.api_key_secret_name:
            await check_secret_access(nhx_sdk, request.api_key_secret_name, workspace)
        if request.model_deployment_id:
            await check_deployment_access(auth_client, request.model_deployment_id, workspace)
        provider = await service.upsert_model_provider(workspace, name, request, auth_context=auth_context)
        return provider
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during model provider upsert: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except ModelProviderValidationError as e:
        logger.warning(f"Model provider validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        logger.warning(f"Permission denied during model provider upsert: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        logger.warning(f"Model provider upsert validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Failed to upsert model provider")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put(
    "/v2/workspaces/{workspace}/providers/{name}/status",
    summary="Update ModelProvider Status Fields",
    response_description="Update status-related fields of a model provider",
    status_code=status.HTTP_200_OK,
)
async def update_provider_status(
    workspace: str,
    name: str,
    request: UpdateModelProviderStatusRequest,
    service: ModelProviderService = Depends(get_model_provider_service),
) -> ModelProvider:
    """
    Update status-related fields of a model provider.

    This endpoint supports partial updates for fields managed by Models Controller:
    - model_deployment_id
    - served_models
    - status
    - status_message

    If status is provided without status_message, status_message will be set to empty string.
    """
    logger.debug(f"Updating model provider status fields: {workspace}/{name}")

    try:
        provider = await service.update_model_provider_status(workspace, name, request)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model provider {workspace}/{name} not found",
            )
        return provider
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during model provider status update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update model provider status fields")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/v2/workspaces/{workspace}/providers/{name}",
    summary="Get ModelProvider",
    response_description="Return model provider details",
    status_code=status.HTTP_200_OK,
)
async def get_provider(
    workspace: str,
    name: str,
    service: ModelProviderService = Depends(get_model_provider_service),
) -> ModelProvider:
    """
    Get a model provider by workspace and name.
    """
    logger.debug(f"Getting model provider: {workspace}/{name}")

    try:
        request = GetModelProviderRequest(workspace=workspace, name=name)
        provider = await service.get_model_provider(request)

        if not provider:
            logger.warning(f"Model provider not found: {workspace}/{name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model provider not found: {workspace}/{name}",
            )

        return provider
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get model provider {workspace}/{name}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    "/v2/workspaces/{workspace}/providers/{name}",
    summary="Delete ModelProvider",
    response_description="Delete a model provider",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_provider(
    workspace: str,
    name: str,
    service: ModelProviderService = Depends(get_model_provider_service),
):
    """
    Delete a model provider by workspace and name.
    """
    logger.info(f"Deleting model provider: {workspace}/{name}")

    try:
        request = DeleteModelProviderRequest(workspace=workspace, name=name)
        deleted = await service.delete_model_provider(request)

        if not deleted:
            logger.warning(f"Model provider not found for deletion: {workspace}/{name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model provider not found: {workspace}/{name}",
            )

        return None
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception as e:
        logger.exception(f"Failed to delete model provider {workspace}/{name}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
