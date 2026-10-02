# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.dependencies import get_model_deployment_config_service
from nhx.core.models.api.permissions import check_model_entity_access
from nhx.core.models.api.service.model_deployment_config_service import (
    ModelDeploymentConfigService,
    ReferentialIntegrityError,
)
from nhx.core.models.api.service.model_entity_service import _has_tool_call_plugin, validate_tool_call_plugin_allowed
from nhx.core.models.api.v2.utils import ERR_DEPLOYMENTS_NOT_ENABLED, deployments_enabled
from nhx.core.models.schemas import (
    CreateModelDeploymentConfigRequest,
    ModelDeploymentConfig,
    ModelDeploymentConfigFilter,
    UpdateModelDeploymentConfigRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/deployment-configs",
    summary="List ModelDeploymentConfigs By Workspace",
    response_description="Return model deployment configurations for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[ModelDeploymentConfig],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ModelDeploymentConfigFilter,
        filter_description=(
            "Filter deployment configs by workspace, project, model_entity_id, "
            "name, description, created_at, and updated_at."
        ),
    ),
)
async def list_deployment_configs(
    workspace: str,
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=100, description="Page size."),
    sort: str = Query(
        default="created_at",
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(ModelDeploymentConfigFilter)),
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
) -> Page[ModelDeploymentConfig]:
    """
    List ModelDeploymentConfigs for a specific workspace.
    Returns only the latest version of each config.
    """
    # Extract workspace — inject from path param if not in filter
    filter_workspace = parsed_filter.remove("workspace") or workspace

    logger.info(f"Listing deployment configs for workspace: {filter_workspace}")

    try:
        result = await service.list_deployment_configs(
            workspace=filter_workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
        return result
    except Exception:
        logger.exception("Unexpected error listing deployment configs by workspace")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list deployment configs"
        )


@router.post(
    "/v2/workspaces/{workspace}/deployment-configs",
    summary="Create ModelDeploymentConfig",
    response_description="Create a new model deployment configuration",
    status_code=status.HTTP_201_CREATED,
)
async def create_deployment_config(
    workspace: str,
    config_input: CreateModelDeploymentConfigRequest,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelDeploymentConfig:
    """
    Create a new ModelDeploymentConfig (version 1).
    """
    logger.info(f"Creating deployment config: {workspace}/{config_input.name}")

    try:
        if config_input.model_entity_id:
            await check_model_entity_access(auth_client, config_input.model_entity_id, workspace)
        if _has_tool_call_plugin(config_input):
            await validate_tool_call_plugin_allowed(auth_client, workspace)
        config = await service.create_deployment_config(config_input, workspace)
        return config
    except PermissionError as e:
        logger.warning(f"Permission denied during deployment config creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Deployment config already exists: {workspace}/{config_input.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Deployment config with workspace '{workspace}' and name '{config_input.name}' already exists",
            )
        logger.warning(f"Deployment config creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during deployment config creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except Exception:
        logger.exception("Unexpected error creating deployment config")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create deployment config"
        )


@router.get(
    "/v2/workspaces/{workspace}/deployment-configs/{name}",
    summary="Get Latest ModelDeploymentConfig Version",
    response_description="Return the latest version of a model deployment configuration",
    status_code=status.HTTP_200_OK,
)
async def get_latest_deployment_config(
    workspace: str,
    name: str,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
) -> ModelDeploymentConfig:
    """
    Get the latest version of a ModelDeploymentConfig.
    """
    deployment_config_name = name
    logger.info(f"Getting latest deployment config: {workspace}/{deployment_config_name}")

    try:
        config = await service.get_deployment_config(workspace, deployment_config_name)
        if not config:
            logger.warning(f"Deployment config not found: {workspace}/{deployment_config_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment config '{workspace}/{deployment_config_name}' not found",
            )
        return config
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error getting deployment config")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get deployment config")


@router.get(
    "/v2/workspaces/{workspace}/deployment-configs/{name}/versions",
    summary="List ModelDeploymentConfig Versions",
    response_description="Return all versions of a model deployment configuration",
    status_code=status.HTTP_200_OK,
)
async def list_deployment_config_versions(
    workspace: str,
    name: str,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
) -> List[ModelDeploymentConfig]:
    """
    List all versions of a ModelDeploymentConfig.
    """
    deployment_config_name = name
    logger.info(f"Listing deployment config versions: {workspace}/{deployment_config_name}")

    try:
        versions = await service.list_deployment_config_versions(workspace, deployment_config_name)
        return versions
    except Exception:
        logger.exception("Unexpected error listing deployment config versions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list deployment config versions"
        )


@router.get(
    "/v2/workspaces/{workspace}/deployment-configs/{config}/versions/{name}",
    summary="Get Specific ModelDeploymentConfig Version",
    response_description="Return a specific version of a model deployment configuration",
    status_code=status.HTTP_200_OK,
)
async def get_deployment_config_version(
    workspace: str,
    config: str,
    name: str,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
) -> ModelDeploymentConfig:
    """
    Get a specific version of a ModelDeploymentConfig.
    """
    deployment_config_name = config
    logger.info(f"Getting deployment config: {workspace}/{deployment_config_name} version {name}")

    try:
        version_int = int(name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid version '{name}': must be a positive integer",
        )

    try:
        result = await service.get_deployment_config(workspace, deployment_config_name, version_int)
        if not result:
            logger.warning(f"Deployment config version not found: {workspace}/{deployment_config_name} version {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment config '{workspace}/{deployment_config_name}' version '{name}' not found",
            )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error getting deployment config version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get deployment config version"
        )


@router.post(
    "/v2/workspaces/{workspace}/deployment-configs/{name}",
    summary="Update ModelDeploymentConfig",
    response_description="Update a model deployment configuration (creates new version)",
    status_code=status.HTTP_201_CREATED,
)
async def update_deployment_config(
    workspace: str,
    name: str,
    config_input: UpdateModelDeploymentConfigRequest,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelDeploymentConfig:
    """
    Update a ModelDeploymentConfig (creates a new immutable version).
    """
    deployment_config_name = name
    logger.info(f"Updating deployment config: {workspace}/{deployment_config_name}")
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    try:
        if config_input.model_entity_id:
            await check_model_entity_access(auth_client, config_input.model_entity_id, workspace)
        if _has_tool_call_plugin(config_input):
            await validate_tool_call_plugin_allowed(auth_client, workspace)
        config = await service.update_deployment_config(workspace, deployment_config_name, config_input)
        return config
    except PermissionError as e:
        logger.warning(f"Permission denied during deployment config update: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        logger.exception("Failed to update deployment config")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during deployment config update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except Exception:
        logger.exception("Unexpected error updating deployment config")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update deployment config"
        )


@router.delete(
    "/v2/workspaces/{workspace}/deployment-configs/{name}",
    summary="Delete All ModelDeploymentConfig Versions",
    response_description="Delete all versions of a model deployment configuration",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Deployment config deleted successfully"},
        404: {"description": "Deployment config not found"},
        409: {"description": "Cannot delete - dependent ModelDeployments exist"},
    },
)
async def delete_all_deployment_config_versions(
    workspace: str,
    name: str,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
):
    """
    Delete all versions of a ModelDeploymentConfig.

    This operation will fail with 409 Conflict if any ModelDeployments currently
    reference this config and are not in DELETED status. Delete or wait for
    dependent deployments to reach DELETED status before deleting the config.
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_config_name = name
    logger.info(f"Deleting all versions of deployment config: {workspace}/{deployment_config_name}")

    try:
        success = await service.delete_deployment_config(workspace, deployment_config_name)
        if not success:
            logger.warning(f"Deployment config not found for deletion: {workspace}/{deployment_config_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment config '{workspace}/{deployment_config_name}' not found",
            )
        return None
    except ReferentialIntegrityError as e:
        logger.warning(f"Referential integrity violation: {e.message}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=e.message)
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception("Unexpected error deleting deployment config")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete deployment config"
        )


@router.delete(
    "/v2/workspaces/{workspace}/deployment-configs/{config}/versions/{name}",
    summary="Delete Specific ModelDeploymentConfig Version",
    response_description="Delete a specific version of a model deployment configuration",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Deployment config version deleted successfully"},
        404: {"description": "Deployment config version not found"},
        409: {"description": "Cannot delete - dependent ModelDeployments exist"},
    },
)
async def delete_deployment_config_version(
    workspace: str,
    config: str,
    name: str,
    service: ModelDeploymentConfigService = Depends(get_model_deployment_config_service),
):
    """
    Delete a specific version of a ModelDeploymentConfig.

    This operation will fail with 409 Conflict if any ModelDeployments currently
    reference this specific version and are not in DELETED status. Delete or wait for
    dependent deployments to reach DELETED status before deleting the config version.
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_config_name = config
    logger.info(f"Deleting deployment config version: {workspace}/{deployment_config_name} version {name}")

    try:
        version_int = int(name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid version '{name}': must be a positive integer",
        )

    try:
        success = await service.delete_deployment_config(workspace, deployment_config_name, version_int)
        if not success:
            logger.warning(
                f"Deployment config version not found for deletion: {workspace}/{deployment_config_name} version {name}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment config '{workspace}/{deployment_config_name}' version '{name}' not found",
            )
        return None
    except ReferentialIntegrityError as e:
        logger.warning(f"Referential integrity violation: {e.message}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=e.message)
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception("Unexpected error deleting deployment config version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete deployment config version"
        )
