# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, AuthContext, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.dependencies import get_model_deployment_service
from nhx.core.models.api.permissions import check_deployment_config_access
from nhx.core.models.api.service.model_deployment_service import DeploymentStatusConflictError, ModelDeploymentService
from nhx.core.models.api.v2.utils import ERR_DEPLOYMENTS_NOT_ENABLED, deployments_enabled
from nhx.core.models.schemas import (
    CreateModelDeploymentRequest,
    ModelDeployment,
    ModelDeploymentFilter,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/deployments",
    summary="List ModelDeployments",
    response_description="Return model deployments for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[ModelDeployment],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=ModelDeploymentFilter,
        filter_description=(
            "Filter deployments by workspace, project, status, config, model_provider_id, "
            "name, status_message, created_at, and updated_at."
        ),
    ),
)
async def list_deployments(
    workspace: str,
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=100, description="Page size."),
    sort: str = Query(
        default="created_at",
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    all_versions: bool = Query(
        default=False,
        description="If true, return all versions of each deployment. If false (default), return only the latest version.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(ModelDeploymentFilter)),
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> Page[ModelDeployment]:
    """
    List ModelDeployments for a specific workspace.

    By default, returns only the latest version of each deployment.
    """
    # Extract workspace — inject from path param if not in filter
    filter_workspace = parsed_filter.remove("workspace") or workspace

    logger.debug(f"Listing deployments for workspace: {filter_workspace}")

    try:
        result = await service.list_deployments(
            workspace=filter_workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
            all_versions=all_versions,
        )
        return result
    except Exception:
        logger.exception("Unexpected error listing deployments by workspace")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list deployments")


@router.post(
    "/v2/workspaces/{workspace}/deployments",
    summary="Create ModelDeployment",
    response_description="Create a new model deployment",
    status_code=status.HTTP_201_CREATED,
)
async def create_deployment(
    workspace: str,
    deployment_input: CreateModelDeploymentRequest,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelDeployment:
    """
    Create a new ModelDeployment (version 1).
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)

    logger.debug(f"Creating deployment: {workspace}/{deployment_input.name}")

    auth_context = AuthContext.from_principal(auth_client.principal)
    try:
        await check_deployment_config_access(auth_client, deployment_input.config, workspace)

        deployment = await service.create_deployment(deployment_input, workspace, auth_context=auth_context)
        return deployment
    except PermissionError as e:
        logger.warning(f"Permission denied during deployment creation: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Deployment already exists: {workspace}/{deployment_input.name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Deployment with workspace '{workspace}' and name '{deployment_input.name}' already exists",
            )
        logger.warning(f"Deployment creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during deployment creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except Exception:
        logger.exception("Unexpected error creating deployment")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create deployment")


@router.get(
    "/v2/workspaces/{workspace}/deployments/{name}",
    summary="Get Latest ModelDeployment",
    response_description="Return the latest version of a model deployment",
    status_code=status.HTTP_200_OK,
)
async def get_latest_deployment(
    workspace: str,
    name: str,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> ModelDeployment:
    """
    Get the latest version of a ModelDeployment.
    """
    deployment_name = name
    logger.debug(f"Getting latest deployment: {workspace}/{deployment_name}")

    try:
        deployment = await service.get_deployment(workspace, deployment_name)
        if not deployment:
            logger.warning(f"Deployment not found: {workspace}/{deployment_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Deployment '{workspace}/{deployment_name}' not found"
            )
        return deployment
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error getting deployment")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get deployment")


@router.get(
    "/v2/workspaces/{workspace}/deployments/{name}/models",
    summary="Get Latest ModelDeployment's Model Entities",
    response_description="Return model entities from Entity Store for the latest deployment",
    status_code=status.HTTP_200_OK,
)
async def get_deployment_models(
    workspace: str,
    name: str,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> dict:
    """
    Get Latest ModelDeployment's Model Entities from Entity Store.
    This provides the API contract that NIMs expect from Entity Store today, for pulling LoRAs,
    but enables us to enforce AuthZ boundaries.

    TODO: Implement model entity retrieval based on deployment config.
    """
    deployment_name = name
    logger.debug(f"Getting deployment models: {workspace}/{deployment_name}")

    try:
        deployment = await service.get_deployment(workspace, deployment_name)
        if not deployment:
            logger.warning(f"Deployment not found for models query: {workspace}/{deployment_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Deployment '{workspace}/{deployment_name}' not found"
            )

        # TODO: Query Entity Store for models based on deployment.config
        # For now, return empty list
        return {"workspace": workspace, "name": deployment_name, "models": []}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error getting deployment models")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get deployment models")


@router.get(
    "/v2/workspaces/{workspace}/deployments/{name}/versions",
    summary="List ModelDeployment Versions",
    response_description="Return all versions of a model deployment",
    status_code=status.HTTP_200_OK,
)
async def list_deployment_versions(
    workspace: str,
    name: str,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> List[ModelDeployment]:
    """
    List all versions of a ModelDeployment.
    """
    deployment_name = name
    logger.debug(f"Listing deployment versions: {workspace}/{deployment_name}")

    try:
        versions = await service.list_deployment_versions(workspace, deployment_name)
        return versions
    except Exception:
        logger.exception("Unexpected error listing deployment versions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list deployment versions"
        )


@router.get(
    "/v2/workspaces/{workspace}/deployments/{deployment}/versions/{name}",
    summary="Get Specific ModelDeployment Version",
    response_description="Return a specific version of a model deployment",
    status_code=status.HTTP_200_OK,
)
async def get_deployment_version(
    workspace: str,
    deployment: str,
    name: str,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> ModelDeployment:
    """
    Get a specific version of a ModelDeployment.
    """
    deployment_name = deployment
    logger.debug(f"Getting deployment: {workspace}/{deployment_name} version {name}")

    try:
        version_int = int(name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid version '{name}': must be a positive integer",
        )

    try:
        result = await service.get_deployment(workspace, deployment_name, version_int)
        if not result:
            logger.warning(f"Deployment version not found: {workspace}/{deployment_name} version {name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment '{workspace}/{deployment_name}' version '{name}' not found",
            )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error getting deployment version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get deployment version"
        )


@router.post(
    "/v2/workspaces/{workspace}/deployments/{name}",
    summary="Update ModelDeployment",
    response_description="Update a model deployment (creates new version)",
    status_code=status.HTTP_201_CREATED,
)
async def update_deployment(
    workspace: str,
    name: str,
    deployment_input: UpdateModelDeploymentRequest,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
    auth_client: AuthClient = Depends(get_auth_client),
) -> ModelDeployment:
    """
    Update a ModelDeployment (creates a new immutable version).
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_name = name
    logger.debug(f"Updating deployment: {workspace}/{deployment_name}")

    try:
        await check_deployment_config_access(auth_client, deployment_input.config, workspace)

        deployment = await service.update_deployment(
            workspace, deployment_name, deployment_input, auth_context=AuthContext.from_principal(auth_client.principal)
        )
        return deployment
    except PermissionError as e:
        logger.warning(f"Permission denied during deployment update: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        logger.exception("Failed to update deployment")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during deployment update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except Exception:
        logger.exception("Unexpected error updating deployment")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update deployment")


@router.post(
    "/v2/workspaces/{workspace}/deployments/{name}/status",
    summary="Update ModelDeployment Status",
    response_description="Update ModelDeployment status and status_message",
    status_code=status.HTTP_200_OK,
)
async def update_deployment_status(
    workspace: str,
    name: str,
    status_input: UpdateModelDeploymentStatusRequest,
    version: str | None = None,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
) -> ModelDeployment:
    """
    Update the status of a ModelDeployment (mutable operation).
    If version is not specified, updates the latest version.
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_name = name
    logger.debug(
        f"Updating deployment status: {workspace}/{deployment_name}"
        + (f" version {version}" if version else " (latest)")
    )

    version_int = None
    if version:
        try:
            version_int = int(version)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid version '{version}': must be a positive integer",
            )

    try:
        deployment = await service.update_deployment_status(workspace, deployment_name, status_input, version_int)
        if not deployment:
            logger.warning(
                f"Deployment not found for status update: {workspace}/{deployment_name}"
                + (f" version {version}" if version else "")
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployment '{workspace}/{deployment_name}'"
                + (f" version '{version}'" if version else "")
                + " not found",
            )
        return deployment
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during deployment status update: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        )
    except DeploymentStatusConflictError as e:
        logger.warning(f"Deployment status conflict: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error updating deployment status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update deployment status"
        )


@router.delete(
    "/v2/workspaces/{workspace}/deployments/{name}",
    summary="Delete All ModelDeployment Versions",
    response_description="Mark deployment for deletion or hard-delete if already DELETED",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Deployment marked for deletion (DELETING status)"},
        204: {"description": "Deployment hard-deleted from database (was already DELETED)"},
        404: {"description": "Deployment not found"},
    },
)
async def delete_all_deployment_versions(
    workspace: str,
    name: str,
    response: Response,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
):
    """
    Delete all versions of a ModelDeployment.

    If the deployment is in any state other than DELETED, this will set its status to DELETING.
    The models controller will then:
    1. Delete the infrastructure (e.g., K8s NimService)
    2. Update the status to DELETED

    If the deployment is already in DELETED status, calling delete again will permanently
    remove it from the database.

    Returns:
    - 202 Accepted: Deployment marked for deletion (status set to DELETING)
    - 204 No Content: Deployment permanently removed from database (was already DELETED)
    - 404 Not Found: Deployment doesn't exist
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_name = name
    logger.debug(f"Deleting all versions of deployment: {workspace}/{deployment_name}")

    try:
        result = await service.delete_deployment(workspace, deployment_name)
        if result is None:
            # Either hard-deleted or not found
            # Check if it was not found by trying to get it
            existing = await service.get_deployment(workspace, deployment_name)
            if existing is None:
                logger.warning(f"Deployment not found for deletion: {workspace}/{deployment_name}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment '{workspace}/{deployment_name}' not found",
                )
            # Hard-deleted (was DELETED)
            response.status_code = status.HTTP_204_NO_CONTENT
            return None
        else:
            # Marked for deletion (status set to DELETING)
            response.status_code = status.HTTP_202_ACCEPTED
            return None
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception("Unexpected error deleting deployment")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete deployment")


@router.delete(
    "/v2/workspaces/{workspace}/deployments/{deployment}/versions/{name}",
    summary="Delete Specific ModelDeployment Version",
    response_description="Mark deployment version for deletion or hard-delete if already DELETED",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Deployment version marked for deletion (DELETING status)"},
        204: {"description": "Deployment version hard-deleted from database (was already DELETED)"},
        404: {"description": "Deployment version not found"},
    },
)
async def delete_deployment_version(
    workspace: str,
    deployment: str,
    name: str,
    response: Response,
    service: ModelDeploymentService = Depends(get_model_deployment_service),
):
    """
    Delete a specific version of a ModelDeployment.

    If the deployment is in any state other than DELETED, this will set its status to DELETING.
    The models controller will then:
    1. Delete the infrastructure (e.g., K8s NimService)
    2. Update the status to DELETED

    If the deployment is already in DELETED status, calling delete again will permanently
    remove it from the database.

    Returns:
    - 202 Accepted: Deployment version marked for deletion (status set to DELETING)
    - 204 No Content: Deployment version permanently removed from database (was already DELETED)
    - 404 Not Found: Deployment version doesn't exist
    """
    if not deployments_enabled():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=ERR_DEPLOYMENTS_NOT_ENABLED)
    deployment_name = deployment
    logger.debug(f"Deleting deployment version: {workspace}/{deployment_name} version {name}")

    try:
        version_int = int(name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid version '{name}': must be a positive integer",
        )

    try:
        result = await service.delete_deployment(workspace, deployment_name, version_int)
        if result is None:
            # Either hard-deleted or not found
            # Check if it was not found by trying to get it
            existing = await service.get_deployment(workspace, deployment_name, version_int)
            if existing is None:
                logger.warning(
                    f"Deployment version not found for deletion: {workspace}/{deployment_name} version {name}"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Deployment '{workspace}/{deployment_name}' version '{name}' not found",
                )
            # Hard-deleted (was DELETED)
            response.status_code = status.HTTP_204_NO_CONTENT
            return None
        else:
            # Marked for deletion (status set to DELETING)
            response.status_code = status.HTTP_202_ACCEPTED
            return None
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception("Unexpected error deleting deployment version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete deployment version"
        )
