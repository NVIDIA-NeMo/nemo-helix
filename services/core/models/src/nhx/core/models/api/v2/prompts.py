# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nhx.common.api.common import Page
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.dependencies import get_prompt_service
from nhx.core.models.api.service.prompt_service import PromptService
from nhx.core.models.schemas import (
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    Prompt,
    PromptFilter,
    PromptSort,
    UpdatePromptRequest,
)

logger = logging.getLogger(__name__)


def _sanitize_for_log(value: object) -> str:
    """Prevent log injection by removing line-break/control characters."""
    return str(value).replace("\r", "").replace("\n", "")


router = APIRouter()


@router.get(
    "/v2/workspaces/{workspace}/prompts",
    summary="List Prompts By Workspace",
    response_description="Return prompts for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[Prompt],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=PromptFilter,
        filter_description=("Filter prompts by workspace, project, name, description, created_at, and updated_at."),
    ),
)
async def list_prompts(
    workspace: str,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    sort: PromptSort = Query(
        default=PromptSort.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(PromptFilter)),
    service: PromptService = Depends(get_prompt_service),
) -> Page[Prompt]:
    """List prompts for a specific workspace."""
    # Discard any workspace override in the filter — always scope to the path workspace.
    parsed_filter.remove("workspace")
    try:
        return await service.list_prompts(
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
    except Exception:
        logger.exception(f"Failed to list prompts for workspace {_sanitize_for_log(workspace)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/v2/workspaces/{workspace}/prompts",
    summary="Create Prompt",
    response_description="Create a new prompt",
    status_code=status.HTTP_201_CREATED,
)
async def create_prompt(
    workspace: str,
    request: CreatePromptRequest,
    service: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Create a new prompt."""
    safe_workspace = _sanitize_for_log(workspace)
    safe_request_name = _sanitize_for_log(request.name)
    logger.info(f"Creating prompt: {safe_workspace}/{safe_request_name}")
    try:
        return await service.create_prompt(request, workspace)
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during prompt creation: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except ValueError as e:
        if "already exists" in str(e).lower():
            logger.warning(f"Prompt already exists: {safe_workspace}/{safe_request_name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Prompt with workspace '{workspace}' and name '{request.name}' already exists",
            )
        logger.warning(f"Prompt creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid prompt data")
    except Exception:
        logger.exception("Failed to create prompt")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.get(
    "/v2/workspaces/{workspace}/prompts/{name}",
    summary="Get Prompt",
    response_description="Return prompt details",
    status_code=status.HTTP_200_OK,
)
async def get_prompt(
    workspace: str,
    name: str,
    service: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Get a prompt by workspace and name."""
    logger.debug(f"Getting prompt: {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
    try:
        prompt = await service.get_prompt(GetPromptRequest(workspace=workspace, name=name))
        if not prompt:
            logger.warning(f"Prompt not found: {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt not found: {workspace}/{name}",
            )
        return prompt
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Failed to get prompt {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.put(
    "/v2/workspaces/{workspace}/prompts/{name}",
    summary="Update Prompt",
    response_description="Update an existing prompt",
    status_code=status.HTTP_200_OK,
)
async def update_prompt(
    workspace: str,
    name: str,
    request: UpdatePromptRequest,
    service: PromptService = Depends(get_prompt_service),
) -> Prompt:
    """Update an existing prompt (full replacement of mutable fields)."""
    logger.debug(f"Updating prompt: {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
    try:
        prompt = await service.update_prompt(workspace, name, request)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt not found: {workspace}/{name}",
            )
        return prompt
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during prompt update: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update prompt")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.delete(
    "/v2/workspaces/{workspace}/prompts/{name}",
    summary="Delete Prompt",
    response_description="Delete a prompt",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_prompt(
    workspace: str,
    name: str,
    service: PromptService = Depends(get_prompt_service),
):
    """Delete a prompt by workspace and name."""
    logger.info(f"Deleting prompt: {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
    try:
        deleted = await service.delete_prompt(DeletePromptRequest(workspace=workspace, name=name))
        if not deleted:
            logger.warning(f"Prompt not found for deletion: {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt not found: {workspace}/{name}",
            )
        return None
    except HTTPException:
        raise
    except EntityConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from e
    except Exception:
        logger.exception(f"Failed to delete prompt {_sanitize_for_log(workspace)}/{_sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
