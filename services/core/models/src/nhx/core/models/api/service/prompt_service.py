# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service layer for Prompt operations using EntityClient."""

import logging

from nhx.common.api.common import Page, PaginationData
from nhx.common.api.filter import FilterOperation
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.models.entities import Prompt as PromptEntity
from nhx.core.models.schemas import (
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    Prompt,
    UpdatePromptRequest,
)

logger = logging.getLogger(__name__)


def _entity_to_schema(entity: PromptEntity) -> Prompt:
    """Convert an EntityBase Prompt to the API schema."""
    return Prompt(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        project=entity.project,
        description=entity.description,
        messages=entity.messages,
        input_variables=entity.input_variables,
        tools=entity.tools,
        tool_choice=entity.tool_choice,
        response_format=entity.response_format,
        inference_params=entity.inference_params,
        tags=entity.tags,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


class PromptService:
    """Service layer for Prompt operations."""

    def __init__(self, entity_client: EntityClient):
        self.entity_client = entity_client

    async def create_prompt(self, request: CreatePromptRequest, workspace: str) -> Prompt:
        """Create a new prompt."""
        logger.debug("Creating prompt", extra={"workspace": workspace, "prompt_name": request.name})

        try:
            await self.entity_client.get(PromptEntity, name=request.name, workspace=workspace)
            logger.warning("Prompt already exists", extra={"workspace": workspace, "prompt_name": request.name})
            raise ValueError(f"Prompt with name '{request.name}' already exists in workspace '{workspace}'")
        except EntityNotFoundError:
            pass  # Expected - prompt doesn't exist, proceed with creation

        entity = PromptEntity(
            name=request.name,
            workspace=workspace,
            project=request.project,
            description=request.description,
            messages=request.messages,
            input_variables=request.input_variables,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            inference_params=request.inference_params,
            tags=request.tags or [],
        )

        try:
            created = await self.entity_client.create(entity)
            logger.info("Prompt created", extra={"workspace": created.workspace, "prompt_name": created.name})
            return _entity_to_schema(created)
        except EntityConflictError as e:
            logger.warning(
                "Prompt already exists (conflict)",
                extra={"workspace": workspace, "prompt_name": request.name},
            )
            raise ValueError(f"Prompt with name '{request.name}' already exists in workspace '{workspace}'") from e

    async def get_prompt(self, request: GetPromptRequest) -> Prompt | None:
        """Get a prompt by workspace and name."""
        logger.debug("Getting prompt", extra={"workspace": request.workspace, "prompt_name": request.name})

        try:
            entity = await self.entity_client.get(
                PromptEntity,
                workspace=request.workspace,
                name=request.name,
            )
            return _entity_to_schema(entity)
        except EntityNotFoundError:
            logger.debug("Prompt not found", extra={"workspace": request.workspace, "prompt_name": request.name})
            return None

    async def list_prompts(
        self,
        workspace: str,
        page: int = 1,
        page_size: int = 100,
        sort: str | None = None,
        filter_operation: FilterOperation | None = None,
    ) -> Page[Prompt]:
        """List prompts with filtering and pagination."""
        logger.debug("Listing prompts", extra={"page": page, "page_size": page_size, "sort": sort})

        result = await self.entity_client.list(
            PromptEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort=sort,
            page=page,
            page_size=page_size,
        )

        prompts = [_entity_to_schema(entity) for entity in result.data]

        return Page(
            data=prompts,
            pagination=PaginationData(
                page=result.pagination.page,
                page_size=result.pagination.page_size,
                current_page_size=len(prompts),
                total_pages=result.pagination.total_pages,
                total_results=result.pagination.total_results,
            ),
            sort=sort,
            filter=None,
        )

    async def update_prompt(self, workspace: str, name: str, request: UpdatePromptRequest) -> Prompt | None:
        """Replace a prompt's mutable fields (full update). Returns None if not found."""
        logger.debug("Updating prompt", extra={"workspace": workspace, "prompt_name": name})

        try:
            entity = await self.entity_client.get(PromptEntity, workspace=workspace, name=name)
        except EntityNotFoundError:
            logger.warning("Prompt not found for update", extra={"workspace": workspace, "prompt_name": name})
            return None

        entity.project = request.project
        entity.description = request.description
        entity.messages = request.messages
        entity.input_variables = request.input_variables
        entity.tools = request.tools
        entity.tool_choice = request.tool_choice
        entity.response_format = request.response_format
        entity.inference_params = request.inference_params
        entity.tags = request.tags or []

        updated = await self.entity_client.update(entity)
        logger.info("Prompt updated", extra={"workspace": updated.workspace, "prompt_name": updated.name})
        return _entity_to_schema(updated)

    async def delete_prompt(self, request: DeletePromptRequest) -> bool:
        """Delete a prompt by workspace and name. Returns False if not found."""
        logger.debug("Deleting prompt", extra={"workspace": request.workspace, "prompt_name": request.name})

        try:
            await self.entity_client.delete(PromptEntity, request.name, workspace=request.workspace)
        except EntityNotFoundError:
            logger.warning(
                "Prompt not found for deletion",
                extra={"workspace": request.workspace, "prompt_name": request.name},
            )
            return False

        logger.info("Prompt deleted", extra={"workspace": request.workspace, "prompt_name": request.name})
        return True
