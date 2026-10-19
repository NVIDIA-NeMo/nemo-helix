# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Prompt service with mocked EntityClient."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.models.api.service.prompt_service import PromptService
from nhx.core.models.entities import Prompt as PromptEntity
from nhx.core.models.schemas import (
    ChatCompletionTool,
    CreatePromptRequest,
    DeletePromptRequest,
    FunctionDefinition,
    GetPromptRequest,
    Prompt,
    PromptMessage,
    PromptMessageRole,
    UpdatePromptRequest,
)


def create_prompt_entity(
    entity_id: str = "prompt-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> PromptEntity:
    """Helper to create a PromptEntity with the store-managed private attributes set."""
    entity = PromptEntity(**kwargs)
    entity._id = entity_id
    entity._created_at = created_at or datetime.now(timezone.utc)
    entity._updated_at = updated_at or datetime.now(timezone.utc)
    return entity


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """Create a mock EntityClient for testing."""
    return AsyncMock(spec=EntityClient)


@pytest.fixture
def prompt_service(mock_entity_client):
    """Create a PromptService with mocked EntityClient."""
    return PromptService(mock_entity_client)


@pytest.fixture
def sample_messages() -> list[PromptMessage]:
    return [
        PromptMessage(role=PromptMessageRole.SYSTEM, content="You are a helpful {{ persona }}."),
        PromptMessage(role=PromptMessageRole.USER, content="Summarize: {{ document }}"),
    ]


@pytest.fixture
def sample_tools() -> list[ChatCompletionTool]:
    return [
        ChatCompletionTool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather for a city.",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )
    ]


@pytest.fixture
def sample_create_request(sample_messages, sample_tools) -> CreatePromptRequest:
    return CreatePromptRequest(
        name="summarizer",
        project="test-project",
        description="A summarization prompt",
        messages=sample_messages,
        input_variables=["persona", "document"],
        tools=sample_tools,
        tool_choice="auto",
        tags=["nlp", "summarize"],
    )


@pytest.fixture
def sample_prompt_entity(sample_messages, sample_tools) -> PromptEntity:
    return create_prompt_entity(
        name="summarizer",
        workspace="default",
        project="test-project",
        description="A summarization prompt",
        messages=sample_messages,
        input_variables=["persona", "document"],
        tools=sample_tools,
        tool_choice="auto",
        tags=["nlp", "summarize"],
    )


@pytest.mark.asyncio
async def test_create_prompt_success(prompt_service, mock_entity_client, sample_create_request, sample_prompt_entity):
    """Test successful prompt creation."""
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = sample_prompt_entity

    result = await prompt_service.create_prompt(sample_create_request, "default")

    assert isinstance(result, Prompt)
    assert result.name == "summarizer"
    assert result.workspace == "default"
    assert result.input_variables == ["persona", "document"]
    assert result.tools is not None
    assert result.tools[0].function.name == "get_weather"
    mock_entity_client.create.assert_called_once()
    created_entity = mock_entity_client.create.call_args[0][0]
    assert isinstance(created_entity, PromptEntity)
    assert created_entity.name == "summarizer"
    assert len(created_entity.messages) == 2


@pytest.mark.asyncio
async def test_create_prompt_conflict_raises_value_error(
    prompt_service, mock_entity_client, sample_create_request, sample_prompt_entity
):
    """Test that an existing prompt causes a ValueError and no create call."""
    mock_entity_client.get.return_value = sample_prompt_entity  # already exists

    with pytest.raises(ValueError, match="already exists"):
        await prompt_service.create_prompt(sample_create_request, "default")

    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_get_prompt_found(prompt_service, mock_entity_client, sample_prompt_entity):
    """Test retrieving an existing prompt."""
    mock_entity_client.get.return_value = sample_prompt_entity

    result = await prompt_service.get_prompt(GetPromptRequest(workspace="default", name="summarizer"))

    assert result is not None
    assert result.name == "summarizer"
    assert result.tool_choice == "auto"


@pytest.mark.asyncio
async def test_get_prompt_not_found(prompt_service, mock_entity_client):
    """Test that a missing prompt returns None."""
    mock_entity_client.get.side_effect = EntityNotFoundError("not found")

    result = await prompt_service.get_prompt(GetPromptRequest(workspace="default", name="missing"))

    assert result is None


@pytest.mark.asyncio
async def test_list_prompts(prompt_service, mock_entity_client, sample_prompt_entity):
    """Test listing prompts returns a Page with mapped schemas."""
    mock_result = AsyncMock()
    mock_result.data = [sample_prompt_entity]
    mock_result.pagination = AsyncMock(page=1, page_size=100, total_pages=1, total_results=1)
    mock_entity_client.list.return_value = mock_result

    page = await prompt_service.list_prompts(workspace="default", page=1, page_size=100, sort="created_at")

    assert page.pagination.total_results == 1
    assert page.pagination.current_page_size == 1
    assert len(page.data) == 1
    assert page.data[0].name == "summarizer"


@pytest.mark.asyncio
async def test_update_prompt_success(prompt_service, mock_entity_client, sample_prompt_entity):
    """Test updating an existing prompt replaces mutable fields."""
    mock_entity_client.get.return_value = sample_prompt_entity
    mock_entity_client.update.return_value = sample_prompt_entity

    request = UpdatePromptRequest(
        description="Updated description",
        messages=[PromptMessage(role=PromptMessageRole.USER, content="New {{ x }}")],
        input_variables=["x"],
        tags=["updated"],
    )

    result = await prompt_service.update_prompt("default", "summarizer", request)

    assert result is not None
    mock_entity_client.update.assert_called_once()
    updated_entity = mock_entity_client.update.call_args[0][0]
    assert updated_entity.description == "Updated description"
    assert updated_entity.input_variables == ["x"]
    assert updated_entity.tags == ["updated"]
    # Full replacement clears fields not present in the request
    assert updated_entity.tools is None
    assert updated_entity.tool_choice is None


@pytest.mark.asyncio
async def test_update_prompt_clears_tags_when_omitted(prompt_service, mock_entity_client, sample_prompt_entity):
    """Test that omitting tags in an update replaces them with an empty list (full replacement)."""
    sample_prompt_entity.tags = ["old-tag"]
    mock_entity_client.get.return_value = sample_prompt_entity
    mock_entity_client.update.return_value = sample_prompt_entity

    request = UpdatePromptRequest(description="no tags")

    await prompt_service.update_prompt("default", "summarizer", request)

    updated_entity = mock_entity_client.update.call_args[0][0]
    assert updated_entity.tags == []


@pytest.mark.asyncio
async def test_update_prompt_not_found(prompt_service, mock_entity_client):
    """Test that updating a missing prompt returns None."""
    mock_entity_client.get.side_effect = EntityNotFoundError("not found")

    result = await prompt_service.update_prompt("default", "missing", UpdatePromptRequest())

    assert result is None
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_prompt_success(prompt_service, mock_entity_client, sample_prompt_entity):
    """Test deleting an existing prompt returns True."""
    mock_entity_client.get.return_value = sample_prompt_entity

    result = await prompt_service.delete_prompt(DeletePromptRequest(workspace="default", name="summarizer"))

    assert result is True
    mock_entity_client.delete.assert_called_once_with(
        PromptEntity,
        "summarizer",
        workspace="default",
    )


@pytest.mark.asyncio
async def test_delete_prompt_not_found(prompt_service, mock_entity_client):
    """Test that deleting a missing prompt returns False."""
    mock_entity_client.delete.side_effect = EntityNotFoundError("not found")

    result = await prompt_service.delete_prompt(DeletePromptRequest(workspace="default", name="missing"))

    assert result is False
    mock_entity_client.delete.assert_called_once_with(PromptEntity, "missing", workspace="default")
