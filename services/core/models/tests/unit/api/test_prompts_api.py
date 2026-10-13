# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Prompt API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.api.common import Page, PaginationData
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.service.prompt_service import PromptService
from nhx.core.models.api.v2.prompts import router
from nhx.core.models.schemas import Prompt, PromptMessage, PromptMessageRole


@pytest.fixture
def mock_prompt_service():
    """Create a mock PromptService."""
    service = Mock(spec=PromptService)
    service.list_prompts = AsyncMock()
    service.get_prompt = AsyncMock()
    service.create_prompt = AsyncMock()
    service.update_prompt = AsyncMock()
    service.delete_prompt = AsyncMock()
    return service


@pytest.fixture
def test_app(mock_prompt_service):
    """Create a FastAPI test app with the prompt service dependency overridden."""
    from nhx.core.models.api.dependencies import get_prompt_service

    app = FastAPI()
    app.dependency_overrides[get_prompt_service] = lambda: mock_prompt_service
    app.include_router(router, prefix="/apis/models")
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def sample_prompt():
    return Prompt(
        id="prompt-1",
        name="summarizer",
        workspace="default",
        description="A summarization prompt",
        messages=[PromptMessage(role=PromptMessageRole.USER, content="Summarize: {{ document }}")],
        input_variables=["document"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_page(sample_prompt):
    return Page(
        data=[sample_prompt],
        pagination=PaginationData(
            page=1,
            page_size=100,
            current_page_size=1,
            total_results=1,
            total_pages=1,
        ),
        sort="created_at",
        filter=None,
    )


def test_list_prompts_default_parameters(client, mock_prompt_service, sample_page):
    mock_prompt_service.list_prompts.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/prompts")

    assert response.status_code == 200
    call_args = mock_prompt_service.list_prompts.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert call_args.kwargs["workspace"] == "default"


def test_list_prompts_with_sort(client, mock_prompt_service, sample_page):
    mock_prompt_service.list_prompts.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/prompts?sort=-name")

    assert response.status_code == 200
    assert mock_prompt_service.list_prompts.call_args.kwargs["sort"] == "-name"


def test_list_prompts_with_name_filter(client, mock_prompt_service, sample_page):
    mock_prompt_service.list_prompts.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/prompts?filter[name][]=summarizer")

    assert response.status_code == 200
    assert mock_prompt_service.list_prompts.call_args.kwargs.get("filter_operation") is not None


def test_list_prompts_workspace_filter_cannot_override_path(client, mock_prompt_service, sample_page):
    mock_prompt_service.list_prompts.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/prompts?filter[workspace][]=other")

    assert response.status_code == 200
    assert mock_prompt_service.list_prompts.call_args.kwargs["workspace"] == "default"


def test_list_prompts_invalid_page_returns_422(client):
    response = client.get("/apis/models/v2/workspaces/default/prompts?page=0")
    assert response.status_code == 422


def test_list_prompts_invalid_page_size_returns_422(client):
    response = client.get("/apis/models/v2/workspaces/default/prompts?page_size=0")
    assert response.status_code == 422


def test_list_prompts_response_structure(client, mock_prompt_service, sample_page):
    mock_prompt_service.list_prompts.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/prompts")

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "pagination" in data
    assert len(data["data"]) == 1
    assert data["data"][0]["name"] == "summarizer"


def test_create_prompt_success(client, mock_prompt_service, sample_prompt):
    mock_prompt_service.create_prompt.return_value = sample_prompt

    request_body = {
        "name": "summarizer",
        "messages": [{"role": "user", "content": "Summarize: {{ document }}"}],
        "input_variables": ["document"],
    }

    response = client.post("/apis/models/v2/workspaces/default/prompts", json=request_body)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "summarizer"
    assert data["messages"][0]["role"] == "user"


def test_create_prompt_with_tools(client, mock_prompt_service, sample_prompt):
    mock_prompt_service.create_prompt.return_value = sample_prompt

    request_body = {
        "name": "weather-bot",
        "messages": [{"role": "system", "content": "You can call tools."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "auto",
    }

    response = client.post("/apis/models/v2/workspaces/default/prompts", json=request_body)

    assert response.status_code == 201
    # The request validated and reached the service with parsed tools.
    sent_request = mock_prompt_service.create_prompt.call_args[0][0]
    assert sent_request.tools[0].function.name == "get_weather"
    assert sent_request.tool_choice == "auto"


def test_create_prompt_conflict_returns_409(client, mock_prompt_service):
    mock_prompt_service.create_prompt.side_effect = ValueError(
        "Prompt with name 'summarizer' already exists in workspace 'default'"
    )

    response = client.post(
        "/apis/models/v2/workspaces/default/prompts",
        json={"name": "summarizer"},
    )

    assert response.status_code == 409


def test_create_prompt_entity_validation_error_returns_422(client, mock_prompt_service):
    mock_prompt_service.create_prompt.side_effect = EntityValidationError("name must match pattern")

    response = client.post(
        "/apis/models/v2/workspaces/default/prompts",
        json={"name": "summarizer"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


def test_get_prompt_success(client, mock_prompt_service, sample_prompt):
    mock_prompt_service.get_prompt.return_value = sample_prompt

    response = client.get("/apis/models/v2/workspaces/default/prompts/summarizer")

    assert response.status_code == 200
    assert response.json()["name"] == "summarizer"


def test_get_prompt_not_found_returns_404(client, mock_prompt_service):
    mock_prompt_service.get_prompt.return_value = None

    response = client.get("/apis/models/v2/workspaces/default/prompts/missing")

    assert response.status_code == 404


def test_update_prompt_success(client, mock_prompt_service, sample_prompt):
    mock_prompt_service.update_prompt.return_value = sample_prompt

    response = client.put(
        "/apis/models/v2/workspaces/default/prompts/summarizer",
        json={"description": "updated", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "summarizer"


def test_update_prompt_not_found_returns_404(client, mock_prompt_service):
    mock_prompt_service.update_prompt.return_value = None

    response = client.put(
        "/apis/models/v2/workspaces/default/prompts/missing",
        json={"description": "updated"},
    )

    assert response.status_code == 404


def test_delete_prompt_success(client, mock_prompt_service):
    mock_prompt_service.delete_prompt.return_value = True

    response = client.delete("/apis/models/v2/workspaces/default/prompts/summarizer")

    assert response.status_code == 204


def test_delete_prompt_not_found_returns_404(client, mock_prompt_service):
    mock_prompt_service.delete_prompt.return_value = False

    response = client.delete("/apis/models/v2/workspaces/default/prompts/missing")

    assert response.status_code == 404


def test_delete_prompt_conflict_returns_409(client, mock_prompt_service):
    mock_prompt_service.delete_prompt.side_effect = EntityConflictError("stale version")

    response = client.delete("/apis/models/v2/workspaces/default/prompts/summarizer")

    assert response.status_code == 409
