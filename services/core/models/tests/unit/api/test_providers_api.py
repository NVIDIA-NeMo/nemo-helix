# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelProvider API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.service.model_provider_service import ModelProviderService, ModelProviderValidationError
from nhx.core.models.api.v2.providers import router
from nhx.core.models.schemas import ModelProvider, ModelProviderStatus


@pytest.fixture
def mock_model_provider_service():
    """Create a mock ModelProviderService."""
    service = Mock(spec=ModelProviderService)
    service.list_model_providers = AsyncMock()
    service.get_model_provider = AsyncMock()
    service.create_model_provider = AsyncMock()
    service.update_model_provider = AsyncMock()
    service.upsert_model_provider = AsyncMock()
    service.update_model_provider_status = AsyncMock()
    service.delete_model_provider = AsyncMock()
    return service


@pytest.fixture
def mock_auth_client():
    """Create a mock AuthClient with auth disabled and a test principal."""
    client = MagicMock(spec=AuthClient)
    client.auth_enabled = False
    client.is_service_principal = False
    client.principal = Principal(id="test-principal", email="test@example.com", groups=[])
    return client


@pytest.fixture
def mock_secrets_client():
    """Create a mock Secrets client for create/upsert endpoints that depend on get_secrets_client."""
    return AsyncMock()


@pytest.fixture
def test_app(mock_model_provider_service, mock_auth_client, mock_secrets_client):
    """Create a FastAPI test app with mocked dependencies."""
    from nhx.core.models.api.dependencies import get_model_provider_service, get_secrets_client

    app = FastAPI()

    app.dependency_overrides[get_model_provider_service] = lambda: mock_model_provider_service
    app.dependency_overrides[get_auth_client] = lambda: mock_auth_client
    app.dependency_overrides[get_secrets_client] = lambda: mock_secrets_client
    app.include_router(router, prefix="/apis/models")

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_model_provider():
    """Create a sample model provider for testing."""
    return ModelProvider(
        id="provider-1",
        name="openai-provider",
        workspace="default",
        host_url="https://api.openai.com",
        status=ModelProviderStatus.CREATED,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_page(sample_model_provider):
    """Create a sample Page response."""
    return Page(
        data=[sample_model_provider],
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


def test_list_providers_default_parameters(client, mock_model_provider_service, sample_page):
    """Test listing providers with default parameters."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers")

    assert response.status_code == 200
    # Workspace-scoped endpoint always passes workspace in filter_obj
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"


def test_list_providers_with_pagination(client, mock_model_provider_service, sample_page):
    """Test listing providers with custom pagination parameters."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?page=2&page_size=50")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert call_args.kwargs["page"] == 2
    assert call_args.kwargs["page_size"] == 50
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"


def test_list_providers_with_workspace_filter(client, mock_model_provider_service, sample_page):
    """Test listing providers filtered by workspace."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/production/providers")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_list_providers_cross_workspace(client, mock_model_provider_service, sample_page):
    """Test listing providers across workspaces."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/-/providers")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "-"


def test_list_providers_with_status_filter(client, mock_model_provider_service, sample_page):
    """Test listing providers filtered by status."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?filter[status]=CREATED")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelProviderStatus.CREATED


def test_list_providers_with_project_filter(client, mock_model_provider_service, sample_page):
    """Test listing providers filtered by project."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?filter[project]=my-project")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs.get("filter_operation") is not None  # filter includes("project") == "my-project"


def test_list_providers_with_multiple_filters(client, mock_model_provider_service, sample_page):
    """Test listing providers with multiple filters."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get(
        "/apis/models/v2/workspaces/production/providers?filter[workspace]=production&filter[status]=CREATED"
    )

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelProviderStatus.CREATED


def test_list_providers_with_search(client, mock_model_provider_service, sample_page):
    """Test listing providers with search parameter."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?filter[name][]=openai")

    assert response.status_code == 200


def test_list_providers_with_host_url_search(client, mock_model_provider_service, sample_page):
    """Test listing providers with host_url search."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?filter[host_url][]=api.openai.com")

    assert response.status_code == 200


def test_list_providers_with_sort(client, mock_model_provider_service, sample_page):
    """Test listing providers with custom sort."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers?sort=-name")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert call_args.kwargs["sort"] == "-name"


def test_list_providers_response_structure(client, mock_model_provider_service, sample_page):
    """Test that the response has the correct structure."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/default/providers")

    assert response.status_code == 200
    data = response.json()

    # Check Page structure
    assert "data" in data
    assert "pagination" in data
    assert "sort" in data

    # Check pagination structure
    assert "page" in data["pagination"]
    assert "page_size" in data["pagination"]
    assert "current_page_size" in data["pagination"]
    assert "total_results" in data["pagination"]
    assert "total_pages" in data["pagination"]

    # Check data
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 1


def test_list_providers_by_workspace_default_parameters(client, mock_model_provider_service, sample_page):
    """Test listing providers by workspace with default parameters."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/apis/models/v2/workspaces/production/providers")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_list_providers_by_workspace_with_additional_filters(client, mock_model_provider_service, sample_page):
    """Test listing providers by workspace with additional filters."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/apis/models/v2/workspaces/production/providers?filter[status]=CREATED")

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelProviderStatus.CREATED


def test_page_parameter_validation(client, mock_model_provider_service, sample_page):
    """Test page parameter validation."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    # Valid page number
    response = client.get("/apis/models/v2/workspaces/default/providers?page=5")
    assert response.status_code == 200


def test_page_size_parameter_validation(client, mock_model_provider_service, sample_page):
    """Test page_size parameter validation."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    # Valid page size
    response = client.get("/apis/models/v2/workspaces/default/providers?page_size=10")
    assert response.status_code == 200


def test_list_providers_filters_and_search_combined(client, mock_model_provider_service, sample_page):
    """Test listing providers with both filters and search."""
    mock_model_provider_service.list_model_providers.return_value = sample_page

    response = client.get(
        "/apis/models/v2/workspaces/production/providers?filter[workspace]=production&filter[name][]=openai"
    )

    assert response.status_code == 200
    call_args = mock_model_provider_service.list_model_providers.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_create_provider_with_default_and_required_extra_fields(client, mock_model_provider_service):
    """Test creating a provider with default and required extra_body and extra_headers via API."""
    default_extra_body = {"temperature": 0.7}
    default_extra_headers = {"X-Provider": "test"}
    required_extra_body = {"stream": True}
    required_extra_headers = {"X-Required": "value"}

    request_body = {
        "name": "test-provider",
        "workspace": "default",
        "host_url": "https://api.example.com/v1",
        "api_key": "sk-test123",
        "default_extra_body": default_extra_body,
        "default_extra_headers": default_extra_headers,
        "required_extra_body": required_extra_body,
        "required_extra_headers": required_extra_headers,
    }

    expected_provider = ModelProvider(
        id="provider-2",
        name="test-provider",
        workspace="default",
        host_url="https://api.example.com/v1",
        status=ModelProviderStatus.CREATED,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        default_extra_body=default_extra_body,
        default_extra_headers=default_extra_headers,
        required_extra_body=required_extra_body,
        required_extra_headers=required_extra_headers,
    )

    mock_model_provider_service.create_model_provider.return_value = expected_provider

    response = client.post("/apis/models/v2/workspaces/default/providers", json=request_body)

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["name"] == "test-provider"
    assert response_data["default_extra_body"] == default_extra_body
    assert response_data["default_extra_headers"] == default_extra_headers
    assert response_data["required_extra_body"] == required_extra_body
    assert response_data["required_extra_headers"] == required_extra_headers


def test_create_provider_without_extra_fields(client, mock_model_provider_service):
    """Test creating a provider without extra_body and extra_headers."""
    request_body = {
        "name": "test-provider",
        "workspace": "default",
        "host_url": "https://api.example.com/v1",
        "api_key": "sk-test123",
    }

    expected_provider = ModelProvider(
        id="provider-3",
        name="test-provider",
        workspace="default",
        host_url="https://api.example.com/v1",
        status=ModelProviderStatus.CREATED,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    mock_model_provider_service.create_model_provider.return_value = expected_provider

    response = client.post("/apis/models/v2/workspaces/default/providers", json=request_body)

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["name"] == "test-provider"
    # Extra fields should not be present or should be null
    assert response_data.get("default_extra_body") is None
    assert response_data.get("default_extra_headers") is None
    assert response_data.get("required_extra_body") is None
    assert response_data.get("required_extra_headers") is None


def test_get_provider_returns_extra_fields(client, mock_model_provider_service):
    """Test that getting a provider returns default and required extra fields."""
    default_extra_body = {"max_tokens": 100}
    default_extra_headers = {"X-Organization": "test-org"}
    required_extra_body = {"stream": True}
    required_extra_headers = {"X-Required": "always"}

    provider = ModelProvider(
        id="provider-4",
        name="test-provider",
        workspace="default",
        host_url="https://api.example.com/v1",
        status=ModelProviderStatus.CREATED,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        default_extra_body=default_extra_body,
        default_extra_headers=default_extra_headers,
        required_extra_body=required_extra_body,
        required_extra_headers=required_extra_headers,
    )

    mock_model_provider_service.get_model_provider.return_value = provider

    response = client.get("/apis/models/v2/workspaces/default/providers/test-provider")

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["default_extra_body"] == default_extra_body
    assert response_data["default_extra_headers"] == default_extra_headers
    assert response_data["required_extra_body"] == required_extra_body
    assert response_data["required_extra_headers"] == required_extra_headers


@pytest.mark.parametrize("field_name", ["default_extra_headers", "required_extra_headers"])
def test_create_provider_rejects_authorization_header(client, mock_model_provider_service, field_name):
    """Test that creating a provider with Authorization header returns 400."""
    mock_model_provider_service.create_model_provider.side_effect = ModelProviderValidationError(
        f"Authorization header is not allowed in {field_name}. Use api_key_secret_name for authentication."
    )

    request_body = {
        "name": "test-provider",
        "host_url": "https://api.example.com/v1",
        field_name: {"Authorization": "Bearer token123"},
    }

    response = client.post("/apis/models/v2/workspaces/default/providers", json=request_body)

    assert response.status_code == 400
    assert "Authorization header is not allowed" in response.json()["detail"]


@pytest.mark.parametrize("field_name", ["default_extra_headers", "required_extra_headers"])
def test_upsert_provider_rejects_authorization_header(client, mock_model_provider_service, field_name):
    """Test that upserting a provider with Authorization header returns 400."""
    mock_model_provider_service.upsert_model_provider = AsyncMock(
        side_effect=ModelProviderValidationError(
            f"Authorization header is not allowed in {field_name}. Use api_key_secret_name for authentication."
        )
    )

    request_body = {
        "host_url": "https://api.example.com/v1",
        field_name: {"Authorization": "Bearer token123"},
    }

    response = client.put("/apis/models/v2/workspaces/default/providers/test-provider", json=request_body)

    assert response.status_code == 400
    assert "Authorization header is not allowed" in response.json()["detail"]


def test_create_provider_entity_validation_error_returns_422(client, mock_model_provider_service):
    """Test that entity store validation errors during provider creation return 422."""
    mock_model_provider_service.create_model_provider.side_effect = EntityValidationError("name must match pattern")

    response = client.post(
        "/apis/models/v2/workspaces/default/providers",
        json={"name": "test-provider", "workspace": "default", "host_url": "https://api.example.com/v1"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


def test_upsert_provider_entity_validation_error_returns_422(client, mock_model_provider_service):
    """Test that entity store validation errors during provider upsert return 422."""
    mock_model_provider_service.upsert_model_provider.side_effect = EntityValidationError("name must match pattern")

    response = client.put(
        "/apis/models/v2/workspaces/default/providers/test-provider",
        json={"host_url": "https://api.example.com/v1"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


def test_update_provider_status_entity_validation_error_returns_422(client, mock_model_provider_service):
    """Test that entity store validation errors during provider status update return 422."""
    mock_model_provider_service.update_model_provider_status.side_effect = EntityValidationError(
        "served_models invalid"
    )

    response = client.put(
        "/apis/models/v2/workspaces/default/providers/test-provider/status",
        json={"status": "CREATED"},
    )

    assert response.status_code == 422
    assert "served_models invalid" in response.json()["detail"]


def test_delete_provider_conflict_returns_409(client, mock_model_provider_service):
    """Test stale provider deletes return 409."""
    mock_model_provider_service.delete_model_provider.side_effect = EntityConflictError("stale version")

    response = client.delete("/apis/models/v2/workspaces/default/providers/test-provider")

    assert response.status_code == 409
