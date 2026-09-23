# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Model (ModelEntity) API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import NemoTransportError
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_entity_service import ModelEntityService
from nhx.core.models.api.v2.models import router, start_update_model_spec_job
from nhx.core.models.schemas import ModelEntity


@pytest.fixture
def mock_model_entity_service():
    """Create a mock ModelEntityService."""
    service = Mock(spec=ModelEntityService)
    service.list_model_entities = AsyncMock()
    service.get_model_entity = AsyncMock()
    service.create_model_entity = AsyncMock()
    service.update_model_entity = AsyncMock()
    service.delete_model_entity = AsyncMock()
    service.entity_client = Mock()
    service.entity_client.get = AsyncMock()
    return service


@pytest.fixture
def mock_adapter_entity_service():
    service = Mock(spec=AdapterEntityService)
    service.create_adapter = AsyncMock()
    service.delete_adapter = AsyncMock()
    service.update_adapter = AsyncMock()
    return service


@pytest.fixture
def mock_auth_client():
    """Create a mock AuthClient with auth disabled."""
    client = MagicMock(spec=AuthClient)
    client.auth_enabled = False
    client.is_service_principal = False
    client.principal = Principal(id="test-principal", email="test@example.com", groups=[])
    return client


@pytest.fixture
def mock_sdk():
    """Create a mock SDK for create/update endpoints that depend on get_sdk_client."""
    sdk = AsyncMock()
    sdk._custom_headers = {"authorization": "Bearer test"}
    sdk.base_url = "http://localhost:8080"
    sdk.workspace = "default"
    return sdk


@pytest.fixture
def test_app(mock_model_entity_service, mock_adapter_entity_service, mock_auth_client, mock_sdk):
    """Create a FastAPI test app with mocked dependencies."""
    from nhx.common.service.dependencies import get_sdk_client
    from nhx.core.models.api.dependencies import get_adapter_entity_service, get_model_entity_service

    app = FastAPI()

    def override_model_entity_service():
        return mock_model_entity_service

    def override_adapter_entity_service():
        return mock_adapter_entity_service

    app.dependency_overrides[get_model_entity_service] = override_model_entity_service
    app.dependency_overrides[get_adapter_entity_service] = override_adapter_entity_service
    app.dependency_overrides[get_auth_client] = lambda: mock_auth_client
    app.dependency_overrides[get_sdk_client] = lambda: mock_sdk
    app.include_router(router, prefix="/apis/models")

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_model_entity():
    """Create a sample model entity for testing."""
    return ModelEntity(
        id="model-1",
        name="llama-3-8b-instruct",
        workspace="nvidia",
        description="Llama 3 8B Instruct model",
        schema_version="1.0",
        model_providers=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_page(sample_model_entity):
    """Create a sample Page response."""
    return Page(
        data=[sample_model_entity],
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


def test_list_models_default_parameters(client, mock_model_entity_service, sample_page):
    """Test listing models with default parameters."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert call_args.kwargs["workspace"] == "nvidia"
    assert call_args.kwargs["parsed_filter"] is not None
    assert call_args.kwargs["verbose"] is False


def test_list_models_with_verbose_true(client, mock_model_entity_service, sample_page):
    """Test listing models with verbose=true query parameter."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?verbose=true")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["verbose"] is True


def test_list_models_with_pagination(client, mock_model_entity_service, sample_page):
    """Test listing models with custom pagination parameters."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?page=2&page_size=50")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["page"] == 2
    assert call_args.kwargs["page_size"] == 50
    assert call_args.kwargs["sort"] == "created_at"
    assert call_args.kwargs["workspace"] == "nvidia"
    assert call_args.kwargs["parsed_filter"] is not None


def test_list_models_with_workspace_filter(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by workspace."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[workspace]=nvidia")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["workspace"] == "nvidia"


def test_list_models_cross_workspace(client, mock_model_entity_service, sample_page):
    """Test listing models across workspaces."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/-/models")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["workspace"] == "-"


def test_list_models_with_project_filter(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by project."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[project]=my-project")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.extract("project") == "my-project"


def test_list_models_with_base_model_filter(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by base_model."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[base_model]=llama-3")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.extract("data.base_model") == "llama-3"


def test_list_models_with_base_model_filter_true(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by base_model=true → $not { data.base_model $eq null }."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[base_model]=true")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    # Bool "true" is coerced to a not-null check
    assert parsed_filter.operation.to_dict() == {"$not": {"data.base_model": {"$eq": None}}}


def test_list_models_with_base_model_filter_false(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by base_model=false → data.base_model $eq null."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[base_model]=false")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    # Bool "false" is coerced to a null check
    assert parsed_filter.operation.to_dict() == {"data.base_model": {"$eq": None}}


def test_list_models_with_base_model_filter_name(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by base_model name using $eq."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[base_model]=llama-3")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.extract("data.base_model") == "llama-3"


def test_list_models_with_fileset_filter_ref(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by fileset reference using $eq."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[fileset]=nvidia/my-fileset")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.extract("data.fileset") == "nvidia/my-fileset"


def test_list_models_with_fileset_filter_true(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by fileset=true → $not { data.fileset $eq null }."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[fileset]=true")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    # Bool "true" is coerced to a not-null check
    assert parsed_filter.operation.to_dict() == {"$not": {"data.fileset": {"$eq": None}}}


def test_list_models_with_fileset_filter_false(client, mock_model_entity_service, sample_page):
    """Test listing models filtered by fileset=false → data.fileset $eq null."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[fileset]=false")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    # Bool "false" is coerced to a null check
    assert parsed_filter.operation.to_dict() == {"data.fileset": {"$eq": None}}


def test_list_models_with_name_filter(client, mock_model_entity_service, sample_page):
    """Test listing models with name filter using $like operator."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[name][$like]=llama")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None


def test_list_models_filter_json_format(client, mock_model_entity_service, sample_page):
    """Test listing models with JSON filter parameter."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get('/apis/models/v2/workspaces/nvidia/models?filter={"name":{"$like":"llama"}}')

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None


def test_list_models_adapters_exists_preserves_relationship_field(client, mock_model_entity_service, sample_page):
    """`adapters` is a relationship, not a data column — the entity-store path must stay `adapters`.

    Regression: translating `adapters` to `data.adapters` breaks relationship detection
    in the entities service.
    """
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get('/apis/models/v2/workspaces/nvidia/models?filter={"adapters":{"$exists":true}}')

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    assert parsed_filter.operation.to_dict() == {"adapters": {"$exists": True}}


def test_list_models_studio_custom_models_default_filter(client, mock_model_entity_service, sample_page):
    """Studio's default Custom Models filter: $or of has-base-model and has-adapters.

    The forwarded filter tree must preserve `adapters` as a relationship key while
    still qualifying `data.base_model` with its data-column prefix.
    """
    mock_model_entity_service.list_model_entities.return_value = sample_page

    studio_filter = '{"$or":[{"data.base_model":{"$not":{"$eq":null}}},{"adapters":{"$exists":true}}]}'
    response = client.get(f"/apis/models/v2/workspaces/nvidia/models?filter={studio_filter}")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None
    assert parsed_filter.operation.to_dict() == {
        "$or": [
            {"$not": {"data.base_model": {"$eq": None}}},
            {"adapters": {"$exists": True}},
        ],
    }


def test_list_models_with_description_filter(client, mock_model_entity_service, sample_page):
    """Test listing models with bracket description filter."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?filter[description][$like]=instruct")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.operation is not None


def test_list_models_with_sort(client, mock_model_entity_service, sample_page):
    """Test listing models with custom sort."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models?sort=-name")

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["sort"] == "-name"


def test_list_models_response_structure(client, mock_model_entity_service, sample_page):
    """Test that the response has the correct structure."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get("/apis/models/v2/workspaces/nvidia/models")

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


def test_list_models_with_multiple_filters(client, mock_model_entity_service, sample_page):
    """Test listing models with multiple filters."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    response = client.get(
        "/apis/models/v2/workspaces/nvidia/models?filter[workspace]=nvidia&filter[base_model]=llama-3"
    )

    assert response.status_code == 200
    call_args = mock_model_entity_service.list_model_entities.call_args
    assert call_args.kwargs["workspace"] == "nvidia"
    parsed_filter = call_args.kwargs["parsed_filter"]
    assert parsed_filter.extract("data.base_model") == "llama-3"


def test_page_parameter_validation(client, mock_model_entity_service, sample_page):
    """Test page parameter validation."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    # Valid page number
    response = client.get("/apis/models/v2/workspaces/nvidia/models?page=5")
    assert response.status_code == 200


def test_page_size_parameter_validation(client, mock_model_entity_service, sample_page):
    """Test page_size parameter validation."""
    mock_model_entity_service.list_model_entities.return_value = sample_page

    # Valid page size
    response = client.get("/apis/models/v2/workspaces/nvidia/models?page_size=10")
    assert response.status_code == 200


def test_create_model_entity_validation_error_returns_422(client, mock_model_entity_service):
    """Test that entity store validation errors during model creation return 422."""
    mock_model_entity_service.create_model_entity.side_effect = EntityValidationError("name must match pattern")

    response = client.post(
        "/apis/models/v2/workspaces/nvidia/models",
        json={"name": "my-model"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


@pytest.mark.asyncio
async def test_model_spec_job_transport_failure_does_not_fail_persisted_model(sample_model_entity):
    request = httpx.Request("POST", "http://test/apis/jobs/v2/workspaces/nvidia/jobs")
    jobs = MagicMock()
    jobs.create_job = AsyncMock(
        side_effect=NemoTransportError(httpx.ConnectError("Connection refused", request=request))
    )

    with (
        patch("nhx.core.models.api.v2.models.get_async_platform_sdk"),
        patch("nhx.core.models.api.v2.models.client_from_platform", return_value=jobs),
    ):
        await start_update_model_spec_job(sample_model_entity)

    jobs.create_job.assert_awaited_once()


def test_update_model_entity_validation_error_returns_422(client, mock_model_entity_service, sample_model_entity):
    """Test that entity store validation errors during model update return 422."""
    mock_model_entity_service.entity_client.get.return_value = Mock()
    mock_model_entity_service.get_model_entity.return_value = sample_model_entity
    mock_model_entity_service.update_model_entity.side_effect = EntityValidationError("name must match pattern")

    response = client.patch(
        "/apis/models/v2/workspaces/nvidia/models/my-model",
        json={"name": "updated-model"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


def test_update_model_default_verbose_false(client, mock_model_entity_service, sample_model_entity):
    """Test update model uses verbose=false by default."""
    mock_model_entity_service.entity_client.get.return_value = Mock()
    mock_model_entity_service.get_model_entity.return_value = sample_model_entity
    mock_model_entity_service.update_model_entity.return_value = sample_model_entity

    response = client.patch(
        "/apis/models/v2/workspaces/nvidia/models/my-model",
        json={"description": "updated"},
    )

    assert response.status_code == 200
    call_args = mock_model_entity_service.update_model_entity.call_args
    assert call_args.kwargs["verbose"] is False


def test_update_model_with_verbose_true(client, mock_model_entity_service, sample_model_entity):
    """Test update model forwards verbose=true query parameter."""
    mock_model_entity_service.entity_client.get.return_value = Mock()
    mock_model_entity_service.get_model_entity.return_value = sample_model_entity
    mock_model_entity_service.update_model_entity.return_value = sample_model_entity

    response = client.patch(
        "/apis/models/v2/workspaces/nvidia/models/my-model?verbose=true",
        json={"description": "updated"},
    )

    assert response.status_code == 200
    call_args = mock_model_entity_service.update_model_entity.call_args
    assert call_args.kwargs["verbose"] is True


def test_delete_model_conflict_returns_409(client, mock_model_entity_service):
    """Test stale model deletes return 409."""
    mock_model_entity_service.delete_model_entity.side_effect = EntityConflictError("stale version")

    response = client.delete("/apis/models/v2/workspaces/default/models/my-model")

    assert response.status_code == 409


def test_delete_model_adapter_conflict_returns_409(client, mock_adapter_entity_service):
    """Test stale model adapter deletes return 409."""
    mock_adapter_entity_service.delete_adapter.side_effect = EntityConflictError("stale version")

    response = client.delete("/apis/models/v2/workspaces/default/models/my-model/adapters/lora")

    assert response.status_code == 409


def test_create_model_adapter_entity_validation_error_returns_422(client, mock_adapter_entity_service):
    """Test that entity store validation errors during adapter creation return 422."""
    mock_adapter_entity_service.create_adapter.side_effect = EntityValidationError("adapter name invalid")

    with patch("nhx.core.models.api.permissions.client_from_platform") as mock_cfp:
        mock_files = AsyncMock()
        mock_files.get_fileset.return_value = MagicMock()
        mock_cfp.return_value = mock_files

        response = client.post(
            "/apis/models/v2/workspaces/nvidia/models/my-model/adapters",
            json={
                "name": "my-adapter",
                "fileset": "nvidia/my-fileset",
                "finetuning_type": "lora",
            },
        )

    assert response.status_code == 422
    assert "adapter name invalid" in response.json()["detail"]


def test_update_model_adapter_entity_validation_error_returns_422(client, mock_adapter_entity_service):
    """Test that entity store validation errors during adapter update return 422."""
    mock_adapter_entity_service.update_adapter.side_effect = EntityValidationError("adapter name invalid")

    response = client.patch(
        "/apis/models/v2/workspaces/nvidia/models/my-model/adapters/my-adapter",
        json={"name": "updated-adapter"},
    )

    assert response.status_code == 422
    assert "adapter name invalid" in response.json()["detail"]


def test_get_model_default_verbose_false(client, mock_model_entity_service, sample_model_entity):
    """Test get model uses verbose=false by default."""
    mock_model_entity_service.get_model_entity.return_value = sample_model_entity

    response = client.get("/apis/models/v2/workspaces/nvidia/models/my-model")

    assert response.status_code == 200
    mock_model_entity_service.get_model_entity.assert_called_once_with("nvidia", "my-model", verbose=False)


def test_get_model_with_verbose_true(client, mock_model_entity_service, sample_model_entity):
    """Test get model forwards verbose=true query parameter."""
    mock_model_entity_service.get_model_entity.return_value = sample_model_entity

    response = client.get("/apis/models/v2/workspaces/nvidia/models/my-model?verbose=true")

    assert response.status_code == 200
    mock_model_entity_service.get_model_entity.assert_called_once_with("nvidia", "my-model", verbose=True)
