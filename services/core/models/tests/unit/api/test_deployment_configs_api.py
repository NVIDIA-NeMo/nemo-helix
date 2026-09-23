# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelDeploymentConfig API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.v2.deployment_configs import router
from nhx.core.models.api.v2.utils import ERR_DEPLOYMENTS_NOT_ENABLED as _DEPLOYMENTS_NOT_ENABLED
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelType,
)


@pytest.fixture
def mock_deployment_config_service():
    """Create a mock ModelDeploymentConfigService."""
    service = Mock(spec=ModelDeploymentConfigService)
    service.list_deployment_configs = AsyncMock()
    service.get_deployment_config = AsyncMock()
    service.create_deployment_config = AsyncMock()
    service.update_deployment_config = AsyncMock()
    service.delete_deployment_config = AsyncMock()
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
def test_app(mock_deployment_config_service, mock_auth_client):
    """Create a FastAPI test app with mocked dependencies."""
    from nhx.core.models.api.dependencies import get_model_deployment_config_service

    app = FastAPI()

    def override_deployment_config_service():
        return mock_deployment_config_service

    app.dependency_overrides[get_model_deployment_config_service] = override_deployment_config_service
    app.dependency_overrides[get_auth_client] = lambda: mock_auth_client
    app.include_router(router)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_model_spec():
    """Create a sample model spec."""
    return ModelDeploymentConfigModelSpec(
        model_type=ModelType.LLM,
        lora_enabled=False,
        model_namespace="nvidia",
        model_name="llama-3-8b-instruct",
    )


@pytest.fixture
def sample_executor_config():
    """Create a sample executor config."""
    return ContainerExecutorConfig(
        gpu=1,
        disk_size="50Gi",
        image_name="nvcr.io/nvidia/nim/llm",
        image_tag="latest",
    )


@pytest.fixture
def sample_deployment_config(sample_model_spec, sample_executor_config):
    """Create a sample deployment config for testing."""
    return ModelDeploymentConfig(
        id="config-1",
        name="test-config",
        workspace="default",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        entity_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_page(sample_deployment_config):
    """Create a sample Page response."""
    return Page(
        data=[sample_deployment_config],
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


def test_list_deployment_configs_default_parameters(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs with default parameters."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs")

    assert response.status_code == 200
    # Workspace-scoped endpoint always passes workspace in filter_obj
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"


def test_list_deployment_configs_with_pagination(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs with custom pagination parameters."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs?page=2&page_size=50")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert call_args.kwargs["page"] == 2
    assert call_args.kwargs["page_size"] == 50
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"


def test_list_deployment_configs_with_workspace_filter(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs filtered by workspace."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/production/deployment-configs")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_list_deployment_configs_with_project_filter(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs filtered by project."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs?filter[project]=my-project")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs.get("filter_operation") is not None  # filter includes("project") == "my-project"


def test_list_deployment_configs_with_search(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs with search parameter."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs?filter[name][]=test")

    assert response.status_code == 200


def test_list_deployment_configs_with_sort(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs with custom sort."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs?sort=-updated_at")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert call_args.kwargs["sort"] == "-updated_at"


def test_list_deployment_configs_response_structure(client, mock_deployment_config_service, sample_page):
    """Test that the response has the correct structure."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployment-configs")

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


def test_list_deployment_configs_by_workspace_default_parameters(client, mock_deployment_config_service, sample_page):
    """Test listing deployment configs by workspace with default parameters."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/v2/workspaces/production/deployment-configs")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_list_deployment_configs_by_workspace_with_additional_filters(
    client, mock_deployment_config_service, sample_page
):
    """Test listing deployment configs by workspace with additional filters."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/v2/workspaces/production/deployment-configs?filter[project]=my-project")

    assert response.status_code == 200
    call_args = mock_deployment_config_service.list_deployment_configs.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert call_args.kwargs.get("filter_operation") is not None  # filter includes("project") == "my-project"


def test_page_parameter_validation(client, mock_deployment_config_service, sample_page):
    """Test page parameter validation."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    # Valid page number
    response = client.get("/v2/workspaces/default/deployment-configs?page=5")
    assert response.status_code == 200


def test_page_size_parameter_validation(client, mock_deployment_config_service, sample_page):
    """Test page_size parameter validation."""
    mock_deployment_config_service.list_deployment_configs.return_value = sample_page

    # Valid page size
    response = client.get("/v2/workspaces/default/deployment-configs?page_size=10")
    assert response.status_code == 200


def test_create_deployment_config_entity_validation_error_returns_422(client, mock_deployment_config_service):
    """Test that entity store validation errors during deployment config creation return 422."""
    mock_deployment_config_service.create_deployment_config.side_effect = EntityValidationError(
        "name must match pattern"
    )

    response = client.post(
        "/v2/workspaces/default/deployment-configs",
        json={
            "name": "my-config",
            "engine": "nim",
            "model_spec": {
                "model_type": "llm",
                "lora_enabled": False,
                "model_namespace": "nvidia",
                "model_name": "llama-3-8b-instruct",
            },
            "executor_config": {
                "gpu": 1,
                "disk_size": "50Gi",
                "image_name": "nvcr.io/nvidia/nim/llm",
                "image_tag": "latest",
            },
        },
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=True)
def test_update_deployment_config_entity_validation_error_returns_422(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Test that entity store validation errors during deployment config update return 422."""
    mock_deployment_config_service.update_deployment_config.side_effect = EntityValidationError(
        "name must match pattern"
    )

    response = client.post(
        "/v2/workspaces/default/deployment-configs/my-config",
        json={
            "engine": "nim",
            "model_spec": {
                "model_type": "llm",
                "lora_enabled": False,
                "model_namespace": "nvidia",
                "model_name": "llama-3-8b-instruct",
            },
            "executor_config": {
                "gpu": 1,
                "disk_size": "50Gi",
                "image_name": "nvcr.io/nvidia/nim/llm",
                "image_tag": "latest",
            },
        },
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


_MINIMAL_MODEL_SPEC = {
    "model_type": "llm",
    "lora_enabled": False,
    "model_namespace": "nvidia",
    "model_name": "llama-3-8b-instruct",
}

_MINIMAL_EXECUTOR_CONFIG = {
    "gpu": 1,
    "disk_size": "50Gi",
    "image_name": "nvcr.io/nvidia/nim/llm",
    "image_tag": "latest",
}


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=False)
def test_update_deployment_config_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Updating a deployment config must return 422 when deployments are not enabled."""
    response = client.post(
        "/v2/workspaces/default/deployment-configs/cfg",
        json={
            "engine": "nim",
            "model_spec": _MINIMAL_MODEL_SPEC,
            "executor_config": _MINIMAL_EXECUTOR_CONFIG,
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_config_service.update_deployment_config.called


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=False)
def test_delete_all_deployment_config_versions_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Deleting all versions of a deployment config must return 422 when deployments are not enabled."""
    response = client.delete("/v2/workspaces/default/deployment-configs/cfg")
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_config_service.delete_deployment_config.called


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=False)
def test_delete_deployment_config_version_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Deleting a single deployment config version must return 422 when deployments are not enabled."""
    response = client.delete("/v2/workspaces/default/deployment-configs/cfg/versions/1")
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_config_service.delete_deployment_config.called


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=True)
def test_delete_deployment_config_conflict_returns_409(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Stale deployment config deletes return 409."""
    mock_deployment_config_service.delete_deployment_config.side_effect = EntityConflictError("stale version")

    response = client.delete("/v2/workspaces/default/deployment-configs/cfg")

    assert response.status_code == 409


@patch("nhx.core.models.api.v2.deployment_configs.deployments_enabled", return_value=True)
def test_delete_deployment_config_version_conflict_returns_409(
    _mock_deployments_enabled, client, mock_deployment_config_service
):
    """Stale deployment config version deletes return 409."""
    mock_deployment_config_service.delete_deployment_config.side_effect = EntityConflictError("stale version")

    response = client.delete("/v2/workspaces/default/deployment-configs/cfg/versions/1")

    assert response.status_code == 409
