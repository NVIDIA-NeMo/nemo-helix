# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelDeployment API endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.core.models.api.service.model_deployment_service import DeploymentStatusConflictError, ModelDeploymentService
from nhx.core.models.api.v2.deployments import router
from nhx.core.models.api.v2.utils import ERR_DEPLOYMENTS_NOT_ENABLED as _DEPLOYMENTS_NOT_ENABLED
from nhx.core.models.schemas import ModelDeployment, ModelDeploymentStatus


@pytest.fixture
def mock_deployment_service():
    """Create a mock ModelDeploymentService."""
    service = Mock(spec=ModelDeploymentService)
    service.list_deployments = AsyncMock()
    service.get_deployment = AsyncMock()
    service.create_deployment = AsyncMock()
    service.update_deployment = AsyncMock()
    service.update_deployment_status = AsyncMock()
    service.delete_deployment = AsyncMock()
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
def test_app(mock_deployment_service, mock_auth_client):
    """Create a FastAPI test app with mocked dependencies."""
    from nhx.core.models.api.dependencies import get_model_deployment_service

    app = FastAPI()

    app.dependency_overrides[get_model_deployment_service] = lambda: mock_deployment_service
    app.dependency_overrides[get_auth_client] = lambda: mock_auth_client
    app.include_router(router)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return ModelDeployment(
        id="deployment-1",
        name="test-deployment",
        workspace="default",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.READY,
        entity_version=1,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_page(sample_deployment):
    """Create a sample Page response."""
    return Page(
        data=[sample_deployment],
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


def test_list_deployments_default_parameters(client, mock_deployment_service, sample_page):
    """Test listing deployments with default parameters."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments")

    assert response.status_code == 200
    # Workspace-scoped endpoint always passes workspace in filter_obj
    call_args = mock_deployment_service.list_deployments.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"
    assert call_args.kwargs["all_versions"] is False


def test_list_deployments_with_pagination(client, mock_deployment_service, sample_page):
    """Test listing deployments with custom pagination parameters."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?page=2&page_size=50")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert call_args.kwargs["page"] == 2
    assert call_args.kwargs["page_size"] == 50
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"
    assert call_args.kwargs["all_versions"] is False


def test_list_deployments_with_all_versions_true(client, mock_deployment_service, sample_page):
    """Test listing deployments with all_versions=true."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?all_versions=true")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"
    assert call_args.kwargs["all_versions"] is True


def test_list_deployments_with_all_versions_false(client, mock_deployment_service, sample_page):
    """Test listing deployments with all_versions=false."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?all_versions=false")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert call_args.kwargs["page"] == 1
    assert call_args.kwargs["page_size"] == 100
    assert call_args.kwargs["sort"] == "created_at"
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "default"
    assert call_args.kwargs["all_versions"] is False


def test_list_deployments_with_status_filter(client, mock_deployment_service, sample_page):
    """Test listing deployments filtered by status."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?filter[status]=READY")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelDeploymentStatus.READY
    assert call_args.kwargs["all_versions"] is False


def test_list_deployments_with_workspace_filter(client, mock_deployment_service, sample_page):
    """Test listing deployments filtered by workspace."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/production/deployments?filter[workspace]=production")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"


def test_list_deployments_with_multiple_filters_and_all_versions(client, mock_deployment_service, sample_page):
    """Test listing deployments with multiple filters and all_versions."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get(
        "/v2/workspaces/production/deployments?filter[status]=READY&filter[workspace]=production&all_versions=true"
    )

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelDeploymentStatus.READY
    assert call_args.kwargs["workspace"] == "production"
    assert call_args.kwargs["all_versions"] is True


def test_list_deployments_with_search(client, mock_deployment_service, sample_page):
    """Test listing deployments with search parameter."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?filter[name][]=test")

    assert response.status_code == 200


def test_list_deployments_with_sort(client, mock_deployment_service, sample_page):
    """Test listing deployments with custom sort."""
    mock_deployment_service.list_deployments.return_value = sample_page

    response = client.get("/v2/workspaces/default/deployments?sort=-updated_at")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert call_args.kwargs["sort"] == "-updated_at"


def test_list_deployments_response_structure(client, mock_deployment_service, sample_page):
    """Test that the response has the correct structure."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Use cross-workspace endpoint to avoid filter_obj being set
    response = client.get("/v2/workspaces/default/deployments")

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


def test_list_deployments_by_workspace_default_parameters(client, mock_deployment_service, sample_page):
    """Test listing deployments by workspace with default parameters."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/v2/workspaces/production/deployments")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert call_args.kwargs["all_versions"] is False


def test_list_deployments_by_workspace_with_all_versions(client, mock_deployment_service, sample_page):
    """Test listing deployments by workspace with all_versions=true."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/v2/workspaces/production/deployments?all_versions=true")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert call_args.kwargs["all_versions"] is True


def test_list_deployments_by_workspace_with_additional_filters(client, mock_deployment_service, sample_page):
    """Test listing deployments by workspace with additional filters."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Use workspace-scoped endpoint
    response = client.get("/v2/workspaces/production/deployments?filter[status]=READY&all_versions=true")

    assert response.status_code == 200
    call_args = mock_deployment_service.list_deployments.call_args
    assert "workspace" in call_args.kwargs
    assert call_args.kwargs["workspace"] == "production"
    assert (
        call_args.kwargs.get("filter_operation") is not None
    )  # filter includes("data.status") == ModelDeploymentStatus.READY
    assert call_args.kwargs["all_versions"] is True


def test_all_versions_parameter_type_validation(client, mock_deployment_service, sample_page):
    """Test that all_versions only accepts boolean values."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Valid boolean values should work
    for value in ["true", "false", "True", "False", "1", "0"]:
        response = client.get(f"/v2/workspaces/default/deployments?all_versions={value}")
        assert response.status_code == 200


def test_page_parameter_validation(client, mock_deployment_service, sample_page):
    """Test page parameter validation."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Valid page number
    response = client.get("/v2/workspaces/default/deployments?page=5")
    assert response.status_code == 200


def test_page_size_parameter_validation(client, mock_deployment_service, sample_page):
    """Test page_size parameter validation."""
    mock_deployment_service.list_deployments.return_value = sample_page

    # Valid page size
    response = client.get("/v2/workspaces/default/deployments?page_size=10")
    assert response.status_code == 200


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_create_deployment_entity_validation_error_returns_422(
    _mock_deployment_enabled, client, mock_deployment_service
):
    """Test that entity store validation errors during deployment creation return 422."""
    mock_deployment_service.create_deployment.side_effect = EntityValidationError("name must match pattern")

    response = client.post(
        "/v2/workspaces/default/deployments",
        json={"name": "my-deployment", "config": "my-config"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_update_deployment_entity_validation_error_returns_422(
    _mock_deployment_enabled, client, mock_deployment_service
):
    """Test that entity store validation errors during deployment update return 422."""
    mock_deployment_service.update_deployment.side_effect = EntityValidationError("name must match pattern")

    response = client.post(
        "/v2/workspaces/default/deployments/my-deployment",
        json={"config": "other-config"},
    )

    assert response.status_code == 422
    assert "name must match pattern" in response.json()["detail"]


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_update_deployment_status_entity_validation_error_returns_422(
    _mock_deployment_enabled, client, mock_deployment_service
):
    """Test that entity store validation errors during deployment status update return 422."""
    mock_deployment_service.update_deployment_status.side_effect = EntityValidationError("status_message invalid")

    response = client.post(
        "/v2/workspaces/default/deployments/my-deployment/status",
        json={"status": "READY"},
    )

    assert response.status_code == 422
    assert "status_message invalid" in response.json()["detail"]


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_update_deployment_status_conflict_returns_409(_mock_deployment_enabled, client, mock_deployment_service):
    """Test that DELETING status conflict during status update returns 409."""
    mock_deployment_service.update_deployment_status.side_effect = DeploymentStatusConflictError(
        "Deployment is marked for deletion (DELETING). Only transition to DELETED is allowed."
    )

    response = client.post(
        "/v2/workspaces/default/deployments/my-deployment/status",
        json={"status": "READY"},
    )

    assert response.status_code == 409
    assert "Only transition to DELETED is allowed" in response.json()["detail"]


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=False)
def test_create_deployment_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_service
):
    """Creating a deployment must return 422 when deployments are not enabled."""
    response = client.post(
        "/v2/workspaces/default/deployments",
        json={"name": "my-deployment", "config": "my-config"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_service.create_deployment.called


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=False)
def test_update_deployment_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_service
):
    """Updating a deployment must return 422 when deployments are not enabled."""
    response = client.post(
        "/v2/workspaces/default/deployments/my-deployment",
        json={"config": "other-config"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_service.update_deployment.called


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=False)
def test_update_deployment_status_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_service
):
    """Updating deployment status must return 422 when deployments are not enabled."""
    response = client.post(
        "/v2/workspaces/default/deployments/my-deployment/status",
        json={"status": "READY"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_service.update_deployment_status.called


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=False)
def test_delete_all_deployment_versions_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_service
):
    """Deleting all versions of a deployment must return 422 when deployments are not enabled."""
    response = client.delete("/v2/workspaces/default/deployments/d")
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_service.delete_deployment.called


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=False)
def test_delete_deployment_version_when_deployments_disabled_returns_422(
    _mock_deployments_enabled, client, mock_deployment_service
):
    """Deleting a single deployment version must return 422 when deployments are not enabled."""
    response = client.delete("/v2/workspaces/default/deployments/d/versions/1")
    assert response.status_code == 422
    assert response.json()["detail"] == _DEPLOYMENTS_NOT_ENABLED
    assert not mock_deployment_service.delete_deployment.called


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_delete_deployment_conflict_returns_409(_mock_deployments_enabled, client, mock_deployment_service):
    """Stale deployment deletes return 409."""
    mock_deployment_service.delete_deployment.side_effect = EntityConflictError("stale version")

    response = client.delete("/v2/workspaces/default/deployments/d")

    assert response.status_code == 409


@patch("nhx.core.models.api.v2.deployments.deployments_enabled", return_value=True)
def test_delete_deployment_version_conflict_returns_409(_mock_deployments_enabled, client, mock_deployment_service):
    """Stale deployment version deletes return 409."""
    mock_deployment_service.delete_deployment.side_effect = EntityConflictError("stale version")

    response = client.delete("/v2/workspaces/default/deployments/d/versions/1")

    assert response.status_code == 409
