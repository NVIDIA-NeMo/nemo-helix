# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ModelProvider service with in-memory EntityClient."""

import pytest
from nhx.common.api.common import Page
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.schemas import (
    CreateModelProviderRequest,
    DeleteModelProviderRequest,
    GetModelProviderRequest,
    ModelProviderStatus,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client


@pytest.fixture
def client_context(secrets_service_config) -> ClientContext:
    """Create a ClientContext backed by in-memory storage for integration testing."""
    workspaces = ["default", "workspace1", "workspace2"]
    with create_test_client(
        SecretsService,
        client_type=ClientContext,
        workspaces=workspaces,
        service_configs={SecretsService: secrets_service_config},
    ) as ctx:
        yield ctx


@pytest.fixture
def model_provider_service(client_context):
    """Create a ModelProviderService with test EntityClient."""
    return ModelProviderService(client_context.entity_client)


@pytest.fixture
def sample_create_request():
    """Create a sample CreateModelProviderRequest for testing."""
    return CreateModelProviderRequest(
        name="test-provider",
        project="test-project",
        description="A test model provider",
        host_url="https://api.example.com/v1",
        api_key_secret_name="test-api-key-secret",
        enabled_models=["model1", "model2"],
    )


@pytest.mark.asyncio
async def test_create_model_provider_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test end-to-end model provider creation."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")

    # Act
    created_provider = await model_provider_service.create_model_provider(sample_create_request, "default")

    # Assert
    assert created_provider is not None
    assert created_provider.name == sample_create_request.name
    assert created_provider.workspace == "default"
    assert created_provider.project == sample_create_request.project
    assert created_provider.description == sample_create_request.description
    assert created_provider.host_url == sample_create_request.host_url
    assert created_provider.api_key_secret_name == "test-api-key-secret"
    assert created_provider.enabled_models == sample_create_request.enabled_models
    assert created_provider.status == ModelProviderStatus.CREATED
    assert created_provider.status_message == "Model provider created"
    assert created_provider.created_at is not None
    assert created_provider.updated_at is not None


@pytest.mark.asyncio
async def test_create_model_provider_duplicate_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test that creating duplicate model providers raises ValueError."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    await model_provider_service.create_model_provider(sample_create_request, "default")

    # Act & Assert - try to create another with same workspace/name
    with pytest.raises(ValueError, match="already exists"):
        await model_provider_service.create_model_provider(sample_create_request, "default")


@pytest.mark.asyncio
async def test_get_model_provider_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test end-to-end model provider retrieval."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    created_provider = await model_provider_service.create_model_provider(sample_create_request, "default")

    # Act
    get_request = GetModelProviderRequest(workspace="default", name=sample_create_request.name)
    retrieved_provider = await model_provider_service.get_model_provider(get_request)

    # Assert
    assert retrieved_provider is not None
    assert retrieved_provider.name == created_provider.name
    assert retrieved_provider.workspace == created_provider.workspace
    assert retrieved_provider.project == created_provider.project
    assert retrieved_provider.description == created_provider.description
    assert retrieved_provider.host_url == created_provider.host_url
    assert retrieved_provider.api_key_secret_name == created_provider.api_key_secret_name
    assert retrieved_provider.enabled_models == created_provider.enabled_models
    assert retrieved_provider.status == created_provider.status


@pytest.mark.asyncio
async def test_get_model_provider_not_found_integration(model_provider_service):
    """Test retrieving a non-existent model provider returns None."""
    # Act
    get_request = GetModelProviderRequest(workspace="nonexistent", name="provider")
    retrieved_provider = await model_provider_service.get_model_provider(get_request)

    # Assert
    assert retrieved_provider is None


@pytest.mark.asyncio
async def test_list_model_providers_empty_integration(model_provider_service):
    """Test listing model providers when none exist."""
    # Act
    result = await model_provider_service.list_model_providers(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert result.data == []
    assert result.pagination.total_results == 0


@pytest.mark.asyncio
async def test_list_model_providers_with_data_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test listing model providers with data."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    created_provider = await model_provider_service.create_model_provider(sample_create_request, "default")

    # Act
    result = await model_provider_service.list_model_providers(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].name == created_provider.name
    assert result.data[0].workspace == created_provider.workspace
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_model_providers_filter_by_workspace_integration(model_provider_service):
    """Test listing model providers filtered by workspace."""
    # Arrange
    request1 = CreateModelProviderRequest(
        name="provider1",
        host_url="https://api1.example.com",
    )
    request2 = CreateModelProviderRequest(
        name="provider2",
        host_url="https://api2.example.com",
    )

    await model_provider_service.create_model_provider(request1, "workspace1")
    await model_provider_service.create_model_provider(request2, "workspace2")

    # Act
    result = await model_provider_service.list_model_providers(workspace="workspace1")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].workspace == "workspace1"
    assert result.data[0].name == "provider1"
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_upsert_model_provider_create_new_integration(model_provider_service, client_context, create_secret):
    """Test upserting a model provider when it doesn't exist (create)."""
    # Arrange
    create_secret(client_context, "upsert-api-key-secret")
    upsert_request = UpsertModelProviderRequest(
        description="A new provider",
        host_url="https://api.example.com/v1",
        api_key_secret_name="upsert-api-key-secret",
        enabled_models=["model1"],
    )

    # Act
    upserted_provider = await model_provider_service.upsert_model_provider(
        workspace="default", name="new-provider", request=upsert_request
    )

    # Assert
    assert upserted_provider is not None
    assert upserted_provider.name == "new-provider"
    assert upserted_provider.workspace == "default"
    assert upserted_provider.description == upsert_request.description
    assert upserted_provider.host_url == upsert_request.host_url
    assert upserted_provider.api_key_secret_name == "upsert-api-key-secret"
    assert upserted_provider.enabled_models == upsert_request.enabled_models


@pytest.mark.asyncio
async def test_upsert_model_provider_update_existing_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test upserting a model provider when it exists (update)."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    create_secret(client_context, "updated-api-key-secret")
    created_provider = await model_provider_service.create_model_provider(sample_create_request, "default")

    upsert_request = UpsertModelProviderRequest(
        description="Updated description",
        host_url="https://api.updated.com/v1",
        api_key_secret_name="updated-api-key-secret",
        enabled_models=["updated-model"],
    )

    # Act
    upserted_provider = await model_provider_service.upsert_model_provider(
        "default", sample_create_request.name, upsert_request
    )

    # Assert
    assert upserted_provider is not None
    assert upserted_provider.name == created_provider.name
    assert upserted_provider.workspace == created_provider.workspace
    assert upserted_provider.description == upsert_request.description
    assert upserted_provider.host_url == upsert_request.host_url
    assert upserted_provider.api_key_secret_name == "updated-api-key-secret"
    assert upserted_provider.enabled_models == upsert_request.enabled_models


@pytest.mark.asyncio
async def test_delete_model_provider_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test end-to-end model provider deletion."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    await model_provider_service.create_model_provider(sample_create_request, "default")

    # Act
    delete_request = DeleteModelProviderRequest(workspace="default", name=sample_create_request.name)
    deleted = await model_provider_service.delete_model_provider(delete_request)

    # Assert
    assert deleted is True

    # Verify it's actually deleted
    get_request = GetModelProviderRequest(workspace="default", name=sample_create_request.name)
    retrieved_provider = await model_provider_service.get_model_provider(get_request)
    assert retrieved_provider is None


@pytest.mark.asyncio
async def test_delete_model_provider_not_found_integration(model_provider_service):
    """Test deleting a non-existent model provider returns False."""
    # Act
    delete_request = DeleteModelProviderRequest(workspace="nonexistent", name="provider")
    deleted = await model_provider_service.delete_model_provider(delete_request)

    # Assert
    assert deleted is False


@pytest.mark.asyncio
async def test_update_model_provider_status_integration(
    model_provider_service, sample_create_request, client_context, create_secret
):
    """Test end-to-end model provider status update."""
    # Arrange
    create_secret(client_context, "test-api-key-secret")
    await model_provider_service.create_model_provider(sample_create_request, "default")

    request = UpdateModelProviderStatusRequest(
        status=ModelProviderStatus.READY,
        status_message="Provider is ready",
    )

    # Act
    updated_provider = await model_provider_service.update_model_provider_status(
        workspace="default",
        name=sample_create_request.name,
        request=request,
    )

    # Assert
    assert updated_provider is not None
    assert updated_provider.status == ModelProviderStatus.READY
    assert updated_provider.status_message == "Provider is ready"


@pytest.mark.asyncio
async def test_update_model_provider_status_not_found_integration(model_provider_service):
    """Test updating status of a non-existent model provider returns None."""
    # Arrange
    request = UpdateModelProviderStatusRequest(status=ModelProviderStatus.READY)

    # Act
    updated_provider = await model_provider_service.update_model_provider_status(
        workspace="nonexistent", name="provider", request=request
    )

    # Assert
    assert updated_provider is None


# Provider Name Tests


@pytest.mark.asyncio
async def test_create_provider_with_long_name(model_provider_service, client_context, create_secret):
    """Test that provider names work correctly with long names."""
    # Arrange
    create_secret(client_context, "long-name-api-key-secret")
    long_name = "p" * 32  # Max valid name length
    create_request = CreateModelProviderRequest(
        name=long_name,
        project="test-project",
        description="Test provider with long name",
        host_url="https://api.example.com/v1",
        api_key_secret_name="long-name-api-key-secret",
        enabled_models=["model1"],
    )

    # Act
    created_provider = await model_provider_service.create_model_provider(create_request, "default")

    # Assert - provider should be created successfully
    assert created_provider is not None
    assert created_provider.name == long_name
    assert created_provider.api_key_secret_name == "long-name-api-key-secret"
