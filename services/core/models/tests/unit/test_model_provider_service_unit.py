# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelProvider service with mocked EntityClient."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.models.api.service.model_provider_service import (
    ModelProviderService,
    ModelProviderValidationError,
)
from nhx.core.models.entities import Model
from nhx.core.models.entities import ModelProvider as ModelProviderEntity
from nhx.core.models.schemas import (
    CreateModelProviderRequest,
    DeleteModelProviderRequest,
    GetModelProviderRequest,
    ModelProvider,
    ModelProviderStatus,
    ServedModelMapping,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)


def create_provider_entity(
    entity_id: str = "provider-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> ModelProviderEntity:
    """Helper to create ModelProviderEntity with proper private attributes."""
    entity = ModelProviderEntity(**kwargs)
    entity._id = entity_id
    entity._created_at = created_at or datetime.now(timezone.utc)
    entity._updated_at = updated_at or datetime.now(timezone.utc)
    return entity


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """Create a mock EntityClient for testing."""
    mock = AsyncMock(spec=EntityClient)
    return mock


@pytest.fixture
def model_provider_service(mock_entity_client):
    """Create a ModelProviderService with mocked EntityClient."""
    return ModelProviderService(mock_entity_client)


@pytest.fixture
def sample_create_request():
    """Create a sample CreateModelProviderRequest for testing."""
    return CreateModelProviderRequest(
        name="test-provider",
        project="test-project",
        description="A test model provider",
        host_url="https://api.example.com/v1",
        api_key="sk-test123456789",
        enabled_models=["model1", "model2"],
    )


@pytest.fixture
def sample_provider_entity():
    """Create a sample ModelProvider entity for testing."""
    return create_provider_entity(
        name="test-provider",
        workspace="default",
        project="test-project",
        description="A test model provider",
        host_url="https://api.example.com/v1",
        api_key_secret_name="test-api-key-id",
        served_models=[],
        enabled_models=["model1", "model2"],
        status=ModelProviderStatus.CREATED,
        status_message="Model provider created",
    )


@pytest.mark.asyncio
async def test_create_model_provider_success(
    model_provider_service, mock_entity_client, sample_create_request, sample_provider_entity
):
    """Test successful model provider creation."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = sample_provider_entity

    # Act
    result = await model_provider_service.create_model_provider(sample_create_request, "default")

    # Assert
    assert result is not None
    assert isinstance(result, ModelProvider)
    assert result.name == sample_create_request.name
    assert result.workspace == sample_provider_entity.workspace
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert isinstance(call_args, ModelProviderEntity)
    assert call_args.name == sample_create_request.name


@pytest.mark.asyncio
async def test_create_model_provider_conflict_error(
    model_provider_service, mock_entity_client, sample_create_request, sample_provider_entity
):
    """Test that EntityConflictError is converted to ValueError."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity  # Provider already exists

    # Act & Assert
    with pytest.raises(ValueError, match="already exists"):
        await model_provider_service.create_model_provider(sample_create_request, "default")

    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_get_model_provider_found(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test retrieving an existing model provider."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity
    get_request = GetModelProviderRequest(workspace="default", name="test-provider")

    # Act
    result = await model_provider_service.get_model_provider(get_request)

    # Assert
    assert result is not None
    assert isinstance(result, ModelProvider)
    assert result.name == sample_provider_entity.name
    mock_entity_client.get.assert_called_once_with(ModelProviderEntity, workspace="default", name="test-provider")


@pytest.mark.asyncio
async def test_get_model_provider_not_found(model_provider_service, mock_entity_client):
    """Test retrieving a non-existent model provider."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    get_request = GetModelProviderRequest(workspace="default", name="nonexistent")

    # Act
    result = await model_provider_service.get_model_provider(get_request)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_list_model_providers_empty(model_provider_service, mock_entity_client):
    """Test listing model providers when none exist."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = []
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 1
    mock_result.pagination.page_size = 100
    mock_result.pagination.total_pages = 0
    mock_result.pagination.total_results = 0
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await model_provider_service.list_model_providers(workspace="default")

    # Assert
    assert result.data == []
    assert result.pagination.total_results == 0
    mock_entity_client.list.assert_called_once()


@pytest.mark.asyncio
async def test_list_model_providers_with_data(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test listing model providers with data."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = [sample_provider_entity]
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 1
    mock_result.pagination.page_size = 100
    mock_result.pagination.total_pages = 1
    mock_result.pagination.total_results = 1
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await model_provider_service.list_model_providers(workspace="default")

    # Assert
    assert len(result.data) == 1
    assert result.data[0].name == sample_provider_entity.name
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_upsert_model_provider_create_new(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test upserting a new model provider (creates it)."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = sample_provider_entity

    upsert_request = UpsertModelProviderRequest(
        description="New description",
        host_url="https://api.new.com/v1",
        api_key="sk-new123456789",
        enabled_models=["new-model"],
    )

    # Act
    result = await model_provider_service.upsert_model_provider("default", "test-provider", upsert_request)

    # Assert
    assert result is not None
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert call_args.name == "test-provider"
    assert call_args.workspace == "default"


@pytest.mark.asyncio
async def test_upsert_model_provider_update_existing(
    model_provider_service, mock_entity_client, sample_provider_entity
):
    """Test upserting an existing model provider (updates it)."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity
    updated_entity = create_provider_entity(
        entity_id=sample_provider_entity.id,
        name=sample_provider_entity.name,
        workspace=sample_provider_entity.workspace,
        project="updated-project",
        description="Updated description",
        host_url="https://api.updated.com/v1",
        api_key_secret_name="new-secret-name",
        served_models=[],
        enabled_models=["updated-model"],
        status=ModelProviderStatus.CREATED,
        status_message="Model provider created",
        created_at=sample_provider_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_entity

    upsert_request = UpsertModelProviderRequest(
        project="updated-project",
        description="Updated description",
        host_url="https://api.updated.com/v1",
        api_key_secret_name="new-secret-name",
        enabled_models=["updated-model"],
    )

    # Act
    result = await model_provider_service.upsert_model_provider("default", "test-provider", upsert_request)

    # Assert
    assert result is not None
    assert result.description == "Updated description"
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_delete_model_provider_success(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test successful model provider deletion."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity
    mock_entity_client.delete.return_value = None
    delete_request = DeleteModelProviderRequest(workspace="default", name="test-provider")

    # Act
    result = await model_provider_service.delete_model_provider(delete_request)

    # Assert
    assert result is True
    mock_entity_client.delete.assert_called_once_with(
        ModelProviderEntity,
        sample_provider_entity.name,
        workspace="default",
        expected_db_version=sample_provider_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_model_provider_not_found(model_provider_service, mock_entity_client):
    """Test deleting a non-existent model provider."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    delete_request = DeleteModelProviderRequest(workspace="default", name="nonexistent")

    # Act
    result = await model_provider_service.delete_model_provider(delete_request)

    # Assert
    assert result is False
    mock_entity_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_provider_status_success(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test successful model provider status fields update."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity

    served_models = [
        ServedModelMapping(model_entity_id="default/model-1", served_model_name="nvidia/model-1"),
        ServedModelMapping(model_entity_id="default/model-2", served_model_name="meta/model-2"),
    ]

    updated_entity = create_provider_entity(
        entity_id=sample_provider_entity.id,
        name=sample_provider_entity.name,
        workspace=sample_provider_entity.workspace,
        project=sample_provider_entity.project,
        description=sample_provider_entity.description,
        host_url=sample_provider_entity.host_url,
        api_key_secret_name=sample_provider_entity.api_key_secret_name,
        served_models=served_models,
        enabled_models=sample_provider_entity.enabled_models,
        model_deployment_id="deployment-123",
        status=ModelProviderStatus.READY,
        status_message="All models discovered",
        created_at=sample_provider_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelProviderStatusRequest(
        model_deployment_id="deployment-123",
        served_models=served_models,
        status=ModelProviderStatus.READY,
        status_message="All models discovered",
    )

    # Act
    result = await model_provider_service.update_model_provider_status("default", "test-provider", request)

    # Assert
    assert result is not None
    assert result.status == ModelProviderStatus.READY
    assert result.status_message == "All models discovered"
    assert result.model_deployment_id == "deployment-123"
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_model_provider_status_partial(model_provider_service, mock_entity_client, sample_provider_entity):
    """Test partial update of status fields (only served_models)."""
    # Arrange
    mock_entity_client.get.return_value = sample_provider_entity
    served_models = [ServedModelMapping(model_entity_id="default/model-1", served_model_name="nvidia/model-1")]

    updated_entity = create_provider_entity(
        entity_id=sample_provider_entity.id,
        name=sample_provider_entity.name,
        workspace=sample_provider_entity.workspace,
        project=sample_provider_entity.project,
        description=sample_provider_entity.description,
        host_url=sample_provider_entity.host_url,
        api_key_secret_name=sample_provider_entity.api_key_secret_name,
        served_models=served_models,
        enabled_models=sample_provider_entity.enabled_models,
        status=sample_provider_entity.status,
        status_message=sample_provider_entity.status_message,
        created_at=sample_provider_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelProviderStatusRequest(served_models=served_models)

    # Act
    result = await model_provider_service.update_model_provider_status("default", "test-provider", request)

    # Assert
    assert result is not None
    assert len(result.served_models) == 1
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_model_provider_status_not_found(model_provider_service, mock_entity_client):
    """Test updating status fields of a non-existent model provider."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    request = UpdateModelProviderStatusRequest(status=ModelProviderStatus.READY, status_message="Ready")

    # Act
    result = await model_provider_service.update_model_provider_status("default", "nonexistent", request)

    # Assert
    assert result is None
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_create_model_provider_without_api_key(model_provider_service, mock_entity_client):
    """Test creating a model provider without an API key secret name."""
    # Arrange
    request = CreateModelProviderRequest(
        name="test-provider",
        host_url="https://api.example.com/v1",
        # No api_key_secret_name provided
    )

    created_entity = create_provider_entity(
        name="test-provider",
        workspace="default",
        host_url="https://api.example.com/v1",
        api_key_secret_name=None,  # No API key secret name
        served_models=[],
        status=ModelProviderStatus.CREATED,
        status_message="Model provider created",
    )
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = created_entity

    # Act
    result = await model_provider_service.create_model_provider(request, "default")

    # Assert
    assert result is not None
    assert result.api_key_secret_name is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_key,field_name,expected_error",
    [
        ("Authorization", "default_extra_headers", "Authorization header is not allowed"),
        ("authorization", "default_extra_headers", "Authorization header is not allowed"),
        ("AUTHORIZATION", "default_extra_headers", "Authorization header is not allowed"),
        ("Authorization", "required_extra_headers", "Authorization header is not allowed"),
        ("Cookie", "default_extra_headers", "Cookie header is not allowed"),
        ("cookie", "default_extra_headers", "Cookie header is not allowed"),
        ("COOKIE", "required_extra_headers", "Cookie header is not allowed"),
    ],
)
async def test_create_model_provider_rejects_reserved_headers(
    model_provider_service, header_key, field_name, expected_error
):
    """Test that create_model_provider rejects reserved headers (case-insensitive)."""
    kwargs = {field_name: {header_key: "some-value"}}
    request = CreateModelProviderRequest(
        name="test-provider",
        host_url="https://api.example.com/v1",
        **kwargs,
    )

    with pytest.raises(ModelProviderValidationError) as exc_info:
        await model_provider_service.create_model_provider(request, "default")

    assert expected_error in str(exc_info.value)
    assert field_name in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_model_provider_accepts_valid_extra_headers(
    model_provider_service, mock_entity_client, sample_provider_entity
):
    """Test that create_model_provider accepts valid headers (non-Authorization, None, empty)."""
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = sample_provider_entity

    request = CreateModelProviderRequest(
        name="test-provider",
        host_url="https://api.example.com/v1",
        default_extra_headers={"X-Custom-Header": "value"},
        required_extra_headers={},
    )

    result = await model_provider_service.create_model_provider(request, "default")
    assert result is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_key,field_name,expected_error",
    [
        ("Authorization", "default_extra_headers", "Authorization header is not allowed"),
        ("Authorization", "required_extra_headers", "Authorization header is not allowed"),
        ("Cookie", "default_extra_headers", "Cookie header is not allowed"),
        ("Cookie", "required_extra_headers", "Cookie header is not allowed"),
    ],
)
async def test_upsert_model_provider_rejects_reserved_headers(
    model_provider_service, header_key, field_name, expected_error
):
    """Test that upsert_model_provider rejects reserved headers."""
    kwargs = {field_name: {header_key: "some-value"}}
    request = UpsertModelProviderRequest(host_url="https://api.example.com/v1", **kwargs)

    with pytest.raises(ModelProviderValidationError) as exc_info:
        await model_provider_service.upsert_model_provider("default", "test-provider", request)

    assert expected_error in str(exc_info.value)
    assert field_name in str(exc_info.value)


# --- Model entity cleanup on provider deletion ---


def _create_model_entity(
    name: str = "test-model",
    workspace: str = "default",
    model_providers: list[str] | None = None,
) -> Model:
    """Helper to create a Model entity for testing."""
    model = Model(
        name=name,
        workspace=workspace,
        model_providers=model_providers or [],
    )
    model._id = f"model-{name}"
    model._created_at = datetime.now(timezone.utc)
    model._updated_at = datetime.now(timezone.utc)
    return model


def _make_entity_get_dispatcher(
    provider: ModelProviderEntity,
    models: dict[str, Model] | None = None,
):
    """Build a mock side_effect for entity_client.get that dispatches by entity type.

    Returns the given provider for ModelProviderEntity lookups.
    For Model lookups, returns matching entries from *models* (keyed by name),
    raising EntityNotFoundError when no match exists.
    """

    async def _dispatch(entity_type, **kwargs):
        if entity_type is ModelProviderEntity:
            return provider
        if entity_type is Model and models:
            name = kwargs.get("name")
            if name in models:
                return models[name]
        raise EntityNotFoundError("not found")

    return _dispatch


@pytest.mark.asyncio
async def test_delete_provider_cleans_up_model_entity_references(model_provider_service, mock_entity_client):
    """Deleting a provider removes it from model_providers on linked model entities."""
    model_entity = _create_model_entity(
        name="my-model",
        workspace="ws",
        model_providers=["ws/provider-to-delete", "ws/other-provider"],
    )
    provider_entity = create_provider_entity(
        name="provider-to-delete",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/my-model", served_model_name="my-model"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.side_effect = _make_entity_get_dispatcher(provider_entity, {"my-model": model_entity})
    mock_entity_client.update.return_value = model_entity
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="provider-to-delete")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    mock_entity_client.update.assert_called_once()
    updated_model = mock_entity_client.update.call_args[0][0]
    assert updated_model.model_providers == ["ws/other-provider"]
    mock_entity_client.delete.assert_called_once_with(
        ModelProviderEntity,
        "provider-to-delete",
        workspace="ws",
        expected_db_version=provider_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_provider_conflict_leaves_model_entity_references(model_provider_service, mock_entity_client):
    """A stale provider version must not clean up linked model references."""
    model_entity = _create_model_entity(
        name="my-model",
        workspace="ws",
        model_providers=["ws/provider-to-delete", "ws/other-provider"],
    )
    provider_entity = create_provider_entity(
        name="provider-to-delete",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/my-model", served_model_name="my-model"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.side_effect = _make_entity_get_dispatcher(provider_entity, {"my-model": model_entity})
    mock_entity_client.delete.side_effect = EntityConflictError("stale version")

    request = DeleteModelProviderRequest(workspace="ws", name="provider-to-delete")
    with pytest.raises(EntityConflictError):
        await model_provider_service.delete_model_provider(request)

    mock_entity_client.update.assert_not_called()
    assert model_entity.model_providers == ["ws/provider-to-delete", "ws/other-provider"]


@pytest.mark.asyncio
async def test_delete_provider_no_served_models_skips_cleanup(
    model_provider_service, mock_entity_client, sample_provider_entity
):
    """When the provider has no served_models the delete proceeds without cleanup queries."""
    mock_entity_client.get.return_value = sample_provider_entity  # served_models=[]
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="default", name="test-provider")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    mock_entity_client.update.assert_not_called()
    mock_entity_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_provider_cleanup_missing_model_entity(model_provider_service, mock_entity_client):
    """Cleanup gracefully skips model entities that no longer exist."""
    provider_entity = create_provider_entity(
        name="my-provider",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/gone-model", served_model_name="gone-model"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.side_effect = _make_entity_get_dispatcher(provider_entity)
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="my-provider")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    mock_entity_client.update.assert_not_called()
    mock_entity_client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_delete_provider_cleanup_provider_not_in_model_providers(model_provider_service, mock_entity_client):
    """Cleanup skips update when the model entity doesn't reference the provider."""
    model_entity = _create_model_entity(
        name="my-model",
        workspace="ws",
        model_providers=["ws/some-other-provider"],
    )
    provider_entity = create_provider_entity(
        name="my-provider",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/my-model", served_model_name="my-model"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.side_effect = _make_entity_get_dispatcher(provider_entity, {"my-model": model_entity})
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="my-provider")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    mock_entity_client.update.assert_not_called()
    mock_entity_client.delete.assert_called_once_with(
        ModelProviderEntity,
        "my-provider",
        workspace="ws",
        expected_db_version=provider_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_provider_cleanup_malformed_model_entity_id(model_provider_service, mock_entity_client):
    """Cleanup skips model entities with malformed model_entity_id (no slash separator)."""
    provider_entity = create_provider_entity(
        name="my-provider",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="no-slash-here", served_model_name="bad-model"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.return_value = provider_entity
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="my-provider")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    mock_entity_client.update.assert_not_called()
    mock_entity_client.delete.assert_called_once_with(
        ModelProviderEntity,
        "my-provider",
        workspace="ws",
        expected_db_version=provider_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_provider_cleanup_error_does_not_block_deletion(model_provider_service, mock_entity_client):
    """A failure during model entity cleanup must not prevent the provider from being deleted."""
    provider_entity = create_provider_entity(
        name="my-provider",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/model-a", served_model_name="model-a"),
        ],
        status=ModelProviderStatus.READY,
    )

    call_count = 0

    async def _failing_model_get(entity_type, **kwargs):
        nonlocal call_count
        if entity_type is ModelProviderEntity:
            return provider_entity
        call_count += 1
        raise RuntimeError("transient entity-store error")

    mock_entity_client.get.side_effect = _failing_model_get
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="my-provider")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    assert call_count == 1
    mock_entity_client.delete.assert_called_once_with(
        ModelProviderEntity,
        "my-provider",
        workspace="ws",
        expected_db_version=provider_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_provider_cleans_up_multiple_model_entities(model_provider_service, mock_entity_client):
    """Deletion cleans up all model entities referenced in served_models."""
    model_a = _create_model_entity(name="model-a", workspace="ws", model_providers=["ws/prov"])
    model_b = _create_model_entity(name="model-b", workspace="ws", model_providers=["ws/prov", "ws/other"])
    provider_entity = create_provider_entity(
        name="prov",
        workspace="ws",
        host_url="https://api.example.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="ws/model-a", served_model_name="model-a"),
            ServedModelMapping(model_entity_id="ws/model-b", served_model_name="model-b"),
        ],
        status=ModelProviderStatus.READY,
    )

    mock_entity_client.get.side_effect = _make_entity_get_dispatcher(
        provider_entity, {"model-a": model_a, "model-b": model_b}
    )
    mock_entity_client.update.return_value = model_a
    mock_entity_client.delete.return_value = None

    request = DeleteModelProviderRequest(workspace="ws", name="prov")
    result = await model_provider_service.delete_model_provider(request)

    assert result is True
    assert mock_entity_client.update.call_count == 2

    updated_models = [call[0][0] for call in mock_entity_client.update.call_args_list]
    for m in updated_models:
        assert "ws/prov" not in m.model_providers
