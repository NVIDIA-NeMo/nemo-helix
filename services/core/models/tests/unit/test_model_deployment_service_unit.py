# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelDeployment service with mocked EntityClient."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.models.api.service.model_deployment_service import DeploymentStatusConflictError, ModelDeploymentService
from nhx.core.models.entities import ModelDeployment as ModelDeploymentEntity
from nhx.core.models.entities import ModelDeploymentConfig as ModelDeploymentConfigEntity
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentRequest,
    ModelDeployment,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelType,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
)
from pydantic import TypeAdapter


def create_deployment_entity(
    entity_id: str = "deployment-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> ModelDeploymentEntity:
    """Helper to create ModelDeploymentEntity with proper private attributes."""
    entity = ModelDeploymentEntity(**kwargs)
    entity._id = entity_id
    entity._created_at = created_at or datetime.now(timezone.utc)
    entity._updated_at = updated_at or datetime.now(timezone.utc)
    return entity


def create_config_entity(
    entity_id: str = "config-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> ModelDeploymentConfigEntity:
    """Helper to create ModelDeploymentConfigEntity with proper private attributes."""
    entity = ModelDeploymentConfigEntity(**kwargs)
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
def deployment_service(mock_entity_client):
    """Create a ModelDeploymentService with mocked EntityClient."""
    return ModelDeploymentService(mock_entity_client)


@pytest.fixture
def sample_config_entity():
    """Create a sample ModelDeploymentConfig entity for testing."""
    return create_config_entity(
        name="test-config-v1",
        workspace="default",
        base_name="test-config",
        entity_version=1,
        project="test-project",
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(
            model_type=ModelType.LLM,
            model_namespace="nvidia",
            model_name="llama-3-8b",
        ),
        executor_config=ContainerExecutorConfig(gpu=1),
    )


@pytest.fixture
def sample_create_request():
    """Create a sample CreateModelDeploymentRequest for testing."""
    return CreateModelDeploymentRequest(
        name="test-deployment",
        project="test-project",
        config="test-config",
        config_version=1,
    )


@pytest.fixture
def sample_deployment_entity():
    """Create a sample ModelDeployment entity for testing."""
    return create_deployment_entity(
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        project="test-project",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.CREATED,
        status_message="Deployment created",
    )


def test_model_deployment_entity_deserializes_with_legacy_hf_token_secret_name():
    """Backward compatibility: stored data with removed hf_token_secret_name still deserializes.

    The datastore may return entity data that still contains hf_token_secret_name until
    migrations run. Pydantic ignores extra keys by default, so old rows can be retrieved
    without error.
    """
    # Dict mimicking entity_client._convert_api_entity_to_model input (entity_dict from API + data)
    stored_data: dict[str, Any] = {
        "name": "test-deployment-v1",
        "workspace": "default",
        "base_name": "test-deployment",
        "entity_version": 1,
        "config": "test-config",
        "config_version": 1,
        "status": ModelDeploymentStatus.CREATED.value,
        "status_message": "Deployment created",
        "project": "test-project",
        # Legacy field that was removed from the entity; should be ignored when deserializing
        "hf_token_secret_name": "my-hf-token-secret",
    }
    adapter = TypeAdapter(ModelDeploymentEntity)
    result = adapter.validate_python(stored_data)
    assert isinstance(result, ModelDeploymentEntity)
    assert result.base_name == "test-deployment"
    assert result.config == "test-config"
    assert not hasattr(result, "hf_token_secret_name")


@pytest.mark.asyncio
async def test_create_deployment_success(
    deployment_service,
    mock_entity_client,
    sample_create_request,
    sample_config_entity,
    sample_deployment_entity,
):
    """Test successful deployment creation."""
    # Arrange - no existing deployment
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity
    mock_entity_client.create.return_value = sample_deployment_entity

    # Act
    result = await deployment_service.create_deployment(sample_create_request, "default")

    # Assert
    assert result is not None
    assert isinstance(result, ModelDeployment)
    assert result.name == "test-deployment"
    assert result.status == ModelDeploymentStatus.CREATED
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert isinstance(call_args, ModelDeploymentEntity)
    assert call_args.name == "test-deployment-v1"
    assert call_args.base_name == "test-deployment"
    assert call_args.entity_version == 1


@pytest.mark.asyncio
async def test_create_deployment_already_exists(
    deployment_service, mock_entity_client, sample_create_request, sample_deployment_entity
):
    """Test that creating duplicate deployment raises ValueError."""
    # Arrange - deployment already exists
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_deployment_entity]
    mock_entity_client.list.return_value = mock_list_result

    # Act & Assert
    with pytest.raises(ValueError, match="already exists"):
        await deployment_service.create_deployment(sample_create_request, "default")

    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_deployment_config_not_found(deployment_service, mock_entity_client, sample_create_request):
    """Test that creating deployment with non-existent config raises ValueError."""
    # Arrange - no deployment exists, no config exists
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.side_effect = EntityNotFoundError("Config not found")

    # Act & Assert
    with pytest.raises(ValueError, match="does not exist"):
        await deployment_service.create_deployment(sample_create_request, "default")


@pytest.mark.asyncio
async def test_create_deployment_without_hf_token(deployment_service, mock_entity_client, sample_config_entity):
    """Test creating a deployment (no HF token field)."""
    # Arrange
    request = CreateModelDeploymentRequest(
        name="test-deployment",
        config="test-config",
        config_version=1,
    )

    created_entity = create_deployment_entity(
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.CREATED,
        status_message="Deployment created",
    )

    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity
    mock_entity_client.create.return_value = created_entity

    # Act
    result = await deployment_service.create_deployment(request, "default")

    # Assert
    assert result is not None
    assert result.name == "test-deployment"


@pytest.mark.asyncio
async def test_get_deployment_found(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test retrieving an existing deployment."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_deployment_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_deployment_entity

    # Act
    result = await deployment_service.get_deployment("default", "test-deployment")

    # Assert
    assert result is not None
    assert result.name == "test-deployment"
    assert result.entity_version == 1


@pytest.mark.asyncio
async def test_get_deployment_with_version(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test retrieving a specific version of a deployment."""
    # Arrange
    mock_entity_client.get.return_value = sample_deployment_entity

    # Act
    result = await deployment_service.get_deployment("default", "test-deployment", version=1)

    # Assert
    assert result is not None
    assert result.entity_version == 1
    mock_entity_client.get.assert_called_once_with(
        ModelDeploymentEntity, workspace="default", name="test-deployment-v1"
    )


@pytest.mark.asyncio
async def test_get_deployment_not_found(deployment_service, mock_entity_client):
    """Test retrieving a non-existent deployment."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    # Act
    result = await deployment_service.get_deployment("default", "nonexistent")

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_list_deployments_empty(deployment_service, mock_entity_client):
    """Test listing deployments when none exist."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = []
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_service.list_deployments(workspace="default")

    # Assert
    assert result.data == []
    assert result.pagination.total_results == 0


@pytest.mark.asyncio
async def test_list_deployments_with_data(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test listing deployments with data (returns only latest versions by default)."""
    # Arrange - two versions of same deployment
    version2 = create_deployment_entity(
        entity_id="deployment-id-124",
        name="test-deployment-v2",
        workspace="default",
        base_name="test-deployment",
        entity_version=2,
        project="test-project",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.PENDING,
        status_message="Update pending",
    )
    mock_result = MagicMock()
    mock_result.data = [sample_deployment_entity, version2]
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_service.list_deployments(workspace="default")

    # Assert - should only return latest version
    assert len(result.data) == 1
    assert result.data[0].entity_version == 2


@pytest.mark.asyncio
async def test_list_deployments_all_versions(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test listing deployments with all_versions=True."""
    # Arrange - two versions of same deployment
    version2 = create_deployment_entity(
        entity_id="deployment-id-124",
        name="test-deployment-v2",
        workspace="default",
        base_name="test-deployment",
        entity_version=2,
        project="test-project",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.PENDING,
        status_message="Update pending",
    )
    mock_result = MagicMock()
    mock_result.data = [sample_deployment_entity, version2]
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 1
    mock_result.pagination.page_size = 100
    mock_result.pagination.total_pages = 1
    mock_result.pagination.total_results = 2
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_service.list_deployments(workspace="default", all_versions=True)

    # Assert - should return all versions
    assert len(result.data) == 2


@pytest.mark.asyncio
async def test_list_deployment_versions(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test listing all versions of a specific deployment."""
    # Arrange - two versions
    version2 = create_deployment_entity(
        entity_id="deployment-id-124",
        name="test-deployment-v2",
        workspace="default",
        base_name="test-deployment",
        entity_version=2,
        project="test-project",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.PENDING,
        status_message="Update pending",
    )
    mock_result = MagicMock()
    mock_result.data = [sample_deployment_entity, version2]
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_service.list_deployment_versions("default", "test-deployment")

    # Assert - should be sorted by version desc
    assert len(result) == 2
    assert result[0].entity_version == 2
    assert result[1].entity_version == 1


@pytest.mark.asyncio
async def test_update_deployment_success(
    deployment_service, mock_entity_client, sample_deployment_entity, sample_config_entity
):
    """Test successful deployment update (creates new version)."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_deployment_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.side_effect = [
        sample_deployment_entity,  # get current deployment
        sample_config_entity,  # get config
    ]

    version2 = create_deployment_entity(
        entity_id="deployment-id-124",
        name="test-deployment-v2",
        workspace="default",
        base_name="test-deployment",
        entity_version=2,
        project="test-project",
        config="updated-config",
        config_version=1,
        status=ModelDeploymentStatus.PENDING,
        status_message="Deployment update pending",
    )
    mock_entity_client.create.return_value = version2

    update_request = UpdateModelDeploymentRequest(
        config="updated-config",
        config_version=1,
    )

    # Act
    result = await deployment_service.update_deployment("default", "test-deployment", update_request)

    # Assert
    assert result is not None
    assert result.entity_version == 2
    assert result.config == "updated-config"
    mock_entity_client.create.assert_called_once()


@pytest.mark.asyncio
async def test_update_deployment_not_found(deployment_service, mock_entity_client):
    """Test updating a non-existent deployment raises error."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    update_request = UpdateModelDeploymentRequest(
        config="test-config",
        config_version=1,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="does not exist"):
        await deployment_service.update_deployment("default", "nonexistent", update_request)


@pytest.mark.asyncio
async def test_update_deployment_status_success(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test successful deployment status update."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_deployment_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_deployment_entity

    updated_entity = create_deployment_entity(
        entity_id=sample_deployment_entity.id,
        name=sample_deployment_entity.name,
        workspace=sample_deployment_entity.workspace,
        base_name=sample_deployment_entity.base_name,
        entity_version=sample_deployment_entity.entity_version,
        project=sample_deployment_entity.project,
        config=sample_deployment_entity.config,
        config_version=sample_deployment_entity.config_version,
        status=ModelDeploymentStatus.READY,
        status_message="Deployment is ready",
        model_provider_id="provider-123",
        status_history=[
            {
                "timestamp": sample_deployment_entity.created_at.isoformat(),
                "status": "CREATED",
                "status_message": "Deployment created",
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "PENDING",
                "status_message": "Deployment pending",
            },
        ],
        created_at=sample_deployment_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.READY,
        status_message="Deployment is ready",
        model_provider_id="provider-123",
    )

    # Act
    result = await deployment_service.update_deployment_status("default", "test-deployment", request)

    # Assert
    assert result is not None
    assert result.status == ModelDeploymentStatus.READY
    assert result.status_message == "Deployment is ready"
    assert result.model_provider_id == "provider-123"
    assert len(result.status_history) == 2
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_deployment_status_not_found(deployment_service, mock_entity_client):
    """Test updating status of a non-existent deployment."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.READY,
        status_message="Ready",
    )

    # Act
    result = await deployment_service.update_deployment_status("default", "nonexistent", request)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_delete_deployment_marks_deleting(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test that deleting a deployment marks it as DELETING."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_deployment_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_deployment_entity

    updated_entity = create_deployment_entity(
        entity_id=sample_deployment_entity.id,
        name=sample_deployment_entity.name,
        workspace=sample_deployment_entity.workspace,
        base_name=sample_deployment_entity.base_name,
        entity_version=sample_deployment_entity.entity_version,
        project=sample_deployment_entity.project,
        config=sample_deployment_entity.config,
        config_version=sample_deployment_entity.config_version,
        status=ModelDeploymentStatus.DELETING,
        status_message="Deployment deletion requested",
        created_at=sample_deployment_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_entity

    # Act
    result = await deployment_service.delete_deployment("default", "test-deployment", version=1)

    # Assert
    assert result is not None
    assert result.status == ModelDeploymentStatus.DELETING
    mock_entity_client.update.assert_called_once()
    mock_entity_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_deployment_appends_deleting_to_status_history(
    deployment_service, mock_entity_client, sample_deployment_entity
):
    """Marking a deployment DELETING must append a status_history entry.

    Clients read the current status from ``status_history[-1]``; if the delete
    path mutates ``status`` without appending, they see a stale status.
    """
    # Arrange
    sample_deployment_entity.status_history = []
    mock_entity_client.get.return_value = sample_deployment_entity
    mock_entity_client.update.side_effect = lambda entity: entity

    # Act
    await deployment_service.delete_deployment("default", "test-deployment", version=1)

    # Assert: the entity handed to update() carries a DELETING history tail.
    (updated_entity,), _ = mock_entity_client.update.call_args
    assert updated_entity.status_history
    last = updated_entity.status_history[-1]
    assert last["status"] == ModelDeploymentStatus.DELETING.value
    assert last["status_message"] == "Deployment deletion requested"


@pytest.mark.asyncio
async def test_delete_deployment_already_deleted_hard_deletes(deployment_service, mock_entity_client):
    """Test that deleting a DELETED deployment performs hard delete."""
    # Arrange
    deleted_entity = create_deployment_entity(
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        project="test-project",
        config="test-config",
        config_version=1,
        status=ModelDeploymentStatus.DELETED,  # Already DELETED
        status_message="Deleted",
    )

    mock_entity_client.get.return_value = deleted_entity
    mock_entity_client.delete.return_value = None

    # Act
    result = await deployment_service.delete_deployment("default", "test-deployment", version=1)

    # Assert
    assert result is None  # Returns None for hard delete
    mock_entity_client.delete.assert_called_once_with(
        ModelDeploymentEntity,
        deleted_entity.name,
        workspace="default",
        expected_db_version=deleted_entity.db_version,
    )


@pytest.mark.asyncio
async def test_delete_deployment_not_found(deployment_service, mock_entity_client):
    """Test deleting a non-existent deployment."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Not found")

    # Act
    result = await deployment_service.delete_deployment("default", "nonexistent", version=1)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_create_deployment_raises_on_conflict(
    deployment_service, mock_entity_client, sample_create_request, sample_config_entity
):
    """Test that deployment creation raises error on conflict."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity
    mock_entity_client.create.side_effect = EntityConflictError("Conflict")

    # Act & Assert
    with pytest.raises(ValueError, match="already exists"):
        await deployment_service.create_deployment(sample_create_request, "default")


@pytest.mark.asyncio
async def test_update_deployment_status_conflict(deployment_service, mock_entity_client, sample_deployment_entity):
    """Test DELETING deployment rejects non-DELETED status transitions."""
    # Arrange
    deleting_entity = create_deployment_entity(
        entity_id=sample_deployment_entity.id,
        name=sample_deployment_entity.name,
        workspace=sample_deployment_entity.workspace,
        base_name=sample_deployment_entity.base_name,
        entity_version=sample_deployment_entity.entity_version,
        project=sample_deployment_entity.project,
        config=sample_deployment_entity.config,
        config_version=sample_deployment_entity.config_version,
        status=ModelDeploymentStatus.DELETING,
        status_message="Deployment deletion requested",
        created_at=sample_deployment_entity.created_at,
        updated_at=sample_deployment_entity.updated_at,
    )

    mock_list_result = MagicMock()
    mock_list_result.data = [deleting_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = deleting_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.READY,
        status_message="Deployment is ready",
    )

    # Act / Assert
    with pytest.raises(
        DeploymentStatusConflictError,
        match="Only transition to DELETED is allowed",
    ):
        await deployment_service.update_deployment_status("default", "test-deployment", request)

    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_update_deployment_status_deleting_to_deleted_allowed(
    deployment_service, mock_entity_client, sample_deployment_entity
):
    """Test DELETING -> DELETED status transition is allowed."""
    # Arrange
    deleting_entity = create_deployment_entity(
        entity_id=sample_deployment_entity.id,
        name=sample_deployment_entity.name,
        workspace=sample_deployment_entity.workspace,
        base_name=sample_deployment_entity.base_name,
        entity_version=sample_deployment_entity.entity_version,
        project=sample_deployment_entity.project,
        config=sample_deployment_entity.config,
        config_version=sample_deployment_entity.config_version,
        status=ModelDeploymentStatus.DELETING,
        status_message="Deployment deletion requested",
        created_at=sample_deployment_entity.created_at,
        updated_at=sample_deployment_entity.updated_at,
    )
    deleted_entity = create_deployment_entity(
        entity_id=sample_deployment_entity.id,
        name=sample_deployment_entity.name,
        workspace=sample_deployment_entity.workspace,
        base_name=sample_deployment_entity.base_name,
        entity_version=sample_deployment_entity.entity_version,
        project=sample_deployment_entity.project,
        config=sample_deployment_entity.config,
        config_version=sample_deployment_entity.config_version,
        status=ModelDeploymentStatus.DELETED,
        status_message="Deployment deleted",
        created_at=sample_deployment_entity.created_at,
        updated_at=datetime.now(timezone.utc),
    )

    mock_list_result = MagicMock()
    mock_list_result.data = [deleting_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = deleting_entity
    mock_entity_client.update.return_value = deleted_entity

    request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.DELETED,
        status_message="Deployment deleted",
    )

    # Act
    result = await deployment_service.update_deployment_status("default", "test-deployment", request)

    # Assert
    assert result is not None
    assert result.status == ModelDeploymentStatus.DELETED
    mock_entity_client.update.assert_called_once()
