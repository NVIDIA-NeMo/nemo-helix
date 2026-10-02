# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelDeploymentConfig service with mocked EntityClient."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.entities.client import EntityClient
from nhx.core.models.api.service.model_deployment_config_service import (
    ModelDeploymentConfigService,
    ReferentialIntegrityError,
)
from nhx.core.models.entities import ModelDeployment as ModelDeploymentEntity
from nhx.core.models.entities import ModelDeploymentConfig as ModelDeploymentConfigEntity
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    ModelDeploymentConfig,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelType,
    UpdateModelDeploymentConfigRequest,
)


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


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """Create a mock EntityClient for testing."""
    mock = AsyncMock(spec=EntityClient)
    return mock


@pytest.fixture
def deployment_config_service(mock_entity_client):
    """Create a ModelDeploymentConfigService with mocked EntityClient."""
    return ModelDeploymentConfigService(mock_entity_client)


@pytest.fixture
def sample_model_spec():
    """Create a sample model spec for testing."""
    return ModelDeploymentConfigModelSpec(
        model_type=ModelType.LLM,
        lora_enabled=False,
        model_namespace="nvidia",
        model_name="llama-3-8b",
    )


@pytest.fixture
def sample_executor_config():
    """Create a sample executor config for testing."""
    return ContainerExecutorConfig(
        gpu=1,
        disk_size="50Gi",
        image_name="nvcr.io/nvidia/nim/llm",
        image_tag="latest",
    )


@pytest.fixture
def sample_create_request(sample_model_spec, sample_executor_config):
    """Create a sample CreateModelDeploymentConfigRequest for testing."""
    return CreateModelDeploymentConfigRequest(
        name="test-config",
        project="test-project",
        description="A test deployment configuration",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-123",
    )


@pytest.fixture
def sample_config_entity(sample_model_spec, sample_executor_config):
    """Create a sample ModelDeploymentConfig entity for testing."""
    return create_config_entity(
        name="test-config-v1",
        workspace="default",
        base_name="test-config",
        entity_version=1,
        project="test-project",
        description="A test deployment configuration",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-123",
    )


@pytest.mark.asyncio
async def test_create_deployment_config_success(
    deployment_config_service, mock_entity_client, sample_create_request, sample_config_entity
):
    """Test successful deployment config creation."""
    # Arrange - no existing config
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.create.return_value = sample_config_entity

    # Act
    result = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Assert
    assert result is not None
    assert isinstance(result, ModelDeploymentConfig)
    assert result.name == "test-config"  # base_name
    assert result.workspace == "default"
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert isinstance(call_args, ModelDeploymentConfigEntity)
    assert call_args.name == "test-config-v1"
    assert call_args.base_name == "test-config"
    assert call_args.entity_version == 1


@pytest.mark.asyncio
async def test_create_deployment_config_already_exists(
    deployment_config_service, mock_entity_client, sample_create_request, sample_config_entity
):
    """Test that creating duplicate deployment config raises ValueError."""
    # Arrange - config already exists
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_config_entity]
    mock_entity_client.list.return_value = mock_list_result

    # Act & Assert
    with pytest.raises(ValueError, match="already exists"):
        await deployment_config_service.create_deployment_config(sample_create_request, "default")

    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_generic_config_requires_image_and_health_path(deployment_config_service, mock_entity_client):
    """A generic config missing image_name or health_check_path is rejected at create."""
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    request = CreateModelDeploymentConfigRequest(
        name="generic-config",
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=0),  # no image_name / health_check_path
    )

    with pytest.raises(ValueError, match="image_name"):
        await deployment_config_service.create_deployment_config(request, "default")
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_generic_config_requires_health_check_path(deployment_config_service, mock_entity_client):
    """A generic config with image_name set but health_check_path missing is rejected."""
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    request = CreateModelDeploymentConfigRequest(
        name="generic-config",
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=0, image_name="my/image"),  # no health_check_path
    )

    with pytest.raises(ValueError, match="health_check_path"):
        await deployment_config_service.create_deployment_config(request, "default")
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_generic_config_rejects_whitespace_padded_fields(deployment_config_service, mock_entity_client):
    """Whitespace-padded generic image/health-path values are rejected, not silently stored."""
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    request = CreateModelDeploymentConfigRequest(
        name="generic-config",
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(
            gpu=0,
            image_name=" my/image ",
            health_check_path=" /health ",
        ),
    )

    with pytest.raises(ValueError, match="whitespace"):
        await deployment_config_service.create_deployment_config(request, "default")
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_generic_config_requires_image_and_health_path(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """The update path enforces the same generic validation as create."""
    # Existing config so update proceeds to validation.
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_config_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity

    request = UpdateModelDeploymentConfigRequest(
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=0),  # no image_name / health_check_path
    )

    with pytest.raises(ValueError, match="image_name"):
        await deployment_config_service.update_deployment_config("default", "test-config", request)
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_generic_config_succeeds_when_image_and_health_path_set(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """A generic config with image_name + health_check_path passes validation."""
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.create.return_value = sample_config_entity

    request = CreateModelDeploymentConfigRequest(
        name="generic-config",
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(
            gpu=0,
            image_name="nvcr.io/nim/nvidia/nemoguard-jailbreak-detect",
            image_tag="1.10.1",
            health_check_path="/v1/health/ready",
        ),
        model_entity_id="model-entity-123",
    )

    result = await deployment_config_service.create_deployment_config(request, "default")
    assert result is not None
    mock_entity_client.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_generic_config_rejects_lora_enabled(deployment_config_service, mock_entity_client):
    """LoRA is unsupported for generic (no compiler to wire the sidecar) -> rejected."""
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    request = CreateModelDeploymentConfigRequest(
        name="generic-config",
        engine="generic",
        model_spec=ModelDeploymentConfigModelSpec(lora_enabled=True),
        executor_config=ContainerExecutorConfig(
            gpu=0,
            image_name="my/image",
            image_tag="1.0",
            health_check_path="/healthz",
        ),
    )

    with pytest.raises(ValueError, match="LoRA"):
        await deployment_config_service.create_deployment_config(request, "default")
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_get_deployment_config_found(deployment_config_service, mock_entity_client, sample_config_entity):
    """Test retrieving an existing deployment config."""
    # Arrange - for get_latest_version
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_config_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity

    # Act
    result = await deployment_config_service.get_deployment_config("default", "test-config")

    # Assert
    assert result is not None
    assert result.name == "test-config"
    assert result.entity_version == 1


@pytest.mark.asyncio
async def test_get_deployment_config_with_version(deployment_config_service, mock_entity_client, sample_config_entity):
    """Test retrieving a specific version of a deployment config."""
    # Arrange
    mock_entity_client.get.return_value = sample_config_entity

    # Act
    result = await deployment_config_service.get_deployment_config("default", "test-config", version=1)

    # Assert
    assert result is not None
    assert result.entity_version == 1
    mock_entity_client.get.assert_called_once_with(
        ModelDeploymentConfigEntity, workspace="default", name="test-config-v1"
    )


@pytest.mark.asyncio
async def test_get_deployment_config_not_found(deployment_config_service, mock_entity_client):
    """Test retrieving a non-existent deployment config."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    # Act
    result = await deployment_config_service.get_deployment_config("default", "nonexistent")

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_list_deployment_configs_empty(deployment_config_service, mock_entity_client):
    """Test listing deployment configs when none exist."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = []
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert
    assert result.data == []
    assert result.pagination.total_results == 0


@pytest.mark.asyncio
async def test_list_deployment_configs_with_data(deployment_config_service, mock_entity_client, sample_config_entity):
    """Test listing deployment configs with data (returns only latest versions)."""
    # Arrange - two versions of same config
    version2 = create_config_entity(
        entity_id="config-id-124",
        name="test-config-v2",
        workspace="default",
        base_name="test-config",
        entity_version=2,
        project="test-project",
        description="Updated description",
        engine="nim",
        model_spec=sample_config_entity.model_spec,
        executor_config=sample_config_entity.executor_config,
        model_entity_id="model-entity-123",
    )
    mock_result = MagicMock()
    mock_result.data = [sample_config_entity, version2]
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert - should only return latest version
    assert len(result.data) == 1
    assert result.data[0].entity_version == 2


@pytest.mark.asyncio
async def test_list_deployment_configs_with_workspace_filter(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """Test listing deployment configs with workspace filter."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = [sample_config_entity]
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert
    assert len(result.data) == 1
    mock_entity_client.list.assert_called_once()
    call_kwargs = mock_entity_client.list.call_args[1]
    assert call_kwargs["workspace"] == "default"


@pytest.mark.asyncio
async def test_list_deployment_config_versions(deployment_config_service, mock_entity_client, sample_config_entity):
    """Test listing all versions of a specific deployment config."""
    # Arrange - two versions
    version2 = create_config_entity(
        entity_id="config-id-124",
        name="test-config-v2",
        workspace="default",
        base_name="test-config",
        entity_version=2,
        project="test-project",
        description="Updated description",
        engine="nim",
        model_spec=sample_config_entity.model_spec,
        executor_config=sample_config_entity.executor_config,
        model_entity_id="model-entity-123",
    )
    mock_result = MagicMock()
    mock_result.data = [sample_config_entity, version2]
    mock_entity_client.list.return_value = mock_result

    # Act
    result = await deployment_config_service.list_deployment_config_versions("default", "test-config")

    # Assert - should be sorted by version desc
    assert len(result) == 2
    assert result[0].entity_version == 2
    assert result[1].entity_version == 1


@pytest.mark.asyncio
async def test_update_deployment_config_success(
    deployment_config_service, mock_entity_client, sample_config_entity, sample_model_spec, sample_executor_config
):
    """Test successful deployment config update (creates new version)."""
    # Arrange - get_latest_version returns 1
    mock_list_result = MagicMock()
    mock_list_result.data = [sample_config_entity]
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.get.return_value = sample_config_entity

    version2 = create_config_entity(
        entity_id="config-id-124",
        name="test-config-v2",
        workspace="default",
        base_name="test-config",
        entity_version=2,
        project="test-project",
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )
    mock_entity_client.create.return_value = version2

    update_request = UpdateModelDeploymentConfigRequest(
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )

    # Act
    result = await deployment_config_service.update_deployment_config("default", "test-config", update_request)

    # Assert
    assert result is not None
    assert result.entity_version == 2
    assert result.description == "Updated description"
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert call_args.name == "test-config-v2"
    assert call_args.entity_version == 2


@pytest.mark.asyncio
async def test_update_deployment_config_not_found(
    deployment_config_service, mock_entity_client, sample_model_spec, sample_executor_config
):
    """Test updating a non-existent deployment config raises error."""
    # Arrange
    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result

    update_request = UpdateModelDeploymentConfigRequest(
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )

    # Act & Assert
    with pytest.raises(ValueError, match="does not exist"):
        await deployment_config_service.update_deployment_config("default", "nonexistent", update_request)


@pytest.mark.asyncio
async def test_delete_deployment_config_success(deployment_config_service, mock_entity_client, sample_config_entity):
    """Test successful deployment config deletion (all versions)."""
    # Arrange - no dependent deployments
    mock_dep_result = MagicMock()
    mock_dep_result.data = []

    mock_config_result = MagicMock()
    mock_config_result.data = [sample_config_entity]

    # First call: check dependent deployments, second call: list config versions
    mock_entity_client.list.side_effect = [mock_dep_result, mock_config_result]
    mock_entity_client.delete.return_value = None

    # Act
    result = await deployment_config_service.delete_deployment_config("default", "test-config")

    # Assert
    assert result is True
    mock_entity_client.delete.assert_called_once_with(
        ModelDeploymentConfigEntity,
        sample_config_entity.name,
        workspace="default",
    )


@pytest.mark.asyncio
async def test_delete_deployment_config_specific_version(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """Test deleting a specific version of a deployment config."""
    # Arrange - no dependent deployments
    mock_dep_result = MagicMock()
    mock_dep_result.data = []
    mock_entity_client.list.return_value = mock_dep_result
    mock_entity_client.get.return_value = sample_config_entity
    mock_entity_client.delete.return_value = None

    # Act
    result = await deployment_config_service.delete_deployment_config("default", "test-config", version=1)

    # Assert
    assert result is True
    mock_entity_client.delete.assert_called_once_with(
        ModelDeploymentConfigEntity,
        sample_config_entity.name,
        workspace="default",
    )


@pytest.mark.asyncio
async def test_delete_deployment_config_not_found(deployment_config_service, mock_entity_client):
    """Test deleting a non-existent deployment config."""
    # Arrange
    mock_dep_result = MagicMock()
    mock_dep_result.data = []

    mock_config_result = MagicMock()
    mock_config_result.data = []

    mock_entity_client.list.side_effect = [mock_dep_result, mock_config_result]

    # Act
    result = await deployment_config_service.delete_deployment_config("default", "nonexistent")

    # Assert
    assert result is False


@pytest.mark.asyncio
async def test_delete_deployment_config_with_dependent_deployments(deployment_config_service, mock_entity_client):
    """Test that deleting a config with dependent deployments raises ReferentialIntegrityError."""
    # Arrange - create a mock deployment that references this config
    mock_deployment = create_deployment_entity(
        entity_id="dep-id-123",
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        config="test-config",
        config_version=1,
        project="test-project",
        status=ModelDeploymentStatus.READY,
        status_message="Running",
    )

    mock_dep_result = MagicMock()
    mock_dep_result.data = [mock_deployment]
    mock_entity_client.list.return_value = mock_dep_result

    # Act & Assert
    with pytest.raises(ReferentialIntegrityError) as exc_info:
        await deployment_config_service.delete_deployment_config("default", "test-config")

    assert "Cannot delete ModelDeploymentConfig" in str(exc_info.value)
    assert "1 ModelDeployment(s) still reference it" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_deployment_config_version_with_dependent_deployments(
    deployment_config_service, mock_entity_client
):
    """Test that deleting a specific version with dependent deployments raises ReferentialIntegrityError."""
    # Arrange - create a mock deployment that references this specific version
    mock_deployment = create_deployment_entity(
        entity_id="dep-id-123",
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        config="test-config",
        config_version=1,  # References version 1
        project="test-project",
        status=ModelDeploymentStatus.READY,
        status_message="Running",
    )

    mock_dep_result = MagicMock()
    mock_dep_result.data = [mock_deployment]
    mock_entity_client.list.return_value = mock_dep_result

    # Act & Assert
    with pytest.raises(ReferentialIntegrityError) as exc_info:
        await deployment_config_service.delete_deployment_config("default", "test-config", version=1)

    assert "Cannot delete ModelDeploymentConfig" in str(exc_info.value)
    assert "version 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_delete_deployment_config_version_no_dependent_on_that_version(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """Test that deleting a version succeeds when dependents reference other versions."""
    # Arrange - deployment references version 2, not version 1
    mock_deployment = create_deployment_entity(
        entity_id="dep-id-123",
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        config="test-config",
        config_version=2,  # References version 2, not 1
        project="test-project",
        status=ModelDeploymentStatus.READY,
        status_message="Running",
    )

    mock_dep_result = MagicMock()
    mock_dep_result.data = [mock_deployment]
    mock_entity_client.list.return_value = mock_dep_result
    mock_entity_client.get.return_value = sample_config_entity
    mock_entity_client.delete.return_value = None

    # Act - delete version 1 (should succeed since deployment references version 2)
    result = await deployment_config_service.delete_deployment_config("default", "test-config", version=1)

    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_delete_deployment_config_with_deleted_dependent_deployments(
    deployment_config_service, mock_entity_client, sample_config_entity
):
    """Test that deleting a config succeeds when dependent deployments are in DELETED status."""
    # Arrange - create a mock deployment in DELETED status
    mock_deployment = create_deployment_entity(
        entity_id="dep-id-123",
        name="test-deployment-v1",
        workspace="default",
        base_name="test-deployment",
        entity_version=1,
        config="test-config",
        config_version=1,
        project="test-project",
        status=ModelDeploymentStatus.DELETED,  # DELETED
        status_message="Deleted",
    )

    mock_dep_result = MagicMock()
    mock_dep_result.data = [mock_deployment]

    mock_config_result = MagicMock()
    mock_config_result.data = [sample_config_entity]

    mock_entity_client.list.side_effect = [mock_dep_result, mock_config_result]
    mock_entity_client.delete.return_value = None

    # Act - delete the config (should succeed since deployment is DELETED)
    result = await deployment_config_service.delete_deployment_config("default", "test-config")

    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_nim_deployment_configuration_preserved(deployment_config_service, mock_entity_client):
    """Test that complex NIM deployment configuration is properly preserved."""
    # Arrange
    complex_model_spec = ModelDeploymentConfigModelSpec(
        model_type=ModelType.LLM,
        lora_enabled=True,
        model_namespace="custom-org",
        model_name="custom-model-70b",
    )
    complex_executor_config = ContainerExecutorConfig(
        gpu=4,
        disk_size="100Gi",
        image_name="nvcr.io/nvidia/nim/custom-llm",
        image_tag="v1.2.3",
    )

    create_request = CreateModelDeploymentConfigRequest(
        name="complex-config",
        engine="nim",
        model_spec=complex_model_spec,
        executor_config=complex_executor_config,
    )

    created_entity = create_config_entity(
        entity_id="config-id-456",
        name="complex-config-v1",
        workspace="production",
        base_name="complex-config",
        entity_version=1,
        engine="nim",
        model_spec=complex_model_spec,
        executor_config=complex_executor_config,
    )

    mock_list_result = MagicMock()
    mock_list_result.data = []
    mock_entity_client.list.return_value = mock_list_result
    mock_entity_client.create.return_value = created_entity

    # Act
    result = await deployment_config_service.create_deployment_config(create_request, "production")

    # Assert
    assert result is not None
    assert result.model_spec.model_type == ModelType.LLM
    assert result.model_spec.lora_enabled is True
    assert result.executor_config.gpu == 4
    assert result.executor_config.disk_size == "100Gi"
    assert result.executor_config.image_name == "nvcr.io/nvidia/nim/custom-llm"
    assert result.executor_config.image_tag == "v1.2.3"
