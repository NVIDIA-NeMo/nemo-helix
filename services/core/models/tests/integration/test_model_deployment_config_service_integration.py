# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ModelDeploymentConfig service with in-memory EntityClient."""

import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nhx.common.api.common import Page
from nhx.common.entities.client import EntityClient
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    ModelDeploymentConfigModelSpec,
    ModelType,
    UpdateModelDeploymentConfigRequest,
)
from nhx.testing import create_test_client


@pytest.fixture
def entity_client() -> EntityClient:
    """Create an EntityClient backed by in-memory storage for integration testing."""
    # Include workspaces needed by tests (default + workspace filter tests)
    workspaces = ["default", "workspace1", "workspace2", "production"]
    # Include projects used in tests
    projects = ["default/test-project", "production/ml-team"]
    with create_test_client(client_type=EntityClient, workspaces=workspaces, projects=projects) as client:
        yield client


@pytest.fixture
def deployment_config_service(entity_client):
    """Create a ModelDeploymentConfigService with MockEntityClient."""
    return ModelDeploymentConfigService(entity_client)


@pytest.fixture
def sample_model_spec():
    """Create a sample ModelDeploymentConfigModelSpec for testing."""
    return ModelDeploymentConfigModelSpec(
        model_type=ModelType.LLM,
        lora_enabled=False,
        model_namespace="nvidia",
        model_name="llama-3-8b",
    )


@pytest.fixture
def sample_executor_config():
    """Create a sample ContainerExecutorConfig for testing."""
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


@pytest.mark.asyncio
async def test_create_deployment_config_integration(deployment_config_service, sample_create_request):
    """Test end-to-end deployment config creation."""
    # Act
    created_config = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Assert
    assert created_config is not None
    assert created_config.name == sample_create_request.name
    assert created_config.workspace == "default"
    assert created_config.project == sample_create_request.project
    assert created_config.description == sample_create_request.description
    assert created_config.entity_version == 1
    assert created_config.model_entity_id == sample_create_request.model_entity_id
    assert created_config.model_spec.model_type == ModelType.LLM
    assert created_config.executor_config.gpu == 1
    assert created_config.executor_config.disk_size == "50Gi"
    assert created_config.created_at is not None
    assert created_config.updated_at is not None


@pytest.mark.asyncio
async def test_create_deployment_config_duplicate_integration(deployment_config_service, sample_create_request):
    """Test that creating duplicate deployment configs raises ValueError."""
    # Arrange - create first config
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Act & Assert - try to create another with same workspace/name
    with pytest.raises(ValueError, match="already exists"):
        await deployment_config_service.create_deployment_config(sample_create_request, "default")


@pytest.mark.asyncio
async def test_get_deployment_config_integration(deployment_config_service, sample_create_request):
    """Test end-to-end deployment config retrieval."""
    # Arrange
    created_config = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Act
    retrieved_config = await deployment_config_service.get_deployment_config("default", sample_create_request.name)

    # Assert
    assert retrieved_config is not None
    assert retrieved_config.name == created_config.name
    assert retrieved_config.workspace == created_config.workspace
    assert retrieved_config.project == created_config.project
    assert retrieved_config.description == created_config.description
    assert retrieved_config.entity_version == created_config.entity_version
    assert retrieved_config.model_entity_id == created_config.model_entity_id


@pytest.mark.asyncio
async def test_get_deployment_config_specific_version_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test retrieving a specific version of a deployment config."""
    # Arrange - create first version
    created_config = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Create second version
    update_request = UpdateModelDeploymentConfigRequest(
        description="Version 2 description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request)

    # Act - retrieve version 1 specifically
    retrieved_config = await deployment_config_service.get_deployment_config(
        "default", sample_create_request.name, version=1
    )

    # Assert
    assert retrieved_config is not None
    assert retrieved_config.entity_version == 1
    assert retrieved_config.description == created_config.description


@pytest.mark.asyncio
async def test_get_deployment_config_latest_version_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test that get without version returns the latest version."""
    # Arrange - create first version
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Create second version
    update_request = UpdateModelDeploymentConfigRequest(
        description="Latest version description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-latest",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request)

    # Act - retrieve without version (should get latest)
    retrieved_config = await deployment_config_service.get_deployment_config("default", sample_create_request.name)

    # Assert
    assert retrieved_config is not None
    assert retrieved_config.entity_version == 2
    assert retrieved_config.description == "Latest version description"
    assert retrieved_config.model_entity_id == "model-entity-latest"


@pytest.mark.asyncio
async def test_get_deployment_config_not_found_integration(deployment_config_service):
    """Test retrieving a non-existent deployment config returns None."""
    # Act
    retrieved_config = await deployment_config_service.get_deployment_config("default", "nonexistent-config")

    # Assert
    assert retrieved_config is None


@pytest.mark.asyncio
async def test_get_deployment_config_workspace_not_found_integration(deployment_config_service):
    """Test retrieving a deployment config from non-existent workspace raises NotFoundError."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_config_service.get_deployment_config("nonexistent", "config")


@pytest.mark.asyncio
async def test_list_deployment_configs_workspace_not_found_integration(deployment_config_service):
    """Test listing deployment configs from non-existent workspace raises NotFoundError."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_config_service.list_deployment_configs(workspace="nonexistent")


@pytest.mark.asyncio
async def test_list_deployment_configs_empty_integration(deployment_config_service):
    """Test listing deployment configs when none exist."""
    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert result.data == []
    assert result.pagination.total_results == 0


@pytest.mark.asyncio
async def test_list_deployment_configs_with_data_integration(deployment_config_service, sample_create_request):
    """Test listing deployment configs with data."""
    # Arrange
    created_config = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].name == created_config.name
    assert result.data[0].workspace == created_config.workspace
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_deployment_configs_returns_only_latest_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test that list returns only the latest version of each config."""
    # Arrange - create first version
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Create second version
    update_request = UpdateModelDeploymentConfigRequest(
        description="Version 2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request)

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].entity_version == 2
    assert result.data[0].description == "Version 2"
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_deployment_configs_filter_by_workspace_integration(
    deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test listing deployment configs filtered by workspace."""
    # Arrange
    request1 = CreateModelDeploymentConfigRequest(
        name="config1",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )
    request2 = CreateModelDeploymentConfigRequest(
        name="config2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )

    await deployment_config_service.create_deployment_config(request1, "workspace1")
    await deployment_config_service.create_deployment_config(request2, "workspace2")

    # Act
    result = await deployment_config_service.list_deployment_configs(workspace="workspace1")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].workspace == "workspace1"
    assert result.data[0].name == "config1"
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_deployment_config_versions_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test listing all versions of a specific deployment config."""
    # Arrange - create multiple versions
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    update_request1 = UpdateModelDeploymentConfigRequest(
        description="Version 2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request1)

    update_request2 = UpdateModelDeploymentConfigRequest(
        description="Version 3",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-789",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request2)

    # Act
    versions = await deployment_config_service.list_deployment_config_versions("default", sample_create_request.name)

    # Assert
    assert len(versions) == 3
    # Should be ordered by version desc
    assert versions[0].entity_version == 3
    assert versions[1].entity_version == 2
    assert versions[2].entity_version == 1


@pytest.mark.asyncio
async def test_list_deployment_config_versions_empty_integration(deployment_config_service):
    """Test listing versions when config doesn't exist."""
    # Act
    versions = await deployment_config_service.list_deployment_config_versions("default", "nonexistent-config")

    # Assert
    assert versions == []


@pytest.mark.asyncio
async def test_list_deployment_config_versions_workspace_not_found_integration(deployment_config_service):
    """Test listing versions from non-existent workspace raises NotFoundError."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_config_service.list_deployment_config_versions("nonexistent", "config")


@pytest.mark.asyncio
async def test_update_deployment_config_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test end-to-end deployment config update (creates new version)."""
    # Arrange
    created_config = await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Act
    update_request = UpdateModelDeploymentConfigRequest(
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-updated",
    )
    updated_config = await deployment_config_service.update_deployment_config(
        "default", sample_create_request.name, update_request
    )

    # Assert
    assert updated_config is not None
    assert updated_config.entity_version == 2
    assert updated_config.description == "Updated description"
    assert updated_config.model_entity_id == "model-entity-updated"

    # Verify old version still exists
    old_version = await deployment_config_service.get_deployment_config(
        "default", sample_create_request.name, version=1
    )
    assert old_version is not None
    assert old_version.description == created_config.description


@pytest.mark.asyncio
async def test_update_deployment_config_not_found_integration(
    deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test updating a non-existent deployment config raises error."""
    # Arrange
    update_request = UpdateModelDeploymentConfigRequest(
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-updated",
    )

    # Act & Assert
    with pytest.raises(ValueError, match="does not exist"):
        await deployment_config_service.update_deployment_config("default", "nonexistent-config", update_request)


@pytest.mark.asyncio
async def test_update_deployment_config_workspace_not_found_integration(
    deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test updating a deployment config in non-existent workspace raises NotFoundError."""
    # Arrange
    update_request = UpdateModelDeploymentConfigRequest(
        description="Updated description",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-updated",
    )

    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_config_service.update_deployment_config("nonexistent", "config", update_request)


@pytest.mark.asyncio
async def test_delete_deployment_config_all_versions_integration(deployment_config_service, sample_create_request):
    """Test end-to-end deployment config deletion (all versions)."""
    # Arrange - create config
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    # Act
    deleted = await deployment_config_service.delete_deployment_config("default", sample_create_request.name)

    # Assert
    assert deleted is True

    # Verify it's actually deleted
    retrieved_config = await deployment_config_service.get_deployment_config("default", sample_create_request.name)
    assert retrieved_config is None


@pytest.mark.asyncio
async def test_delete_deployment_config_specific_version_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test deleting a specific version of a deployment config."""
    # Arrange - create multiple versions
    await deployment_config_service.create_deployment_config(sample_create_request, "default")

    update_request = UpdateModelDeploymentConfigRequest(
        description="Version 2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-456",
    )
    await deployment_config_service.update_deployment_config("default", sample_create_request.name, update_request)

    # Act - delete version 1
    deleted = await deployment_config_service.delete_deployment_config("default", sample_create_request.name, version=1)

    # Assert
    assert deleted is True

    # Verify version 1 is deleted
    version1 = await deployment_config_service.get_deployment_config("default", sample_create_request.name, version=1)
    assert version1 is None

    # Verify version 2 still exists
    version2 = await deployment_config_service.get_deployment_config("default", sample_create_request.name, version=2)
    assert version2 is not None
    assert version2.entity_version == 2


@pytest.mark.asyncio
async def test_delete_deployment_config_not_found_integration(deployment_config_service):
    """Test deleting a non-existent deployment config returns False."""
    # Act
    deleted = await deployment_config_service.delete_deployment_config("default", "nonexistent-config")

    # Assert
    assert deleted is False


@pytest.mark.asyncio
async def test_delete_deployment_config_workspace_not_found_integration(deployment_config_service):
    """Test deleting a deployment config from non-existent workspace raises NotFoundError."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_config_service.delete_deployment_config("nonexistent", "config")


@pytest.mark.asyncio
async def test_versioning_workflow_integration(
    deployment_config_service, sample_create_request, sample_model_spec, sample_executor_config
):
    """Test complete versioning workflow: create, update multiple times, list versions, delete."""
    # Create initial version
    v1 = await deployment_config_service.create_deployment_config(sample_create_request, "default")
    assert v1.entity_version == 1

    # Create version 2
    update_request1 = UpdateModelDeploymentConfigRequest(
        description="Version 2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-v2",
    )
    v2 = await deployment_config_service.update_deployment_config(
        "default", sample_create_request.name, update_request1
    )
    assert v2.entity_version == 2

    # Create version 3
    update_request2 = UpdateModelDeploymentConfigRequest(
        description="Version 3",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-v3",
    )
    v3 = await deployment_config_service.update_deployment_config(
        "default", sample_create_request.name, update_request2
    )
    assert v3.entity_version == 3

    # List all versions
    versions = await deployment_config_service.list_deployment_config_versions("default", sample_create_request.name)
    assert len(versions) == 3

    # Get latest (should be v3)
    latest = await deployment_config_service.get_deployment_config("default", sample_create_request.name)
    assert latest.entity_version == 3

    # Delete version 2
    deleted = await deployment_config_service.delete_deployment_config("default", sample_create_request.name, version=2)
    assert deleted is True

    # Verify version 2 is gone
    versions_after_delete = await deployment_config_service.list_deployment_config_versions(
        "default", sample_create_request.name
    )
    assert len(versions_after_delete) == 2
    version_numbers = [v.entity_version for v in versions_after_delete]
    assert 2 not in version_numbers
    assert 1 in version_numbers
    assert 3 in version_numbers


@pytest.mark.asyncio
async def test_complex_deployment_config_integration(deployment_config_service):
    """Test creating and retrieving a config with complex NIM deployment settings."""
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
        project="ml-team",
        description="Complex deployment configuration",
        engine="nim",
        model_spec=complex_model_spec,
        executor_config=complex_executor_config,
        model_entity_id="model-xyz-789",
    )

    # Act
    await deployment_config_service.create_deployment_config(create_request, "production")
    retrieved_config = await deployment_config_service.get_deployment_config("production", "complex-config")

    # Assert
    assert retrieved_config is not None
    assert retrieved_config.model_spec.model_type == ModelType.LLM
    assert retrieved_config.model_spec.lora_enabled is True
    assert retrieved_config.executor_config.gpu == 4
    assert retrieved_config.executor_config.disk_size == "100Gi"
    assert retrieved_config.executor_config.image_name == "nvcr.io/nvidia/nim/custom-llm"
    assert retrieved_config.executor_config.image_tag == "v1.2.3"
    assert retrieved_config.model_spec.model_namespace == "custom-org"
    assert retrieved_config.model_spec.model_name == "custom-model-70b"


@pytest.mark.asyncio
async def test_minimal_config_integration(deployment_config_service):
    """Test creating a config with minimal required fields."""
    # Arrange
    create_request = CreateModelDeploymentConfigRequest(
        name="minimal-config",
        engine="nim",
        model_spec=ModelDeploymentConfigModelSpec(),
        executor_config=ContainerExecutorConfig(gpu=1),
    )

    # Act
    await deployment_config_service.create_deployment_config(create_request, "default")
    retrieved_config = await deployment_config_service.get_deployment_config("default", "minimal-config")

    # Assert
    assert retrieved_config is not None
    assert retrieved_config.name == "minimal-config"
    assert retrieved_config.workspace == "default"
    assert retrieved_config.project is None
    assert retrieved_config.description is None
    assert retrieved_config.model_entity_id is None
    assert retrieved_config.executor_config.gpu == 1
