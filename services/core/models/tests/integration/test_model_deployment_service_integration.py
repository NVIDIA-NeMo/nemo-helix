# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ModelDeployment service with in-memory EntityClient."""

import pytest
import pytest_asyncio
from nemo_helix_plugin.client.errors import NotFoundError
from nhx.common.api.common import Page
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.service.model_deployment_service import ModelDeploymentService
from nhx.core.models.schemas import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    ModelDeploymentConfigModelSpec,
    ModelDeploymentStatus,
    ModelType,
    UpdateModelDeploymentRequest,
    UpdateModelDeploymentStatusRequest,
)
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client


@pytest.fixture
def client_context(secrets_service_config) -> ClientContext:
    """Create a ClientContext backed by in-memory storage for integration testing."""
    workspaces = ["default", "workspace1", "workspace2", "production"]
    with create_test_client(
        SecretsService,
        client_type=ClientContext,
        workspaces=workspaces,
        service_configs={SecretsService: secrets_service_config},
    ) as ctx:
        yield ctx


@pytest.fixture
def deployment_config_service(client_context):
    """Create a ModelDeploymentConfigService with test EntityClient."""
    return ModelDeploymentConfigService(client_context.entity_client)


@pytest.fixture
def deployment_service(client_context):
    """Create a ModelDeploymentService with test EntityClient."""
    return ModelDeploymentService(client_context.entity_client)


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


@pytest_asyncio.fixture
async def sample_deployment_config(deployment_config_service, sample_model_spec, sample_executor_config):
    """Create a sample deployment config for testing."""
    create_request = CreateModelDeploymentConfigRequest(
        name="test-config",
        project="test-project",
        description="A test deployment configuration",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        model_entity_id="model-entity-123",
    )
    return await deployment_config_service.create_deployment_config(create_request, "default")


@pytest.fixture
def sample_create_request():
    """Create a sample CreateModelDeploymentRequest for testing."""
    return CreateModelDeploymentRequest(
        name="test-deployment",
        project="test-project",
        config="test-config",
        config_version=1,
    )


@pytest.mark.asyncio
async def test_create_deployment_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request
):
    """Test end-to-end deployment creation."""
    # Act
    created_deployment = await deployment_service.create_deployment(sample_create_request, "default")

    # Assert
    assert created_deployment is not None
    assert created_deployment.name == sample_create_request.name
    assert created_deployment.workspace == "default"
    assert created_deployment.project == sample_create_request.project
    assert created_deployment.entity_version == 1
    assert created_deployment.config == sample_create_request.config
    assert created_deployment.config_version == 1
    assert created_deployment.status == ModelDeploymentStatus.CREATED
    assert created_deployment.status_message == "Deployment created"
    assert created_deployment.created_at is not None
    assert created_deployment.updated_at is not None


@pytest.mark.asyncio
async def test_create_deployment_duplicate_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request
):
    """Test that creating duplicate deployment raises ValueError."""
    # Arrange
    await deployment_service.create_deployment(sample_create_request, "default")

    # Act & Assert - try to create another with same workspace/name
    with pytest.raises(ValueError, match="already exists"):
        await deployment_service.create_deployment(sample_create_request, "default")


@pytest.mark.asyncio
async def test_create_deployment_config_not_found_integration(deployment_service):
    """Test creating deployment with non-existent config raises ValueError."""
    # Arrange
    request = CreateModelDeploymentRequest(
        name="test-deployment",
        config="nonexistent-config",
        config_version=1,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="does not exist"):
        await deployment_service.create_deployment(request, "default")


@pytest.mark.asyncio
async def test_get_deployment_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request
):
    """Test end-to-end deployment retrieval."""
    # Arrange
    created_deployment = await deployment_service.create_deployment(sample_create_request, "default")

    # Act
    retrieved_deployment = await deployment_service.get_deployment("default", sample_create_request.name)

    # Assert
    assert retrieved_deployment is not None
    assert retrieved_deployment.name == created_deployment.name
    assert retrieved_deployment.workspace == created_deployment.workspace
    assert retrieved_deployment.entity_version == created_deployment.entity_version


@pytest.mark.asyncio
async def test_get_deployment_not_found_integration(deployment_service):
    """Test retrieving a non-existent deployment returns None."""
    # Act
    retrieved_deployment = await deployment_service.get_deployment("default", "nonexistent-deployment")

    # Assert
    assert retrieved_deployment is None


@pytest.mark.asyncio
async def test_get_deployment_workspace_not_found_integration(deployment_service):
    """Test retrieving a deployment from non-existent workspace raises NotFoundError."""
    # Act & Assert
    with pytest.raises(NotFoundError):
        await deployment_service.get_deployment("nonexistent", "deployment")


@pytest.mark.asyncio
async def test_list_deployments_empty_integration(deployment_service):
    """Test listing deployments when none exist."""
    # Act
    result = await deployment_service.list_deployments(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert result.data == []
    assert result.pagination.total_results == 0


@pytest.mark.asyncio
async def test_list_deployments_with_data_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request
):
    """Test listing deployments with data."""
    # Arrange
    created_deployment = await deployment_service.create_deployment(sample_create_request, "default")

    # Act
    result = await deployment_service.list_deployments(workspace="default")

    # Assert
    assert isinstance(result, Page)
    assert len(result.data) == 1
    assert result.data[0].name == created_deployment.name
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_versioning_workflow_integration(
    client_context, deployment_service, deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test the complete versioning workflow."""
    # Arrange - Create deployment config
    config_request = CreateModelDeploymentConfigRequest(
        name="versioning-config",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )
    await deployment_config_service.create_deployment_config(config_request, "default")

    # Create version 1
    create_request = CreateModelDeploymentRequest(
        name="versioned-deployment",
        config="versioning-config",
        config_version=1,
    )
    v1 = await deployment_service.create_deployment(create_request, "default")
    assert v1.entity_version == 1

    # Update to create version 2
    update_request = UpdateModelDeploymentRequest(
        config="versioning-config",
        config_version=1,
    )
    v2 = await deployment_service.update_deployment("default", "versioned-deployment", update_request)
    assert v2.entity_version == 2

    # Get latest (should be v2)
    latest = await deployment_service.get_deployment("default", "versioned-deployment")
    assert latest.entity_version == 2

    # Get specific version (v1)
    specific = await deployment_service.get_deployment("default", "versioned-deployment", version=1)
    assert specific.entity_version == 1

    # List all versions
    versions = await deployment_service.list_deployment_versions("default", "versioned-deployment")
    assert len(versions) == 2
    assert versions[0].entity_version == 2  # Latest first
    assert versions[1].entity_version == 1


@pytest.mark.asyncio
async def test_list_deployments_returns_latest_only_integration(
    deployment_service, deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test that list returns only the latest version of each deployment by default."""
    # Create config
    config_request = CreateModelDeploymentConfigRequest(
        name="test-config-latest",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )
    await deployment_config_service.create_deployment_config(config_request, "default")

    # Create deployment
    create_request = CreateModelDeploymentRequest(
        name="test-deployment-latest",
        config="test-config-latest",
        config_version=1,
    )
    await deployment_service.create_deployment(create_request, "default")

    # Update to create version 2
    update_request = UpdateModelDeploymentRequest(
        config="test-config-latest",
        config_version=1,
    )
    await deployment_service.update_deployment("default", "test-deployment-latest", update_request)

    # List should return only latest version
    result = await deployment_service.list_deployments(workspace="default")
    deployment = next((d for d in result.data if d.name == "test-deployment-latest"), None)

    assert deployment is not None
    assert deployment.entity_version == 2  # Latest version


@pytest.mark.asyncio
async def test_status_lifecycle_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request, create_secret
):
    """Test the deployment status lifecycle."""
    # Arrange
    create_secret(client_context, "hf-token-secret")

    # Create deployment (CREATED)
    deployment = await deployment_service.create_deployment(sample_create_request, "default")
    assert deployment.status == ModelDeploymentStatus.CREATED
    assert deployment.status_history == []

    # Update to PENDING
    pending_request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.PENDING,
        status_message="Provisioning resources",
    )
    deployment = await deployment_service.update_deployment_status(
        deployment.workspace, deployment.name, pending_request
    )
    assert deployment.status == ModelDeploymentStatus.PENDING
    assert len(deployment.status_history) >= 1
    assert deployment.status_history[0].status == ModelDeploymentStatus.CREATED

    # Update to READY
    ready_request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.READY,
        status_message="Deployment is ready",
        model_provider_id="provider-123",
    )
    deployment = await deployment_service.update_deployment_status(deployment.workspace, deployment.name, ready_request)
    assert deployment.status == ModelDeploymentStatus.READY
    assert deployment.model_provider_id == "provider-123"
    assert len(deployment.status_history) >= 2

    # Deduplication: same status and message should not add history entry
    deployment_before = await deployment_service.get_deployment(deployment.workspace, deployment.name, version=1)
    history_len_before = len(deployment_before.status_history)
    deployment_after = await deployment_service.update_deployment_status(
        deployment.workspace, deployment.name, ready_request
    )
    assert deployment_after.status == ModelDeploymentStatus.READY
    assert len(deployment_after.status_history) == history_len_before

    # Chronological order (oldest first); normalize for tz-naive vs tz-aware comparison
    from datetime import timezone as tz

    def _ts_key(dt):
        return dt.timestamp() if dt.tzinfo else dt.replace(tzinfo=tz.utc).timestamp()

    for i in range(len(deployment_after.status_history) - 1):
        assert _ts_key(deployment_after.status_history[i].timestamp) <= _ts_key(
            deployment_after.status_history[i + 1].timestamp
        )

    # Mark for deletion (DELETING)
    deployment = await deployment_service.delete_deployment(deployment.workspace, deployment.name, version=1)
    assert deployment.status == ModelDeploymentStatus.DELETING


@pytest.mark.asyncio
async def test_delete_deployment_marks_deleting_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request, create_secret
):
    """Test that deleting a deployment marks it as DELETING."""
    # Arrange
    create_secret(client_context, "hf-token-secret")
    created = await deployment_service.create_deployment(sample_create_request, "default")

    # Act
    result = await deployment_service.delete_deployment(created.workspace, created.name, version=1)

    # Assert
    assert result is not None
    assert result.status == ModelDeploymentStatus.DELETING

    # Verify status change persisted
    retrieved = await deployment_service.get_deployment(created.workspace, created.name, version=1)
    assert retrieved.status == ModelDeploymentStatus.DELETING


@pytest.mark.asyncio
async def test_delete_deployment_already_deleted_hard_deletes_integration(
    client_context, deployment_service, sample_deployment_config, sample_create_request, create_secret
):
    """Test that deleting a DELETED deployment performs hard delete."""
    # Arrange
    create_secret(client_context, "hf-token-secret")
    created = await deployment_service.create_deployment(sample_create_request, "default")

    # First mark as DELETING
    await deployment_service.delete_deployment(created.workspace, created.name, version=1)

    # Simulate controller marking as DELETED
    deleted_request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.DELETED,
        status_message="Deployment deleted",
    )
    await deployment_service.update_deployment_status(created.workspace, created.name, deleted_request, version=1)

    # Act - delete again (should hard delete)
    result = await deployment_service.delete_deployment(created.workspace, created.name, version=1)

    # Assert
    assert result is None  # Hard delete returns None

    # Verify it's actually gone
    retrieved = await deployment_service.get_deployment(created.workspace, created.name, version=1)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_deployment_not_found_integration(deployment_service):
    """Test deleting a non-existent deployment returns None."""
    # Act
    result = await deployment_service.delete_deployment("nonexistent", "deployment", version=1)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_filter_by_workspace_integration(
    deployment_service, deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test filtering deployments by workspace."""
    # Create configs in different workspaces
    config1 = CreateModelDeploymentConfigRequest(
        name="config1",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )
    config2 = CreateModelDeploymentConfigRequest(
        name="config2",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
    )

    await deployment_config_service.create_deployment_config(config1, "workspace1")
    await deployment_config_service.create_deployment_config(config2, "workspace2")

    # Create deployments in different workspaces
    deploy1 = CreateModelDeploymentRequest(
        name="deployment1",
        config="config1",
        config_version=1,
    )
    deploy2 = CreateModelDeploymentRequest(
        name="deployment2",
        config="config2",
        config_version=1,
    )

    await deployment_service.create_deployment(deploy1, "workspace1")
    await deployment_service.create_deployment(deploy2, "workspace2")

    # Filter by workspace1
    result = await deployment_service.list_deployments(workspace="workspace1")

    assert len(result.data) == 1
    assert result.data[0].workspace == "workspace1"
    assert result.data[0].name == "deployment1"


@pytest.mark.asyncio
async def test_filter_by_status_integration(
    deployment_service, sample_deployment_config, sample_model_spec, sample_executor_config
):
    """Test filtering deployments by status."""
    # Create a deployment
    request = CreateModelDeploymentRequest(
        name="status-test-deployment",
        config="test-config",
        config_version=1,
    )
    created = await deployment_service.create_deployment(request, "default")

    # Update status to READY
    update_request = UpdateModelDeploymentStatusRequest(
        status=ModelDeploymentStatus.READY,
        status_message="Ready",
    )
    await deployment_service.update_deployment_status(created.workspace, created.name, update_request, version=1)

    # Filter by READY status
    from nhx.common.api.filter import ComparisonOperation, FilterOperator

    status_filter = ComparisonOperation(operator=FilterOperator.EQ, field="data.status", value="READY")
    result = await deployment_service.list_deployments(workspace="default", filter_operation=status_filter)

    ready_deployments = [d for d in result.data if d.name == "status-test-deployment"]
    assert len(ready_deployments) == 1
    assert ready_deployments[0].status == ModelDeploymentStatus.READY


@pytest.mark.asyncio
async def test_create_deployment_without_hf_token_integration(deployment_service, sample_deployment_config):
    """Test creating a deployment with minimal request (no optional fields)."""
    # Arrange
    request = CreateModelDeploymentRequest(
        name="no-token-deployment",
        config="test-config",
        config_version=1,
    )

    # Act
    created = await deployment_service.create_deployment(request, "default")

    # Assert
    assert created is not None
    assert created.name == "no-token-deployment"


async def test_update_deployment_with_new_config_version_integration(
    deployment_service, deployment_config_service, sample_model_spec, sample_executor_config
):
    """Test updating a deployment to use a new config version."""
    # Create config v1
    config_request = CreateModelDeploymentConfigRequest(
        name="update-test-config",
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        description="Version 1",
    )
    await deployment_config_service.create_deployment_config(config_request, "default")

    # Create config v2
    from nhx.core.models.schemas import UpdateModelDeploymentConfigRequest as UpdateConfigRequest

    update_config_request = UpdateConfigRequest(
        engine="nim",
        model_spec=sample_model_spec,
        executor_config=sample_executor_config,
        description="Version 2",
    )
    await deployment_config_service.update_deployment_config("default", "update-test-config", update_config_request)

    # Create deployment using config v1
    deploy_request = CreateModelDeploymentRequest(
        name="update-test-deployment",
        config="update-test-config",
        config_version=1,
    )
    created = await deployment_service.create_deployment(deploy_request, "default")
    assert created.config_version == 1

    # Update deployment to use config v2
    update_request = UpdateModelDeploymentRequest(
        config="update-test-config",
        config_version=2,
    )
    updated = await deployment_service.update_deployment("default", "update-test-deployment", update_request)

    # Assert
    assert updated.entity_version == 2  # New deployment version
    assert updated.config_version == 2  # Using new config version
