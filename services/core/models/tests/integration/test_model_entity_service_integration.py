# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Model Entity service with in-memory EntityClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.files.metadata import FilesetMetadata
from nemo_helix_plugin.files.storage_config import LocalStorageConfig
from nemo_helix_plugin.files.types import FilesetFileOutput, FilesetOutput, ListFilesetFilesResponse
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.entities.client import EntityClient
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.service.model_entity_service import ModelEntityService
from nhx.core.models.entities import Model
from nhx.core.models.schemas import (
    APIEndpointData,
    ContainerExecutorConfig,
    CreateModelAdapterRequest,
    CreateModelDeploymentConfigRequest,
    CreateModelEntityRequest,
    ModelDeploymentConfigModelSpec,
    ModelSpec,
    ModelType,
    PromptData,
    UpdateAdapterRequest,
    UpdateModelEntityRequest,
)
from nhx.testing import create_test_client


@pytest.fixture
def entity_client() -> EntityClient:
    """Create an EntityClient backed by in-memory storage for integration testing."""
    with create_test_client(client_type=EntityClient) as client:
        yield client


@pytest.fixture
def _mock_files_client():
    """Mock the FilesClient returned by client_from_platform in the permissions module."""
    fileset_output = FilesetOutput(
        id="fileset-id-123",
        name="test-fileset",
        workspace="default",
        description="Test fileset",
        storage=LocalStorageConfig(path="test-path"),
        purpose="generic",
        project="test-project",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        custom_fields={"key": "value"},
        metadata=FilesetMetadata(),
    )
    mock_fc = AsyncMock()
    mock_response = MagicMock()
    mock_response.data.return_value = fileset_output
    mock_fc.get_fileset.return_value = mock_response
    with patch("nhx.core.models.api.permissions.client_from_platform", return_value=mock_fc):
        yield mock_fc


@pytest.fixture
def model_entity_service(entity_client, _mock_files_client):
    """Create a ModelEntityService with MockEntityClient for integration testing."""
    async_sdk = AsyncMock(spec=AsyncNeMoHelix)
    async_sdk.files.list = AsyncMock(
        return_value=ListFilesetFilesResponse(
            data=[
                FilesetFileOutput(
                    file_ref="file-ref-123",
                    file_url="file-url-123",
                    path="path-123",
                    size=123,
                    cache_status="cached",
                )
            ]
        )
    )
    return ModelEntityService(entity_client, sdk=async_sdk)


@pytest.fixture
def adapter_entity_service(model_entity_service):
    from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService

    return AdapterEntityService(model_entity_service.entity_client, sdk=model_entity_service.sdk)


@pytest.fixture
def deployment_config_service(model_entity_service):
    return ModelDeploymentConfigService(model_entity_service.entity_client)


@pytest.fixture
def sample_create_request():
    """Create a sample CreateModelEntityRequest for testing."""
    return CreateModelEntityRequest(
        name="test-model",
        project="test-project",
        description="A test model entity",
        fileset="default/test-fileset",
        model_providers=["provider1", "provider2"],
    )


@pytest.mark.asyncio
async def test_create_model_entity_integration(model_entity_service, sample_create_request):
    """Test end-to-end model entity creation."""
    # Act
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Assert
    assert created_entity is not None
    assert created_entity.name == sample_create_request.name
    assert created_entity.workspace == "default"
    assert created_entity.project == sample_create_request.project
    assert created_entity.description == sample_create_request.description
    assert created_entity.model_providers == sample_create_request.model_providers
    assert created_entity.created_at is not None
    assert created_entity.updated_at is not None


@pytest.mark.asyncio
async def test_create_model_entity_duplicate_integration(model_entity_service, sample_create_request):
    """Test that creating duplicate model entities raises ValueError."""
    # Arrange - create first entity
    await model_entity_service.create_model_entity(sample_create_request, "default")

    # Act & Assert - try to create another with same workspace/name
    with pytest.raises(ValueError, match="already exists"):
        await model_entity_service.create_model_entity(sample_create_request, "default")


@pytest.mark.asyncio
async def test_get_model_entity_integration(model_entity_service, sample_create_request):
    """Test end-to-end model entity retrieval."""
    # Arrange
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Act
    retrieved_entity = await model_entity_service.get_model_entity("default", sample_create_request.name)

    # Assert
    assert retrieved_entity is not None
    assert retrieved_entity.name == created_entity.name
    assert retrieved_entity.workspace == created_entity.workspace
    assert retrieved_entity.project == created_entity.project
    assert retrieved_entity.description == created_entity.description
    assert retrieved_entity.model_providers == created_entity.model_providers


@pytest.mark.asyncio
async def test_get_model_entity_not_found_integration(model_entity_service):
    """Test retrieving a non-existent model entity returns None."""
    # Act
    retrieved_entity = await model_entity_service.get_model_entity("nonexistent", "model")

    # Assert
    assert retrieved_entity is None


@pytest.mark.asyncio
async def test_list_model_entities_empty_integration(model_entity_service):
    """Test listing model entities when none exist."""
    # Act
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None)
    )

    # Assert
    assert result.data == []
    assert result.pagination.total_results == 0
    assert result.pagination.page == 1
    assert result.pagination.page_size == 100


@pytest.mark.asyncio
async def test_list_model_entities_with_data_integration(model_entity_service, sample_create_request):
    """Test listing model entities with data."""
    # Arrange
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Act
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None)
    )

    # Assert
    assert len(result.data) == 1
    assert result.data[0].name == created_entity.name
    assert result.data[0].workspace == created_entity.workspace
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_model_entities_pagination_integration(model_entity_service):
    """Test pagination functionality."""
    # Arrange - create 5 entities
    entities = []
    for i in range(5):
        request = CreateModelEntityRequest(
            name=f"model-{i:02d}",
            description=f"Model {i}",
            fileset=f"default/model-{i:02d}-fileset",
            model_providers=[],
        )
        created_entity = await model_entity_service.create_model_entity(request, "default")
        entities.append(created_entity)

    # Test first page
    result_page1 = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), page=1, page_size=2
    )
    assert len(result_page1.data) == 2
    assert result_page1.pagination.page == 1
    assert result_page1.pagination.page_size == 2
    assert result_page1.pagination.total_results == 5

    # Test second page
    result_page2 = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), page=2, page_size=2
    )
    assert len(result_page2.data) == 2
    assert result_page2.pagination.page == 2

    # Test third page (partial)
    result_page3 = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), page=3, page_size=2
    )
    assert len(result_page3.data) == 1

    # Test large page size
    result_large = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), page=1, page_size=100
    )
    assert len(result_large.data) == 5
    assert result_large.pagination.total_pages == 1


@pytest.mark.asyncio
async def test_list_model_entities_sorting_integration(model_entity_service):
    """Test listing model entities with sorting."""
    # Arrange - create entities with different names
    entity1 = CreateModelEntityRequest(
        name="alpha-model",
        description="Alpha model",
        fileset="default/alpha-fileset",
        model_providers=[],
    )
    entity2 = CreateModelEntityRequest(
        name="beta-model",
        description="Beta model",
        fileset="default/beta-fileset",
        model_providers=[],
    )
    entity3 = CreateModelEntityRequest(
        name="gamma-model",
        description="Gamma model",
        fileset="default/gamma-fileset",
        model_providers=[],
    )

    await model_entity_service.create_model_entity(entity3, "default")  # Create out of order
    await model_entity_service.create_model_entity(entity1, "default")
    await model_entity_service.create_model_entity(entity2, "default")

    # Act - sort by name ascending
    result_asc = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), sort="name"
    )

    # Assert
    assert len(result_asc.data) == 3
    assert result_asc.data[0].name == "alpha-model"
    assert result_asc.data[1].name == "beta-model"
    assert result_asc.data[2].name == "gamma-model"

    # Act - sort by name descending
    result_desc = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), sort="-name"
    )

    # Assert
    assert len(result_desc.data) == 3
    assert result_desc.data[0].name == "gamma-model"
    assert result_desc.data[1].name == "beta-model"
    assert result_desc.data[2].name == "alpha-model"


@pytest.mark.asyncio
async def test_list_model_entities_sort_by_created_at_integration(model_entity_service):
    """Test sorting model entities by created_at (default sort field)."""
    # Arrange - create entities in order; µs-level timestamps ensure distinct created_at values
    names = []
    for i in range(3):
        request = CreateModelEntityRequest(
            name=f"model-{i:02d}",
            description=f"Model {i}",
            fileset=f"default/model-{i:02d}-fileset",
            model_providers=[],
        )
        await model_entity_service.create_model_entity(request, "default")
        names.append(f"model-{i:02d}")

    # Act - sort by created_at ascending (default)
    result_asc = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), sort="created_at"
    )

    # Assert - should be in creation order
    assert [e.name for e in result_asc.data] == names

    # Act - sort by created_at descending
    result_desc = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), sort="-created_at"
    )

    # Assert - should be in reverse creation order
    assert [e.name for e in result_desc.data] == list(reversed(names))


@pytest.mark.asyncio
async def test_list_model_entities_sort_by_updated_at_integration(model_entity_service):
    """Test sorting model entities by updated_at."""
    # Arrange - create three entities
    for i in range(3):
        request = CreateModelEntityRequest(
            name=f"model-{i:02d}",
            description=f"Model {i}",
            fileset=f"default/model-{i:02d}-fileset",
            model_providers=[],
        )
        await model_entity_service.create_model_entity(request, "default")

    # Update the first entity so it has the latest updated_at
    model = await model_entity_service.entity_client.get(Model, workspace="default", name="model-00")
    update_request = UpdateModelEntityRequest(description="Updated")
    await model_entity_service.update_model_entity(model, "default", "model-00", update_request)

    # Act - sort by updated_at descending (most recently updated first)
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), sort="-updated_at"
    )

    # Assert - model-00 should be first since it was most recently updated
    assert result.data[0].name == "model-00"


@pytest.mark.asyncio
async def test_created_model_entity_preserves_timestamps_integration(model_entity_service, sample_create_request):
    """Test that timestamps survive a create -> get round-trip (not replaced by NOW())."""
    # Act - create and immediately retrieve
    created = await model_entity_service.create_model_entity(sample_create_request, "default")
    retrieved = await model_entity_service.get_model_entity("default", sample_create_request.name)

    # Assert - the retrieved timestamps must match the created timestamps exactly
    assert retrieved is not None
    assert retrieved.created_at == created.created_at
    assert retrieved.updated_at == created.updated_at
    assert retrieved.id == created.id


@pytest.mark.asyncio
async def test_update_model_entity_integration(model_entity_service, sample_create_request):
    """Test end-to-end model entity update."""
    # Arrange
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")
    model: Model = await model_entity_service.entity_client.get(
        Model,
        workspace="default",
        name=sample_create_request.name,
    )
    # Act
    update_request = UpdateModelEntityRequest(
        description="Updated description",
        model_providers=["provider1", "provider2", "provider3"],
    )
    updated_entity = await model_entity_service.update_model_entity(model, model.workspace, model.name, update_request)

    # Assert
    assert updated_entity is not None
    assert updated_entity.description == "Updated description"
    assert updated_entity.model_providers == ["provider1", "provider2", "provider3"]
    assert updated_entity.name == created_entity.name  # Should not change
    assert updated_entity.workspace == created_entity.workspace  # Should not change
    assert updated_entity.updated_at is not None


@pytest.mark.asyncio
async def test_delete_model_entity_integration(model_entity_service, sample_create_request):
    """Test end-to-end model entity deletion."""
    # Arrange
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Act
    deleted = await model_entity_service.delete_model_entity(created_entity.workspace, created_entity.name)

    # Assert
    assert deleted is True

    # Verify entity is actually deleted
    retrieved_entity = await model_entity_service.get_model_entity(created_entity.workspace, created_entity.name)
    assert retrieved_entity is None


@pytest.mark.asyncio
async def test_delete_model_entity_not_found_integration(model_entity_service):
    """Test deleting a non-existent model entity returns False."""
    # Act
    deleted = await model_entity_service.delete_model_entity("nonexistent", "model")

    # Assert
    assert deleted is False


@pytest.mark.asyncio
async def test_model_entity_exists_integration(model_entity_service, sample_create_request):
    """Test checking if a model entity exists."""
    # Arrange
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Act & Assert - entity exists
    exists = await model_entity_service.model_entity_exists(created_entity.workspace, created_entity.name)
    assert exists is True

    # Act & Assert - entity doesn't exist
    exists = await model_entity_service.model_entity_exists("nonexistent", "model")
    assert exists is False


@pytest.mark.asyncio
async def test_model_entity_complex_fields_integration(model_entity_service):
    """Test model entity with complex fields."""
    # Arrange
    complex_request = CreateModelEntityRequest(
        name="complex-model",
        description="Model with complex fields",
        spec=ModelSpec(
            context_size=65536,  # 64K context window
            num_virtual_tokens=100,
            is_chat=False,
            checkpoint_model_name="meta-llama/Llama-2-70b",
            family="llama",
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,
            ffn_hidden_size=28672,
            vocab_size=32000,
            tied_embeddings=True,
            gated_mlp=True,
            base_num_parameters=70000000000,
            precision="fp16",
        ),
        fileset="https://huggingface.co/meta-llama/Llama-2-70b",
        api_endpoint=APIEndpointData(
            url="https://api.openai.com/v1/chat/completions",
            model_id="gpt-4",
            api_key="sk-openai-key",
            format="openai",
        ),
        prompt=PromptData(
            system_prompt="You are a medical assistant.",
            icl_few_shot_examples="Example 1\nExample 2",
        ),
        custom_fields={"domain": "medical", "accuracy": 0.95},
        ownership={
            "created_by": "medical-team",
            "updated_by": "medical-team",
            "access_policies": {"read": "medical-staff", "write": "doctors"},
        },
        model_providers=["openai", "azure-openai"],
    )

    # Act
    created_entity = await model_entity_service.create_model_entity(complex_request, "default")

    # Assert
    assert created_entity is not None
    assert created_entity.name == "complex-model"
    assert created_entity.spec is not None
    assert created_entity.spec.base_num_parameters == 70000000000
    assert created_entity.fileset is not None
    assert created_entity.fileset == "https://huggingface.co/meta-llama/Llama-2-70b"
    assert created_entity.api_endpoint is not None
    assert str(created_entity.api_endpoint.url) == "https://api.openai.com/v1/chat/completions"
    assert created_entity.prompt is not None
    assert created_entity.prompt.system_prompt == "You are a medical assistant."
    assert created_entity.custom_fields == {"domain": "medical", "accuracy": 0.95}
    assert created_entity.model_providers == ["openai", "azure-openai"]


# =============================================================================
# Adapter endpoints
# =============================================================================


@pytest.mark.asyncio
async def test_create_model_adapter_integration(model_entity_service, adapter_entity_service, sample_create_request):
    """Test creating an adapter on a model entity."""
    # Arrange: create a model first
    created_entity = await model_entity_service.create_model_entity(sample_create_request, "default")
    assert created_entity is not None

    adapter_request = CreateModelAdapterRequest(
        name="lora-adapter",
        fileset="default/my-adapter-fileset",
        finetuning_type="lora",
        enabled=True,
    )

    # Act
    updated_entity = await adapter_entity_service.create_adapter(
        "default", adapter_request, base_model=created_entity.name
    )

    # Assert
    assert updated_entity is not None

    assert updated_entity.name == "lora-adapter"
    assert updated_entity.fileset == "default/my-adapter-fileset"
    assert updated_entity.finetuning_type == "lora"
    assert updated_entity.enabled is True
    assert updated_entity.created_at is not None
    assert updated_entity.updated_at is not None


@pytest.mark.asyncio
async def test_create_model_adapter_duplicate_name_integration(
    model_entity_service, adapter_entity_service, sample_create_request
):
    """Test that creating an adapter with duplicate name raises."""
    # Arrange: create model and first adapter
    await model_entity_service.create_model_entity(sample_create_request, "default")
    adapter_request = CreateModelAdapterRequest(
        name="my-adapter",
        fileset="default/fileset",
        finetuning_type="lora",
    )
    await adapter_entity_service.create_adapter("default", adapter_request, base_model="test-model")

    # Act & Assert: same name raises
    with pytest.raises(ValueError, match="already exists"):
        await adapter_entity_service.create_adapter("default", adapter_request, base_model="test-model")


@pytest.mark.asyncio
async def test_update_model_adapter_integration(model_entity_service, adapter_entity_service, sample_create_request):
    """Test updating an adapter's description and enabled flag."""
    # Arrange: create model and adapter
    await model_entity_service.create_model_entity(sample_create_request, "default")
    await adapter_entity_service.create_adapter(
        "default",
        CreateModelAdapterRequest(
            name="my-adapter",
            fileset="default/fileset",
            finetuning_type="lora",
            enabled=False,
        ),
        base_model=sample_create_request.name,
    )
    entity = await model_entity_service.get_model_entity("default", sample_create_request.name)
    assert entity is not None
    assert entity.adapters is None or len(entity.adapters) == 1

    update_request = UpdateAdapterRequest(
        description="Updated adapter description",
        enabled=True,
        fileset="default/new-fileset",
    )

    # Act
    updated_entity = await adapter_entity_service.update_adapter(
        "default", sample_create_request.name, "my-adapter", update_request
    )

    # Assert
    assert updated_entity is not None
    assert updated_entity.name == "my-adapter"
    assert updated_entity.description == "Updated adapter description"
    assert updated_entity.enabled is True
    assert updated_entity.fileset == "default/new-fileset"


@pytest.mark.asyncio
async def test_delete_model_adapter_integration(model_entity_service, adapter_entity_service, sample_create_request):
    """Test deleting an adapter from a model entity."""
    # Arrange: create model and adapter
    await model_entity_service.create_model_entity(sample_create_request, "default")
    await adapter_entity_service.create_adapter(
        "default",
        CreateModelAdapterRequest(
            name="to-delete",
            fileset="default/fileset",
            finetuning_type="lora",
        ),
        base_model=sample_create_request.name,
    )

    entity = await model_entity_service.get_model_entity("default", sample_create_request.name)
    assert entity is not None
    assert entity.adapters is None or len(entity.adapters) == 1

    # Act
    error_code = await adapter_entity_service.delete_adapter("default", sample_create_request.name, "to-delete")

    # Assert
    assert error_code == 0
    entity = await model_entity_service.get_model_entity("default", sample_create_request.name)
    assert entity is not None
    assert entity.adapters is None or len(entity.adapters) == 0


@pytest.mark.asyncio
async def test_create_model_adapter_model_not_found_integration(adapter_entity_service):
    """Test create adapter when model does not exist returns None."""
    adapter_request = CreateModelAdapterRequest(
        name="lora-adapter",
        fileset="default/fileset",
        finetuning_type="lora",
    )
    result = await adapter_entity_service.create_adapter("default", adapter_request, base_model="nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_update_model_adapter_not_found_integration(
    model_entity_service, adapter_entity_service, sample_create_request
):
    """Test update adapter when adapter name does not exist returns None."""
    await model_entity_service.create_model_entity(sample_create_request, "default")
    update_request = UpdateAdapterRequest(description="New description")
    result = await adapter_entity_service.update_adapter("default", "test-model", "nonexistent-adapter", update_request)
    assert result == -2


@pytest.mark.asyncio
async def test_delete_model_adapter_not_found_integration(
    model_entity_service, adapter_entity_service, sample_create_request
):
    """Test delete adapter when adapter does not exist returns False."""
    await model_entity_service.create_model_entity(sample_create_request, "default")
    result = await adapter_entity_service.delete_adapter("default", "test-model", "nonexistent")
    assert result == -2


# ---------------------------------------------------------------------------
# lora_enabled cross-entity filter — combined with other filters
#
# Regression for the bug where the lora condition was passed as `search` while
# other user filters went through `filter_operation`; the EntityClient picked
# one and silently dropped the other. Both clauses must now reach the entity
# store as a single AND-merged filter operation.
# ---------------------------------------------------------------------------


def _lora_model_spec(lora_enabled: bool) -> ModelDeploymentConfigModelSpec:
    return ModelDeploymentConfigModelSpec(
        model_type=ModelType.LLM,
        lora_enabled=lora_enabled,
        model_namespace="nvidia",
        model_name="llama-3-8b",
    )


def _lora_executor_config() -> ContainerExecutorConfig:
    return ContainerExecutorConfig(
        gpu=1,
        disk_size="50Gi",
        image_name="nvcr.io/nvidia/nim/llm",
        image_tag="latest",
    )


async def _create_model_with_deployment_config(
    model_entity_service: ModelEntityService,
    deployment_config_service: ModelDeploymentConfigService,
    workspace: str,
    model_name: str,
    lora_enabled: bool,
) -> None:
    await model_entity_service.create_model_entity(
        CreateModelEntityRequest(
            name=model_name,
            description=f"Model {model_name}",
            fileset=f"{workspace}/{model_name}-fileset",
            model_providers=[],
        ),
        workspace,
    )
    await deployment_config_service.create_deployment_config(
        CreateModelDeploymentConfigRequest(
            name=f"{model_name}-config",
            description=f"Deployment config for {model_name}",
            engine="nim",
            model_spec=_lora_model_spec(lora_enabled),
            executor_config=_lora_executor_config(),
            model_entity_id=f"{workspace}/{model_name}",
        ),
        workspace,
    )


@pytest.mark.asyncio
async def test_list_model_entities_lora_combined_with_name_filter_integration(
    model_entity_service, deployment_config_service
):
    """lora_enabled=true combined with name=lora-a returns only lora-a (the matching lora model)."""
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "lora-a", lora_enabled=True
    )
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "lora-b", lora_enabled=True
    )
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "plain-c", lora_enabled=False
    )

    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=True),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="lora-a"),
            ],
        )
    )

    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    names = sorted(m.name for m in result.data)
    assert names == ["lora-a"]


@pytest.mark.asyncio
async def test_list_model_entities_lora_true_excludes_non_lora_with_name_match_integration(
    model_entity_service, deployment_config_service
):
    """lora_enabled=true combined with name=plain-c returns nothing.

    Pre-fix this would return ``plain-c`` because the lora condition was
    silently dropped; only the name filter would reach the entity store.
    """
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "lora-a", lora_enabled=True
    )
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "plain-c", lora_enabled=False
    )

    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=True),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="plain-c"),
            ],
        )
    )

    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    assert result.data == []


@pytest.mark.asyncio
async def test_list_model_entities_lora_false_combined_with_name_filter_integration(
    model_entity_service, deployment_config_service
):
    """lora_enabled=false AND name=plain-c returns plain-c (excludes lora models)."""
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "lora-a", lora_enabled=True
    )
    await _create_model_with_deployment_config(
        model_entity_service, deployment_config_service, "default", "plain-c", lora_enabled=False
    )

    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=False),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="plain-c"),
            ],
        )
    )

    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    names = sorted(m.name for m in result.data)
    assert names == ["plain-c"]
