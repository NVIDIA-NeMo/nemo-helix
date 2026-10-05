# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Model Entity service with mocked EntityClient."""

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix_plugin.files.metadata import FilesetMetadata
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig, LocalStorageConfig, NGCStorageConfig
from nemo_helix_plugin.files.types import FilesetFileOutput, FilesetOutput, ListFilesetFilesResponse
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.models.api.service.model_entity_service import (
    ModelEntityService,
    _model_to_model_entity,
    _repo_id_matches_trusted,
)
from nhx.core.models.api.v2.models import is_trusted_repo_id
from nhx.core.models.config import config
from nhx.core.models.entities import Adapter, Model, ModelDeploymentConfig
from nhx.core.models.schemas import (
    CreateModelAdapterRequest,
    CreateModelEntityRequest,
    LinearLayerSpec,
    MambaConfig,
    ModelEntity,
    ModelSpec,
    MoEConfig,
    SlidingWindowConfig,
    UpdateAdapterRequest,
    UpdateModelEntityRequest,
)


def create_model_entity(
    entity_id: str = "model-id-123",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    **kwargs: Any,
) -> Model:
    """Helper to create Model entity with proper private attributes."""

    created_at = created_at or datetime.now(timezone.utc)
    updated_at = updated_at or datetime.now(timezone.utc)
    entity = Model(**kwargs)
    entity._id = entity_id
    entity._created_at = created_at
    entity._updated_at = updated_at
    assert entity.id == entity_id
    assert entity.created_at == created_at
    assert entity.updated_at == updated_at
    return entity


def create_adapter_entity(
    parent: Model,
    name: str = "adapter-1",
    *,
    entity_id: str | None = None,
    workspace: str | None = None,
    fileset: str | None = None,
) -> Adapter:
    """Create an Adapter entity for testing.

    ``workspace`` defaults to the base model's workspace so existing tests retain
    their single-workspace semantics. Pass an explicit value to construct
    cross-workspace adapter rows (AALGO-129) — the entity store stores adapters
    parented by id, not by workspace, so ``Adapter.workspace`` may differ from
    ``parent.workspace``.
    """
    adapter_workspace = workspace if workspace is not None else parent.workspace
    adapter = Adapter(
        workspace=adapter_workspace,
        name=name,
        fileset=fileset or f"{adapter_workspace}/fileset",
        finetuning_type="lora",
        enabled=True,
        model=f"{parent.workspace}/{parent.name}",
    )
    adapter._id = entity_id or f"adapter-entity-id-{adapter_workspace}-{name}"
    adapter._created_at = datetime.now(timezone.utc)
    adapter._updated_at = datetime.now(timezone.utc)
    adapter._parent = parent.id
    assert adapter.parent == parent.id
    return adapter


@pytest.mark.parametrize(
    ("stored_fields", "expected_enabled"),
    [
        pytest.param({"auto_deploy": False}, False),
        pytest.param({"auto_deploy": True}, True),
        pytest.param({"enabled": False, "auto_deploy": True}, False),
    ],
)
def test_adapter_entity_migrates_auto_deploy_to_enabled(stored_fields: dict, expected_enabled: bool):
    """Backward compatibility: legacy ``auto_deploy`` in stored data maps to ``enabled``.

    The entities datastore may still contain rows written with the previous
    ``auto_deploy`` field name.  A model_validator on Adapter translates the
    old key so old rows are deserialized without data loss.  When both fields
    are present, ``enabled`` takes precedence.
    """
    from pydantic import TypeAdapter

    stored_data: dict[str, Any] = {
        "name": "test-adapter",
        "workspace": "default",
        "fileset": "default/my-fileset",
        "finetuning_type": "lora",
        "model": "default/mymod",
        **stored_fields,
    }
    adapter = TypeAdapter(Adapter).validate_python(stored_data)
    assert isinstance(adapter, Adapter)
    assert adapter.enabled is expected_enabled


def create_model_spec(*, with_default_linear_layer: bool = False, **overrides: Any) -> ModelSpec:
    """Create a ModelSpec with common llama defaults and optional overrides."""
    spec_data: dict[str, Any] = {
        "checkpoint_model_name": "meta-llama/Llama-3.2-1b-instruct",
        "family": "llama",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "ffn_hidden_size": 16384,
        "vocab_size": 32000,
        "tied_embeddings": True,
        "gated_mlp": True,
        "base_num_parameters": 1000,
        "precision": "fp16",
    }
    if with_default_linear_layer:
        spec_data["linear_layers"] = [
            LinearLayerSpec(name="model.layers.0.self_attn.q_proj", in_features=4096, out_features=4096),
        ]
    spec_data.update(overrides)
    return ModelSpec(**spec_data)


def _empty_adapters_page():
    """Page with real pagination for get_adapters(); avoids AsyncMock in total_results > 1000."""
    return Page(
        data=[],
        pagination=PaginationData(
            page=1,
            page_size=1000,
            total_pages=1,
            total_results=0,
            current_page_size=0,
        ),
    )


def _adapters_page(adapters: list[Adapter]):
    """Page wrapper for a non-empty adapter list; matches the shape returned by entity_client.list."""
    return Page(
        data=adapters,
        pagination=PaginationData(
            page=1,
            page_size=1000,
            total_pages=1,
            total_results=len(adapters),
            current_page_size=len(adapters),
        ),
    )


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """Create a mock EntityClient for testing."""
    mock = AsyncMock(spec=EntityClient)
    return mock


@pytest.fixture
def mock_files_client():
    """Mock the Files client the service uses for fileset checks."""
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
    files = AsyncMock()
    fileset_response = MagicMock()
    fileset_response.data.return_value = fileset_output
    files.get_fileset.return_value = fileset_response
    list_response = MagicMock()
    list_response.data.return_value = ListFilesetFilesResponse(
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
    files.list_files.return_value = list_response
    return files


@pytest.fixture
def model_entity_service(mock_entity_client, mock_files_client):
    """Create a ModelEntityService with mocked EntityClient and Files client."""
    return ModelEntityService(mock_entity_client, files=mock_files_client)


@pytest.fixture
def adapter_entity_service(model_entity_service):
    from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService

    return AdapterEntityService(model_entity_service.entity_client, files=model_entity_service.files)


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


@pytest.fixture
def sample_model():
    """Create a sample Model entity for testing."""
    return create_model_entity(
        name="test-model",
        workspace="default",
        project="test-project",
        description="A test model entity",
        model_providers=["provider1", "provider2"],
    )


@pytest.mark.asyncio
async def test_create_model_entity_success(
    model_entity_service, mock_entity_client, sample_create_request, sample_model
):
    """Test successful model entity creation."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = sample_model

    # Act
    result = await model_entity_service.create_model_entity(sample_create_request, "default")

    # Assert
    assert result is not None
    assert isinstance(result, ModelEntity)
    assert result.name == sample_create_request.name
    assert result.workspace == sample_model.workspace
    mock_entity_client.create.assert_called_once()
    call_args = mock_entity_client.create.call_args[0][0]
    assert isinstance(call_args, Model)
    assert call_args.name == sample_create_request.name
    assert call_args.workspace == "default"


@pytest.mark.asyncio
async def test_create_model_entity_conflict_error(
    model_entity_service, mock_entity_client, sample_create_request, sample_model
):
    """Test that EntityConflictError is converted to ValueError."""
    # Arrange
    mock_entity_client.get.return_value = sample_model  # Entity already exists

    # Act & Assert
    with pytest.raises(ValueError, match="already exists"):
        await model_entity_service.create_model_entity(sample_create_request, "default")

    mock_entity_client.get.assert_called_once()
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_get_model_entity_found(model_entity_service, mock_entity_client, sample_model):
    """Test retrieving an existing model entity."""
    # Arrange
    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())
    # Act
    result = await model_entity_service.get_model_entity("default", "test-model")

    # Assert
    assert result is not None
    assert isinstance(result, ModelEntity)
    assert result.name == sample_model.name
    assert result.workspace == sample_model.workspace
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="test-model")


@pytest.mark.asyncio
async def test_get_model_entity_not_found(model_entity_service, mock_entity_client):
    """Test retrieving a non-existent model entity."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")

    # Act
    result = await model_entity_service.get_model_entity("default", "nonexistent")

    # Assert
    assert result is None
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="nonexistent")


@pytest.mark.asyncio
async def test_list_model_entities_empty(model_entity_service, mock_entity_client):
    """Test listing model entities when none exist."""
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
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), verbose=True
    )

    # Assert
    assert result.data == []
    assert result.pagination.total_results == 0
    mock_entity_client.list.assert_called_once()


@pytest.mark.asyncio
async def test_list_model_entities_with_data(model_entity_service, mock_entity_client, sample_model):
    """Test listing model entities with data."""
    # Arrange
    mock_result_models = MagicMock()
    mock_result_models.data = [sample_model]
    mock_result_models.pagination = MagicMock()
    mock_result_models.pagination.page = 1
    mock_result_models.pagination.page_size = 100
    mock_result_models.pagination.total_pages = 1
    mock_result_models.pagination.total_results = 1

    mock_result_adapters = MagicMock()
    mock_result_adapters.data = [
        create_adapter_entity(parent=sample_model, name="test-adapter"),
        create_adapter_entity(parent=sample_model, name="test-adapter-2"),
    ]
    mock_result_adapters.pagination = MagicMock()
    mock_result_adapters.pagination.page = 1
    mock_result_adapters.pagination.page_size = 100
    mock_result_adapters.pagination.total_pages = 1
    mock_result_adapters.pagination.total_results = 2

    mock_entity_client.list.side_effect = [mock_result_models, mock_result_adapters]

    # Act
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), verbose=True
    )

    # Assert
    assert len(result.data) == 1
    assert result.data[0].name == sample_model.name
    assert len(result.data[0].adapters) == 2
    assert result.data[0].adapters[0].name == "test-adapter"
    assert result.data[0].adapters[1].name == "test-adapter-2"
    assert result.pagination.total_results == 1


@pytest.mark.asyncio
async def test_list_model_entities_with_filter(model_entity_service, mock_entity_client, sample_model):
    """Test listing model entities with workspace filter."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = [sample_model]
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 1
    mock_result.pagination.page_size = 100
    mock_result.pagination.total_pages = 1
    mock_result.pagination.total_results = 1

    mock_result_adapters = MagicMock()
    mock_result_adapters.data = [
        create_adapter_entity(parent=sample_model, name="test-adapter"),
    ]
    mock_result_adapters.pagination = MagicMock()
    mock_result_adapters.pagination.page = 1
    mock_result_adapters.pagination.page_size = 100
    mock_result_adapters.pagination.total_pages = 1
    mock_result_adapters.pagination.total_results = 1
    mock_entity_client.list.side_effect = [mock_result, mock_result_adapters]

    parsed_filter = ParsedFilter(operation=None)

    # Act
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=parsed_filter, verbose=True
    )

    # Assert
    assert len(result.data) == 1
    # Verify workspace was passed to EntityClient
    assert mock_entity_client.list.call_count == 2
    call_kwargs = mock_entity_client.list.call_args_list[0][1]
    assert call_kwargs["workspace"] == "default"


@pytest.mark.asyncio
async def test_list_model_entities_with_search(model_entity_service, mock_entity_client, sample_model):
    """Test listing model entities with search."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = [sample_model]
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 1
    mock_result.pagination.page_size = 100
    mock_result.pagination.total_pages = 1
    mock_result.pagination.total_results = 1

    mock_result_adapters = MagicMock()
    mock_result_adapters.data = [
        create_adapter_entity(parent=sample_model, name="test-adapter"),
    ]
    mock_result_adapters.pagination = MagicMock()
    mock_result_adapters.pagination.page = 1
    mock_result_adapters.pagination.page_size = 100
    mock_result_adapters.pagination.total_pages = 1
    mock_result_adapters.pagination.total_results = 1
    mock_entity_client.list.side_effect = [mock_result, mock_result_adapters]

    filter_op = ComparisonOperation(operator=FilterOperator.LIKE, field="name", value="test")
    parsed_filter = ParsedFilter(operation=filter_op)

    # Act
    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=parsed_filter, verbose=True
    )

    # Assert
    assert len(result.data) == 1
    assert mock_entity_client.list.call_count == 2
    call_kwargs = mock_entity_client.list.call_args_list[0][1]
    assert call_kwargs["filter_operation"] == filter_op
    call_kwargs = mock_entity_client.list.call_args_list[1][1]
    assert call_kwargs["filter_str"] == json.dumps({"parent": {"$in": [sample_model.id]}})


@pytest.mark.asyncio
async def test_list_model_entities_with_pagination(model_entity_service, mock_entity_client):
    """Test listing model entities with custom pagination."""
    # Arrange
    mock_result = MagicMock()
    mock_result.data = []
    mock_result.pagination = MagicMock()
    mock_result.pagination.page = 2
    mock_result.pagination.page_size = 50
    mock_result.pagination.total_pages = 0
    mock_result.pagination.total_results = 0
    mock_entity_client.list.return_value = mock_result

    # Act
    await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), page=2, page_size=50, verbose=True
    )

    # Assert
    mock_entity_client.list.assert_called_once()
    call_kwargs = mock_entity_client.list.call_args[1]
    assert call_kwargs["page"] == 2
    assert call_kwargs["page_size"] == 50


@pytest.mark.asyncio
async def test_update_model_entity_success(model_entity_service, mock_entity_client, sample_model):
    """Test successful model entity update."""
    # Arrange
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())
    updated_model = create_model_entity(
        entity_id=sample_model.id,
        name=sample_model.name,
        workspace=sample_model.workspace,
        project=sample_model.project,
        description="Updated description",
        model_providers=["provider1", "provider2", "provider3"],
        created_at=sample_model.created_at,
        updated_at=datetime.now(timezone.utc),
    )
    mock_entity_client.update.return_value = updated_model

    update_request = UpdateModelEntityRequest(
        description="Updated description",
        model_providers=["provider1", "provider2", "provider3"],
    )

    # Act
    result = await model_entity_service.update_model_entity(sample_model, "default", "test-model", update_request)

    # Assert
    assert result is not None
    assert result.description == "Updated description"
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_model_entity_can_clear_backend_format(model_entity_service, mock_entity_client, sample_model):
    """Explicit ``backend_format=null`` should clear the stored override."""
    sample_model.backend_format = "ANTHROPIC_MESSAGES"
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())
    mock_entity_client.update.side_effect = lambda model: model

    result = await model_entity_service.update_model_entity(
        sample_model,
        "default",
        "test-model",
        UpdateModelEntityRequest(backend_format=None),
    )

    assert result is not None
    assert sample_model.backend_format is None
    assert result.backend_format is None
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_model_entity_non_verbose_keeps_adapters(model_entity_service, mock_entity_client, sample_model):
    """Non-verbose update should still fetch adapters and strip linear_layers."""
    sample_model.spec = create_model_spec(with_default_linear_layer=True)
    mock_entity_client.update.return_value = sample_model
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())

    result = await model_entity_service.update_model_entity(
        sample_model,
        "default",
        "test-model",
        UpdateModelEntityRequest(description="Updated description"),
        verbose=False,
    )

    assert result is not None
    assert result.spec is not None
    assert result.spec.linear_layers is None
    mock_entity_client.update.assert_called_once()
    mock_entity_client.list.assert_called_once()


@pytest.mark.asyncio
async def test_delete_model_entity_success(model_entity_service, mock_entity_client, sample_model):
    """Test successful model entity deletion."""
    # Arrange
    mock_entity_client.delete.return_value = None

    # Act
    result = await model_entity_service.delete_model_entity("default", "test-model")

    # Assert
    assert result is True
    mock_entity_client.delete.assert_called_once_with(
        Model,
        "test-model",
        workspace="default",
    )


@pytest.mark.asyncio
async def test_delete_model_entity_not_found(model_entity_service, mock_entity_client):
    """Test deleting a non-existent model entity."""
    # Arrange
    mock_entity_client.delete.side_effect = EntityNotFoundError("Entity not found")

    # Act
    result = await model_entity_service.delete_model_entity("default", "nonexistent")

    # Assert
    assert result is False
    mock_entity_client.delete.assert_called_once_with(Model, "nonexistent", workspace="default")


@pytest.mark.asyncio
async def test_model_entity_exists_true(model_entity_service, mock_entity_client, sample_model):
    """Test checking if a model entity exists when it does."""
    # Arrange
    mock_entity_client.get.return_value = sample_model

    # Act
    result = await model_entity_service.model_entity_exists("default", "test-model")

    # Assert
    assert result is True
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="test-model")


@pytest.mark.asyncio
async def test_model_entity_exists_false(model_entity_service, mock_entity_client):
    """Test checking if a model entity exists when it doesn't."""
    # Arrange
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")

    # Act
    result = await model_entity_service.model_entity_exists("default", "nonexistent")

    # Assert
    assert result is False
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="nonexistent")


@pytest.mark.asyncio
async def test_create_model_entity_with_all_fields(model_entity_service, mock_entity_client):
    """Test creating a model entity with all optional fields."""
    # Arrange
    model_spec = create_model_spec(
        base_num_parameters=7000000000,
        context_size=4096,
        num_virtual_tokens=0,
        is_chat=True,
        moe_config=MoEConfig(
            num_experts=128,
            num_experts_per_tok=128,
            num_expert_layers=128,
            expert_ffn_size=16384,
            num_shared_experts=128,
        ),
        mamba_config=MambaConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            ffn_hidden_size=16384,
            vocab_size=32000,
            is_hybrid=True,
            num_mamba_layers=32,
        ),
        sliding_window_config=SlidingWindowConfig(
            window_size=1024,
        ),
        minimum_gpus_all_weights=1,
        minimum_gpus_lora=1,
        linear_layers=[
            LinearLayerSpec(
                name="linear-layer-1",
                in_features=4096,
                out_features=4096,
            )
        ],
    )

    create_request = CreateModelEntityRequest(
        name="complex-model",
        project="test-project",
        description="A complex model",
        spec=model_spec,
        fileset="default/complex-fileset",
        model_providers=["openai", "azure"],
        custom_fields={"domain": "medical"},
    )

    created_model = create_model_entity(
        entity_id="model-id-456",
        name="complex-model",
        workspace="default",
        project="test-project",
        description="A complex model",
        spec=model_spec,
        model_providers=["openai", "azure"],
        custom_fields={"domain": "medical"},
    )
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")
    mock_entity_client.create.return_value = created_model

    # Act
    result = await model_entity_service.create_model_entity(create_request, "default")

    # Assert
    assert result is not None
    assert result.name == "complex-model"
    assert result.spec is not None
    assert result.spec.base_num_parameters == 7000000000
    assert result.custom_fields == {"domain": "medical"}
    mock_entity_client.create.assert_called_once()


class TestModelToModelEntity:
    """Tests for _model_to_model_entity ensuring entity fields are preserved."""

    def test_preserves_id(self):
        """The schema id must come from the entity, not be auto-generated."""
        model = create_model_entity(
            entity_id="model-fixed-id-abc",
            name="test-model",
            workspace="default",
        )
        schema = _model_to_model_entity(model)
        assert schema.id == "model-fixed-id-abc"

    def test_preserves_created_at(self):
        """created_at must come from the entity, not datetime.now()."""
        fixed_time = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        model = create_model_entity(
            name="test-model",
            workspace="default",
            created_at=fixed_time,
        )
        schema = _model_to_model_entity(model)
        assert schema.created_at == fixed_time

    def test_preserves_updated_at(self):
        """updated_at must come from the entity, not datetime.now()."""
        fixed_time = datetime(2025, 6, 20, 14, 0, 0, tzinfo=timezone.utc)
        model = create_model_entity(
            name="test-model",
            workspace="default",
            updated_at=fixed_time,
        )
        schema = _model_to_model_entity(model)
        assert schema.updated_at == fixed_time

    def test_preserves_all_metadata_fields(self):
        """id, created_at, and updated_at must all be preserved from the entity."""
        created = datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc)
        updated = datetime(2025, 9, 15, 16, 45, 0, tzinfo=timezone.utc)
        model = create_model_entity(
            entity_id="model-specific-id",
            name="my-model",
            workspace="ws",
            created_at=created,
            updated_at=updated,
        )
        schema = _model_to_model_entity(model)
        assert schema.id == "model-specific-id"
        assert schema.created_at == created
        assert schema.updated_at == updated

    def test_non_verbose_filters_only_linear_layers(self):
        """Non-verbose mode keeps spec fields but strips linear_layers."""
        model_spec = create_model_spec(with_default_linear_layer=True)
        model = create_model_entity(name="my-model", workspace="ws", spec=model_spec)

        schema = _model_to_model_entity(model, verbose=False)

        assert schema.spec is not None
        assert schema.spec.family == "llama"
        assert schema.spec.linear_layers is None


@pytest.mark.asyncio
async def test_get_model_entity_non_verbose_keeps_adapters(model_entity_service, mock_entity_client):
    """Non-verbose get should still fetch adapters and strip linear_layers."""
    model_spec = create_model_spec(with_default_linear_layer=True)
    model = create_model_entity(name="test-model", workspace="default", spec=model_spec)

    mock_entity_client.get.return_value = model
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())

    result = await model_entity_service.get_model_entity("default", "test-model", verbose=False)

    assert result is not None
    assert result.spec is not None
    assert result.spec.linear_layers is None
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="test-model")
    mock_entity_client.list.assert_called_once()


# =============================================================================
# Cross-workspace adapter resolution (AALGO-129)
#
# Adapters can live in a workspace different from their parent model. The
# entity-store query that resolves adapters for a ModelEntity therefore must
# fan out across all workspaces — scoping it to the base model's workspace
# silently drops cross-workspace rows from the response, which in turn
# prevents the NIM sidecar from materializing the corresponding
# {adapter_ws}--{name} directories.
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_entity_includes_cross_workspace_adapters(model_entity_service, mock_entity_client):
    """Adapters in other workspaces parented to this model must appear in ``ModelEntity.adapters``."""
    from nhx.common.entities import ALL_WORKSPACES

    base_model = create_model_entity(entity_id="model-base-id", name="llama-3-2-1b-instruct", workspace="ws-base")
    same_ws = create_adapter_entity(parent=base_model, name="lora-english", workspace="ws-base")
    cross_ws_collision = create_adapter_entity(parent=base_model, name="lora-english", workspace="ws-a")
    cross_ws_distinct = create_adapter_entity(parent=base_model, name="lora-spanish", workspace="ws-a")

    mock_entity_client.get.return_value = base_model
    mock_entity_client.list = AsyncMock(return_value=_adapters_page([same_ws, cross_ws_collision, cross_ws_distinct]))

    result = await model_entity_service.get_model_entity("ws-base", "llama-3-2-1b-instruct")

    assert result is not None
    assert {(a.workspace, a.name) for a in result.adapters} == {
        ("ws-base", "lora-english"),
        ("ws-a", "lora-english"),
        ("ws-a", "lora-spanish"),
    }
    list_call = mock_entity_client.list.call_args
    assert list_call.kwargs["workspace"] == ALL_WORKSPACES, (
        "get_adapters must query with ALL_WORKSPACES so cross-workspace adapters "
        "parented to this base model are not silently dropped"
    )
    assert json.loads(list_call.kwargs["filter_str"]) == {"parent": {"$in": [base_model.id]}}


@pytest.mark.asyncio
async def test_list_model_entities_uses_all_workspaces_for_adapter_lookup(
    model_entity_service, mock_entity_client, sample_model
):
    """List path also resolves adapters cross-workspace so listed entities show all their adapters."""
    from nhx.common.entities import ALL_WORKSPACES

    cross_ws_adapter = create_adapter_entity(parent=sample_model, name="cross-ws-adapter", workspace="other-ws")

    models_page = MagicMock()
    models_page.data = [sample_model]
    models_page.pagination = MagicMock()
    models_page.pagination.page = 1
    models_page.pagination.page_size = 100
    models_page.pagination.total_pages = 1
    models_page.pagination.total_results = 1

    mock_entity_client.list.side_effect = [models_page, _adapters_page([cross_ws_adapter])]

    result = await model_entity_service.list_model_entities(
        workspace="default", parsed_filter=ParsedFilter(operation=None), verbose=True
    )

    assert len(result.data) == 1
    assert len(result.data[0].adapters) == 1
    assert result.data[0].adapters[0].workspace == "other-ws"

    # Two entity_client.list calls: (1) models scoped to "default", (2) adapters cross-workspace.
    assert mock_entity_client.list.call_count == 2
    assert mock_entity_client.list.call_args_list[0].kwargs["workspace"] == "default"
    assert mock_entity_client.list.call_args_list[1].kwargs["workspace"] == ALL_WORKSPACES


@pytest.mark.asyncio
async def test_update_model_entity_uses_all_workspaces_for_adapter_lookup(
    model_entity_service, mock_entity_client, sample_model
):
    """Update path resolves adapters cross-workspace so the response carries all of them."""
    from nhx.common.entities import ALL_WORKSPACES

    cross_ws_adapter = create_adapter_entity(parent=sample_model, name="cross-ws-adapter", workspace="other-ws")
    mock_entity_client.update.return_value = sample_model
    mock_entity_client.list = AsyncMock(return_value=_adapters_page([cross_ws_adapter]))

    result = await model_entity_service.update_model_entity(
        sample_model,
        "default",
        "test-model",
        UpdateModelEntityRequest(description="updated"),
    )

    assert result is not None
    assert len(result.adapters) == 1
    assert result.adapters[0].workspace == "other-ws"
    assert mock_entity_client.list.call_args.kwargs["workspace"] == ALL_WORKSPACES


@pytest.mark.asyncio
async def test_get_adapters_legacy_workspace_fallback_uses_caller_workspace(
    model_entity_service, mock_entity_client, sample_model
):
    """Adapter rows without their own ``workspace`` (pre-AALGO-117) fall back to the caller-supplied workspace.

    This pins that the ``workspace`` parameter to ``get_adapters`` is still used
    as the schema-level fallback for legacy rows even though it is no longer
    the entity-store filter. Without this fallback, legacy adapters would have
    an empty ``workspace`` field after the cross-workspace query change.

    Uses a ``MagicMock``-shaped adapter to bypass the entity store's required
    ``workspace`` field — it represents a row written before AALGO-117 added
    first-class adapter workspaces.
    """
    legacy_adapter = MagicMock()
    legacy_adapter.parent = sample_model.id
    legacy_adapter.name = "legacy-adapter"
    legacy_adapter.workspace = None
    legacy_adapter.description = None
    legacy_adapter.fileset = "default/fileset"
    legacy_adapter.finetuning_type = "lora"
    legacy_adapter.enabled = True
    legacy_adapter.lora_config = None
    legacy_adapter.model = None
    legacy_adapter.created_at = datetime.now(timezone.utc)
    legacy_adapter.updated_at = datetime.now(timezone.utc)

    mock_entity_client.list = AsyncMock(return_value=_adapters_page([legacy_adapter]))

    adapters_map = await model_entity_service.get_adapters(
        workspace="caller-ws-fallback",
        ids=[sample_model.id],
        model_name_map={sample_model.id: f"{sample_model.workspace}/{sample_model.name}"},
        schema=True,
    )

    rendered = adapters_map[sample_model.id]
    assert len(rendered) == 1
    assert rendered[0].workspace == "caller-ws-fallback"


# =============================================================================
# Adapter endpoints
# =============================================================================


@pytest.fixture
def create_adapter_request():
    """CreateModelAdapterRequest for adapter create tests; base model is always passed as base_model= to the service."""
    return CreateModelAdapterRequest(
        name="lora-adapter",
        fileset="default/adapter-fileset",
        finetuning_type="lora",
        enabled=True,
    )


@pytest.mark.asyncio
async def test_create_model_adapter_success(
    adapter_entity_service, mock_entity_client, sample_model, create_adapter_request
):
    """Test successful adapter creation."""
    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list.return_value = Page(
        data=[],
        pagination=PaginationData(
            page=1,
            page_size=100,
            total_pages=1,
            total_results=0,
            current_page_size=1,
        ),
    )
    mock_entity_client.create.return_value = create_adapter_entity(
        parent=sample_model,
        name=create_adapter_request.name,
    )

    result = await adapter_entity_service.create_adapter("default", create_adapter_request, base_model="test-model")

    assert result is not None
    assert result.name == "lora-adapter"
    assert result.fileset == "default/fileset"
    assert result.finetuning_type == "lora"
    assert result.enabled is True
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="test-model")
    mock_entity_client.list.assert_called_once()
    mock_entity_client.create.assert_called_once()


@pytest.mark.asyncio
async def test_create_model_adapter_model_not_found(adapter_entity_service, mock_entity_client, create_adapter_request):
    """Test create adapter when model does not exist returns None."""
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")

    result = await adapter_entity_service.create_adapter("default", create_adapter_request, base_model="nonexistent")

    assert result is None
    mock_entity_client.get.assert_called_once_with(Model, workspace="default", name="nonexistent")
    mock_entity_client.list.assert_not_called()
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_create_model_adapter_duplicate_name_raises(
    adapter_entity_service, mock_entity_client, create_adapter_request
):
    """Test create adapter with duplicate name raises ValueError."""
    model_with_adapter = create_model_entity(
        name="test-model",
        workspace="default",
    )
    mock_entity_client.get.return_value = model_with_adapter
    mock_entity_client.list.return_value = Page(
        data=[
            create_adapter_entity(parent=model_with_adapter, name="lora-adapter"),
        ],
        pagination=PaginationData(
            page=1,
            page_size=100,
            total_pages=1,
            total_results=0,
            current_page_size=1,
        ),
    )

    with pytest.raises(ValueError, match="already exists"):
        await adapter_entity_service.create_adapter("default", create_adapter_request, base_model="test-model")

    mock_entity_client.get.assert_called_once()
    mock_entity_client.list.assert_called_once()
    mock_entity_client.create.assert_not_called()


@pytest.mark.asyncio
async def test_update_model_adapter_success(adapter_entity_service, mock_entity_client, sample_model):
    """Test successful adapter update."""
    adapter = create_adapter_entity(parent=sample_model, name="lora-adapter")

    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list.return_value = Page(
        data=[adapter],
        pagination=PaginationData(
            page=1,
            page_size=100,
            total_pages=1,
            total_results=0,
            current_page_size=1,
        ),
    )
    updated_adapter = create_adapter_entity(parent=sample_model, name="lora-adapter")
    updated_adapter.description = "Updated description"
    updated_adapter.enabled = False
    mock_entity_client.update.return_value = updated_adapter

    update_request = UpdateAdapterRequest(description="Updated description", enabled=False)
    result = await adapter_entity_service.update_adapter("default", "test-model", "lora-adapter", update_request)

    assert result is not None

    assert result.description == "Updated description"
    assert result.enabled is False
    assert mock_entity_client.list.call_count == 1
    mock_entity_client.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_model_adapter_not_found(adapter_entity_service, mock_entity_client, sample_model):
    """Test update adapter when adapter name does not exist returns None."""
    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())

    update_request = UpdateAdapterRequest(description="New description")
    result = await adapter_entity_service.update_adapter("default", "test-model", "nonexistent-adapter", update_request)

    assert result == -2
    mock_entity_client.update.assert_not_called()
    # One list (get_adapters) then empty name match for the adapter row
    assert mock_entity_client.list.call_count == 1


@pytest.mark.asyncio
async def test_update_model_adapter_model_not_found(adapter_entity_service, mock_entity_client):
    """Test update adapter when model does not exist returns None."""
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")

    result = await adapter_entity_service.update_adapter(
        "default", "nonexistent", "adapter-name", UpdateAdapterRequest(description="x")
    )

    assert result == -1
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_model_adapter_success(
    adapter_entity_service, mock_entity_client, sample_model, create_adapter_request
):
    """Test successful adapter deletion."""
    adapter = create_adapter_entity(parent=sample_model, name="lora-adapter")
    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list.return_value = Page(
        data=[adapter],
        pagination=PaginationData(
            page=1,
            page_size=100,
            total_pages=1,
            total_results=0,
            current_page_size=1,
        ),
    )
    result = await adapter_entity_service.delete_adapter("default", "test-model", "lora-adapter")

    assert result == 0
    mock_entity_client.delete.assert_called_once_with(
        Adapter,
        adapter.name,
        workspace="default",
        parent=adapter.parent,
    )


@pytest.mark.asyncio
async def test_delete_model_adapter_not_found(adapter_entity_service, mock_entity_client, sample_model):
    """Test delete adapter when adapter name does not exist returns False."""
    mock_entity_client.get.return_value = sample_model
    mock_entity_client.list = AsyncMock(return_value=_empty_adapters_page())

    result = await adapter_entity_service.delete_adapter("default", "test-model", "nonexistent")

    assert result == -2
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_delete_model_adapter_model_not_found(adapter_entity_service, mock_entity_client):
    """Test delete adapter when model does not exist returns False."""
    mock_entity_client.get.side_effect = EntityNotFoundError("Entity not found")

    result = await adapter_entity_service.delete_adapter("default", "nonexistent", "adapter-name")

    assert result == -1
    mock_entity_client.update.assert_not_called()


@pytest.mark.asyncio
async def test_list_adapters_resolves_parent_models_with_canonical_all_workspaces(
    adapter_entity_service, mock_entity_client, sample_model
):
    """The parent-models lookup must use the canonical ``ALL_WORKSPACES = "-"`` constant.

    Earlier revisions of ``adapter_entity_service`` defined a local
    ``ALL_WORKSPACES = "*"`` which the entity store does not recognize as the
    cross-workspace wildcard. That made the parent-model lookup return empty,
    so adapters listed with cross-workspace parents rendered with their parent
    *id* in the ``model`` field instead of ``"{ws}/{name}"``. Pin the canonical
    ``"-"`` to prevent that regression.
    """
    from nhx.common.entities import ALL_WORKSPACES

    assert ALL_WORKSPACES == "-"  # canonical wildcard used by the entity store

    # Adapter lives in ws-a but its parent model lives in ws-base. The list
    # endpoint resolves parent model names so the response shows
    # ``model="ws-base/test-model"`` instead of ``model="model-id-..."``.
    cross_ws_adapter = create_adapter_entity(parent=sample_model, name="cross-ws-adapter", workspace="ws-a")

    adapters_page = _adapters_page([cross_ws_adapter])
    parent_models_page = _adapters_page([sample_model])  # reuse the helper shape — list of entities + pagination

    mock_entity_client.list.side_effect = [adapters_page, parent_models_page]

    result = await adapter_entity_service.list_adapters(
        adapter_workspace="ws-a", parsed_filter=ParsedFilter(operation=None)
    )

    assert len(result.data) == 1
    rendered = result.data[0]
    assert rendered.workspace == "ws-a"
    assert rendered.model == f"{sample_model.workspace}/{sample_model.name}"

    # First call: list adapters scoped to the requested workspace.
    assert mock_entity_client.list.call_args_list[0].kwargs["workspace"] == "ws-a"
    # Second call: resolve parent models cross-workspace using the canonical wildcard.
    assert mock_entity_client.list.call_args_list[1].kwargs["workspace"] == ALL_WORKSPACES


def _hf_fileset(repo_id: str) -> FilesetOutput:
    """Create a FilesetOutput with HuggingFace storage for is_trusted_repo_id tests."""
    return FilesetOutput(
        id="fs-1",
        name="test-fileset",
        workspace="default",
        description="",
        storage=HuggingfaceStorageConfig(repo_id=repo_id, type="huggingface"),
        purpose="generic",
        project="default",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        custom_fields={},
        metadata=FilesetMetadata(),
    )


def _ngc_fileset(org: str, team: str, target: str) -> FilesetOutput:
    """Create a FilesetOutput with NGC storage for is_trusted_repo_id tests (path: org/team/target)."""
    return FilesetOutput(
        id="fs-1",
        name="test-fileset",
        workspace="default",
        description="",
        storage=NGCStorageConfig(
            api_key_secret="dummy",
            org=org,
            team=team,
            target=target,
            type="ngc",
        ),
        purpose="generic",
        project="default",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        custom_fields={},
        metadata=FilesetMetadata(),
    )


def test_repo_id_matches_trusted_direct_string_match():
    """Exact repo_id in patterns returns True."""
    assert _repo_id_matches_trusted("nvidia/Llama-3.1-8B", ["nvidia/Llama-3.1-8B"]) is True


def test_repo_id_matches_trusted_regex_fullmatch():
    """Pattern as regex matching full repo_id returns True."""
    assert _repo_id_matches_trusted("nvidia/Llama-3.1-8B", [r"nvidia/.*"]) is True
    assert _repo_id_matches_trusted("meta-llama/Llama-3-8B", [r"meta-llama/.*"]) is True


def test_repo_id_matches_trusted_regex_no_match():
    """Repo_id not matching any pattern returns False."""
    assert _repo_id_matches_trusted("other/Model-7B", [r"nvidia/.*"]) is False
    assert _repo_id_matches_trusted("nvidia/Llama", [r"nvidia/.*-8B"]) is False


def test_repo_id_matches_trusted_no_match():
    """Repo_id not in patterns and no regex match returns False."""
    assert _repo_id_matches_trusted("unknown/repo", ["nvidia/foo", r"meta-llama/.*"]) is False


def test_repo_id_matches_trusted_empty_patterns():
    """Empty patterns list returns False."""
    assert _repo_id_matches_trusted("nvidia/any", []) is False


def test_repo_id_matches_trusted_invalid_regex_treated_as_literal_only():
    """Invalid regex does not match; only direct equality would match."""
    assert _repo_id_matches_trusted("nvidia/foo", ["[invalid"]) is False
    assert _repo_id_matches_trusted("[invalid]", ["[invalid]"]) is True


@pytest.mark.asyncio
async def test_is_trusted_repo_id_direct_match(model_entity_service):
    """When repo_id is in trust_remote_code.hf_allow_list (direct match), returns True."""
    # Arrange
    fileset = _hf_fileset("nvidia/trusted-model")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "hf_allow_list", ["nvidia/trusted-model"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_is_trusted_repo_id_regex_match(model_entity_service):
    """When repo_id fullmatches a regex in trust_remote_code.hf_allow_list, returns True."""
    # Arrange
    fileset = _hf_fileset("nvidia/Llama-3.1-70B")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "hf_allow_list", [r"nvidia/.*"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_is_trusted_repo_id_no_match(model_entity_service):
    """When repo_id does not match any pattern, returns False."""
    # Arrange
    fileset = _hf_fileset("unknown/repo")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "hf_allow_list", ["nvidia/only-this", r"meta-llama/.*"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is False


@pytest.mark.asyncio
async def test_is_trusted_repo_id_non_huggingface_storage(model_entity_service):
    """When fileset storage is not huggingface or ngc, returns False."""
    # Arrange
    fileset = FilesetOutput(
        id="fs-1",
        name="test-fileset",
        workspace="default",
        description="",
        storage=LocalStorageConfig(path="/some/path"),
        purpose="generic",
        project="default",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        custom_fields={},
        metadata=FilesetMetadata(),
    )
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "hf_allow_list", [r"nvidia/.*"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is False


@pytest.mark.asyncio
async def test_is_trusted_repo_id_ngc_direct_match(model_entity_service):
    """When org/team/target is in trust_remote_code.ngc_allow_list (direct match), returns True."""
    # Arrange: path is org/team/target -> nvidia/nemotron/nemotron-4-340b
    fileset = _ngc_fileset("nvidia", "nemotron", "nemotron-4-340b")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "ngc_allow_list", ["nvidia/nemotron/nemotron-4-340b"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_is_trusted_repo_id_ngc_regex_match(model_entity_service):
    """When org/team/target fullmatches a regex in trust_remote_code.ngc_allow_list, returns True."""
    # Arrange
    fileset = _ngc_fileset("nvidia", "team", "some-model")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(config.trust_remote_code, "ngc_allow_list", [r"nvidia/.*/some-model"]):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is True


@pytest.mark.asyncio
async def test_is_trusted_repo_id_ngc_no_match(model_entity_service):
    """When org/team/target does not match any ngc_allow_list pattern, returns False."""
    # Arrange
    fileset = _ngc_fileset("other-org", "team", "model")
    mock_get = AsyncMock(return_value=(fileset, []))
    # Act
    with patch("nhx.core.models.api.service.model_entity_service.get_fileset_and_files_list", mock_get):
        with patch.object(
            config.trust_remote_code, "ngc_allow_list", ["nvidia/only-this/only", r"nvidia/.*/ngc-model"]
        ):
            result = await is_trusted_repo_id(model_entity_service.files, "default", "default/fileset")
    # Assert
    assert result is False


# --- Tests for _resolve_lora_filter ---


def _make_deployment_config(
    base_name: str,
    entity_version: int = 1,
    lora_enabled: bool = False,
    model_entity_id: str | None = None,
    nim_deployment_present: bool = True,
) -> ModelDeploymentConfig:
    """Helper to create a ModelDeploymentConfig entity for testing."""
    from nhx.core.models.schemas import ContainerExecutorConfig, ModelDeploymentConfigModelSpec

    model_spec = ModelDeploymentConfigModelSpec(lora_enabled=lora_enabled) if nim_deployment_present else None
    executor_config = ContainerExecutorConfig(gpu=1) if nim_deployment_present else None
    cfg = ModelDeploymentConfig(
        workspace="default",
        name=f"{base_name}-v{entity_version}",
        base_name=base_name,
        entity_version=entity_version,
        engine="nim" if nim_deployment_present else None,
        model_spec=model_spec,
        executor_config=executor_config,
        model_entity_id=model_entity_id,
    )
    cfg._id = f"cfg-{base_name}-v{entity_version}"
    cfg._created_at = datetime.now(timezone.utc)
    cfg._updated_at = datetime.now(timezone.utc)
    return cfg


def _configs_list_response(configs):
    """Wrap configs in a ListResponse-like MagicMock."""
    mock = MagicMock()
    mock.data = configs
    mock.pagination = MagicMock()
    mock.pagination.page = 1
    mock.pagination.page_size = 1000
    mock.pagination.total_pages = 1
    mock.pagination.total_results = len(configs)
    return mock


@pytest.mark.asyncio
async def test_resolve_lora_filter_returns_lora_model_ids(model_entity_service, mock_entity_client):
    """_resolve_lora_filter returns (workspace, name) pairs when lora-enabled configs exist."""
    configs = [
        _make_deployment_config("cfg-a", lora_enabled=True, model_entity_id="default/model-a"),
        _make_deployment_config("cfg-b", lora_enabled=False, model_entity_id="default/model-b"),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == [("default", "model-a")]


@pytest.mark.asyncio
async def test_resolve_lora_filter_empty_when_no_lora(model_entity_service, mock_entity_client):
    """_resolve_lora_filter returns empty list when no configs have lora enabled."""
    configs = [
        _make_deployment_config("cfg-a", lora_enabled=False, model_entity_id="default/model-a"),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == []


@pytest.mark.asyncio
async def test_resolve_lora_filter_respects_versioning(model_entity_service, mock_entity_client):
    """_resolve_lora_filter uses latest version — older lora=true is overridden by newer lora=false."""
    configs = [
        _make_deployment_config("cfg-a", entity_version=1, lora_enabled=True, model_entity_id="default/model-a"),
        _make_deployment_config("cfg-a", entity_version=2, lora_enabled=False, model_entity_id="default/model-a"),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == []


@pytest.mark.asyncio
async def test_resolve_lora_filter_latest_version_enables_lora(model_entity_service, mock_entity_client):
    """_resolve_lora_filter includes model when latest version has lora enabled."""
    configs = [
        _make_deployment_config("cfg-a", entity_version=1, lora_enabled=False, model_entity_id="default/model-a"),
        _make_deployment_config("cfg-a", entity_version=2, lora_enabled=True, model_entity_id="default/model-a"),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == [("default", "model-a")]


@pytest.mark.asyncio
async def test_resolve_lora_filter_skips_none_nim_deployment(model_entity_service, mock_entity_client):
    """_resolve_lora_filter skips configs where model_spec is None."""
    configs = [
        _make_deployment_config(
            "cfg-a", lora_enabled=False, model_entity_id="default/model-a", nim_deployment_present=False
        ),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == []


@pytest.mark.asyncio
async def test_resolve_lora_filter_skips_none_model_entity_id(model_entity_service, mock_entity_client):
    """_resolve_lora_filter skips configs where model_entity_id is None."""
    configs = [
        _make_deployment_config("cfg-a", lora_enabled=True, model_entity_id=None),
    ]
    mock_entity_client.list.return_value = _configs_list_response(configs)

    result = await model_entity_service._resolve_lora_filter("default")

    assert result == []


@pytest.mark.asyncio
async def test_list_model_entities_lora_true_short_circuits_when_empty(model_entity_service, mock_entity_client):
    """list_model_entities with lora_enabled=true short-circuits with empty result when no configs match."""
    # _resolve_lora_filter will return empty list
    mock_entity_client.list.return_value = _configs_list_response([])

    lora_filter = ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=True)
    parsed_filter = ParsedFilter(operation=lora_filter)
    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    assert result.data == []
    assert result.pagination.total_results == 0
    # entity_client.list called once for deployment configs, NOT for models
    mock_entity_client.list.assert_called_once()


@pytest.mark.asyncio
async def test_list_model_entities_lora_true_adds_in_constraint(model_entity_service, mock_entity_client, sample_model):
    """list_model_entities with lora_enabled=true passes $in constraint to search query."""
    lora_configs = [
        _make_deployment_config("cfg-a", lora_enabled=True, model_entity_id="default/test-model"),
    ]
    configs_response = _configs_list_response(lora_configs)

    mock_models_result = MagicMock()
    mock_models_result.data = [sample_model]
    mock_models_result.pagination = MagicMock()
    mock_models_result.pagination.page = 1
    mock_models_result.pagination.page_size = 100
    mock_models_result.pagination.total_pages = 1
    mock_models_result.pagination.total_results = 1

    mock_adapters_result = _empty_adapters_page()

    # First call: deployment configs, second: models, third: adapters
    mock_entity_client.list.side_effect = [configs_response, mock_models_result, mock_adapters_result]

    lora_filter = ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=True)
    parsed_filter = ParsedFilter(operation=lora_filter)
    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    assert len(result.data) == 1
    # Verify the models list call merged the lora condition into filter_operation,
    # not into a separate search/filter_str kwarg.
    models_list_call = mock_entity_client.list.call_args_list[1]
    assert "search" not in models_list_call.kwargs
    assert "filter_str" not in models_list_call.kwargs
    filter_op = models_list_call.kwargs["filter_operation"]
    assert filter_op is not None
    op_dict = filter_op.to_dict()
    assert op_dict == {
        "$or": [
            {
                "$and": [
                    {"workspace": {"$eq": "default"}},
                    {"name": {"$in": ["test-model"]}},
                ]
            }
        ]
    }


@pytest.mark.asyncio
async def test_list_model_entities_lora_false_adds_not_constraint(
    model_entity_service, mock_entity_client, sample_model
):
    """list_model_entities with lora_enabled=false passes $not constraint to exclude lora models."""
    lora_configs = [
        _make_deployment_config("cfg-a", lora_enabled=True, model_entity_id="default/some-lora-model"),
    ]
    configs_response = _configs_list_response(lora_configs)

    mock_models_result = MagicMock()
    mock_models_result.data = [sample_model]
    mock_models_result.pagination = MagicMock()
    mock_models_result.pagination.page = 1
    mock_models_result.pagination.page_size = 100
    mock_models_result.pagination.total_pages = 1
    mock_models_result.pagination.total_results = 1

    mock_adapters_result = _empty_adapters_page()

    # First call: deployment configs, second: models, third: adapters
    mock_entity_client.list.side_effect = [configs_response, mock_models_result, mock_adapters_result]

    lora_filter = ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=False)
    parsed_filter = ParsedFilter(operation=lora_filter)
    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    assert len(result.data) == 1
    # Verify the models list call merged the $not condition into filter_operation.
    models_list_call = mock_entity_client.list.call_args_list[1]
    assert "search" not in models_list_call.kwargs
    assert "filter_str" not in models_list_call.kwargs
    filter_op = models_list_call.kwargs["filter_operation"]
    assert filter_op is not None
    op_dict = filter_op.to_dict()
    assert op_dict == {
        "$not": {
            "$or": [
                {
                    "$and": [
                        {"workspace": {"$eq": "default"}},
                        {"name": {"$in": ["some-lora-model"]}},
                    ]
                }
            ]
        }
    }


@pytest.mark.asyncio
async def test_list_model_entities_lora_false_no_lora_configs_skips_filter(
    model_entity_service, mock_entity_client, sample_model
):
    """list_model_entities with lora_enabled=false and no lora configs skips the name filter."""
    configs_response = _configs_list_response([])

    mock_models_result = MagicMock()
    mock_models_result.data = [sample_model]
    mock_models_result.pagination = MagicMock()
    mock_models_result.pagination.page = 1
    mock_models_result.pagination.page_size = 100
    mock_models_result.pagination.total_pages = 1
    mock_models_result.pagination.total_results = 1

    mock_adapters_result = _empty_adapters_page()

    # First call: deployment configs (empty), second: models, third: adapters
    mock_entity_client.list.side_effect = [configs_response, mock_models_result, mock_adapters_result]

    lora_filter = ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=False)
    parsed_filter = ParsedFilter(operation=lora_filter)
    result = await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)

    assert len(result.data) == 1
    # No lora configs to exclude → no filter conditions added at all.
    models_list_call = mock_entity_client.list.call_args_list[1]
    assert "search" not in models_list_call.kwargs
    assert "filter_str" not in models_list_call.kwargs
    assert models_list_call.kwargs["filter_operation"] is None


# Field map matching what make_filter_dep(ModelEntityFilter) populates in production.
_LORA_FIELD_MAP = {"lora_enabled": "data.lora_enabled"}


@pytest.mark.asyncio
async def test_list_model_entities_rejects_lora_inside_or(model_entity_service, mock_entity_client):
    """lora_enabled inside an $or is rejected — only top-level equality is supported."""
    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.lora_enabled", value=True),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
            ],
        ),
        _field_map=_LORA_FIELD_MAP,
    )
    with pytest.raises(ValueError, match="lora_enabled"):
        await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)
    mock_entity_client.list.assert_not_called()


@pytest.mark.asyncio
async def test_list_model_entities_rejects_lora_inside_not(model_entity_service, mock_entity_client):
    """lora_enabled inside a $not is rejected — only top-level equality is supported."""
    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.NOT,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="data.lora_enabled", value=True),
            ],
        ),
        _field_map=_LORA_FIELD_MAP,
    )
    with pytest.raises(ValueError, match="lora_enabled"):
        await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)
    mock_entity_client.list.assert_not_called()


@pytest.mark.asyncio
async def test_list_model_entities_rejects_lora_in_nested_and(model_entity_service, mock_entity_client):
    """lora_enabled inside a nested AND (not the top-level AND) is rejected."""
    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.AND,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
                LogicalOperation(
                    operator=FilterOperator.OR,
                    operations=[
                        ComparisonOperation(operator=FilterOperator.EQ, field="data.lora_enabled", value=True),
                        ComparisonOperation(operator=FilterOperator.EQ, field="project", value="p"),
                    ],
                ),
            ],
        ),
        _field_map=_LORA_FIELD_MAP,
    )
    with pytest.raises(ValueError, match="lora_enabled"):
        await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)
    mock_entity_client.list.assert_not_called()


@pytest.mark.asyncio
async def test_list_model_entities_rejects_untranslated_lora_inside_or(model_entity_service, mock_entity_client):
    """A service-internal caller that builds a ParsedFilter with the un-translated
    name ``lora_enabled`` (rather than the post-``translate_operation`` name
    ``data.lora_enabled``) is still rejected when the field appears in a
    nested position. The guard resolves both forms via the field map.

    Not a flow we hit in production today (``make_filter_dep`` always
    translates), but it's the case CodeRabbit flagged: if a future internal
    caller forgets to translate, we still want the safety net.
    """
    parsed_filter = ParsedFilter(
        operation=LogicalOperation(
            operator=FilterOperator.OR,
            operations=[
                ComparisonOperation(operator=FilterOperator.EQ, field="lora_enabled", value=True),
                ComparisonOperation(operator=FilterOperator.EQ, field="name", value="foo"),
            ],
        ),
        _field_map=_LORA_FIELD_MAP,
    )
    with pytest.raises(ValueError, match="lora_enabled"):
        await model_entity_service.list_model_entities(workspace="default", parsed_filter=parsed_filter)
    mock_entity_client.list.assert_not_called()
