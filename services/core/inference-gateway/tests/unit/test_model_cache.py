# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model cache functionality."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from nemo_helix.types.inference import ServedModelMapping
from nemo_helix_plugin.inference_middleware import BackendFormat
from nhx.core.inference_gateway.api import model_cache as model_cache_module
from nhx.core.inference_gateway.api.backend_format import resolve_backend_format
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry
from nhx.core.inference_gateway.api.model_cache import (
    ModelCache,
    ModelEntityInfo,
    ModelProvider,
    ModelProviderInfo,
    ModelProviderRefreshError,
    refresh_model_cache,
    refresh_model_cache_task,
)
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache


def new_model_infos() -> list[ModelProviderInfo]:
    """Create new model provider infos for testing."""
    return [
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="default",
                name="new",
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ),
    ]


async def async_new_model_providers() -> list[ModelProvider]:
    return [mpi.model_provider for mpi in new_model_infos()]


def _model_provider_getter_for(
    model_entity_id: str = "test-ns/claude-sonnet",
    served_model_name: str = "claude-sonnet",
):
    async def provider_getter():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://test.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id=model_entity_id,
                        served_model_name=served_model_name,
                    ),
                ],
            )
        ]

    return provider_getter


def _model_entity_getter_for(*model_entities: SimpleNamespace):
    async def model_entity_getter():
        return list(model_entities)

    return model_entity_getter


def _model_entity(
    *,
    workspace: str = "test-ns",
    name: str = "claude-sonnet",
    spec: object | None = None,
    finetuning_type: str | None = None,
    backend_format: object | None = "ANTHROPIC_MESSAGES",
) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=workspace,
        name=name,
        spec=spec,
        finetuning_type=finetuning_type,
        backend_format=backend_format,
    )


def test_get_from_workspace_name(model_cache: ModelCache):
    tot = model_cache.get_from_provider("default", "tot")
    assert tot
    assert "mock-nim.example.invalid" in tot.model_provider.host_url


def test_get_from_workspace_name_not_exist(model_cache: ModelCache):
    model_provider = model_cache.get_from_provider("default", "notexist")
    assert model_provider is None


def test_add_model_provider(model_cache: ModelCache):
    model_provider_info = new_model_infos()[0]
    model_cache.update_model_info(model_provider_info)

    retrieved = model_cache.get_from_provider("default", "new")
    assert retrieved
    assert retrieved.model_provider.name == "new"

    # validate that overwriting for a given workspace/name overwrites the existing entry
    model_provider_info.model_provider.host_url = "http://localhost:4242"
    model_cache.update_model_info(model_provider_info)
    retrieved = model_cache.get_from_provider("default", "new")
    assert retrieved
    assert "4242" in retrieved.model_provider.host_url


@pytest.mark.asyncio
async def test_refresh_model_cache(model_cache: ModelCache, mock_nhx_sdk):
    await refresh_model_cache(model_cache, async_new_model_providers, secrets_sdk=mock_nhx_sdk)
    assert model_cache.get_from_provider("default", "new")


@pytest.mark.asyncio
async def test_refresh_model_cache_getter_failure(model_cache: ModelCache, mock_nhx_sdk):
    """Test model cache refresh when getter fails."""
    mock_getter = AsyncMock(side_effect=Exception("API error"))

    with pytest.raises(ModelProviderRefreshError) as exc_info:
        await refresh_model_cache(model_cache, mock_getter, secrets_sdk=mock_nhx_sdk)

    assert "Error trying to refresh model provider cache" in str(exc_info.value)


@pytest.mark.asyncio
async def test_refresh_model_cache_task(mocker, model_cache: ModelCache, mock_nhx_sdk):
    # Create side effect that allows looping twice, then raises exception to break loop
    mock_logger = Mock()
    mocker.patch.object(model_cache_module, "logger", mock_logger)
    mock_pause = AsyncMock(side_effect=[None, None, Exception("Break loop")])
    mocker.patch.object(model_cache_module, "_async_pause", mock_pause)

    # Throw an error first, and then return successful data
    mock_getter = AsyncMock(side_effect=[Exception("API Error"), await async_new_model_providers()])
    with pytest.raises(Exception, match="Break loop"):
        await refresh_model_cache_task(model_cache, mock_getter, secrets_sdk=mock_nhx_sdk, sleep_duration_s=0)

    assert mock_pause.call_count == 3
    for call in mock_pause.call_args_list:
        assert call[0][0] == 0

    # ensure that we both error logged and properly added data to the cache
    mock_logger.exception.assert_called()
    assert model_cache.get_from_provider("default", "new")


@pytest.mark.asyncio
async def test_refresh_model_cache_task_max_consecutive_failures(mocker, model_cache: ModelCache, mock_nhx_sdk):
    """Test that refresh_model_cache_task raises an exception after max_consecutive_failures."""
    mock_logger = Mock()
    mocker.patch.object(model_cache_module, "logger", mock_logger)

    # Mock sleep to return immediately
    mock_pause = AsyncMock(return_value=None)
    mocker.patch.object(model_cache_module, "_async_pause", mock_pause)

    # Create a getter that always fails
    mock_getter = AsyncMock(side_effect=Exception("API Error"))

    # Test with max_consecutive_failures=3 for faster testing
    with pytest.raises(ModelProviderRefreshError) as exc_info:
        await refresh_model_cache_task(
            model_cache, mock_getter, secrets_sdk=mock_nhx_sdk, sleep_duration_s=0, max_consecutive_failures=3
        )

    # Verify the error message mentions consecutive failures
    assert "failed 3 consecutive times" in str(exc_info.value)
    assert "persistently unavailable" in str(exc_info.value)

    # Verify we logged errors - note that each failure logs twice:
    # once in refresh_model_cache and once in refresh_model_cache_task
    assert mock_logger.exception.call_count == 6  # 3 failures * 2 logs per failure
    assert mock_logger.error.call_count == 1


@pytest.mark.asyncio
async def test_refresh_model_cache_task_failure_counter_resets(mocker, model_cache: ModelCache, mock_nhx_sdk):
    """Test that the failure counter resets after a successful refresh."""
    mock_logger = Mock()
    mocker.patch.object(model_cache_module, "logger", mock_logger)

    # Mock sleep - we'll let it run 6 times before breaking
    mock_pause = AsyncMock(side_effect=[None, None, None, None, None, None, Exception("Break loop")])
    mocker.patch.object(model_cache_module, "_async_pause", mock_pause)

    # Create a getter that fails twice, succeeds once, fails twice again, then succeeds
    # This ensures we never hit max_consecutive_failures because success resets the counter
    mock_getter = AsyncMock(
        side_effect=[
            Exception("API Error 1"),
            Exception("API Error 2"),
            await async_new_model_providers(),  # Success - resets counter
            Exception("API Error 3"),
            Exception("API Error 4"),
            await async_new_model_providers(),  # Success - resets counter again
        ]
    )

    # Test with max_consecutive_failures=3, but we never hit it because of successes
    with pytest.raises(Exception, match="Break loop"):
        await refresh_model_cache_task(
            model_cache, mock_getter, secrets_sdk=mock_nhx_sdk, sleep_duration_s=0, max_consecutive_failures=3
        )

    # Should have completed without hitting max_consecutive_failures
    # We should see 4 error logs (2 failures + 2 failures), but each failure logs twice:
    # once in refresh_model_cache and once in refresh_model_cache_task
    assert mock_logger.exception.call_count == 8  # 4 failures * 2 logs per failure
    # The logger.error call happens only when we hit max_consecutive_failures, which shouldn't happen
    assert mock_logger.error.call_count == 0

    # Verify the cache was updated with the successful refreshes
    assert model_cache.get_from_provider("default", "new")


def test_get_from_model_entity(model_cache: ModelCache):
    """Test getting model entity from cache."""
    entity_info = model_cache.get_from_model_entity("e2e-test", "meta_llama-3.2-1b-instruct")
    assert entity_info is not None
    assert entity_info.workspace == "e2e-test"
    assert entity_info.name == "meta_llama-3.2-1b-instruct"
    assert len(entity_info.model_providers) == 2  # Both ollama and tot serve this model


def test_get_from_model_entity_not_exist(model_cache: ModelCache):
    """Test getting non-existent model entity returns None."""
    entity_info = model_cache.get_from_model_entity("nonexistent", "model")
    assert entity_info is None


def test_rebuild_model_entity_map():
    """Test rebuild_model_entity_map populates the model entity map correctly."""
    cache = ModelCache()

    # Add model providers with served models
    provider1 = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="test-ns",
            name="provider1",
            host_url="http://provider1.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="workspace1/model-a",
                    served_model_name="model-a-v1",
                ),
                ServedModelMapping(
                    model_entity_id="workspace1/model-b",
                    served_model_name="model-b-v1",
                ),
            ],
        )
    )

    provider2 = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="test-ns",
            name="provider2",
            host_url="http://provider2.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="workspace1/model-a",
                    served_model_name="model-a-v2",
                ),
            ],
        )
    )

    cache.update_model_info(provider1)
    cache.update_model_info(provider2)
    cache.rebuild_model_entity_map()

    # Check model-a has two providers
    model_a_info = cache.get_from_model_entity("workspace1", "model-a")
    assert model_a_info is not None
    assert len(model_a_info.model_providers) == 2
    assert model_a_info.workspace == "workspace1"
    assert model_a_info.name == "model-a"

    # Check served model names
    served_names = [name for name, _ in model_a_info.model_providers]
    assert "model-a-v1" in served_names
    assert "model-a-v2" in served_names

    # Check model-b has one provider
    model_b_info = cache.get_from_model_entity("workspace1", "model-b")
    assert model_b_info is not None
    assert len(model_b_info.model_providers) == 1
    assert model_b_info.model_providers[0][0] == "model-b-v1"


def test_rebuild_model_entity_map_clears_existing():
    """Test that rebuild_model_entity_map clears existing entries."""
    cache = ModelCache()

    # Add a provider with served models
    provider1 = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="test-ns",
            name="provider1",
            host_url="http://provider1.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="workspace1/model-a",
                    served_model_name="model-a-v1",
                ),
            ],
        )
    )

    cache.update_model_info(provider1)
    cache.rebuild_model_entity_map()

    # Verify model-a exists
    assert cache.get_from_model_entity("workspace1", "model-a") is not None

    # Remove the served_models and rebuild
    provider1.model_provider.served_models = []
    cache.update_model_info(provider1)
    cache.rebuild_model_entity_map()

    # Verify model-a no longer exists
    assert cache.get_from_model_entity("workspace1", "model-a") is None


def test_rebuild_model_entity_map_handles_none_served_models():
    """Test that rebuild_model_entity_map handles None served_models gracefully."""
    cache = ModelCache()

    # Add a provider with None served_models
    provider = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="test-ns",
            name="provider1",
            host_url="http://provider1.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=None,
        )
    )

    cache.update_model_info(provider)
    cache.rebuild_model_entity_map()

    # Should not raise an exception and entity map should be empty
    assert len(cache.model_entity_info_map) == 0


def test_rebuild_model_entity_map_lora_entity_id():
    """Test rebuild_model_entity_map with model_entity_id containing &adapters/ (split on first / only)."""
    cache = ModelCache()
    provider = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="ws",
            name="nim-provider",
            host_url="http://nim.example.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="ws/base&adapters/ws/adder",
                    served_model_name="adapter-backend-id",
                ),
            ],
        )
    )
    cache.update_model_info(provider)
    cache.rebuild_model_entity_map()

    entity_info = cache.get_from_model_entity("ws", "base&adapters/ws/adder")
    assert entity_info is not None
    assert entity_info.workspace == "ws"
    assert entity_info.name == "base&adapters/ws/adder"
    assert len(entity_info.model_providers) == 1
    assert entity_info.model_providers[0][0] == "adapter-backend-id"


def test_rebuild_model_entity_map_lora_cross_workspace():
    """Cross-workspace LoRA: ``provider.workspace="ws-a"``, ``base_ws="ws-a"``, ``adapter_ws="ws-b"``.

    Pins down the invariant that ``ModelCache.rebuild_model_entity_map`` splits
    ``model_entity_id`` on the first ``/`` only and does NOT promote the
    ``adapter_ws`` segment to the cache key's workspace slot — so the entity
    keys live under ``base_ws`` (here ``ws-a``), regardless of where the
    adapter is homed (here ``ws-b``).

    This guards the wire format ``{base_ws}/{base_name}&adapters/{adapter_ws}/{adapter_name}``
    against a regression that silently clamps ``adapter_ws == provider.workspace``
    or ``adapter_ws == base_ws`` — every existing LoRA test collapses all three
    onto the same workspace, so this case is the cross-workspace gap.
    """
    cache = ModelCache()
    provider_info = ModelProviderInfo(
        model_provider=ModelProvider(
            workspace="ws-a",
            name="nim-provider",
            host_url="http://nim.workspace-a.example.com",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            served_models=[
                ServedModelMapping(
                    model_entity_id="ws-a/base&adapters/ws-b/adapter",
                    served_model_name="ws-b--adapter",
                ),
            ],
        )
    )
    cache.update_model_info(provider_info)
    cache.rebuild_model_entity_map()

    entity_info = cache.get_from_model_entity("ws-a", "base&adapters/ws-b/adapter")
    assert entity_info is not None
    assert entity_info.workspace == "ws-a"
    assert entity_info.name == "base&adapters/ws-b/adapter"
    assert len(entity_info.model_providers) == 1
    served_name, provider_pair = entity_info.model_providers[0]
    assert served_name == "ws-b--adapter"
    assert provider_pair is provider_info

    # The adapter_ws ("ws-b") segment must NOT be promoted to the key's workspace
    # slot — looking up under ``ws-b`` for any sub-key is a miss.
    assert cache.get_from_model_entity("ws-b", "adapter") is None
    assert cache.get_from_model_entity("ws-b", "base&adapters/ws-b/adapter") is None
    assert cache.get_from_model_entity("ws-b", "base&adapters/ws-a/adapter") is None


def test_rebuild_model_entity_map_skips_malformed_entity_ids(caplog):
    """Malformed model_entity_ids are skipped with a WARNING, good ones are still cached."""
    cache = ModelCache()
    cache.update_model_info(
        ModelProviderInfo(
            model_provider=ModelProvider(
                workspace="ws",
                name="nim-provider",
                host_url="http://nim.example.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(model_entity_id="ws/good", served_model_name="good-backend"),
                    # No workspace separator at all.
                    ServedModelMapping(model_entity_id="bare-name", served_model_name="bad-1"),
                    # Empty workspace (leading slash).
                    ServedModelMapping(model_entity_id="/foo", served_model_name="bad-2"),
                    # Empty entity-name segment (trailing slash, no remainder).
                    ServedModelMapping(model_entity_id="ws/", served_model_name="bad-3"),
                ],
            )
        )
    )

    with caplog.at_level("WARNING", logger=model_cache_module.logger.name):
        cache.rebuild_model_entity_map()

    # Only the valid id survives; malformed ids don't leak unreachable entries.
    assert cache.get_from_model_entity("ws", "good") is not None
    assert len(cache.model_entity_info_map) == 1


@pytest.mark.asyncio
async def test_refresh_model_cache_rebuilds_entity_map(mock_nhx_sdk):
    """Test that refresh_model_cache rebuilds the model entity map."""
    cache = ModelCache()

    async def provider_getter():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://test.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="test-ns/test-model",
                        served_model_name="test-model-v1",
                    ),
                ],
            )
        ]

    await refresh_model_cache(cache, provider_getter, secrets_sdk=mock_nhx_sdk)

    # Verify the entity map was rebuilt
    entity_info = cache.get_from_model_entity("test-ns", "test-model")
    assert entity_info is not None
    assert len(entity_info.model_providers) == 1
    assert entity_info.model_providers[0][0] == "test-model-v1"


@pytest.mark.asyncio
async def test_refresh_model_cache_populates_model_entity_backend_format(mock_nhx_sdk):
    cache = ModelCache()

    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(_model_entity()),
    )

    entity_info = cache.get_from_model_entity("test-ns", "claude-sonnet")
    assert entity_info is not None
    assert entity_info.backend_format is BackendFormat.ANTHROPIC_MESSAGES


@pytest.mark.asyncio
async def test_refresh_model_cache_preserves_backend_format_when_metadata_refresh_fails(mock_nhx_sdk):
    cache = ModelCache()

    async def failing_model_entity_getter():
        raise RuntimeError("models service unavailable")

    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(_model_entity()),
    )
    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=failing_model_entity_getter,
    )

    entity_info = cache.get_from_model_entity("test-ns", "claude-sonnet")
    assert entity_info is not None
    assert entity_info.backend_format is BackendFormat.ANTHROPIC_MESSAGES


@pytest.mark.asyncio
async def test_refresh_model_cache_preserves_omitted_model_entity_metadata_fields(mock_nhx_sdk):
    cache = ModelCache()
    updated_spec = SimpleNamespace(context_length=2048)
    partial_model_entity = SimpleNamespace(
        workspace="test-ns",
        name="claude-sonnet",
        spec=updated_spec,
        finetuning_type=None,
        backend_format=None,
        model_fields_set={"workspace", "name", "spec"},
    )

    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(_model_entity(finetuning_type="lora")),
    )
    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(partial_model_entity),
    )

    entity_info = cache.get_from_model_entity("test-ns", "claude-sonnet")
    assert entity_info is not None
    assert entity_info.spec is updated_spec
    assert entity_info.finetuning_type == "lora"
    assert entity_info.backend_format is BackendFormat.ANTHROPIC_MESSAGES


@pytest.mark.asyncio
async def test_refresh_model_cache_clears_backend_format_when_successful_metadata_refresh_omits_model(mock_nhx_sdk):
    cache = ModelCache()

    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(_model_entity()),
    )
    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(),
    )

    entity_info = cache.get_from_model_entity("test-ns", "claude-sonnet")
    assert entity_info is not None
    assert entity_info.backend_format is None


@pytest.mark.asyncio
async def test_refresh_model_cache_ignores_invalid_model_entity_backend_format(mock_nhx_sdk):
    cache = ModelCache()

    await refresh_model_cache(
        cache,
        _model_provider_getter_for(),
        secrets_sdk=mock_nhx_sdk,
        model_entity_getter=_model_entity_getter_for(_model_entity(backend_format="NOT_A_FORMAT")),
    )

    entity_info = cache.get_from_model_entity("test-ns", "claude-sonnet")
    assert entity_info is not None
    assert entity_info.backend_format is None


def test_resolve_backend_format_precedence():
    entity_info = ModelEntityInfo(workspace="ws", name="model-a", backend_format=BackendFormat.OPENAI_CHAT)
    virtual_model = SimpleNamespace(
        models=[
            SimpleNamespace(model="ws/model-a", backend_format="ANTHROPIC_MESSAGES"),
        ]
    )

    assert resolve_backend_format(entity_info, virtual_model) is BackendFormat.ANTHROPIC_MESSAGES
    assert resolve_backend_format(entity_info) is BackendFormat.OPENAI_CHAT
    assert resolve_backend_format(ModelEntityInfo(workspace="ws", name="model-b")) is None


def test_resolve_backend_format_returns_none_for_invalid_values():
    entity_info = SimpleNamespace(workspace="ws", name="model-a", backend_format="NOT_A_FORMAT")
    virtual_model = SimpleNamespace(
        models=[
            SimpleNamespace(model="ws/model-a", backend_format="ALSO_NOT_A_FORMAT"),
        ]
    )

    assert resolve_backend_format(entity_info, virtual_model) is None


@pytest.mark.asyncio
async def test_refresh_model_cache_removes_stale_providers(mock_nhx_sdk):
    """Test that refresh_model_cache removes providers that are no longer in the fetched list."""
    cache = ModelCache()

    # First refresh: add two providers
    async def provider_getter_initial():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://test1.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            ModelProvider(
                workspace="test",
                name="provider2",
                host_url="http://test2.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    await refresh_model_cache(cache, provider_getter_initial, secrets_sdk=mock_nhx_sdk)

    # Verify both providers are in cache
    assert cache.get_from_provider("test", "provider1") is not None
    assert cache.get_from_provider("test", "provider2") is not None
    assert len(cache.workspace_name_provider_map) == 2

    # Second refresh: only provider1 remains
    async def provider_getter_updated():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://test1.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

    await refresh_model_cache(cache, provider_getter_updated, secrets_sdk=mock_nhx_sdk)

    # Verify provider1 still exists but provider2 was removed
    assert cache.get_from_provider("test", "provider1") is not None
    assert cache.get_from_provider("test", "provider2") is None
    assert len(cache.workspace_name_provider_map) == 1


@pytest.mark.asyncio
async def test_refresh_model_cache_updates_existing_provider_config(mock_nhx_sdk):
    """Test that refresh_model_cache updates existing providers with fresh configuration."""
    cache = ModelCache()

    # First refresh: add provider with initial config
    async def provider_getter_initial():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://initial.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                default_extra_body=None,
                required_extra_body=None,
            ),
        ]

    await refresh_model_cache(cache, provider_getter_initial, secrets_sdk=mock_nhx_sdk)

    # Verify initial config
    provider_info = cache.get_from_provider("test", "provider1")
    assert provider_info is not None
    assert provider_info.model_provider.host_url == "http://initial.com"
    assert provider_info.model_provider.default_extra_body is None
    assert provider_info.model_provider.required_extra_body is None

    # Second refresh: same provider with updated config
    async def provider_getter_updated():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://updated.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                default_extra_body={"temperature": 0.7},
                required_extra_body={"stream": True},
            ),
        ]

    await refresh_model_cache(cache, provider_getter_updated, secrets_sdk=mock_nhx_sdk)

    # Verify config was updated
    provider_info = cache.get_from_provider("test", "provider1")
    assert provider_info is not None
    assert provider_info.model_provider.host_url == "http://updated.com"
    assert provider_info.model_provider.default_extra_body == {"temperature": 0.7}
    assert provider_info.model_provider.required_extra_body == {"stream": True}
    assert len(cache.workspace_name_provider_map) == 1


@pytest.mark.asyncio
async def test_refresh_model_cache_updates_existing_provider_served_models(mock_nhx_sdk):
    """Test that refresh_model_cache updates served_models for existing providers.

    This catches a bug where existing providers in cache would not get their
    model_provider updated with fresh data (like newly populated served_models),
    causing the model entity map to be built from stale empty served_models.
    """
    cache = ModelCache()

    # First refresh: provider exists but has no served_models yet (simulating
    # the state before autodiscovery populates served_models)
    async def provider_getter_initial():
        return [
            ModelProvider(
                workspace="e2e-test",
                name="llama-deployment",
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[],  # Empty - autodiscovery hasn't run yet
            ),
        ]

    await refresh_model_cache(cache, provider_getter_initial, secrets_sdk=mock_nhx_sdk)

    # Verify provider is cached but no model entities exist
    assert cache.get_from_provider("e2e-test", "llama-deployment") is not None
    assert cache.get_from_model_entity("e2e-test", "meta-llama-3-2-1b-instruct") is None
    assert len(cache.model_entity_info_map) == 0

    # Second refresh: same provider but now served_models is populated
    # (simulating autodiscovery having run and updated the provider)
    async def provider_getter_with_served_models():
        return [
            ModelProvider(
                workspace="e2e-test",
                name="llama-deployment",
                host_url="http://localhost:8080",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(
                        model_entity_id="e2e-test/meta-llama-3-2-1b-instruct",
                        served_model_name="meta/llama-3.2-1b-instruct",
                    ),
                ],
            ),
        ]

    await refresh_model_cache(cache, provider_getter_with_served_models, secrets_sdk=mock_nhx_sdk)

    # Verify the model entity map was updated with the new served_models
    entity_info = cache.get_from_model_entity("e2e-test", "meta-llama-3-2-1b-instruct")
    assert entity_info is not None, "Model entity should exist after provider's served_models is populated"
    assert entity_info.workspace == "e2e-test"
    assert entity_info.name == "meta-llama-3-2-1b-instruct"
    assert len(entity_info.model_providers) == 1
    assert entity_info.model_providers[0][0] == "meta/llama-3.2-1b-instruct"


# ---------------------------------------------------------------------------
# middleware_registry parameter threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_model_cache_without_registry_is_backward_compatible(model_cache: ModelCache, mock_nhx_sdk):
    """refresh_model_cache without middleware_registry still works (backward compat)."""
    await refresh_model_cache(model_cache, async_new_model_providers, secrets_sdk=mock_nhx_sdk)
    assert model_cache.get_from_provider("default", "new")


@pytest.mark.asyncio
async def test_refresh_model_cache_passes_registry_to_vm_cache_refresh(model_cache: ModelCache, mock_nhx_sdk):
    """When middleware_registry is provided it is forwarded to refresh_virtual_model_cache."""
    vm_cache = VirtualModelCache()
    registry = MiddlewareRegistry()

    with patch(
        "nhx.core.inference_gateway.api.model_cache.refresh_virtual_model_cache",
        new_callable=AsyncMock,
    ) as mock_vm_refresh:
        await refresh_model_cache(
            model_cache,
            async_new_model_providers,
            secrets_sdk=mock_nhx_sdk,
            virtual_model_cache=vm_cache,
            middleware_registry=registry,
        )

    mock_vm_refresh.assert_awaited_once()
    assert mock_vm_refresh.call_args.kwargs.get("registry") is registry


@pytest.mark.asyncio
async def test_refresh_model_cache_task_passes_registry(mocker, model_cache: ModelCache, mock_nhx_sdk):
    """refresh_model_cache_task threads middleware_registry into each refresh cycle."""
    vm_cache = VirtualModelCache()
    registry = MiddlewareRegistry()

    mock_pause = AsyncMock(side_effect=[None, Exception("stop")])
    mocker.patch.object(model_cache_module, "_async_pause", mock_pause)

    with patch(
        "nhx.core.inference_gateway.api.model_cache.refresh_virtual_model_cache",
        new_callable=AsyncMock,
    ) as mock_vm_refresh:
        with pytest.raises(Exception, match="stop"):
            await refresh_model_cache_task(
                model_cache,
                async_new_model_providers,
                secrets_sdk=mock_nhx_sdk,
                sleep_duration_s=1,
                virtual_model_cache=vm_cache,
                middleware_registry=registry,
            )

    # At least one cycle ran and passed the registry through
    assert mock_vm_refresh.await_count >= 1
    assert mock_vm_refresh.call_args.kwargs.get("registry") is registry


# ---------------------------------------------------------------------------
# refresh_model_cache: no-op rebuild skip (provider-layer signature)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_skips_rebuild_when_provider_layer_unchanged(mocker, model_cache: ModelCache, mock_nmp_sdk):
    """A second refresh with an identical provider layer skips rebuild_model_entity_map.

    The rebuild is compute-only work; when the provider set + served_models are byte-for-byte
    unchanged, the entity map cannot have changed, so the rebuild is skipped.
    """
    getter = _model_provider_getter_for()
    spy = mocker.spy(ModelCache, "rebuild_model_entity_map")

    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk)
    assert spy.call_count == 1  # first cycle always rebuilds (signature was None)

    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk)
    assert spy.call_count == 1  # unchanged provider layer -> rebuild skipped


@pytest.mark.asyncio
async def test_refresh_rebuilds_when_served_model_changes(mocker, model_cache: ModelCache, mock_nmp_sdk):
    """Changing a provider's served-model id changes the signature and forces a rebuild."""
    spy = mocker.spy(ModelCache, "rebuild_model_entity_map")

    await refresh_model_cache(
        model_cache, _model_provider_getter_for(model_entity_id="test/a"), secrets_sdk=mock_nmp_sdk
    )
    assert spy.call_count == 1

    # Different served-model id => different signature => rebuild runs again.
    await refresh_model_cache(
        model_cache, _model_provider_getter_for(model_entity_id="test/b"), secrets_sdk=mock_nmp_sdk
    )
    assert spy.call_count == 2


@pytest.mark.asyncio
async def test_refresh_skips_rebuild_when_map_empty_and_signature_unchanged(
    mocker, model_cache: ModelCache, mock_nmp_sdk
):
    """An empty entity map with an unchanged signature is a valid steady state and is skipped.

    Providers can legitimately produce an empty map (no/empty served_models before autodiscovery,
    or only malformed ids). Cold start is already covered by the initial signature being None, so
    once the signature is set, a still-empty map with a matching signature must NOT trigger a
    rebuild every cycle (which would defeat the optimization and re-spam malformed-id warnings).
    """
    getter = _model_provider_getter_for()

    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk)
    # Simulate a cache whose entity map is empty while the (matching) signature is retained.
    model_cache.model_entity_info_map = {}
    spy = mocker.spy(ModelCache, "rebuild_model_entity_map")

    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk)
    assert spy.call_count == 0  # unchanged signature -> skipped even though the map is empty


@pytest.mark.asyncio
async def test_refresh_runs_metadata_update_even_when_rebuild_skipped(mocker, model_cache: ModelCache, mock_nmp_sdk):
    """update_model_entity_metadata runs every cycle, independent of the rebuild skip.

    Metadata (spec/finetuning_type/backend_format) is applied in place, so it must not be
    gated by the rebuild signature.
    """
    getter = _model_provider_getter_for(model_entity_id="test-ns/claude-sonnet")
    entity_getter = _model_entity_getter_for(_model_entity())
    meta_spy = mocker.spy(ModelCache, "update_model_entity_metadata")
    rebuild_spy = mocker.spy(ModelCache, "rebuild_model_entity_map")

    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk, model_entity_getter=entity_getter)
    await refresh_model_cache(model_cache, getter, secrets_sdk=mock_nmp_sdk, model_entity_getter=entity_getter)

    assert rebuild_spy.call_count == 1  # second rebuild skipped (unchanged)
    assert meta_spy.call_count == 2  # metadata update still ran both cycles


def _model_provider_getter_with_mappings(mappings: list[tuple[str, str]]):
    """Provider getter whose single provider serves `mappings` (entity_id, served_name) in order."""

    async def provider_getter():
        return [
            ModelProvider(
                workspace="test",
                name="provider1",
                host_url="http://test.com",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(model_entity_id=eid, served_model_name=sname) for eid, sname in mappings
                ],
            )
        ]

    return provider_getter


@pytest.mark.asyncio
async def test_refresh_rebuilds_when_served_models_reordered(mocker, model_cache: ModelCache, mock_nmp_sdk):
    """Reordering same-entity served-model mappings must invalidate the signature and rebuild.

    The rebuild appends providers to an entity's model_providers list in served_models order,
    and consumers select model_providers[0] (proxy / middleware_registry). A signature that
    sorted the mappings would treat a reorder as unchanged and skip the rebuild, silently
    retaining the stale first backend. The signature preserves order, so a reorder rebuilds.
    """
    entity = "test-ns/claude-sonnet"
    getter_ab = _model_provider_getter_with_mappings([(entity, "backend-a"), (entity, "backend-b")])
    getter_ba = _model_provider_getter_with_mappings([(entity, "backend-b"), (entity, "backend-a")])

    await refresh_model_cache(model_cache, getter_ab, secrets_sdk=mock_nmp_sdk)
    # The first-listed backend is selected as model_providers[0].
    first_before = model_cache.model_entity_info_map[("test-ns", "claude-sonnet")].model_providers[0][0]
    assert first_before == "backend-a"

    spy = mocker.spy(ModelCache, "rebuild_model_entity_map")
    await refresh_model_cache(model_cache, getter_ba, secrets_sdk=mock_nmp_sdk)
    assert spy.call_count == 1  # reorder changed the signature -> rebuild ran
    first_after = model_cache.model_entity_info_map[("test-ns", "claude-sonnet")].model_providers[0][0]
    assert first_after == "backend-b"  # routing now reflects the new order
