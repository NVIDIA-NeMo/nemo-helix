# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IGW cache synchronization with Models Service.

These tests verify that the Inference Gateway's model cache correctly syncs
with providers created/updated/deleted in the Models Service, and that the
VirtualModel cache is populated from the entity store.
"""

from __future__ import annotations

import asyncio
import uuid

from nemo_helix import AsyncNeMoHelix, NeMoHelix
from nemo_helix.types.inference.model_provider import ModelProvider
from nhx.core.inference_gateway.api.dependencies import global_model_cache, global_virtual_model_cache
from nhx.core.inference_gateway.api.model_cache import ModelCache, model_provider_getter_from_sdk, refresh_model_cache
from nhx.core.inference_gateway.api.virtual_model_cache import VirtualModelCache, refresh_virtual_model_cache
from nhx.testing import ClientContext

DEFAULT_WORKSPACE = "default"


def _create_provider(
    sdk: NeMoHelix,
    provider_name: str,
    host_url: str,
) -> ModelProvider:
    """Create a provider via the SDK."""
    return sdk.inference.providers.create(
        workspace=DEFAULT_WORKSPACE,
        name=provider_name,
        host_url=host_url,
    )


def _run_cache_refresh(
    model_cache: ModelCache,
    async_sdk: AsyncNeMoHelix,
) -> None:
    """Run cache refresh synchronously."""

    async def refresh() -> None:
        model_provider_getter = model_provider_getter_from_sdk(async_sdk)
        await refresh_model_cache(
            model_cache=model_cache,
            model_provider_getter=model_provider_getter,
            secrets_sdk=async_sdk,
        )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(refresh())
    finally:
        loop.close()


def test_cache_syncs_providers_from_models_service(test_clients: ClientContext):
    """Test that IGW cache syncs providers created via Models Service.

    Verifies that when a provider is created directly via the API,
    a manual cache refresh picks up the new provider.
    """
    model_cache = global_model_cache()
    test_uuid = uuid.uuid4().hex[:8]
    provider_name = f"test-cache-sync-{test_uuid}"
    host_url = "http://localhost:9000"

    # Verify cache is initially empty for our test provider
    assert model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name) is None

    # Create provider
    provider = _create_provider(test_clients.sdk, provider_name, host_url)
    assert provider.name == provider_name

    # Refresh the cache
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify the provider is now in the cache
    cached_provider = model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name)
    assert cached_provider is not None
    assert cached_provider.model_provider.host_url == host_url


def test_cache_includes_served_models_mapping(test_clients: ClientContext):
    """Test that served_models are correctly mapped in the cache.

    When a provider has served_models, the cache should build the
    model_entity_info_map for routing by model entity.
    """
    model_cache = global_model_cache()
    test_uuid = uuid.uuid4().hex[:8]
    provider_name = f"test-served-models-{test_uuid}"
    model_entity_name = f"test-model-entity-{test_uuid}"
    served_model_name = "gpt-test"

    # Create provider
    provider = _create_provider(test_clients.sdk, provider_name, "http://localhost:9001")
    assert provider.name == provider_name

    # Update provider with served_models
    test_clients.sdk.inference.providers.update_status(
        provider_name,
        workspace=DEFAULT_WORKSPACE,
        served_models=[
            {
                "model_entity_id": f"{DEFAULT_WORKSPACE}/{model_entity_name}",
                "served_model_name": served_model_name,
            }
        ],
    )

    # Refresh cache
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify the provider is in the cache with served_models
    cached_provider = model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name)
    assert cached_provider is not None, "Provider should be in cache"
    assert cached_provider.model_provider.served_models is not None
    assert len(cached_provider.model_provider.served_models) == 1

    # Verify model entity is in the cache
    model_entity_info = model_cache.get_from_model_entity(DEFAULT_WORKSPACE, model_entity_name)
    assert model_entity_info is not None, "Model entity info should be in cache"
    assert model_entity_info.workspace == DEFAULT_WORKSPACE
    assert model_entity_info.name == model_entity_name
    assert len(model_entity_info.model_providers) == 1

    cached_served_model_name, provider_info = model_entity_info.model_providers[0]
    assert cached_served_model_name == served_model_name
    assert provider_info.model_provider.name == provider_name


def test_cache_invalidates_deleted_providers(test_clients: ClientContext):
    """Test that deleted providers are removed from cache after refresh.

    When a provider is deleted, a cache refresh should remove it
    from the workspace_name_provider_map.
    """
    model_cache = global_model_cache()
    test_uuid = uuid.uuid4().hex[:8]
    provider_name = f"test-delete-cache-{test_uuid}"

    # Create provider
    provider = _create_provider(test_clients.sdk, provider_name, "http://localhost:9002")
    assert provider.name == provider_name

    # Refresh cache to pick up the provider
    _run_cache_refresh(model_cache, test_clients.async_sdk)
    assert model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name) is not None

    # Delete the provider
    test_clients.sdk.inference.providers.delete(provider_name, workspace=DEFAULT_WORKSPACE)

    # Refresh cache again
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify provider is no longer in cache
    assert model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name) is None


def test_cache_updates_provider_host_url_on_refresh(test_clients: ClientContext):
    """Test that cache updates provider data when it changes.

    When a provider's host_url changes, a cache refresh should update
    the cached value.
    """
    model_cache = global_model_cache()
    test_uuid = uuid.uuid4().hex[:8]
    provider_name = f"test-update-cache-{test_uuid}"

    # Create provider
    provider = _create_provider(test_clients.sdk, provider_name, "http://localhost:9003")
    assert provider.name == provider_name

    # Refresh cache
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify initial host_url
    cached = model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name)
    assert cached.model_provider.host_url == "http://localhost:9003"

    # Update provider host_url using update (PUT)
    test_clients.sdk.inference.providers.update(
        provider_name,
        workspace=DEFAULT_WORKSPACE,
        host_url="http://localhost:9999",
    )

    # Refresh cache again
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify updated host_url
    cached = model_cache.get_from_provider(DEFAULT_WORKSPACE, provider_name)
    assert cached.model_provider.host_url == "http://localhost:9999"


def _run_vm_cache_refresh(
    vm_cache: VirtualModelCache,
    async_sdk: AsyncNeMoHelix,
) -> None:
    """Run VirtualModel cache refresh synchronously."""

    async def refresh() -> None:
        await refresh_virtual_model_cache(cache=vm_cache, sdk=async_sdk)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(refresh())
    finally:
        loop.close()


def test_virtual_model_cache_syncs_from_entity_store(test_clients: ClientContext):
    """VirtualModel cache is populated from VirtualModels stored in the entity store.

    Creates a VirtualModel via the CRUD API, triggers a cache refresh, and
    verifies the VM appears in the in-memory VirtualModelCache.
    """
    vm_cache = global_virtual_model_cache()
    test_uuid = uuid.uuid4().hex[:8]
    vm_name = f"test-vm-cache-{test_uuid}"

    # Verify not already in cache
    assert vm_cache.get(DEFAULT_WORKSPACE, vm_name) is None

    # Create a VirtualModel via the IGW CRUD API
    test_clients.sdk.inference.virtual_models.create(
        workspace=DEFAULT_WORKSPACE,
        name=vm_name,
        default_model_entity=f"{DEFAULT_WORKSPACE}/some-model",
    )

    # Refresh the VM cache
    _run_vm_cache_refresh(vm_cache, test_clients.async_sdk)

    # Verify it's now in the cache
    cached_vm = vm_cache.get(DEFAULT_WORKSPACE, vm_name)
    assert cached_vm is not None
    assert cached_vm.name == vm_name
    assert cached_vm.workspace == DEFAULT_WORKSPACE
    assert cached_vm.default_model_entity == f"{DEFAULT_WORKSPACE}/some-model"


def test_cache_handles_multiple_providers(test_clients: ClientContext):
    """Test that cache correctly handles multiple providers.

    Verifies that all providers are included in the cache after refresh.
    """
    model_cache = global_model_cache()
    test_uuid = uuid.uuid4().hex[:8]

    # Create multiple providers
    provider_names = []
    for i in range(3):
        provider_name = f"test-multi-{test_uuid}-{i}"
        provider_names.append(provider_name)

        provider = _create_provider(test_clients.sdk, provider_name, f"http://localhost:900{i}")
        assert provider.name == provider_name

    # Refresh cache
    _run_cache_refresh(model_cache, test_clients.async_sdk)

    # Verify all providers are in cache
    for name in provider_names:
        cached = model_cache.get_from_provider(DEFAULT_WORKSPACE, name)
        assert cached is not None, f"Provider {name} should be in cache"
