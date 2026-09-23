# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VirtualModelCache and refresh_virtual_model_cache."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_helix import APIConnectionError, APIStatusError, AsyncNeMoHelix
from nemo_helix.types.inference.middleware_call import MiddlewareCall
from nemo_helix.types.inference.virtual_model import VirtualModel
from nhx.core.inference_gateway.api.middleware_registry import (
    MiddlewareConfigRef,
    MiddlewareRegistry,
    PrefetchResult,
)
from nhx.core.inference_gateway.api.virtual_model_cache import (
    VirtualModelCache,
    VirtualModelCacheRefreshError,
    refresh_virtual_model_cache,
    sync_config_ref_versions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vm(workspace: str, name: str, default_model_entity: str | None = None) -> VirtualModel:
    return VirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity=default_model_entity or f"{workspace}/{name}",
    )


def _make_sdk_with_vms(vms: list[VirtualModel]) -> AsyncNeMoHelix:
    """Return a mock SDK whose inference.virtual_models.list() yields *vms* as an async iterator."""
    sdk = MagicMock(spec=AsyncNeMoHelix)

    async def _async_iter(_self=None):
        for vm in vms:
            yield vm

    paginator = MagicMock()
    paginator.__aiter__ = lambda self: _async_iter()
    sdk.inference.virtual_models.list = MagicMock(return_value=paginator)
    return sdk


# ---------------------------------------------------------------------------
# VirtualModelCache unit tests
# ---------------------------------------------------------------------------


def test_get_returns_none_for_missing():
    cache = VirtualModelCache()
    assert cache.get("ws", "nonexistent") is None


def test_get_returns_entry_after_rebuild():
    cache = VirtualModelCache()
    vm = _make_vm("ws", "model-a")
    cache.rebuild([vm])
    assert cache.get("ws", "model-a") is vm


def test_rebuild_replaces_stale_entries():
    cache = VirtualModelCache()
    cache.rebuild([_make_vm("ws", "old-model")])
    cache.rebuild([_make_vm("ws", "new-model")])
    assert cache.get("ws", "old-model") is None
    assert cache.get("ws", "new-model") is not None


def test_rebuild_with_empty_list_clears_cache():
    cache = VirtualModelCache()
    cache.rebuild([_make_vm("ws", "model-a")])
    cache.rebuild([])
    assert cache.get("ws", "model-a") is None
    assert cache.virtual_model_map == {}


def test_get_is_workspace_scoped():
    cache = VirtualModelCache()
    cache.rebuild([_make_vm("ws-a", "model"), _make_vm("ws-b", "model")])
    assert cache.get("ws-a", "model") is not None
    assert cache.get("ws-b", "model") is not None
    assert cache.get("ws-c", "model") is None


# ---------------------------------------------------------------------------
# refresh_virtual_model_cache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_populates_cache():
    """Refresh with two VMs rebuilds the cache correctly."""
    vms = [_make_vm("ws", "model-a"), _make_vm("ws", "model-b")]
    sdk = _make_sdk_with_vms(vms)
    cache = VirtualModelCache()

    await refresh_virtual_model_cache(cache, sdk)

    assert cache.get("ws", "model-a") is not None
    assert cache.get("ws", "model-b") is not None
    sdk.inference.virtual_models.list.assert_called_once_with(workspace="-", page_size=200)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_empty_result_clears_cache():
    """Refresh with no VMs leaves cache empty."""
    sdk = _make_sdk_with_vms([])
    cache = VirtualModelCache()
    cache.rebuild([_make_vm("ws", "stale")])

    await refresh_virtual_model_cache(cache, sdk)

    assert cache.get("ws", "stale") is None


@pytest.mark.asyncio
async def test_refresh_replaces_stale_entries():
    """Second refresh with a different set removes the old entries."""
    sdk_first = _make_sdk_with_vms([_make_vm("ws", "model-a"), _make_vm("ws", "model-b")])
    sdk_second = _make_sdk_with_vms([_make_vm("ws", "model-b"), _make_vm("ws", "model-c")])
    cache = VirtualModelCache()

    await refresh_virtual_model_cache(cache, sdk_first)
    await refresh_virtual_model_cache(cache, sdk_second)

    assert cache.get("ws", "model-a") is None  # removed
    assert cache.get("ws", "model-b") is not None  # retained
    assert cache.get("ws", "model-c") is not None  # added


@pytest.mark.asyncio
async def test_refresh_raises_on_api_connection_error():
    """APIConnectionError is wrapped in VirtualModelCacheRefreshError."""
    sdk = MagicMock(spec=AsyncNeMoHelix)
    paginator = MagicMock()

    async def _raise():
        raise APIConnectionError.__new__(APIConnectionError)
        yield  # make it an async generator

    paginator.__aiter__ = lambda self: _raise()
    sdk.inference.virtual_models.list = MagicMock(return_value=paginator)

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), sdk)


@pytest.mark.asyncio
async def test_refresh_raises_on_api_status_error():
    """APIStatusError is wrapped in VirtualModelCacheRefreshError."""
    sdk = MagicMock(spec=AsyncNeMoHelix)
    mock_response = MagicMock()
    mock_response.status_code = 503
    paginator = MagicMock()

    async def _raise():
        raise APIStatusError("service unavailable", response=mock_response, body={})
        yield

    paginator.__aiter__ = lambda self: _raise()
    sdk.inference.virtual_models.list = MagicMock(return_value=paginator)

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), sdk)


@pytest.mark.asyncio
async def test_refresh_raises_on_unexpected_error():
    """Any unexpected exception is wrapped in VirtualModelCacheRefreshError."""
    sdk = MagicMock(spec=AsyncNeMoHelix)
    paginator = MagicMock()

    async def _raise():
        raise RuntimeError("something exploded")
        yield

    paginator.__aiter__ = lambda self: _raise()
    sdk.inference.virtual_models.list = MagicMock(return_value=paginator)

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), sdk)


@pytest.mark.asyncio
async def test_refresh_does_not_mutate_cache_on_error():
    """If refresh fails, the existing cache entries are preserved."""
    sdk_good = _make_sdk_with_vms([_make_vm("ws", "model-a")])
    cache = VirtualModelCache()
    await refresh_virtual_model_cache(cache, sdk_good)

    # Now simulate a failure on the second refresh
    sdk_bad = MagicMock(spec=AsyncNeMoHelix)
    paginator = MagicMock()

    async def _raise():
        raise RuntimeError("network error")
        yield

    paginator.__aiter__ = lambda self: _raise()
    sdk_bad.inference.virtual_models.list = MagicMock(return_value=paginator)

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(cache, sdk_bad)

    # Original entry still present because rebuild() was never called
    assert cache.get("ws", "model-a") is not None


# ---------------------------------------------------------------------------
# refresh_virtual_model_cache — diff/notify behaviour
# ---------------------------------------------------------------------------


def _make_registry() -> MiddlewareRegistry:
    """Empty registry with all async hooks mocked."""
    registry = MiddlewareRegistry()
    registry.prefetch_configs = AsyncMock(return_value=PrefetchResult())  # type: ignore[method-assign]
    registry.resolve_configs_for_virtual_model = AsyncMock()  # type: ignore[method-assign]
    registry.notify_upserted = AsyncMock()  # type: ignore[method-assign]
    registry.notify_destroyed = AsyncMock()  # type: ignore[method-assign]
    registry.evict = MagicMock()
    return registry


def _make_vm_at(workspace: str, name: str, updated_at: str = "2026-01-01T00:00:00Z") -> VirtualModel:
    return VirtualModel(
        id=f"{workspace}/{name}",
        entity_id=f"{workspace}/{name}",
        name=name,
        workspace=workspace,
        parent=workspace,
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
        default_model_entity=f"{workspace}/{name}",
    )


@pytest.mark.asyncio
async def test_refresh_without_registry_is_backward_compatible():
    """refresh_virtual_model_cache with no registry argument works as before."""
    vms = [_make_vm("ws", "model-a")]
    sdk = _make_sdk_with_vms(vms)
    cache = VirtualModelCache()

    # Must not raise; registry hooks are not called
    await refresh_virtual_model_cache(cache, sdk)
    assert cache.get("ws", "model-a") is not None


@pytest.mark.asyncio
async def test_refresh_calls_resolve_and_notify_upserted_for_added_vms():
    """Newly added VMs get their configs resolved and notify_upserted called."""
    vm = _make_vm_at("ws", "new-vm")
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    registry.prefetch_configs.assert_awaited_once()  # type: ignore[attr-defined]
    call_kwargs = registry.resolve_configs_for_virtual_model.call_args.kwargs  # type: ignore[union-attr]
    assert registry.resolve_configs_for_virtual_model.await_count == 1  # type: ignore[union-attr]
    assert registry.resolve_configs_for_virtual_model.call_args.args == (vm,)  # type: ignore[union-attr]
    assert isinstance(call_kwargs["prefetch"], PrefetchResult)
    registry.notify_upserted.assert_awaited_once_with(vm)
    registry.notify_destroyed.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_calls_resolve_and_notify_upserted_for_changed_vms():
    """VMs with a newer updated_at are treated as changed and re-resolved."""
    vm_old = _make_vm_at("ws", "my-vm", updated_at="2026-01-01T00:00:00Z")
    vm_new = _make_vm_at("ws", "my-vm", updated_at="2026-06-01T00:00:00Z")

    sdk_first = _make_sdk_with_vms([vm_old])
    sdk_second = _make_sdk_with_vms([vm_new])
    cache = VirtualModelCache()
    registry = _make_registry()

    # First refresh — adds vm_old
    await refresh_virtual_model_cache(cache, sdk_first, registry=registry)
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.notify_upserted.reset_mock()

    # Second refresh — vm_new has a newer updated_at → treated as changed
    await refresh_virtual_model_cache(cache, sdk_second, registry=registry)

    assert registry.resolve_configs_for_virtual_model.await_count == 1  # type: ignore[union-attr]
    call = registry.resolve_configs_for_virtual_model.call_args  # type: ignore[union-attr]
    assert call.args == (vm_new,)
    assert isinstance(call.kwargs["prefetch"], PrefetchResult)
    registry.notify_upserted.assert_awaited_once_with(vm_new)


@pytest.mark.asyncio
async def test_refresh_does_not_resolve_unchanged_vms():
    """VMs with the same updated_at and no changed middleware config refs are skipped."""
    vm = _make_vm_at("ws", "stable-vm", updated_at="2026-01-01T00:00:00Z")
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, sdk, registry=registry)
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.notify_upserted.reset_mock()
    registry.prefetch_configs.reset_mock()  # type: ignore[attr-defined]

    # Same VM, same updated_at — no external ref changes from sync_config_ref_versions
    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    registry.resolve_configs_for_virtual_model.assert_not_awaited()
    registry.notify_upserted.assert_not_awaited()
    registry.prefetch_configs.assert_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_calls_evict_and_notify_destroyed_for_removed_vms():
    """VMs absent from the new set get evicted and notify_destroyed called."""
    vm_a = _make_vm_at("ws", "vm-a")
    vm_b = _make_vm_at("ws", "vm-b")

    sdk_with_both = _make_sdk_with_vms([vm_a, vm_b])
    sdk_without_a = _make_sdk_with_vms([vm_b])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, sdk_with_both, registry=registry)
    # Reset all mocks so second-refresh assertions are isolated
    registry.notify_upserted.reset_mock()
    registry.notify_destroyed.reset_mock()
    registry.evict.reset_mock()
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.prefetch_configs.reset_mock()  # type: ignore[attr-defined]

    await refresh_virtual_model_cache(cache, sdk_without_a, registry=registry)

    registry.evict.assert_called_once_with(("ws", "vm-a"))
    registry.notify_destroyed.assert_awaited_once_with(vm_a)
    registry.notify_upserted.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_notification_errors_do_not_fail_refresh():
    """Errors in notify_upserted or notify_destroyed are swallowed — refresh succeeds."""
    vm = _make_vm_at("ws", "vm-a")
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()
    registry.notify_upserted = AsyncMock(side_effect=RuntimeError("hook exploded"))  # type: ignore[method-assign]

    # Must not raise even though the hook raises
    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    # Cache is still updated despite the hook error
    assert cache.get("ws", "vm-a") is not None


@pytest.mark.asyncio
async def test_refresh_resolve_errors_do_not_fail_refresh():
    """Errors in resolve_configs_for_virtual_model are swallowed — refresh succeeds.

    The VM is also evicted and marked broken so the proxy returns 503 rather than
    silently bypassing the middleware chain.
    """
    vm = _make_vm_at("ws", "vm-a")
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()
    registry.resolve_configs_for_virtual_model = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("resolution failed")
    )

    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    assert cache.get("ws", "vm-a") is not None
    registry.evict.assert_called_once_with(("ws", "vm-a"))  # type: ignore[attr-defined]
    assert ("ws", "vm-a") in registry.broken_vms


@pytest.mark.asyncio
async def test_refresh_middleware_references_receives_deduped_config_refs():
    """All unique (plugin, config_type, config_id) triples from the list are de-duplicated."""
    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vms = [
        VirtualModel(
            id="ws/a",
            entity_id="ws/a",
            name="a",
            workspace="ws",
            parent="ws",
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity="ws/m",
            request_middleware=[call],
        ),
        VirtualModel(
            id="ws/b",
            entity_id="ws/b",
            name="b",
            workspace="ws",
            parent="ws",
            db_version=1,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            default_model_entity="ws/m",
            request_middleware=[call],
        ),
    ]
    sdk = _make_sdk_with_vms(vms)
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    registry.prefetch_configs.assert_awaited_once_with({mref})  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_re_resolves_unchanged_vm_when_middleware_config_entity_version_changes():
    """VM updated_at stable but a referenced config entity has a newer version → upsert."""

    class _Entity:
        def __init__(self, ts):
            self.updated_at = ts

    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vm = VirtualModel(
        id="ws/only",
        entity_id="ws/only",
        name="only",
        workspace="ws",
        parent="ws",
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity="ws/m",
        request_middleware=[call],
    )
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    first = PrefetchResult(fetched={mref: _Entity(t1)})
    second = PrefetchResult(fetched={mref: _Entity(t2)})
    registry.prefetch_configs = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    await refresh_virtual_model_cache(cache, sdk, registry=registry)
    await refresh_virtual_model_cache(cache, sdk, registry=registry)

    assert registry.resolve_configs_for_virtual_model.await_count == 2  # type: ignore[union-attr]
    last = registry.resolve_configs_for_virtual_model.call_args  # type: ignore[union-attr]
    assert last.args == (vm,)
    assert last.kwargs["prefetch"] is second


@pytest.mark.asyncio
async def test_refresh_marks_vm_broken_when_config_deleted_and_clears_on_recreate():
    """End-to-end deletion → broken → recreate → healthy round trip through `refresh_virtual_model_cache`."""

    class _Entity:
        def __init__(self, ts):
            self.updated_at = ts

    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vm = VirtualModel(
        id="ws/only",
        entity_id="ws/only",
        name="only",
        workspace="ws",
        parent="ws",
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity="ws/m",
        request_middleware=[call],
    )
    sdk = _make_sdk_with_vms([vm])
    cache = VirtualModelCache()

    # Use a real registry so the actual broken_vms / resolve interaction runs.
    plugin = MagicMock()
    plugin.validate_middleware_config = AsyncMock(side_effect=lambda ct, c: c)
    plugin.on_virtual_model_upserted = AsyncMock()
    plugin.on_virtual_model_destroyed = AsyncMock()
    registry = MiddlewareRegistry(plugins={"my-plugin": plugin})

    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    healthy = PrefetchResult(fetched={mref: _Entity(t1)})
    deleted = PrefetchResult(missing={mref})
    recreated = PrefetchResult(fetched={mref: _Entity(t2)})
    registry.prefetch_configs = AsyncMock(side_effect=[healthy, deleted, recreated])  # type: ignore[method-assign]

    # Cycle 1: healthy — VM resolves cleanly.
    await refresh_virtual_model_cache(cache, sdk, registry=registry)
    assert ("ws", "only") in registry.request_middleware_calls
    assert ("ws", "only") not in registry.broken_vms

    # Cycle 2: config deleted upstream — VM should be evicted and flagged broken.
    await refresh_virtual_model_cache(cache, sdk, registry=registry)
    assert ("ws", "only") not in registry.request_middleware_calls
    assert ("ws", "only") in registry.broken_vms

    # Cycle 3: config recreated — VM should recover.
    await refresh_virtual_model_cache(cache, sdk, registry=registry)
    assert ("ws", "only") in registry.request_middleware_calls
    assert ("ws", "only") not in registry.broken_vms


@pytest.mark.asyncio
async def test_refresh_removed_vm_clears_broken_state():
    """Deleting the VM itself purges both resolved state and any broken flag."""
    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vm = VirtualModel(
        id="ws/only",
        entity_id="ws/only",
        name="only",
        workspace="ws",
        parent="ws",
        db_version=1,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        default_model_entity="ws/m",
        request_middleware=[call],
    )
    cache = VirtualModelCache()
    registry = _make_registry()
    # Seed broken state so we can verify cleanup.
    registry.broken_vms.add(("ws", "only"))
    cache.rebuild([vm])
    cache.config_ref_versions[mref] = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Second refresh returns no VMs at all (the VM was deleted).
    sdk_empty = _make_sdk_with_vms([])
    await refresh_virtual_model_cache(cache, sdk_empty, registry=registry)

    registry.evict.assert_called_once_with(("ws", "only"))  # type: ignore[attr-defined]
    assert ("ws", "only") not in registry.broken_vms


# ---------------------------------------------------------------------------
# sync_config_ref_versions (per-ref version state on VirtualModelCache)
# ---------------------------------------------------------------------------


class _VCEntity:
    def __init__(self, updated_at: datetime) -> None:
        self.updated_at = updated_at


class TestApplyMiddlewareConfigRefFetches:
    def test_first_success_marks_ref_changed(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state: dict = {}
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: _VCEntity(t1)}))
        assert changed == {mref}
        assert state[mref] == t1

    def test_same_version_second_time_not_in_changed(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        ent = _VCEntity(t1)
        state: dict = {}
        c1 = sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: ent}))
        assert c1 == {mref}
        c2 = sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: ent}))
        assert c2 == set()
        assert state[mref] == t1

    def test_bumped_updated_at_marks_changed(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state: dict = {}
        sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: _VCEntity(t1)}))
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: _VCEntity(t2)}))
        assert changed == {mref}
        assert state[mref] == t2

    def test_prune_drops_unused_refs(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state = {mref: t1}
        sync_config_ref_versions(state, set(), PrefetchResult())
        assert mref not in state

    def test_missing_ref_marks_changed_and_prunes_state(self):
        """A ref reported as deleted is flagged as changed and its version is dropped.

        Pruning is important: recreating the config later must look "newly seen"
        so a subsequent ``fetched`` observation re-enters the changed set even
        if the recreated entity carries a stale-looking ``updated_at``.
        """
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state = {mref: t1}
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(missing={mref}))
        assert changed == {mref}
        assert mref not in state

    def test_already_missing_ref_does_not_rechurn(self):
        """Second consecutive missing cycle is a no-op — no changed flag, no state mutation.

        Once a deletion has been processed (ref pruned, VMs marked broken), subsequent
        cycles seeing the same ref as missing should not re-flag it. The dependent VMs
        are already broken and re-resolving them every poll cycle would only produce
        redundant warning logs.
        """
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state = {mref: t1}
        # Cycle 1: first deletion — flagged as changed, pruned from state.
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(missing={mref}))
        assert changed == {mref}
        assert mref not in state
        # Cycle 2: still deleted — already absent from state, so no churn.
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(missing={mref}))
        assert changed == set()
        assert mref not in state

    def test_transient_ref_leaves_known_versions_alone(self):
        """A transient ref must not toggle the changed set or wipe its prior version.

        This is the "don't flap on transient failures" invariant: a brief
        network blip during prefetch must not cascade into a re-resolve of
        every dependent VirtualModel.
        """
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state = {mref: t1}
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(transient={mref}))
        assert changed == set()
        assert state[mref] == t1

    def test_recreate_after_delete_re_emerges_as_changed(self):
        """Deletion + recreation flow: missing prunes, fresh fetch re-emerges as changed."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        mref = MiddlewareConfigRef("p1", "my_config", "ws/cfg")
        state: dict = {}
        sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: _VCEntity(t1)}))
        assert state[mref] == t1

        # Delete
        sync_config_ref_versions(state, {mref}, PrefetchResult(missing={mref}))
        assert mref not in state

        # Recreate — even if the new updated_at happens to equal the original t1,
        # the prune step above means this is "newly seen" so it flags as changed.
        changed = sync_config_ref_versions(state, {mref}, PrefetchResult(fetched={mref: _VCEntity(t2)}))
        assert changed == {mref}
        assert state[mref] == t2

    def test_mixed_buckets_partition_correctly(self):
        """A single call with all three buckets returns only fetched+missing as changed."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ok = MiddlewareConfigRef("p1", "x", "ws/ok")
        gone = MiddlewareConfigRef("p1", "x", "ws/gone")
        flake = MiddlewareConfigRef("p1", "x", "ws/flake")

        state = {gone: t1, flake: t1}
        changed = sync_config_ref_versions(
            state,
            {ok, gone, flake},
            PrefetchResult(fetched={ok: _VCEntity(t1)}, missing={gone}, transient={flake}),
        )
        assert changed == {ok, gone}
        assert state == {ok: t1, flake: t1}
