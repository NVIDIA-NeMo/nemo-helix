# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VirtualModelCache and refresh_virtual_model_cache."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.inference_middleware_models import MiddlewareCall, VirtualModel
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
from nhx.testing.client import mock_async_nemo_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vm_at(workspace: str, name: str, updated_at: str = "2026-01-01T00:00:00Z", **fields: object) -> VirtualModel:
    """Build a VirtualModel the way an API response does, carrying entity-store metadata."""
    return VirtualModel.model_validate(
        {
            "id": f"{workspace}/{name}",
            "name": name,
            "workspace": workspace,
            "parent": workspace,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": updated_at,
            "default_model_entity": f"{workspace}/{name}",
            **fields,
        }
    )


def _make_vm(workspace: str, name: str, default_model_entity: str | None = None) -> VirtualModel:
    return _make_vm_at(workspace, name, default_model_entity=default_model_entity or f"{workspace}/{name}")


def _make_client_with_vms(vms: list[VirtualModel]) -> AsyncNemoClient:
    """Return a client whose VirtualModel list endpoint serves *vms* in one page."""

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/workspaces/-/virtual-models")
        assert request.url.params["page_size"] == "200"
        return httpx.Response(
            200,
            json={
                "data": [vm.model_dump(mode="json") for vm in vms],
                "pagination": {
                    "page": 1,
                    "page_size": 200,
                    "current_page_size": len(vms),
                    "total_pages": 1,
                    "total_results": len(vms),
                },
            },
        )

    return mock_async_nemo_client(_handler, AsyncNemoClient)


def _make_failing_client(handler_error: Exception | int) -> AsyncNemoClient:
    """Return a client whose every request raises *handler_error* or answers with that HTTP status."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if isinstance(handler_error, int):
            return httpx.Response(handler_error, json={"detail": "failure"})
        raise handler_error

    return mock_async_nemo_client(_handler, AsyncNemoClient)


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
    client = _make_client_with_vms(vms)
    cache = VirtualModelCache()

    await refresh_virtual_model_cache(cache, client)

    assert cache.get("ws", "model-a") is not None
    assert cache.get("ws", "model-b") is not None


@pytest.mark.asyncio
async def test_refresh_empty_result_clears_cache():
    """Refresh with no VMs leaves cache empty."""
    client = _make_client_with_vms([])
    cache = VirtualModelCache()
    cache.rebuild([_make_vm("ws", "stale")])

    await refresh_virtual_model_cache(cache, client)

    assert cache.get("ws", "stale") is None


@pytest.mark.asyncio
async def test_refresh_replaces_stale_entries():
    """Second refresh with a different set removes the old entries."""
    client_first = _make_client_with_vms([_make_vm("ws", "model-a"), _make_vm("ws", "model-b")])
    client_second = _make_client_with_vms([_make_vm("ws", "model-b"), _make_vm("ws", "model-c")])
    cache = VirtualModelCache()

    await refresh_virtual_model_cache(cache, client_first)
    await refresh_virtual_model_cache(cache, client_second)

    assert cache.get("ws", "model-a") is None  # removed
    assert cache.get("ws", "model-b") is not None  # retained
    assert cache.get("ws", "model-c") is not None  # added


@pytest.mark.asyncio
async def test_refresh_raises_on_transport_error():
    """Transport errors are wrapped in VirtualModelCacheRefreshError."""
    client = _make_failing_client(httpx.ConnectError("connection refused"))

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), client)


@pytest.mark.asyncio
async def test_refresh_raises_on_http_status_error():
    """HTTP error responses are wrapped in VirtualModelCacheRefreshError."""
    client = _make_failing_client(503)

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), client)


@pytest.mark.asyncio
async def test_refresh_raises_on_unexpected_error():
    """Any unexpected exception is wrapped in VirtualModelCacheRefreshError."""
    client = _make_failing_client(RuntimeError("something exploded"))

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(VirtualModelCache(), client)


@pytest.mark.asyncio
async def test_refresh_does_not_mutate_cache_on_error():
    """If refresh fails, the existing cache entries are preserved."""
    good_client = _make_client_with_vms([_make_vm("ws", "model-a")])
    cache = VirtualModelCache()
    await refresh_virtual_model_cache(cache, good_client)

    # Now simulate a failure on the second refresh
    bad_client = _make_failing_client(RuntimeError("network error"))

    with pytest.raises(VirtualModelCacheRefreshError):
        await refresh_virtual_model_cache(cache, bad_client)

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


@pytest.mark.asyncio
async def test_refresh_without_registry_is_backward_compatible():
    """refresh_virtual_model_cache with no registry argument works as before."""
    vms = [_make_vm("ws", "model-a")]
    client = _make_client_with_vms(vms)
    cache = VirtualModelCache()

    # Must not raise; registry hooks are not called
    await refresh_virtual_model_cache(cache, client)
    assert cache.get("ws", "model-a") is not None


@pytest.mark.asyncio
async def test_refresh_calls_resolve_and_notify_upserted_for_added_vms():
    """Newly added VMs get their configs resolved and notify_upserted called."""
    vm = _make_vm_at("ws", "new-vm")
    client = _make_client_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, client, registry=registry)

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

    client_first = _make_client_with_vms([vm_old])
    client_second = _make_client_with_vms([vm_new])
    cache = VirtualModelCache()
    registry = _make_registry()

    # First refresh — adds vm_old
    await refresh_virtual_model_cache(cache, client_first, registry=registry)
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.notify_upserted.reset_mock()

    # Second refresh — vm_new has a newer updated_at → treated as changed
    await refresh_virtual_model_cache(cache, client_second, registry=registry)

    assert registry.resolve_configs_for_virtual_model.await_count == 1  # type: ignore[union-attr]
    call = registry.resolve_configs_for_virtual_model.call_args  # type: ignore[union-attr]
    assert call.args == (vm_new,)
    assert isinstance(call.kwargs["prefetch"], PrefetchResult)
    registry.notify_upserted.assert_awaited_once_with(vm_new)


@pytest.mark.asyncio
async def test_refresh_does_not_resolve_unchanged_vms():
    """VMs with the same updated_at and no changed middleware config refs are skipped."""
    vm = _make_vm_at("ws", "stable-vm", updated_at="2026-01-01T00:00:00Z")
    client = _make_client_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, client, registry=registry)
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.notify_upserted.reset_mock()
    registry.prefetch_configs.reset_mock()  # type: ignore[attr-defined]

    # Same VM, same updated_at — no external ref changes from sync_config_ref_versions
    await refresh_virtual_model_cache(cache, client, registry=registry)

    registry.resolve_configs_for_virtual_model.assert_not_awaited()
    registry.notify_upserted.assert_not_awaited()
    registry.prefetch_configs.assert_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_calls_evict_and_notify_destroyed_for_removed_vms():
    """VMs absent from the new set get evicted and notify_destroyed called."""
    vm_a = _make_vm_at("ws", "vm-a")
    vm_b = _make_vm_at("ws", "vm-b")

    client_with_both = _make_client_with_vms([vm_a, vm_b])
    client_without_a = _make_client_with_vms([vm_b])
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, client_with_both, registry=registry)
    # Reset all mocks so second-refresh assertions are isolated
    registry.notify_upserted.reset_mock()
    registry.notify_destroyed.reset_mock()
    registry.evict.reset_mock()
    registry.resolve_configs_for_virtual_model.reset_mock()
    registry.prefetch_configs.reset_mock()  # type: ignore[attr-defined]

    await refresh_virtual_model_cache(cache, client_without_a, registry=registry)

    registry.evict.assert_called_once_with(("ws", "vm-a"))
    registry.notify_destroyed.assert_awaited_once_with(vm_a)
    registry.notify_upserted.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_notification_errors_do_not_fail_refresh():
    """Errors in notify_upserted or notify_destroyed are swallowed — refresh succeeds."""
    vm = _make_vm_at("ws", "vm-a")
    client = _make_client_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()
    registry.notify_upserted = AsyncMock(side_effect=RuntimeError("hook exploded"))  # type: ignore[method-assign]

    # Must not raise even though the hook raises
    await refresh_virtual_model_cache(cache, client, registry=registry)

    # Cache is still updated despite the hook error
    assert cache.get("ws", "vm-a") is not None


@pytest.mark.asyncio
async def test_refresh_resolve_errors_do_not_fail_refresh():
    """Errors in resolve_configs_for_virtual_model are swallowed — refresh succeeds.

    The VM is also evicted and marked broken so the proxy returns 503 rather than
    silently bypassing the middleware chain.
    """
    vm = _make_vm_at("ws", "vm-a")
    client = _make_client_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()
    registry.resolve_configs_for_virtual_model = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("resolution failed")
    )

    await refresh_virtual_model_cache(cache, client, registry=registry)

    assert cache.get("ws", "vm-a") is not None
    registry.evict.assert_called_once_with(("ws", "vm-a"))  # type: ignore[attr-defined]
    assert ("ws", "vm-a") in registry.broken_vms


@pytest.mark.asyncio
async def test_refresh_middleware_references_receives_deduped_config_refs():
    """All unique (plugin, config_type, config_id) triples from the list are de-duplicated."""
    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vms = [
        _make_vm_at("ws", "a", default_model_entity="ws/m", request_middleware=[call]),
        _make_vm_at("ws", "b", default_model_entity="ws/m", request_middleware=[call]),
    ]
    client = _make_client_with_vms(vms)
    cache = VirtualModelCache()
    registry = _make_registry()

    await refresh_virtual_model_cache(cache, client, registry=registry)

    registry.prefetch_configs.assert_awaited_once_with({mref})  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refresh_re_resolves_unchanged_vm_when_middleware_config_entity_version_changes():
    """VM updated_at stable but a referenced config entity has a newer version → upsert."""

    class _Entity:
        def __init__(self, ts):
            self.updated_at = ts

    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vm = _make_vm_at("ws", "only", default_model_entity="ws/m", request_middleware=[call])
    client = _make_client_with_vms([vm])
    cache = VirtualModelCache()
    registry = _make_registry()

    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    first = PrefetchResult(fetched={mref: _Entity(t1)})
    second = PrefetchResult(fetched={mref: _Entity(t2)})
    registry.prefetch_configs = AsyncMock(side_effect=[first, second])  # type: ignore[method-assign]

    await refresh_virtual_model_cache(cache, client, registry=registry)
    await refresh_virtual_model_cache(cache, client, registry=registry)

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
    vm = _make_vm_at("ws", "only", default_model_entity="ws/m", request_middleware=[call])
    client = _make_client_with_vms([vm])
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
    await refresh_virtual_model_cache(cache, client, registry=registry)
    assert ("ws", "only") in registry.request_middleware_calls
    assert ("ws", "only") not in registry.broken_vms

    # Cycle 2: config deleted upstream — VM should be evicted and flagged broken.
    await refresh_virtual_model_cache(cache, client, registry=registry)
    assert ("ws", "only") not in registry.request_middleware_calls
    assert ("ws", "only") in registry.broken_vms

    # Cycle 3: config recreated — VM should recover.
    await refresh_virtual_model_cache(cache, client, registry=registry)
    assert ("ws", "only") in registry.request_middleware_calls
    assert ("ws", "only") not in registry.broken_vms


@pytest.mark.asyncio
async def test_refresh_removed_vm_clears_broken_state():
    """Deleting the VM itself purges both resolved state and any broken flag."""
    mref = MiddlewareConfigRef("my-plugin", "gcfg", "ws/cfg-1")
    call = MiddlewareCall(name="my-plugin", config_type="gcfg", config_id="ws/cfg-1", config=None)
    vm = _make_vm_at("ws", "only", default_model_entity="ws/m", request_middleware=[call])
    cache = VirtualModelCache()
    registry = _make_registry()
    # Seed broken state so we can verify cleanup.
    registry.broken_vms.add(("ws", "only"))
    cache.rebuild([vm])
    cache.config_ref_versions[mref] = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Second refresh returns no VMs at all (the VM was deleted).
    client_empty = _make_client_with_vms([])
    await refresh_virtual_model_cache(cache, client_empty, registry=registry)

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
