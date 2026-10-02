# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelEntityCache."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.models.types import CreateModelEntityRequest, UpdateModelEntityRequest
from nhx.core.models.controllers.entity_cache import ModelEntityCache, UnflushedMutationsError

from .conftest import (
    _AsyncPage,
    _ModelResponse,
    _status_error,
    make_async_models_client,
    make_entity,
    seed_entity_cache,
)


def _entity(workspace="ws", name="model", model_providers=None, **attrs):
    return make_entity(workspace, name, model_providers=model_providers, **attrs)


@pytest.fixture
async def mock_models_client():
    return make_async_models_client()


@pytest.fixture
async def patch_models_client(mock_models_client):
    """Route ``client_from_platform(sdk, AsyncModelsClient)`` in the entity_cache
    module back to :data:`mock_models_client`."""
    with patch("nhx.core.models.controllers.entity_cache.client_from_platform", return_value=mock_models_client):
        yield mock_models_client


@pytest.fixture
async def mock_models_sdk(mock_models_client, patch_models_client):
    sdk = MagicMock(spec=AsyncNeMoHelix)
    sdk.models_client = mock_models_client
    return sdk


@pytest.fixture
def heartbeat_calls():
    """Collects heartbeat emissions so tests can assert progress was reported."""
    return []


@pytest.fixture
def cache(mock_models_sdk, heartbeat_calls):
    return ModelEntityCache(models_sdk=mock_models_sdk, emit_heartbeat=lambda: heartbeat_calls.append(1))


async def _load(mock_models_sdk, cache, entities=()):
    await seed_entity_cache(mock_models_sdk, cache, entities)


def _page(items):
    return _AsyncPage(list(items))


def _resp(value=None, exc=None):
    return _ModelResponse(value=value, exc=exc)


def _configure_models_client(mock_models_sdk, **kwargs):
    """Configure the mock typed client's methods; ``create_model``/``get_model``/
    ``update_model``/``list_models`` return ``_ModelResponse`` / ``_AsyncPage``
    wrappers unless overridden here."""
    client = mock_models_sdk.models_client
    if "list_models" in kwargs:
        client.list_models = kwargs["list_models"]
    if "create_model" in kwargs:
        client.create_model = kwargs["create_model"]
    if "get_model" in kwargs:
        client.get_model = kwargs["get_model"]
    if "update_model" in kwargs:
        client.update_model = kwargs["update_model"]
    return client


@pytest.mark.asyncio
async def test_refresh_loads_entities_keyed_by_workspace_and_name(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws-a", "m1"), _entity("ws-b", "m1")])

    assert cache.loaded
    assert cache.get("ws-a", "m1") is not None
    assert cache.get("ws-b", "m1") is not None
    assert cache.get("ws-c", "m1") is None


@pytest.mark.asyncio
async def test_refresh_rejects_unflushed_mutations(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache)
    cache.stage_provider_link("ws", "model", "ws/p1")

    with pytest.raises(UnflushedMutationsError):
        await cache.refresh()


@pytest.mark.asyncio
async def test_get_reflects_staged_provider_link_within_a_phase(mock_models_sdk, cache):
    """A read after a stage in the same phase must observe the staged change."""
    await _load(mock_models_sdk, cache, [_entity("ws", "model", ["ws/p1"])])

    cache.stage_provider_link("ws", "model", "ws/p2")

    assert cache.get("ws", "model").model_providers == ["ws/p1", "ws/p2"]


@pytest.mark.asyncio
async def test_get_reflects_staged_provider_unlink_within_a_phase(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws", "model", ["ws/p1", "ws/p2"])])

    cache.stage_provider_unlink("ws", "model", "ws/p1")

    assert cache.get("ws", "model").model_providers == ["ws/p2"]


@pytest.mark.asyncio
async def test_entity_staged_for_creation_still_reads_as_absent(mock_models_sdk, cache):
    """A staged creation is not fabricated, so callers keep treating it as new."""
    await _load(mock_models_sdk, cache)

    cache.stage_create("ws", "new-model", description="d", backend_format="OPENAI_CHAT")
    cache.stage_provider_link("ws", "new-model", "ws/p1")

    assert cache.get("ws", "new-model") is None


@pytest.mark.asyncio
async def test_multiple_providers_produce_a_single_update(mock_models_sdk, cache):
    """An entity linked by several providers is written once, not once per provider."""
    await _load(mock_models_sdk, cache, [_entity("ws", "model", [])])
    client = _configure_models_client(
        mock_models_sdk, update_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/p1", "ws/p2"])))
    )

    cache.stage_provider_link("ws", "model", "ws/p1")
    cache.stage_provider_link("ws", "model", "ws/p2")
    await cache.flush()

    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "model"
    assert call.kwargs["body"].model_providers == ["ws/p1", "ws/p2"]


@pytest.mark.asyncio
async def test_no_write_when_already_converged(mock_models_sdk, cache):
    """Staging state that already matches the entity performs no write."""
    await _load(mock_models_sdk, cache, [_entity("ws", "model", ["ws/p1"])])
    client = _configure_models_client(mock_models_sdk)

    cache.stage_provider_link("ws", "model", "ws/p1")
    await cache.flush()

    client.update_model.assert_not_awaited()
    client.create_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_two_providers_creating_the_same_entity_collapse_to_one_create(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache)
    client = _configure_models_client(
        mock_models_sdk,
        create_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/p1", "ws/p2"]))),
    )

    cache.stage_create("ws", "model", description="from p1", backend_format="OPENAI_CHAT")
    cache.stage_provider_link("ws", "model", "ws/p1")
    cache.stage_create("ws", "model", description="from p2", backend_format="OPENAI_CHAT")
    cache.stage_provider_link("ws", "model", "ws/p2")
    await cache.flush()

    client.create_model.assert_awaited_once()
    call = client.create_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    body: CreateModelEntityRequest = call.kwargs["body"]
    assert body.name == "model"
    assert body.description == "from p1"
    assert body.backend_format == "OPENAI_CHAT"
    assert body.model_providers == ["ws/p1", "ws/p2"]


@pytest.mark.asyncio
async def test_create_conflict_falls_back_to_updating_the_existing_entity(mock_models_sdk, cache):
    """An entity created concurrently is adopted rather than reported as an error."""
    await _load(mock_models_sdk, cache)
    client = _configure_models_client(
        mock_models_sdk,
        create_model=AsyncMock(side_effect=_status_error(409, "exists")),
        get_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/other"]))),
        update_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/other", "ws/p1"]))),
    )

    cache.stage_create("ws", "model", description="d", backend_format="OPENAI_CHAT")
    cache.stage_provider_link("ws", "model", "ws/p1")
    await cache.flush()

    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "model"
    assert call.kwargs["body"].model_providers == ["ws/other", "ws/p1"]


@pytest.mark.asyncio
async def test_create_conflict_with_vanished_entity_is_ignored(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache)
    client = _configure_models_client(
        mock_models_sdk,
        create_model=AsyncMock(side_effect=_status_error(409, "exists")),
        get_model=AsyncMock(side_effect=_status_error(404, "gone")),
    )

    cache.stage_create("ws", "model", description="d")
    await cache.flush()

    client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_failing_entity_does_not_stop_the_others(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", []), _entity("ws", "m2", [])])
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(side_effect=[Exception("boom"), _resp(_entity("ws", "m2", ["ws/p1"]))]),
    )

    cache.stage_provider_link("ws", "m1", "ws/p1")
    cache.stage_provider_link("ws", "m2", "ws/p1")
    await cache.flush()

    assert client.update_model.await_count == 2


@pytest.mark.asyncio
async def test_failed_write_is_kept_for_retry_and_succeeds_later(mock_models_sdk, cache):
    """A write that fails must not be lost.

    Some staged changes cannot be recomputed by a later pass -- unlinking a provider
    that is being deleted is derived from that provider -- so a dropped failure
    would leave the entity permanently inconsistent.
    """
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", ["ws/p1"]), _entity("ws", "m2", ["ws/p1"])])
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(side_effect=[Exception("boom"), _resp(_entity("ws", "m2", []))]),
    )

    cache.stage_provider_unlink("ws", "m1", "ws/p1")
    cache.stage_provider_unlink("ws", "m2", "ws/p1")
    await cache.flush()

    # The successful entity is done; the failed one is still staged.
    assert cache.get("ws", "m1").model_providers == []
    assert ("ws", "m1") in cache._pending
    assert ("ws", "m2") not in cache._pending

    # A later flush retries it, and this time it lands.
    client.update_model = AsyncMock(return_value=_resp(_entity("ws", "m1", [])))
    await cache.flush()

    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "m1"
    assert call.kwargs["body"].model_providers == []
    assert cache._pending == {}


@pytest.mark.asyncio
async def test_refresh_allows_retained_failures_but_still_rejects_unflushed_work(mock_models_sdk, cache):
    """Refresh distinguishes "flushed and failed" from "staged and forgotten"."""
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", ["ws/p1"])])
    _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(side_effect=Exception("boom")),
    )

    cache.stage_provider_unlink("ws", "m1", "ws/p1")
    await cache.flush()
    assert ("ws", "m1") in cache._pending

    # A retained failure does not block the next phase from re-reading.
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", ["ws/p1"])])

    # Work that no flush has attempted still does.
    cache.stage_provider_link("ws", "m2", "ws/p2")
    with pytest.raises(UnflushedMutationsError):
        await cache.refresh()


@pytest.mark.asyncio
async def test_retained_failure_replays_against_a_newer_snapshot(mock_models_sdk, cache):
    """Staged changes are differences, so replaying them after a refresh stays correct."""
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", ["ws/p1", "ws/p2"])])
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(side_effect=Exception("boom")),
    )

    cache.stage_provider_unlink("ws", "m1", "ws/p1")
    await cache.flush()

    # Snapshot moves on: another writer added a third provider meanwhile.
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", ["ws/p1", "ws/p2", "ws/p3"])])
    client.update_model = AsyncMock(return_value=_resp(_entity("ws", "m1", ["ws/p2"])))
    await cache.flush()

    # The unlink applies to the newer state rather than reinstating the old list.
    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "m1"
    assert call.kwargs["body"].model_providers == ["ws/p2", "ws/p3"]


@pytest.mark.asyncio
async def test_flush_clears_staged_changes(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws", "model", [])])
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/p1"]))),
    )

    cache.stage_provider_link("ws", "model", "ws/p1")
    await cache.flush()
    client.update_model.reset_mock()

    # Nothing left staged, so a second flush writes nothing and a refresh is allowed.
    await cache.flush()
    client.update_model.assert_not_awaited()
    await cache.refresh()


@pytest.mark.asyncio
async def test_link_then_unlink_for_the_same_provider_cancels_out(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws", "model", ["ws/p1"])])
    client = _configure_models_client(mock_models_sdk)

    cache.stage_provider_unlink("ws", "model", "ws/p1")
    cache.stage_provider_link("ws", "model", "ws/p1")
    await cache.flush()

    client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_field_updates_are_written_as_staged(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache, [_entity("ws", "model", ["ws/p1"])])
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/p1"], fileset="hub/model"))),
    )

    cache.stage_field_updates("ws", "model", fileset="hub/model", api_endpoint=None)
    await cache.flush()

    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "model"
    assert call.kwargs["body"].fileset == "hub/model"


@pytest.mark.asyncio
async def test_staged_change_for_missing_entity_without_create_is_skipped(mock_models_sdk, cache):
    await _load(mock_models_sdk, cache)
    client = _configure_models_client(mock_models_sdk)

    cache.stage_provider_unlink("ws", "ghost", "ws/p1")
    await cache.flush()

    client.create_model.assert_not_awaited()
    client.update_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_after_flush_does_not_reapply_earlier_state(mock_models_sdk, cache):
    """A link removed in one phase is not reinstated by the next phase.

    The second phase must decide from a snapshot that already reflects the first
    phase's writes, otherwise it would re-add what was just removed.
    """
    store = {("ws", "model"): _entity("ws", "model", ["ws/p1"])}
    client = _configure_models_client(mock_models_sdk)

    async def _list_models(**kwargs):
        return _page(list(store.values()))

    async def _update_model(**kwargs):
        body: UpdateModelEntityRequest = kwargs["body"]
        workspace, name = kwargs["workspace"], kwargs["name"]
        current = store[(workspace, name)]
        store[(workspace, name)] = _entity(
            workspace, name, body.model_providers if body.model_providers is not None else current.model_providers
        )
        return _resp(store[(workspace, name)])

    client.list_models = _list_models
    client.update_model = _update_model

    # Phase one removes the provider link and applies it.
    await cache.refresh()
    cache.stage_provider_unlink("ws", "model", "ws/p1")
    await cache.flush()
    assert store[("ws", "model")].model_providers == []

    # Phase two re-reads, so it sees the removal instead of the stale link.
    await cache.refresh()
    assert cache.get("ws", "model").model_providers == []


@pytest.mark.asyncio
async def test_refresh_reports_progress_per_entity_read(mock_models_sdk, cache, heartbeat_calls):
    """Reading a large batch has to report progress as it goes."""
    await _load(mock_models_sdk, cache, [_entity("ws", f"m{i}") for i in range(25)])

    assert len(heartbeat_calls) == 25


@pytest.mark.asyncio
async def test_flush_reports_progress_per_entity_written(mock_models_sdk, cache, heartbeat_calls):
    """Writing a large batch has to report progress as it goes.

    Writes are the slowest part of a pass, so a flush that reported nothing would
    make a long but advancing pass indistinguishable from a stalled one.
    """
    await _load(mock_models_sdk, cache, [_entity("ws", f"m{i}", []) for i in range(25)])
    heartbeat_calls.clear()
    client = _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(return_value=_resp(_entity("ws", "m0", ["ws/p1"]))),
    )

    for i in range(25):
        cache.stage_provider_link("ws", f"m{i}", "ws/p1")
    await cache.flush()

    assert client.update_model.await_count == 25
    assert len(heartbeat_calls) == 25


@pytest.mark.asyncio
async def test_flush_reports_progress_even_when_an_entity_write_fails(mock_models_sdk, cache, heartbeat_calls):
    """Moving past a failed entity is still progress."""
    await _load(mock_models_sdk, cache, [_entity("ws", "m1", []), _entity("ws", "m2", [])])
    _configure_models_client(
        mock_models_sdk,
        update_model=AsyncMock(side_effect=[Exception("boom"), _resp(_entity("ws", "m2", ["ws/p1"]))]),
    )
    heartbeat_calls.clear()

    cache.stage_provider_link("ws", "m1", "ws/p1")
    cache.stage_provider_link("ws", "m2", "ws/p1")
    await cache.flush()

    assert len(heartbeat_calls) == 2


@pytest.mark.asyncio
async def test_conflict_adoption_does_not_overwrite_the_existing_entity_attributes(mock_models_sdk, cache):
    """Adopting a concurrently-created entity leaves its own attributes alone.

    Attributes supplied for creation describe an entity we expected to create. When
    another writer got there first, theirs win; the owning reconciler re-evaluates
    what is still missing on a later pass.
    """
    await _load(mock_models_sdk, cache)
    client = _configure_models_client(
        mock_models_sdk,
        create_model=AsyncMock(side_effect=_status_error(409, "exists")),
        get_model=AsyncMock(
            return_value=_resp(_entity("ws", "model", ["ws/other"], backend_format="ANTHROPIC_MESSAGES"))
        ),
        update_model=AsyncMock(return_value=_resp(_entity("ws", "model", ["ws/other", "ws/p1"]))),
    )

    cache.stage_create("ws", "model", description="ours", backend_format="OPENAI_CHAT")
    cache.stage_provider_link("ws", "model", "ws/p1")
    await cache.flush()

    # Only the provider link is written; description/backend_format are not forced.
    client.update_model.assert_awaited_once()
    call = client.update_model.await_args
    assert call is not None
    assert call.kwargs["workspace"] == "ws"
    assert call.kwargs["name"] == "model"
    assert call.kwargs["body"].model_providers == ["ws/other", "ws/p1"]
