# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.common.config import Configuration
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.service import AuthService


@pytest.fixture(autouse=True)
def clear_config_overrides():
    yield
    Configuration.clear_override(AuthServiceConfig)


@pytest.mark.asyncio
async def test_on_startup_starts_owned_embedded_policy_off_loop(monkeypatch: pytest.MonkeyPatch):
    calls: list[None] = []
    Configuration.set_override(
        AuthServiceConfig(
            enabled=True,
            policy_decision_point_provider="embedded",
        )
    )
    loop_thread = threading.get_ident()

    def start():
        assert threading.get_ident() != loop_thread
        calls.append(None)

    monkeypatch.setattr("nhx.core.auth.service.start_policy", start)
    service = AuthService()
    await service.on_startup()

    assert calls == [None]
    assert service._owns_policy


@pytest.mark.parametrize(
    "config",
    [
        AuthServiceConfig(enabled=False, policy_decision_point_provider="embedded"),
        AuthServiceConfig(enabled=True, policy_decision_point_provider="opa"),
    ],
)
@pytest.mark.asyncio
async def test_on_startup_skips_policy_preflight_when_not_needed(
    config: AuthServiceConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[None] = []
    Configuration.set_override(config)
    monkeypatch.setattr("nhx.core.auth.service.start_policy", lambda: calls.append(None))

    await AuthService().on_startup()

    assert calls == []


def _block_policy_refresh(monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, asyncio.Event, list[str]]:
    """Replace the policy refresh with one that blocks until released and records how it ended."""
    started = asyncio.Event()
    release = asyncio.Event()
    outcomes: list[str] = []

    async def refresh(service_client: object, *, skip_static_bootstrap: bool) -> dict[str, object]:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            outcomes.append("cancelled")
            raise
        outcomes.append("completed")
        return {}

    monkeypatch.setattr("nhx.core.auth.service.apply_embedded_policy_document", refresh)
    return started, release, outcomes


async def _start_refresh_loop(monkeypatch: pytest.MonkeyPatch) -> AuthService:
    config = AuthServiceConfig(enabled=True, policy_decision_point_provider="embedded")
    Configuration.set_override(config)
    service = AuthService().with_config(config)
    monkeypatch.setattr(service.dependency_provider, "get_service_entity_client", lambda name: object())
    await service.startup()
    return service


@pytest.mark.asyncio
async def test_on_shutdown_lets_an_in_flight_policy_refresh_finish(monkeypatch: pytest.MonkeyPatch):
    # Cancelling a refresh mid-request cancels the entity store query it is waiting on, which can
    # leave SQLAlchemy unable to return that connection to the pool.
    started, release, outcomes = _block_policy_refresh(monkeypatch)
    service = await _start_refresh_loop(monkeypatch)
    await started.wait()

    shutdown = asyncio.create_task(service.on_shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()

    release.set()
    await shutdown

    assert outcomes == ["completed"]


@pytest.mark.asyncio
async def test_on_shutdown_cancels_a_policy_refresh_that_does_not_finish(monkeypatch: pytest.MonkeyPatch):
    started, _release, outcomes = _block_policy_refresh(monkeypatch)
    monkeypatch.setattr("nhx.core.auth.service._REFRESH_STOP_TIMEOUT_SECONDS", 0.01)
    service = await _start_refresh_loop(monkeypatch)
    await started.wait()

    await service.on_shutdown()

    assert outcomes == ["cancelled"]


@pytest.mark.asyncio
async def test_shutdown_closes_owned_policy_after_drain_even_if_config_changed(monkeypatch):
    started, release, outcomes = _block_policy_refresh(monkeypatch)
    monkeypatch.setattr("nhx.core.auth.service.start_policy", lambda: None)
    service = await _start_refresh_loop(monkeypatch)
    await service.on_startup()
    await started.wait()
    Configuration.set_override(AuthServiceConfig(enabled=False))
    loop_thread = threading.get_ident()

    def close():
        assert outcomes == ["completed"]
        assert threading.get_ident() != loop_thread
        outcomes.append("closed")

    monkeypatch.setattr("nhx.core.auth.service.close_policy", close)
    shutdown = asyncio.create_task(service.on_shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    await shutdown
    assert outcomes == ["completed", "closed"]
    assert not service._owns_policy
    assert service._refresh_task is None


@pytest.mark.asyncio
async def test_shutdown_retains_ownership_when_cleanup_fails_for_retry(monkeypatch):
    Configuration.set_override(AuthServiceConfig(enabled=True))
    monkeypatch.setattr("nhx.core.auth.service.start_policy", lambda: None)
    attempts = []

    def close():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("child reap timed out")

    monkeypatch.setattr("nhx.core.auth.service.close_policy", close)
    service = AuthService()
    await service.on_startup()
    with pytest.raises(RuntimeError, match="child reap timed out"):
        await service.on_shutdown()
    assert service._owns_policy
    assert not service._policy_refresh_healthy
    assert service._refresh_task is None
    # Configuration changes must not prevent retrying resources we still own.
    Configuration.set_override(AuthServiceConfig(enabled=False))
    await service.on_shutdown()
    assert attempts == [True, True]
    assert not service._owns_policy


@pytest.mark.asyncio
async def test_shutdown_does_not_close_unowned_policy(monkeypatch):
    monkeypatch.setattr("nhx.core.auth.service.close_policy", lambda: pytest.fail("closed unowned policy"))
    await AuthService().on_shutdown()


@pytest.mark.asyncio
async def test_readiness_requires_live_child_and_healthy_refresh(monkeypatch):
    Configuration.set_override(AuthServiceConfig(enabled=True))
    service = AuthService()
    loop_thread = threading.get_ident()

    def ready():
        assert threading.get_ident() != loop_thread
        return True

    monkeypatch.setattr("nhx.core.auth.service.is_policy_ready", ready)
    assert not await service.is_ready()
    service._policy_refresh_healthy = True
    assert await service.is_ready()
    monkeypatch.setattr("nhx.core.auth.service.is_policy_ready", lambda: False)
    assert not await service.is_ready()


@pytest.mark.asyncio
async def test_failed_refresh_invalidates_and_bootstraps_again(monkeypatch):
    calls = []
    invalidated = threading.Event()
    recovered = asyncio.Event()
    proceed = asyncio.Event()

    async def refresh(client, *, skip_static_bootstrap):
        calls.append(skip_static_bootstrap)
        if len(calls) == 1:
            return {}
        if len(calls) == 2:
            raise RuntimeError("entity fetch failed")
        recovered.set()
        await proceed.wait()
        return {}

    monkeypatch.setattr("nhx.core.auth.service.apply_embedded_policy_document", refresh)
    monkeypatch.setattr("nhx.core.auth.service.invalidate_policy_data", invalidated.set)
    service = await _start_refresh_loop(monkeypatch)
    # Shorten the loop without relying on wall-clock sleeps.
    original_wait_for = asyncio.wait_for

    async def immediate_interval(awaitable, timeout):
        return await original_wait_for(awaitable, timeout=0.001 if timeout == 30 else timeout)

    monkeypatch.setattr("nhx.core.auth.service.asyncio.wait_for", immediate_interval)
    await asyncio.wait_for(recovered.wait(), timeout=2)
    assert invalidated.is_set()
    assert calls == [False, True, False]
    assert not service._policy_refresh_healthy
    proceed.set()
    await service.on_shutdown()


@pytest.mark.asyncio
async def test_start_failure_does_not_claim_ownership(monkeypatch):
    Configuration.set_override(AuthServiceConfig(enabled=True))

    def fail():
        raise RuntimeError("OPA startup failed")

    monkeypatch.setattr("nhx.core.auth.service.start_policy", fail)
    service = AuthService()
    with pytest.raises(RuntimeError, match="OPA startup failed"):
        await service.on_startup()
    assert not service._owns_policy


@pytest.mark.asyncio
async def test_cancelled_startup_closes_child_started_in_worker(monkeypatch):
    Configuration.set_override(AuthServiceConfig(enabled=True))
    started = threading.Event()
    release = threading.Event()
    closed = []

    def start():
        started.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr("nhx.core.auth.service.start_policy", start)
    monkeypatch.setattr("nhx.core.auth.service.close_policy", lambda: closed.append(True))
    service = AuthService()
    task = asyncio.create_task(service.on_startup())
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]
    assert not service._owns_policy


@pytest.mark.asyncio
async def test_owned_child_closes_after_refresh_timeout(monkeypatch):
    started, _release, outcomes = _block_policy_refresh(monkeypatch)
    monkeypatch.setattr("nhx.core.auth.service._REFRESH_STOP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("nhx.core.auth.service.start_policy", lambda: None)
    monkeypatch.setattr("nhx.core.auth.service.close_policy", lambda: outcomes.append("closed"))
    service = await _start_refresh_loop(monkeypatch)
    await service.on_startup()
    await started.wait()
    await service.on_shutdown()
    assert outcomes == ["cancelled", "closed"]


@pytest.mark.asyncio
async def test_api_evaluation_runs_off_event_loop(monkeypatch):
    from nemo_helix_plugin.iam.types import AuthzRequest
    from nhx.core.auth.api.v2.authz import endpoints

    Configuration.set_override(AuthServiceConfig(enabled=True, bundle_cache_seconds=5))
    monkeypatch.setattr(
        endpoints,
        "resolve_authz_account_context",
        AsyncMock(return_value=SimpleNamespace(to_policy_fields=lambda: {})),
    )
    loop_thread = threading.get_ident()

    def evaluate(entrypoint, data):
        assert threading.get_ident() != loop_thread
        assert entrypoint == "allow"
        return {"allow": True}

    monkeypatch.setattr(endpoints, "evaluate", evaluate)
    result = await endpoints.evaluate_policy(AuthzRequest(input={}), entrypoint="allow", entity_client=MagicMock())
    assert result.result == {"allow": True}


@pytest.mark.asyncio
async def test_data_publication_runs_off_event_loop(monkeypatch):
    from nhx.core.auth.app.embedded_pdp import data

    bootstrap = {"authz": {"principals": {}}}
    dynamic = {"authz": {"principals": {"user": {}}}}
    monkeypatch.setattr(data, "build_authorization_data", AsyncMock(side_effect=[bootstrap, dynamic]))
    loop_thread = threading.get_ident()
    applied = []

    def publish(document):
        assert threading.get_ident() != loop_thread
        applied.append(document)

    monkeypatch.setattr(data, "set_policy_data", publish)
    assert await data.apply_embedded_policy_document(None) == dynamic
    assert applied == [bootstrap, dynamic]


@pytest.mark.asyncio
async def test_data_reload_invalidates_even_when_static_fallback_fails(monkeypatch):
    from nhx.core.auth.app.embedded_pdp import data

    monkeypatch.setattr(data, "build_authorization_data", AsyncMock(side_effect=RuntimeError("no data")))
    invalidated = []
    monkeypatch.setattr(data, "invalidate_policy_data", lambda: invalidated.append(True))
    assert not await data.load_policy_data()
    assert invalidated == [True, True]
