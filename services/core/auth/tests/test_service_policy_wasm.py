# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from nhx.common.config import Configuration
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.auth.service import AuthService


@pytest.fixture(autouse=True)
def clear_config_overrides():
    yield
    Configuration.clear_override(AuthServiceConfig)


@pytest.mark.asyncio
async def test_on_startup_preflights_embedded_policy_wasm(monkeypatch: pytest.MonkeyPatch):
    calls: list[bool] = []
    Configuration.set_override(
        AuthServiceConfig(
            enabled=True,
            policy_decision_point_provider="embedded",
            embedded_pdp_auto_build_wasm=False,
        )
    )
    monkeypatch.setattr(
        "nhx.core.auth.service.ensure_embedded_policy_wasm", lambda *, auto_build: calls.append(auto_build)
    )

    await AuthService().on_startup()

    assert calls == [False]


@pytest.mark.parametrize(
    "config",
    [
        AuthServiceConfig(enabled=False, policy_decision_point_provider="embedded"),
        AuthServiceConfig(enabled=True, policy_decision_point_provider="opa"),
    ],
)
@pytest.mark.asyncio
async def test_on_startup_skips_policy_wasm_preflight_when_not_needed(
    config: AuthServiceConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[bool] = []
    Configuration.set_override(config)
    monkeypatch.setattr(
        "nhx.core.auth.service.ensure_embedded_policy_wasm", lambda *, auto_build: calls.append(auto_build)
    )

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
