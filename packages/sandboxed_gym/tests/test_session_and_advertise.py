# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for cross-job session handoff and broker advertise_url."""

from __future__ import annotations

import asyncio
import atexit
import signal
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import sandboxed_gym.orchestrator as orchestrator_module
from sandboxed_gym.broker import EpisodeBrokerServer
from sandboxed_gym.config import BrokerEndpoint, EpisodeBrokerConfig
from sandboxed_gym.host.models import GymHostHandle
from sandboxed_gym.orchestrator import (
    _INSTALLED,
    _TERMINATION_SHUTDOWNS,
    TERMINATION_SIGNALS,
    SandboxedGymOrchestrator,
    SandboxedGymSession,
)
from sandboxed_gym.serve_config import SandboxedGymServeConfig


@pytest.fixture(autouse=True)
def _drop_the_reaper_the_session_arms():
    """start() arms a process-exit sweep. These tests must not leave it installed."""
    original = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    registered: list[Callable[..., object]] = []
    real_register = atexit.register
    installed = set(_INSTALLED)
    prior_shutdowns = list(_TERMINATION_SHUTDOWNS)

    def _tracking_register(func, *args, **kwargs):
        registered.append(func)
        return real_register(func, *args, **kwargs)

    atexit.register = _tracking_register  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        atexit.register = real_register
        for func in registered:
            atexit.unregister(func)
        for signum, handler in original.items():
            signal.signal(signum, handler)
        _INSTALLED.clear()
        _INSTALLED.update(installed)
        _TERMINATION_SHUTDOWNS.clear()
        _TERMINATION_SHUTDOWNS.extend(prior_shutdowns)


def test_advertise_url_preferred_over_host():
    cfg = EpisodeBrokerConfig(
        job_id="job-1",
        backend="memory",
        allow_insecure_memory_backend=True,
        host="10.0.0.9",
        advertise_url="http://broker.job-1.svc:8741",
        port=8741,
    )
    server = EpisodeBrokerServer(cfg)
    url, host, port = server._resolve_advertise(8741)
    assert url == "http://broker.job-1.svc:8741"
    assert host == "broker.job-1.svc"
    assert port == 8741


def test_advertise_url_keeps_its_path_when_the_port_is_implied():
    """The bind port has to be spliced in ahead of the path, not in place of it: the Gym host
    treats this URL as a base and appends its own routes to it."""
    cfg = EpisodeBrokerConfig(
        job_id="job-1",
        backend="memory",
        allow_insecure_memory_backend=True,
        advertise_url="http://ingress.svc/broker/job-1",
    )
    url, host, port = EpisodeBrokerServer(cfg)._resolve_advertise(8741)
    assert url == "http://ingress.svc:8741/broker/job-1"
    assert (host, port) == ("ingress.svc", 8741)


def test_session_descriptor_orchestrator_mode():
    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
            },
            "rollout_auth_token": "secret",
        }
    )
    broker = BrokerEndpoint(url="http://broker.svc:1", host="broker.svc", port=1, token="btok")
    host = GymHostHandle(
        host_id="h1",
        health_url="http://host/health",
        rollout_url="http://host/rollouts/run",
        headers={},
    )
    session = SandboxedGymSession(
        cfg=cfg,
        broker_server=MagicMock(),
        broker=broker,
        host_provider=MagicMock(),
        host=host,
        orchestrator_url="http://orch.svc:8090",
    )
    desc = session.descriptor(mode="orchestrator")
    assert desc.orchestrator_url == "http://orch.svc:8090"
    assert desc.rollout_url == "http://orch.svc:8090/rollouts/run"
    assert desc.broker_token == "btok"
    assert desc.rollout_auth_token == "secret"
    assert desc.health_url == "http://host/health"


def test_host_sdk_lifecycle_uses_one_event_loop(monkeypatch):
    loops = []
    host = GymHostHandle(
        host_id="h1",
        health_url="http://host/health",
        rollout_url="http://host/rollouts/run",
        headers={},
    )

    class HostProvider:
        async def create_host(self, spec):
            loops.append(asyncio.get_running_loop())
            return host

        async def wait_ready(self, handle, timeout_s):
            loops.append(asyncio.get_running_loop())

        async def destroy_host(self, handle):
            loops.append(asyncio.get_running_loop())

    broker_server = MagicMock()
    broker_server.start.return_value = BrokerEndpoint(
        url="http://broker.svc:8741",
        host="broker.svc",
        port=8741,
        token="btok",
    )
    monkeypatch.setattr(orchestrator_module, "EpisodeBrokerServer", lambda config: broker_server)
    monkeypatch.setattr(orchestrator_module, "get_host_provider", lambda name, options: HostProvider())
    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
            },
        }
    )

    session = SandboxedGymOrchestrator().start(cfg)
    session.shutdown()

    assert len(loops) == 3
    assert loops[0] is loops[1] is loops[2]


# --------------------------------------------------------------------------------------------
# Choosing where the broker runs
# --------------------------------------------------------------------------------------------


def _minimal_cfg() -> SandboxedGymServeConfig:
    return SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
            },
        }
    )


def _stub_host_provider(monkeypatch, *, ready_fails: bool = False):
    """Patch out host provisioning so only broker selection is under test."""
    host = GymHostHandle(host_id="h1", health_url="http://host/health", rollout_url="http://host/rollouts/run")
    destroyed: list[str] = []

    class HostProvider:
        async def create_host(self, spec):
            return host

        async def wait_ready(self, handle, timeout_s):
            if ready_fails:
                raise RuntimeError("host never came up")

        async def destroy_host(self, handle):
            destroyed.append(handle.host_id)

    monkeypatch.setattr(orchestrator_module, "get_host_provider", lambda name, options: HostProvider())
    return destroyed


def _fake_broker() -> MagicMock:
    broker = MagicMock()
    broker.start.return_value = BrokerEndpoint(url="http://injected:9000", host="injected", port=9000, token="itok")
    return broker


def test_a_supplied_broker_is_used_instead_of_starting_one_here(monkeypatch):
    """The whole point: the caller decides where the broker runs."""
    _stub_host_provider(monkeypatch)
    in_process = MagicMock()
    monkeypatch.setattr(orchestrator_module, "EpisodeBrokerServer", lambda config: in_process)
    injected = _fake_broker()

    session = SandboxedGymOrchestrator().start(_minimal_cfg(), broker=injected)
    try:
        injected.start.assert_called_once()
        # Not merely "ours was used" -- nothing may start a second broker holding the same
        # credential.
        in_process.start.assert_not_called()
        assert session.broker.url == "http://injected:9000"
    finally:
        # The only test here that reaches a live session; closing it releases the runner thread.
        session.shutdown()


def test_a_supplied_broker_is_shut_down_with_the_session(monkeypatch):
    _stub_host_provider(monkeypatch)
    injected = _fake_broker()

    SandboxedGymOrchestrator().start(_minimal_cfg(), broker=injected).shutdown()

    injected.shutdown.assert_called_once()


def test_a_supplied_broker_is_shut_down_when_the_host_never_becomes_ready(monkeypatch):
    """A broker left running past a failed spinup holds its port and its credential."""
    destroyed = _stub_host_provider(monkeypatch, ready_fails=True)
    injected = _fake_broker()

    with pytest.raises(RuntimeError, match="never came up"):
        SandboxedGymOrchestrator().start(_minimal_cfg(), broker=injected)

    injected.shutdown.assert_called_once()
    assert destroyed == ["h1"]


def test_a_supplied_broker_is_shut_down_when_the_host_provider_is_unknown(monkeypatch):
    """Provisioning is not the only step that can fail after the broker is already serving.

    `get_host_provider` rejects an unknown name and a bad option -- ordinary config mistakes,
    reached before any host exists.
    """
    monkeypatch.setattr(
        orchestrator_module,
        "get_host_provider",
        lambda name, options: (_ for _ in ()).throw(ValueError("Unknown sandboxed gym host provider: 'nope'")),
    )
    injected = _fake_broker()

    with pytest.raises(ValueError, match="Unknown sandboxed gym host provider"):
        SandboxedGymOrchestrator().start(_minimal_cfg(), broker=injected)

    injected.shutdown.assert_called_once()


def test_a_broker_that_fails_to_shut_down_does_not_hide_why_startup_failed(monkeypatch):
    """The startup error is the diagnosis; a failed cleanup of caller code must not replace it."""
    _stub_host_provider(monkeypatch, ready_fails=True)
    injected = _fake_broker()
    injected.shutdown.side_effect = RuntimeError("broker actor already dead")

    closed: list[bool] = []
    real_runner = orchestrator_module._SessionAsyncRunner

    class RecordingRunner(real_runner):
        def close(self) -> None:
            closed.append(True)
            super().close()

    monkeypatch.setattr(orchestrator_module, "_SessionAsyncRunner", RecordingRunner)

    with pytest.raises(RuntimeError, match="never came up"):
        SandboxedGymOrchestrator().start(_minimal_cfg(), broker=injected)

    # Without this the test would also pass if cleanup skipped the broker entirely: the raising
    # shutdown is only interesting once we know it was reached.
    injected.shutdown.assert_called_once()
    # The loop thread is released before the broker is asked to stop, so a broker that raises
    # cannot strand it.
    assert closed == [True]


@pytest.mark.parametrize(
    ("sandbox_timeouts", "expected_pull_s", "expected_health_s"),
    [
        ({"ready_timeout_s": 900, "bootstrap_timeout_s": 240}, 900, 240),
        ({"ready_timeout_s": 900}, 900, 900),
    ],
)
def test_the_bootstrap_timeout_bounds_the_health_wait_and_not_the_image_pull(
    monkeypatch, sandbox_timeouts, expected_pull_s, expected_health_s
):
    """A Gym-startup budget must not also cap how long the pod may take to pull its image."""
    host = GymHostHandle(host_id="h1", health_url="http://host/health", rollout_url="http://host/rollouts/run")
    seen: dict[str, float] = {}

    class HostProvider:
        async def create_host(self, spec):
            seen["pull"] = spec.ready_timeout_s
            return host

        async def wait_ready(self, handle, timeout_s):
            seen["health"] = timeout_s

        async def destroy_host(self, handle):
            return None

    monkeypatch.setattr(orchestrator_module, "get_host_provider", lambda name, options: HostProvider())
    cfg = _minimal_cfg().model_copy(update={"sandbox": _minimal_cfg().sandbox.model_copy(update=sandbox_timeouts)})

    SandboxedGymOrchestrator().start(cfg, broker=_fake_broker()).shutdown()

    assert seen == {"pull": expected_pull_s, "health": expected_health_s}
