# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspection tests for the Docker Compose sandbox provider."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxCreateError, SandboxSpec
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_cli as compose_cli
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_inspection as compose_inspection
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers.compose import ComposeServiceTopology

from packages.nhx_evals_sdk.tests.agent_eval._compose_testkit import _compose_suffix, _create, _provider, _Runner


async def test_preflight_rejects_mismatched_topology_before_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.config["services"]["unexpected"] = {}
    provider = _provider(tmp_path)

    with pytest.raises(SandboxCreateError, match="unexpected=.*unexpected"):
        await _create(monkeypatch, provider, runner)
    assert not any(_compose_suffix(argv)[:1] == ("up",) for argv, _, _ in runner.calls)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"Service": "agent", "State": "running", "Health": "unhealthy"},
                {"Service": "redis", "State": "running", "Health": ""},
                {"Service": "init", "State": "exited", "ExitCode": 0},
            ],
            "not healthy",
        ),
        (
            [
                {"Service": "agent", "State": "running", "Health": "healthy"},
                {"Service": "redis", "State": "running", "Health": ""},
                {"Service": "init", "State": "exited", "ExitCode": 1},
            ],
            "did not exit successfully",
        ),
    ],
)
async def test_readiness_enforces_service_roles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    runner = _Runner()
    runner.rows = rows
    provider = _provider(tmp_path)

    with pytest.raises(SandboxCreateError, match=message):
        await _create(monkeypatch, provider, runner)
    assert any(_compose_suffix(argv)[:1] == ("down",) for argv, _, _ in runner.calls)


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_readiness_checks_every_scaled_service_replica(reverse_rows: bool) -> None:
    topology = ComposeServiceTopology(
        target_service="agent",
        long_running_services=frozenset({"agent"}),
        one_shot_services=frozenset({"init"}),
    )
    healthy_agent = {"Service": "agent", "State": "running", "Health": "healthy"}
    failed_agent = {"Service": "agent", "State": "exited", "Health": "", "ExitCode": 1}
    successful_init = {"Service": "init", "State": "exited", "ExitCode": 0}
    rows = [healthy_agent, failed_agent, successful_init]
    if reverse_rows:
        rows.reverse()

    problem = compose_inspection._services_ready(rows, topology)

    assert problem is not None
    assert "agent" in problem
    assert "not running" in problem


def test_readiness_checks_every_scaled_one_shot_replica() -> None:
    topology = ComposeServiceTopology(
        target_service="agent",
        long_running_services=frozenset({"agent"}),
        one_shot_services=frozenset({"init"}),
    )
    rows = [
        {"Service": "agent", "State": "running", "Health": "healthy"},
        {"Service": "init", "State": "exited", "ExitCode": 0},
        {"Service": "init", "State": "exited", "ExitCode": 1},
    ]

    assert (
        compose_inspection._services_ready(rows, topology)
        == "Compose one-shot service 'init' did not exit successfully"
    )


async def test_provider_options_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    monkeypatch.setattr(compose_cli, "_run_command", runner)

    with pytest.raises(SandboxCreateError, match="provider_options"):
        await provider.create(SandboxSpec(provider_options={"build": True}))
    assert runner.calls == []


async def test_port_conflict_uses_caller_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.config["services"]["agent"] = {
        "ports": [
            {
                "host_ip": "127.0.0.1",
                "published": "18080",
                "target": 8080,
                "protocol": "tcp",
            }
        ]
    }
    monkeypatch.setattr(compose_inspection, "_published_port_available", lambda _port: False)
    provider = _provider(tmp_path, port_override_hints={"agent": "AGENT_PORT"})

    with pytest.raises(SandboxCreateError, match="AGENT_PORT") as caught:
        await _create(monkeypatch, provider, runner)
    assert "127.0.0.1:18080 -> 8080/tcp" in str(caught.value)


@pytest.mark.parametrize(
    ("protocol", "socket_type", "expected_options"),
    [
        (
            "tcp",
            compose_inspection.socket.SOCK_STREAM,
            [(compose_inspection.socket.SOL_SOCKET, compose_inspection.socket.SO_REUSEADDR, 1)],
        ),
        ("udp", compose_inspection.socket.SOCK_DGRAM, []),
    ],
)
def test_published_port_probe_reuses_only_tcp_addresses(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    socket_type: int,
    expected_options: list[tuple[int, int, int]],
) -> None:
    options: list[tuple[int, int, int]] = []
    bindings: list[tuple[str, int]] = []

    class ProbeSocket:
        def __enter__(self) -> ProbeSocket:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def setsockopt(self, level: int, option: int, value: int) -> None:
            options.append((level, option, value))

        def bind(self, address: tuple[str, int]) -> None:
            bindings.append(address)

    def socket_factory(family: int, kind: int) -> ProbeSocket:
        assert family == compose_inspection.socket.AF_INET
        assert kind == socket_type
        return ProbeSocket()

    monkeypatch.setattr(compose_inspection.socket, "socket", socket_factory)
    published_port = compose_inspection._PublishedPort("agent", "127.0.0.1", 18080, 8080, protocol)

    assert compose_inspection._published_port_available(published_port)
    assert options == expected_options
    assert bindings == [("127.0.0.1", 18080)]


def test_published_port_probe_rejects_active_tcp_listener() -> None:
    with compose_inspection.socket.socket(
        compose_inspection.socket.AF_INET,
        compose_inspection.socket.SOCK_STREAM,
    ) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        host, port = listener.getsockname()
        published_port = compose_inspection._PublishedPort("agent", host, port, 8080, "tcp")

        assert not compose_inspection._published_port_available(published_port)


async def test_preflight_parses_rendered_config_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    monkeypatch.setattr(compose_cli, "_run_command", runner)
    original_loads = json.loads
    payloads: list[str] = []

    def tracked_loads(text: str) -> Any:
        payloads.append(text)
        return original_loads(text)

    monkeypatch.setattr(compose_inspection.json, "loads", tracked_loads)

    await provider._preflight(dict(os.environ))

    assert payloads == [json.dumps(runner.config)]


async def test_preflight_exception_cancels_sibling_and_preserves_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    ps_started = asyncio.Event()
    ps_cancelled = asyncio.Event()

    async def run_compose(args: Sequence[str], **_kwargs: object) -> None:
        if tuple(args[:1]) == ("config",):
            await ps_started.wait()
            raise OSError("config transport failed")
        if tuple(args[:1]) == ("ps",):
            ps_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                ps_cancelled.set()
                raise
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(provider._cli, "run_compose", run_compose)

    with pytest.raises(SandboxCreateError, match="config transport failed") as caught:
        await provider.create(SandboxSpec())

    assert isinstance(caught.value.__cause__, OSError)
    assert ps_cancelled.is_set()
    assert provider._session is None


async def test_preflight_drain_finishes_sibling_cleanup_before_restoring_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    ps_started = asyncio.Event()
    ps_cleanup_started = asyncio.Event()
    ps_cleanup_finished = asyncio.Event()
    ps_cancelled_during_cleanup = asyncio.Event()
    release_ps_cleanup = asyncio.Event()

    async def run_compose(args: Sequence[str], **_kwargs: object) -> None:
        if tuple(args[:1]) == ("config",):
            await ps_started.wait()
            raise OSError("config transport failed")
        if tuple(args[:1]) == ("ps",):
            ps_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                ps_cleanup_started.set()
                try:
                    await release_ps_cleanup.wait()
                except asyncio.CancelledError:
                    ps_cancelled_during_cleanup.set()
                    raise
                ps_cleanup_finished.set()
                raise
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(provider._cli, "run_compose", run_compose)

    create_task = asyncio.create_task(provider.create(SandboxSpec()))
    await ps_cleanup_started.wait()
    create_task.cancel()
    release_ps_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert ps_cleanup_finished.is_set()
    assert not ps_cancelled_during_cleanup.is_set()
    assert provider._session is None


@pytest.mark.parametrize(
    ("config_stdout", "cause_type"),
    [
        ("{", json.JSONDecodeError),
        ("[]", TypeError),
        ('{"services": []}', TypeError),
    ],
)
async def test_invalid_rendered_config_preserves_create_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_stdout: str,
    cause_type: type[Exception],
) -> None:
    runner = _Runner()
    runner.config_stdout = config_stdout
    provider = _provider(tmp_path)

    with pytest.raises(SandboxCreateError, match="Could not inspect rendered Compose configuration") as caught:
        await _create(monkeypatch, provider, runner)

    assert isinstance(caught.value.__cause__, cause_type)
