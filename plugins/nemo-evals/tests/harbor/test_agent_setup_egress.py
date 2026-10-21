# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent wrapper that runs a Harbor agent's setup() inside the sandbox's setup egress window."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.nop import NopAgent
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor.agent_setup_egress import (
    INNER_AGENT_KWARG,
    SETUP_EGRESS_AGENT,
    SETUP_HOSTS_KWARG,
    AgentSetupEgress,
)

SETUP_HOSTS = ["registry.npmjs.org", "pypi.org"]
NO_NETWORK = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
DYNAMIC_ALLOWLIST = EnvironmentCapabilities(
    disable_internet=True, network_allowlist=True, network_allowlist_hostnames=True, dynamic_network_policy=True
)


class _WindowEnvironment:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    @asynccontextmanager
    async def agent_setup_egress(self) -> AsyncGenerator[None, None]:
        self.events.append("open")
        try:
            yield
        finally:
            self.events.append("close")


class _PolicyEnvironment:
    """The part of Harbor's ``BaseEnvironment`` network policy API the wrapper uses, recording each switch."""

    def __init__(
        self,
        events: list[Any],
        baseline: NetworkPolicy = NO_NETWORK,
        capabilities: EnvironmentCapabilities = DYNAMIC_ALLOWLIST,
    ) -> None:
        self.events = events
        self.network_policy = baseline
        self.capabilities = capabilities

    async def set_network_policy(self, network_policy: NetworkPolicy) -> None:
        self.events.append(network_policy)
        self.network_policy = network_policy


class _PolicyAndWindowEnvironment(_PolicyEnvironment, _WindowEnvironment):
    pass


def _agent(tmp_path, inner: str = "nop", hosts: list[str] | None = None) -> Any:
    kwargs: dict[str, Any] = {INNER_AGENT_KWARG: inner}
    if hosts is not None:
        kwargs[SETUP_HOSTS_KWARG] = hosts
    config = AgentConfig(import_path=SETUP_EGRESS_AGENT, kwargs=kwargs)
    return AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)


def _record_setup(monkeypatch, events: list[Any]) -> None:
    async def setup(self, environment) -> None:
        events.append(("setup", environment.network_policy))

    monkeypatch.setattr(NopAgent, "setup", setup)


def _allowlist(*hosts: str) -> NetworkPolicy:
    return NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=list(hosts))


def test_harbor_builds_the_real_agent_from_the_wrapper_import_path(tmp_path):
    agent = _agent(tmp_path)
    assert isinstance(agent, NopAgent)
    assert type(agent).__name__ == "NopAgent" and agent.name() == NopAgent.name()


async def test_setup_runs_inside_the_window(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        events.append("setup")

    monkeypatch.setattr(NopAgent, "setup", setup)
    await _agent(tmp_path).setup(_WindowEnvironment(events))
    assert events == ["open", "setup", "close"]


async def test_a_failed_setup_still_closes_the_window(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        raise RuntimeError("install failed")

    monkeypatch.setattr(NopAgent, "setup", setup)
    with pytest.raises(RuntimeError, match="install failed"):
        await _agent(tmp_path).setup(_WindowEnvironment(events))
    assert events == ["open", "close"]


async def test_an_environment_without_a_window_runs_setup_unchanged(tmp_path, monkeypatch):
    events: list[str] = []

    async def setup(self, environment) -> None:
        events.append("setup")

    monkeypatch.setattr(NopAgent, "setup", setup)
    await _agent(tmp_path).setup(object())
    assert events == ["setup"]


async def test_a_dynamic_policy_environment_opens_the_setup_hosts_and_restores_no_network(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    opened = _allowlist(*SETUP_HOSTS)
    assert events == [opened, ("setup", opened), NO_NETWORK]
    assert environment.network_policy == NO_NETWORK


async def test_the_setup_hosts_are_added_to_an_allowlist_baseline(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    baseline = _allowlist("api.example.com", "pypi.org")
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(_PolicyEnvironment(events, baseline))
    opened = _allowlist("api.example.com", "pypi.org", "registry.npmjs.org")
    assert events == [opened, ("setup", opened), baseline]


async def test_a_failed_setup_restores_the_policy(tmp_path, monkeypatch):
    events: list[Any] = []

    async def setup(self, environment) -> None:
        raise RuntimeError("install failed")

    monkeypatch.setattr(NopAgent, "setup", setup)
    environment = _PolicyEnvironment(events)
    with pytest.raises(RuntimeError, match="install failed"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == [_allowlist(*SETUP_HOSTS), NO_NETWORK]
    assert environment.network_policy == NO_NETWORK


async def test_a_cancelled_setup_restores_the_policy(tmp_path, monkeypatch):
    events: list[Any] = []
    started = asyncio.Event()

    async def setup(self, environment) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(NopAgent, "setup", setup)
    environment = _PolicyEnvironment(events)
    task = asyncio.create_task(_agent(tmp_path, hosts=SETUP_HOSTS).setup(environment))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == [_allowlist(*SETUP_HOSTS), NO_NETWORK]
    assert environment.network_policy == NO_NETWORK


async def test_a_public_baseline_is_left_alone(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    public = NetworkPolicy(network_mode=NetworkMode.PUBLIC)
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(_PolicyEnvironment(events, public))
    assert events == [("setup", public)]


@pytest.mark.parametrize(
    "capabilities",
    [
        EnvironmentCapabilities(disable_internet=True, network_allowlist=True, network_allowlist_hostnames=True),
        EnvironmentCapabilities(disable_internet=True, dynamic_network_policy=True),
    ],
    ids=["static-policy", "no-allowlist"],
)
async def test_an_environment_that_cannot_switch_to_an_allowlist_keeps_the_task_policy(
    tmp_path, monkeypatch, capabilities
):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(_PolicyEnvironment(events, capabilities=capabilities))
    assert events == [("setup", NO_NETWORK)]


async def test_no_setup_hosts_leaves_the_policy_alone(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    await _agent(tmp_path).setup(_PolicyEnvironment(events))
    assert events == [("setup", NO_NETWORK)]


async def test_the_environments_own_window_is_preferred_over_the_network_policy(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(_PolicyAndWindowEnvironment(events))
    assert events == ["open", ("setup", NO_NETWORK), "close"]


def test_preflight_is_the_real_agents_without_the_wrapper_kwarg(monkeypatch):
    seen: list[dict[str, Any]] = []

    def preflight(cls, kwargs=None, env=None) -> None:
        seen.append(dict(kwargs or {}))

    monkeypatch.setattr(NopAgent, "preflight", classmethod(preflight))
    AgentFactory.run_preflight(
        AgentConfig(
            import_path=SETUP_EGRESS_AGENT,
            kwargs={INNER_AGENT_KWARG: "nop", SETUP_HOSTS_KWARG: SETUP_HOSTS, "version": "1"},
        )
    )
    assert seen == [{"version": "1"}]


def test_the_inner_agent_may_be_an_import_path(tmp_path):
    agent = _agent(tmp_path, inner="harbor.agents.nop:NopAgent")
    assert isinstance(agent, NopAgent)


def test_the_wrapper_requires_the_inner_agent(tmp_path):
    with pytest.raises(KeyError, match=INNER_AGENT_KWARG):
        AgentSetupEgress(logs_dir=tmp_path)


def test_an_acp_shorthand_keeps_its_registry_spec(tmp_path):
    from harbor.agents.installed.acp import AcpAgent

    assert isinstance(_agent(tmp_path, inner="acp:claude-code"), AcpAgent)
