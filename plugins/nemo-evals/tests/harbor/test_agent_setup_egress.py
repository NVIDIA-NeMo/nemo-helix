# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent wrapper that runs a Harbor agent's setup() inside the sandbox's setup egress window."""

import asyncio
import shlex
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.nop import NopAgent
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor.agent_setup_egress import (
    INNER_AGENT_KWARG,
    SETUP_EGRESS_AGENT,
    SETUP_HOSTS_KWARG,
    AgentSetupEgress,
)
from nemo_evals.harbor.setup_cleanup import KILL_CONNECTION_HOLDERS_SCRIPT, LIST_PROCESSES_SCRIPT, SYSTEM_PATH

SETUP_HOSTS = ["registry.npmjs.org", "pypi.org"]
SNAPSHOT = ("1:100", "7:250")
KEEP = "1:100 7:250"
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
    """The part of Harbor's ``BaseEnvironment`` the wrapper uses, recording each policy switch and command.

    The cleanup scripts are recognised by their text: listing processes prints :data:`SNAPSHOT`, and the
    kill sweep records the ``KEEP`` it was given.
    """

    def __init__(
        self,
        events: list[Any],
        baseline: NetworkPolicy = NO_NETWORK,
        capabilities: EnvironmentCapabilities = DYNAMIC_ALLOWLIST,
    ) -> None:
        self.events = events
        self.network_policy = baseline
        self.capabilities = capabilities
        self.list_exit = 0
        self.kill_exit = 0
        self.policy_hook: Callable[[NetworkPolicy], Awaitable[None]] | None = None

    async def set_network_policy(self, network_policy: NetworkPolicy) -> None:
        self.events.append(network_policy)
        if self.policy_hook is not None:
            await self.policy_hook(network_policy)
        self.network_policy = network_policy

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if shlex.quote(LIST_PROCESSES_SCRIPT) in command:
            self.events.append("list")
            assert (env or {}).get("PATH") == SYSTEM_PATH and user == "root"
            return ExecResult(stdout="\n".join(SNAPSHOT) + "\n", stderr="no /proc", return_code=self.list_exit)
        if shlex.quote(KILL_CONNECTION_HOLDERS_SCRIPT) in command:
            self.events.append(("kill", (env or {}).get("KEEP")))
            assert (env or {}).get("PATH") == SYSTEM_PATH and user == "root"
            return ExecResult(stderr="could not kill: 42", return_code=self.kill_exit)
        self.events.append(("exec", command))
        return ExecResult(return_code=0)

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)


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
    assert events == ["list", opened, ("setup", opened), NO_NETWORK, ("kill", KEEP)]
    assert environment.network_policy == NO_NETWORK


async def test_the_setup_hosts_are_added_to_an_allowlist_baseline(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    baseline = _allowlist("api.example.com", "pypi.org")
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(_PolicyEnvironment(events, baseline))
    opened = _allowlist("api.example.com", "pypi.org", "registry.npmjs.org")
    assert events == ["list", opened, ("setup", opened), baseline, ("kill", KEEP)]


async def test_a_failed_setup_restores_the_policy(tmp_path, monkeypatch):
    events: list[Any] = []

    async def setup(self, environment) -> None:
        raise RuntimeError("install failed")

    monkeypatch.setattr(NopAgent, "setup", setup)
    environment = _PolicyEnvironment(events)
    with pytest.raises(RuntimeError, match="install failed"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list", _allowlist(*SETUP_HOSTS), NO_NETWORK, ("kill", KEEP)]
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
    assert events == ["list", _allowlist(*SETUP_HOSTS), NO_NETWORK, ("kill", KEEP)]
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


async def _refuses_commands(environment: _PolicyEnvironment) -> None:
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        await environment.exec("echo task")
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        await environment.service_exec("echo task", service="main")


async def test_commands_run_while_open_and_after_a_clean_close(tmp_path, monkeypatch):
    events: list[Any] = []

    async def setup(self, environment) -> None:
        await environment.exec("npm install -g agent")
        await environment.service_exec("pip install agent")

    monkeypatch.setattr(NopAgent, "setup", setup)
    environment = _PolicyEnvironment(events)
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    await environment.exec("echo task")
    await environment.service_exec("echo verifier")
    assert events == [
        "list",
        _allowlist(*SETUP_HOSTS),
        ("exec", "npm install -g agent"),
        ("exec", "pip install agent"),
        NO_NETWORK,
        ("kill", KEEP),
        ("exec", "echo task"),
        ("exec", "echo verifier"),
    ]
    assert "exec" not in vars(environment) and "service_exec" not in vars(environment)


async def test_a_failed_kill_raises_and_keeps_refusing_commands(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    environment.kill_exit = 1
    with pytest.raises(RuntimeError, match="could not stop processes left by agent setup"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events[-2:] == [NO_NETWORK, ("kill", KEEP)]
    await _refuses_commands(environment)


async def test_a_failed_restore_keeps_refusing_commands(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    opened = _allowlist(*SETUP_HOSTS)

    async def reject_restore(network_policy: NetworkPolicy) -> None:
        if network_policy != opened:
            raise RuntimeError("provider rejected the policy")

    environment.policy_hook = reject_restore
    with pytest.raises(RuntimeError, match="provider rejected the policy"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list", opened, ("setup", opened), NO_NETWORK]
    await _refuses_commands(environment)


async def test_a_restore_cancelled_by_the_setup_timeout_keeps_refusing_commands(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    restoring = asyncio.Event()
    opened = _allowlist(*SETUP_HOSTS)

    async def hang_restore(network_policy: NetworkPolicy) -> None:
        if network_policy != opened:
            restoring.set()
            await asyncio.Event().wait()

    environment.policy_hook = hang_restore
    task = asyncio.create_task(_agent(tmp_path, hosts=SETUP_HOSTS).setup(environment))
    await restoring.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ("kill", KEEP) not in events
    await _refuses_commands(environment)


async def test_an_open_that_fails_partway_still_restores_and_sweeps(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    opened = _allowlist(*SETUP_HOSTS)

    async def fail_partway(network_policy: NetworkPolicy) -> None:
        if network_policy == opened:
            environment.network_policy = network_policy
            raise RuntimeError("applied some rules, then failed")

    environment.policy_hook = fail_partway
    with pytest.raises(RuntimeError, match="applied some rules"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list", opened, NO_NETWORK, ("kill", KEEP)]
    assert environment.network_policy == NO_NETWORK
    await environment.exec("echo task")


async def test_a_failed_snapshot_opens_nothing_and_leaves_commands_alone(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    environment.list_exit = 1
    with pytest.raises(RuntimeError, match="could not list environment processes"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list"]
    assert "exec" not in vars(environment) and "service_exec" not in vars(environment)


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
