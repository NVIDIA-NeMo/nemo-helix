# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent wrapper that runs a Harbor agent's setup() inside the sandbox's setup egress window."""

import asyncio
import shlex
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.nop import NopAgent
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.config import AgentConfig
from nemo_evals.harbor import agent_setup_egress
from nemo_evals.harbor.agent_setup_egress import (
    GUARDED_METHODS,
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
    """The part of Harbor's ``BaseEnvironment`` the wrapper uses, recording each policy switch, command and transfer.

    Like Harbor's, ``set_network_policy`` returns early for the cached policy and caches a new one only once
    it was applied. The cleanup scripts are recognised by their text: listing processes prints
    :data:`SNAPSHOT`, and the kill sweep records the ``KEEP`` it was given.
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
        self.policy_supported = True
        self.policy_hook: Callable[[NetworkPolicy], Awaitable[None]] | None = None
        self._uses_compose = False

    def validate_network_policy_support(self, network_policy: NetworkPolicy) -> None:
        if not self.policy_supported:
            raise ValueError("network_mode='allowlist' with wildcard hostnames is not supported")

    async def set_network_policy(self, network_policy: NetworkPolicy) -> None:
        self.validate_network_policy_support(network_policy)
        if network_policy == self.network_policy:
            return
        self.events.append(network_policy)
        if self.policy_hook is not None:
            await self.policy_hook(network_policy)
        self.network_policy = network_policy

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.events.append(("upload_file", target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.events.append(("upload_dir", target_dir))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.events.append(("download_file", source_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        self.events.append(("download_dir", source_dir))

    def __getattr__(self, name: str) -> Callable[..., Awaitable[None]]:
        """The rest of the guarded methods, which these tests don't call."""
        if name not in GUARDED_METHODS:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> None:
            self.events.append((name, args))

        return method

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
        # Not through self.exec: providers with sidecars run service commands their own way.
        self.events.append(("service_exec", command))
        return ExecResult(return_code=0)


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


class _CodexInstallEnvironment:
    """Records each command and whether the setup window was open when it ran; codex starts uninstalled."""

    def __init__(self) -> None:
        self.open = False
        self.commands: list[tuple[str, bool]] = []

    @asynccontextmanager
    async def agent_setup_egress(self) -> AsyncGenerator[None, None]:
        self.open = True
        try:
            yield
        finally:
            self.open = False

    async def exec(self, command: str, **_: Any) -> ExecResult:
        self.commands.append((command, self.open))
        if command.startswith("for manager in"):
            return ExecResult(stdout="apt-get", return_code=0)
        if command.startswith("command -v") or command.endswith("command -v codex >/dev/null 2>&1"):
            return ExecResult(return_code=1)  # neither codex nor its system packages are in the image
        return ExecResult(stdout="codex-cli 0.153.0", return_code=0)


async def test_codex_installs_inside_the_window(tmp_path):
    """Harbor's codex downloads itself in setup, so every install command must run while the hosts are open."""
    from harbor.agents.installed.codex import Codex

    agent = _agent(tmp_path, inner="codex", hosts=SETUP_HOSTS)
    assert isinstance(agent, Codex)
    environment = _CodexInstallEnvironment()
    await agent.setup(environment)
    commands = [command for command, _ in environment.commands]
    assert any("apt-get" in command and "install" in command for command in commands), commands
    assert any("npm install -g @openai/codex" in command for command in commands), commands
    assert all(open_ for _, open_ in environment.commands)
    assert not environment.open


CODEX_DOWNLOAD = "npm install -g @openai/codex"


class _FlakyCodexInstallEnvironment(_CodexInstallEnvironment):
    """The codex download fails ``failures`` times with npm's network error, then installs; counts window opens."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.opens = 0

    @asynccontextmanager
    async def agent_setup_egress(self) -> AsyncGenerator[None, None]:
        self.opens += 1
        self.open = True
        try:
            yield
        finally:
            self.open = False

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        if CODEX_DOWNLOAD in command and self.failures:
            self.failures -= 1
            self.commands.append((command, self.open))
            return ExecResult(stderr="npm error code ENETUNREACH", return_code=1)
        return await super().exec(command, **kwargs)

    def downloads(self) -> list[bool]:
        """Whether the window was open for each codex download attempt."""
        return [open_ for command, open_ in self.commands if CODEX_DOWNLOAD in command]


def _codex(tmp_path, monkeypatch) -> Any:
    monkeypatch.setattr(agent_setup_egress, "SETUP_RETRY_DELAY_SEC", 0)
    agent = _agent(tmp_path, inner="codex", hosts=SETUP_HOSTS)
    agent.logger = Mock()
    return agent


async def test_a_failed_install_is_retried_once_inside_the_same_window(tmp_path, monkeypatch):
    agent = _codex(tmp_path, monkeypatch)
    environment = _FlakyCodexInstallEnvironment(failures=1)
    await agent.setup(environment)
    assert environment.downloads() == [True, True]
    assert environment.opens == 1 and not environment.open
    (warning,) = agent.logger.warning.call_args_list
    assert "attempt 1 of 2" in warning.args[0] % warning.args[1:]


async def test_a_second_failed_install_raises_and_closes_the_window(tmp_path, monkeypatch):
    agent = _codex(tmp_path, monkeypatch)
    environment = _FlakyCodexInstallEnvironment(failures=2)
    with pytest.raises(NonZeroAgentExitCodeError, match="ENETUNREACH"):
        await agent.setup(environment)
    assert environment.downloads() == [True, True]
    assert environment.opens == 1 and not environment.open


async def test_an_install_that_succeeds_runs_once(tmp_path, monkeypatch):
    agent = _codex(tmp_path, monkeypatch)
    environment = _FlakyCodexInstallEnvironment(failures=0)
    await agent.setup(environment)
    assert environment.downloads() == [True]
    agent.logger.warning.assert_not_called()


class _HangingCodexInstallEnvironment(_FlakyCodexInstallEnvironment):
    """The codex download never finishes."""

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        if CODEX_DOWNLOAD in command:
            self.commands.append((command, self.open))
            await asyncio.Event().wait()
        return await super().exec(command, **kwargs)


async def test_an_install_that_times_out_is_not_retried(tmp_path, monkeypatch):
    agent = _codex(tmp_path, monkeypatch)
    environment = _HangingCodexInstallEnvironment(failures=0)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(agent.setup(environment), timeout=0.1)  # as Harbor's setup timeout does
    assert environment.downloads() == [True]
    assert not environment.open


async def test_a_failed_setup_of_an_agent_that_is_not_installed_is_not_retried(tmp_path, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(agent_setup_egress, "SETUP_RETRY_DELAY_SEC", 0)

    async def setup(self, environment) -> None:
        events.append("setup")
        raise NonZeroAgentExitCodeError("install failed")

    monkeypatch.setattr(NopAgent, "setup", setup)
    with pytest.raises(NonZeroAgentExitCodeError):
        await _agent(tmp_path).setup(_WindowEnvironment(events))
    assert events == ["open", "setup", "close"]


async def test_a_dynamic_policy_environment_retries_inside_one_policy_switch(tmp_path, monkeypatch):
    from harbor.agents.installed.codex import Codex

    events: list[Any] = []
    attempts = iter([NonZeroAgentExitCodeError("npm error code ENETUNREACH"), None])

    async def setup(self, environment) -> None:
        events.append(("setup", environment.network_policy))
        if error := next(attempts):
            raise error

    monkeypatch.setattr(Codex, "setup", setup)
    environment = _PolicyEnvironment(events)
    await _codex(tmp_path, monkeypatch).setup(environment)
    opened = _allowlist(*SETUP_HOSTS)
    assert events == ["list", opened, ("setup", opened), ("setup", opened), NO_NETWORK, ("kill", KEEP)]


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


REFUSED_CALLS = {
    "exec": lambda environment: environment.exec("echo task"),
    "service_exec": lambda environment: environment.service_exec("echo task", service="main"),
    "upload_file": lambda environment: environment.upload_file("task.sh", "/tests/task.sh"),
    "upload_dir": lambda environment: environment.upload_dir("tests", "/tests"),
    "download_file": lambda environment: environment.download_file("/logs/agent.log", "agent.log"),
    "download_dir": lambda environment: environment.download_dir("/logs/agent", "agent"),
}


async def _refuses_commands(environment: _PolicyEnvironment) -> None:
    for call in REFUSED_CALLS.values():
        with pytest.raises(RuntimeError, match="did not close cleanly"):
            await call(environment)


@pytest.mark.parametrize("name", list(REFUSED_CALLS))
async def test_each_command_and_transfer_is_refused_after_a_failed_close(tmp_path, monkeypatch, name):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    environment.kill_exit = 1
    with pytest.raises(RuntimeError, match="could not stop processes"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    recorded = len(events)
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        await REFUSED_CALLS[name](environment)
    assert len(events) == recorded


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
    await environment.download_dir("/logs/agent", "agent")
    assert events == [
        "list",
        _allowlist(*SETUP_HOSTS),
        ("exec", "npm install -g agent"),
        ("service_exec", "pip install agent"),
        NO_NETWORK,
        ("kill", KEEP),
        ("exec", "echo task"),
        ("service_exec", "echo verifier"),
        ("download_dir", "/logs/agent"),
    ]
    assert not set(GUARDED_METHODS) & set(vars(environment))


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


async def test_an_open_that_fails_partway_keeps_refusing_commands(tmp_path, monkeypatch):
    """Harbor still caches the baseline, so a restore would be a no-op; nothing may run afterwards."""
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    opened = _allowlist(*SETUP_HOSTS)

    async def fail_partway(network_policy: NetworkPolicy) -> None:
        raise RuntimeError("applied some rules, then failed")

    environment.policy_hook = fail_partway
    with pytest.raises(RuntimeError, match="applied some rules"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list", opened]
    assert environment.network_policy == NO_NETWORK
    await _refuses_commands(environment)


async def test_an_open_cancelled_partway_keeps_refusing_commands(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    opening = asyncio.Event()

    async def hang_open(network_policy: NetworkPolicy) -> None:
        opening.set()
        await asyncio.Event().wait()

    environment.policy_hook = hang_open
    task = asyncio.create_task(_agent(tmp_path, hosts=SETUP_HOSTS).setup(environment))
    await opening.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["list", _allowlist(*SETUP_HOSTS)]
    await _refuses_commands(environment)


async def test_a_policy_the_provider_cannot_express_fails_before_anything_changes(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    environment.policy_supported = False
    with pytest.raises(ValueError, match="not supported"):
        await _agent(tmp_path, hosts=["*.npmjs.org"]).setup(environment)
    assert events == []
    assert not set(GUARDED_METHODS) & set(vars(environment))


async def test_a_compose_environment_keeps_the_task_policy(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events, capabilities=DYNAMIC_ALLOWLIST.model_copy(update={"docker_compose": True}))
    environment._uses_compose = True
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == [("setup", NO_NETWORK)]


async def test_a_compose_capable_environment_without_compose_services_opens_the_window(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events, capabilities=DYNAMIC_ALLOWLIST.model_copy(update={"docker_compose": True}))
    await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    opened = _allowlist(*SETUP_HOSTS)
    assert events == ["list", opened, ("setup", opened), NO_NETWORK, ("kill", KEEP)]


async def test_a_failed_snapshot_opens_nothing_and_leaves_commands_alone(tmp_path, monkeypatch):
    events: list[Any] = []
    _record_setup(monkeypatch, events)
    environment = _PolicyEnvironment(events)
    environment.list_exit = 1
    with pytest.raises(RuntimeError, match="could not list environment processes"):
        await _agent(tmp_path, hosts=SETUP_HOSTS).setup(environment)
    assert events == ["list"]
    assert not set(GUARDED_METHODS) & set(vars(environment))


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
