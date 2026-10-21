# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent channel for no-network tasks, and codex installed by upload instead of download."""

import asyncio
import ipaddress
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

import pytest
from harbor.environments.base import ExecResult
from harbor.models.task.config import NetworkMode, NetworkPolicy
from nemo_evals.harbor import agents, opensandbox_environment
from nemo_evals.harbor.agents import CODEX_ARCHIVE_ENV, REMOTE_ARCHIVE, REMOTE_BINARY, SandboxedCodex
from nemo_evals.harbor.opensandbox_environment import AgentChannelOpenSandboxEnvironment
from nhx_sandbox import egress
from nhx_sandbox.harbor_opensandbox_environment import NemoOpenSandboxEnvironment
from nhx_sandbox.opensandbox_policy import EgressVerificationError

IGW = "nemo-helix-api.ns.svc.cluster.local"


def _environment(mode: NetworkMode, channel: tuple[str, ...], trusted: tuple[str, ...] = ()) -> Any:
    """The environment's egress state only; ``egress_policy`` reads nothing else."""
    environment = object.__new__(AgentChannelOpenSandboxEnvironment)
    environment._network_policy = NetworkPolicy(network_mode=mode)
    environment._agent_channel_hosts = channel
    environment._trusted_allowed_hosts = trusted
    environment._agent_setup_hosts = ()
    environment._resolver_addresses = None
    return environment


def _allowed(policy: Any) -> set[str]:
    return {rule.target for rule in policy.rules if rule.action == "allow"}


def test_no_network_without_channel_stays_deny_all():
    policy = _environment(NetworkMode.NO_NETWORK, ()).egress_policy()
    assert policy.default_action == "deny" and policy.rules == ()


def test_no_network_with_channel_allows_only_the_channel():
    policy = _environment(NetworkMode.NO_NETWORK, (IGW,), trusted=(IGW, "pypi.org")).egress_policy()
    assert policy.default_action == "deny"
    assert IGW in _allowed(policy) and "pypi.org" not in _allowed(policy)


def test_public_tasks_keep_the_profile_allowlist():
    policy = _environment(NetworkMode.PUBLIC, (IGW,), trusted=(IGW, "pypi.org")).egress_policy()
    assert {IGW, "pypi.org"} <= _allowed(policy)


@pytest.mark.parametrize("web_search", ["cached", "live"])
def test_sandboxed_codex_refuses_web_search(tmp_path, web_search):
    with pytest.raises(ValueError, match="web_search=disabled"):
        SandboxedCodex(logs_dir=tmp_path, model_name="m", web_search=web_search)


def test_sandboxed_codex_disables_web_search_by_default(tmp_path):
    agent = SandboxedCodex(logs_dir=tmp_path, model_name="m")
    assert "-c web_search=disabled" in agent.build_cli_flags()


def test_a_gateway_alias_has_no_price_unless_a_pricing_model_names_one(tmp_path):
    tokens = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000, "cached_tokens": 0, "cache_write_tokens": 0}
    alias = "azure-openai-gpt-6-luna"  # what codex calls through the gateway; not in LiteLLM's table
    assert SandboxedCodex(logs_dir=tmp_path, model_name=alias)._compute_cost_from_pricing(**tokens) is None
    priced = SandboxedCodex(logs_dir=tmp_path, model_name=alias, pricing_model="azure/gpt-6-luna")
    cost = priced._compute_cost_from_pricing(**tokens, model_name=alias)
    assert cost is not None and cost > 0
    assert "azure/gpt-6-luna" not in priced.build_cli_flags()  # codex still calls the gateway alias


def test_harbor_accepts_pricing_model_before_queueing_trials():
    # Job.create validates agent_kwargs against the agent's options model before any trial runs.
    SandboxedCodex.preflight(kwargs={"pricing_model": "azure/gpt-6-luna"}, env={})
    with pytest.raises(ValueError, match="Unknown option 'pricing'"):
        SandboxedCodex.preflight(kwargs={"pricing": "x"}, env={})


class _RecordingEnvironment:
    def __init__(self) -> None:
        self.uploads: list[tuple[Path, str]] = []
        self.commands: list[str] = []

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploads.append((Path(source_path), target_path))

    async def exec(self, command: str, **_: Any) -> ExecResult:
        self.commands.append(command)
        return ExecResult(stdout="codex-cli 0.162.1", stderr="", return_code=0)


async def test_sandboxed_codex_uploads_the_pinned_archive(tmp_path, monkeypatch):
    archive = tmp_path / "codex.tar.gz"
    archive.write_bytes(b"archive")
    monkeypatch.setenv(CODEX_ARCHIVE_ENV, str(archive))
    environment = _RecordingEnvironment()
    await SandboxedCodex(logs_dir=tmp_path, model_name="m").install(environment)  # ty: ignore[invalid-argument-type]
    assert environment.uploads == [(archive, REMOTE_ARCHIVE)]
    assert any(REMOTE_BINARY in command and "apt-get" not in command for command in environment.commands)


async def test_sandboxed_codex_fails_without_the_archive(tmp_path, monkeypatch):
    monkeypatch.setenv(CODEX_ARCHIVE_ENV, str(tmp_path / "missing.tar.gz"))
    with pytest.raises(RuntimeError, match="codex archive not found"):
        await SandboxedCodex(logs_dir=tmp_path, model_name="m").install(_RecordingEnvironment())  # ty: ignore[invalid-argument-type]


def test_default_archive_path_is_in_the_dispatcher_image():
    assert agents.DEFAULT_CODEX_ARCHIVE == Path("/opt/nemo/codex/codex-x86_64-unknown-linux-musl.tar.gz")


SETUP = ("registry.npmjs.org", "*.pypi.org")


class _FakeSandbox:
    """The live sandbox's egress API: deny by default, rules keyed by target, as the sidecar keeps them."""

    def __init__(self, rules: tuple[Any, ...], calls: list[str]) -> None:
        self.rules = {rule.target: rule.action for rule in rules}
        self.calls = calls

    async def get_egress_policy(self) -> Any:
        rules = [SimpleNamespace(action=action, target=target) for target, action in self.rules.items()]
        return SimpleNamespace(default_action="deny", egress=rules)

    async def delete_egress_rules(self, targets: list[str]) -> None:
        self.calls.append("delete")
        for target in targets:
            self.rules.pop(target, None)

    async def patch_egress_rules(self, rules: list[Any]) -> None:
        self.calls.append("patch")
        self.rules.update({rule.target: rule.action for rule in rules})


class _SetupWindowEnvironment(AgentChannelOpenSandboxEnvironment):
    """The window's collaborators: a fake sidecar and an exec that records cleanup commands."""

    calls: list[str]
    execs: list[tuple[str, dict[str, Any]]]
    kill_exit: int

    @override
    async def _exec_cleanup(self, script: str, **env: str) -> ExecResult:
        command = f"{opensandbox_environment.SHELL} -c {script}"
        self.execs.append((command, {"env": {"PATH": opensandbox_environment.SYSTEM_PATH, **env}}))
        if "kill -9" in command:
            self.calls.append("kill")
            return ExecResult(stdout="", stderr="could not kill: 9", return_code=self.kill_exit)
        self.calls.append("list")
        return ExecResult(stdout="1:100\n7:200\n", stderr="", return_code=0)


def _setup_environment(
    monkeypatch, *, mode: NetworkMode = NetworkMode.NO_NETWORK, setup: tuple[str, ...] = SETUP, kill_exit: int = 0
) -> _SetupWindowEnvironment:
    monkeypatch.setattr(opensandbox_environment, "network_rules", lambda rules: list(rules))
    monkeypatch.setattr(egress, "_resolve_host", lambda target: ())  # no DNS from unit tests
    environment = object.__new__(_SetupWindowEnvironment)
    environment._network_policy = NetworkPolicy(network_mode=mode)
    environment._agent_channel_hosts = (IGW,)
    environment._agent_setup_hosts = setup
    environment._trusted_allowed_hosts = ("registry.npmjs.org",)
    environment._trusted_targets = frozenset(environment._trusted_allowed_hosts)
    environment._resolver_addresses = ()
    environment._egress_verification = "strict"
    environment._services = None
    environment.session_id = "s"
    environment.logger = logging.getLogger("test")
    environment.calls = []
    environment.execs = []
    environment.kill_exit = kill_exit
    environment._sandbox = _FakeSandbox(environment.egress_policy().rules, environment.calls)
    return environment


def _allowed_now(environment: Any) -> set[str]:
    return {target for target, action in environment._sandbox.rules.items() if action == "allow"}


async def test_setup_window_opens_the_setup_hosts_only_inside_the_block(monkeypatch):
    environment = _setup_environment(monkeypatch)
    assert _allowed_now(environment) == {IGW}
    async with environment.agent_setup_egress():
        assert _allowed_now(environment) == {IGW, *SETUP}
    assert _allowed_now(environment) == {IGW}
    # Processes are listed before opening; rules leave before leftovers are killed, so nothing reconnects.
    assert environment.calls == ["list", "patch", "delete", "kill"]


async def test_setup_window_closes_when_setup_fails(monkeypatch):
    environment = _setup_environment(monkeypatch)
    with pytest.raises(RuntimeError, match="install failed"):
        async with environment.agent_setup_egress():
            raise RuntimeError("install failed")
    assert _allowed_now(environment) == {IGW}
    assert environment.calls[-2:] == ["delete", "kill"]


async def test_cleanup_keeps_the_processes_from_before_setup_and_ignores_the_task_path(monkeypatch):
    environment = _setup_environment(monkeypatch)
    async with environment.agent_setup_egress():
        pass
    command, kwargs = environment.execs[-1]
    assert command.startswith("/bin/sh -c ")
    assert kwargs["env"] == {"PATH": opensandbox_environment.SYSTEM_PATH, "KEEP": "1:100 7:200"}


async def test_a_process_that_survives_the_kill_fails_the_trial(monkeypatch):
    environment = _setup_environment(monkeypatch, kill_exit=1)
    with pytest.raises(RuntimeError, match="could not stop processes left by agent setup"):
        async with environment.agent_setup_egress():
            pass
    assert _allowed_now(environment) == {IGW}  # the hosts were removed before the kill was attempted


async def test_setup_window_fails_when_the_sidecar_keeps_a_setup_host(monkeypatch):
    environment = _setup_environment(monkeypatch)

    async def ignore_delete(targets: list[str]) -> None:
        return None

    monkeypatch.setattr(environment._sandbox, "delete_egress_rules", ignore_delete)
    with pytest.raises(EgressVerificationError, match="still allows"):
        async with environment.agent_setup_egress():
            pass


async def test_a_setup_host_the_task_already_allows_is_left_in_place(monkeypatch):
    environment = _setup_environment(monkeypatch, mode=NetworkMode.PUBLIC)
    before = _allowed_now(environment)
    assert "registry.npmjs.org" in before  # the trusted allowlist already has it
    async with environment.agent_setup_egress():
        assert "*.pypi.org" in _allowed_now(environment)
    assert _allowed_now(environment) == before


async def test_without_setup_hosts_the_window_does_nothing(monkeypatch):
    environment = _setup_environment(monkeypatch, setup=())
    async with environment.agent_setup_egress():
        pass
    assert environment.calls == []


def test_a_task_cannot_claim_the_setup_hosts(monkeypatch):
    """The setup hosts are operator config for the window only; a task allowlist naming them is still checked."""
    environment = _setup_environment(monkeypatch)
    with pytest.raises(ValueError, match="trusted allowlist"):
        environment.validate_network_policy_support(
            NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["*.pypi.org"])
        )
    assert not environment.capabilities.dynamic_network_policy


def _denied_networks(environment: Any) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    return [
        ipaddress.ip_network(target)
        for target, action in environment._sandbox.rules.items()
        if action == "deny" and "/" in target
    ]


async def test_a_private_setup_host_is_carved_out_of_the_deny_ranges(monkeypatch):
    """An internal mirror on a private address must be reachable during setup, then denied again."""
    environment = _setup_environment(monkeypatch, setup=("10.77.0.9",))
    registry = ipaddress.ip_address("10.77.0.9")
    assert any(registry in network for network in _denied_networks(environment))
    async with environment.agent_setup_egress():
        assert "10.77.0.9" in _allowed_now(environment)
        assert not any(registry in network for network in _denied_networks(environment))
    assert "10.77.0.9" not in _allowed_now(environment)
    assert any(registry in network for network in _denied_networks(environment))


@pytest.mark.parametrize("failure", [asyncio.CancelledError, RuntimeError], ids=["cancelled", "failed"])
async def test_a_close_that_does_not_finish_refuses_every_later_command(monkeypatch, failure):
    """Harbor collects task artifacts after a failed setup; they must not run while the hosts may be open."""
    environment = _setup_environment(monkeypatch)

    async def interrupted_delete(targets: list[str]) -> None:
        raise failure("interrupted")

    monkeypatch.setattr(environment._sandbox, "delete_egress_rules", interrupted_delete)
    with pytest.raises(failure):
        async with environment.agent_setup_egress():
            pass
    with pytest.raises(RuntimeError, match="did not close cleanly"):
        await environment.exec("cat /app/answer")


async def test_commands_run_normally_while_the_window_is_open_and_after_it_closes(monkeypatch):
    environment = _setup_environment(monkeypatch)
    ran: list[str] = []

    async def parent_exec(self, command: str, **kwargs: Any) -> ExecResult:
        ran.append(command)
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(NemoOpenSandboxEnvironment, "exec", parent_exec)
    async with environment.agent_setup_egress():
        await environment.exec("apt-get install -y curl")
    await environment.exec("run the agent")
    assert ran == ["apt-get install -y curl", "run the agent"]
