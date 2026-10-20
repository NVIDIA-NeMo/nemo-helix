# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent channel for no-network tasks, and codex installed by upload instead of download."""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.environments.base import ExecResult
from harbor.models.task.config import NetworkMode, NetworkPolicy
from nemo_evals.harbor import agents
from nemo_evals.harbor.agents import CODEX_ARCHIVE_ENV, REMOTE_ARCHIVE, REMOTE_BINARY, SandboxedCodex
from nemo_evals.harbor.opensandbox_environment import AgentChannelOpenSandboxEnvironment

IGW = "nemo-helix-api.ns.svc.cluster.local"


def _environment(mode: NetworkMode, channel: tuple[str, ...], trusted: tuple[str, ...] = ()) -> Any:
    """The environment's egress state only; ``egress_policy`` reads nothing else."""
    environment = object.__new__(AgentChannelOpenSandboxEnvironment)
    environment._network_policy = NetworkPolicy(network_mode=mode)
    environment._agent_channel_hosts = channel
    environment._trusted_allowed_hosts = trusted
    environment._agent_setup_hosts = ()
    environment._agent_setup_targets = frozenset()
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

    def __init__(self, rules: tuple[Any, ...]) -> None:
        self.rules = {rule.target: rule.action for rule in rules}
        self.calls: list[str] = []

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


def _setup_environment(monkeypatch, channel: tuple[str, ...] = (IGW,)) -> Any:
    module = pytest.importorskip("nemo_evals.harbor.opensandbox_environment")
    monkeypatch.setattr(module, "network_rules", lambda rules: list(rules))
    environment = object.__new__(module.AgentChannelOpenSandboxEnvironment)
    environment._network_policy = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=list(SETUP))
    environment._agent_channel_hosts = channel
    environment._agent_setup_hosts = SETUP
    environment._agent_setup_targets = frozenset(SETUP)
    environment._trusted_allowed_hosts = ()
    environment._trusted_targets = frozenset()
    environment._resolver_addresses = ()
    environment._services = None
    environment._egress_verification = "strict"
    environment._startup_pids = ("1:100", "7:200")
    environment.session_id = "s"
    environment.logger = logging.getLogger("test")
    environment._sandbox = _FakeSandbox(environment.egress_policy().rules)
    commands: list[str] = []

    async def exec(command: str, **_: Any) -> ExecResult:
        commands.append(command)
        environment._sandbox.calls.append("kill")
        return ExecResult(stdout="", stderr="", return_code=0)

    environment.exec = exec
    environment.commands = commands
    return environment


def test_setup_policy_allows_setup_hosts_and_the_channel(monkeypatch):
    environment = _setup_environment(monkeypatch)
    assert {*SETUP, IGW} <= _allowed(environment.egress_policy())
    assert environment.capabilities.dynamic_network_policy


async def test_switch_to_no_network_drops_setup_hosts_then_kills_leftovers(monkeypatch):
    environment = _setup_environment(monkeypatch)
    await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    sandbox = environment._sandbox
    assert {target for target, action in sandbox.rules.items() if action == "allow"} == {IGW}
    # Rules go before the kill, so nothing the install left behind can reconnect in between.
    assert sandbox.calls[:2] == ["delete", "kill"]
    assert "KEEP='1:100 7:200'" in environment.commands[0]


async def test_switch_back_to_setup_kills_before_reopening(monkeypatch):
    environment = _setup_environment(monkeypatch)
    await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    environment._sandbox.calls.clear()
    await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=list(SETUP)))
    assert environment._sandbox.calls == ["kill", "patch"]
    assert set(SETUP) <= set(environment._sandbox.rules)


async def test_switch_without_a_channel_closes_everything(monkeypatch):
    environment = _setup_environment(monkeypatch, channel=())
    await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    assert not any(action == "allow" for action in environment._sandbox.rules.values())


async def test_switch_fails_when_the_sidecar_keeps_a_setup_host(monkeypatch):
    from nhx_sandbox.opensandbox_policy import EgressVerificationError

    environment = _setup_environment(monkeypatch)

    async def ignore_delete(targets: list[str]) -> None:
        return None

    environment._sandbox.delete_egress_rules = ignore_delete
    with pytest.raises(EgressVerificationError, match="still allows"):
        await environment.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))


def test_setup_hosts_pass_validation_without_being_trusted(monkeypatch):
    environment = _setup_environment(monkeypatch)
    environment.validate_network_policy_support()
    with pytest.raises(ValueError, match="trusted allowlist"):
        environment.validate_network_policy_support(
            NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["evil.example.com"])
        )


def test_without_setup_hosts_the_policy_stays_fixed(monkeypatch):
    environment = _setup_environment(monkeypatch)
    environment._agent_setup_hosts = ()
    environment._agent_setup_targets = frozenset()
    assert not environment.capabilities.dynamic_network_policy
