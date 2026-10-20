# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent channel for no-network tasks, and codex installed by upload instead of download."""

from pathlib import Path
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
