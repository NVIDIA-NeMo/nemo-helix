# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoOpenSandboxEnvironment against a fake OpenSandbox SDK."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("harbor.environments.opensandbox")
pytest.importorskip("nhx_sandbox")
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

import harbor.environments.opensandbox as harbor_opensandbox
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from scaled_evals.harbor_opensandbox_cleanup import APPLIED_EGRESS_FILENAME
from scaled_evals.harbor_opensandbox_environment import (
    CREATE_ATTEMPT_METADATA_KEY,
    MANAGED_BY_METADATA_KEY,
    MANAGED_BY_METADATA_VALUE,
    NemoOpenSandboxEnvironment,
)
from tenacity import wait_none

TRUSTED = ["10.96.0.20", "*.nvidia.com", "2001:db8::/48"]


class SandboxApiException(Exception):
    """Name-matched by Harbor as a transient SDK error."""


class _Server:
    def __init__(self) -> None:
        self.sandboxes: dict[str, dict[str, str]] = {}
        self.killed: list[str] = []
        self.creates: list[dict[str, Any]] = []
        self.create_failures: list[BaseException] = []
        self.applied_default: str | None = None
        self.readback = True
        self._ids = (f"sbx-{n}" for n in itertools.count(1))

    def live(self) -> set[str]:
        return set(self.sandboxes) - set(self.killed)


class _Sandbox:
    def __init__(self, server: _Server, sandbox_id: str, policy: dict[str, Any]) -> None:
        self._server = server
        self.id = sandbox_id
        self._policy = policy
        if server.readback:
            self.get_egress_policy = self._get_egress_policy

    async def _get_egress_policy(self) -> SimpleNamespace:
        default = self._server.applied_default or self._policy["defaultAction"]
        rules = [SimpleNamespace(**rule) for rule in self._policy["egress"]]
        return SimpleNamespace(default_action=default, egress=rules)

    async def get_info(self) -> SimpleNamespace:
        return SimpleNamespace(status=SimpleNamespace(state="Running"))

    async def is_healthy(self) -> bool:
        return True

    async def kill(self) -> None:
        self._server.killed.append(self.id)

    async def close(self) -> None:
        pass


class _Manager:
    def __init__(self, server: _Server) -> None:
        self._server = server

    async def list_sandbox_infos(self, selector: SimpleNamespace) -> SimpleNamespace:
        matches = [
            SimpleNamespace(id=sandbox_id)
            for sandbox_id, metadata in self._server.sandboxes.items()
            if sandbox_id not in self._server.killed and selector.metadata.items() <= metadata.items()
        ]
        return SimpleNamespace(sandbox_infos=matches, pagination=SimpleNamespace(has_next_page=False))

    async def kill_sandbox(self, sandbox_id: str) -> None:
        self._server.killed.append(sandbox_id)

    async def close(self) -> None:
        pass


class _RunCommandOpts:
    model_fields = {"working_directory": None, "timeout": None, "envs": None, "uid": None}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Commands:
    """``sandbox.commands`` whose default user is ``image_uid``; ``None`` makes ``id -u`` fail."""

    def __init__(self, image_uid: int | None) -> None:
        self.image_uid = image_uid
        self.runs: list[tuple[str, dict[str, Any]]] = []

    async def run(self, command: str, opts: _RunCommandOpts) -> SimpleNamespace:
        self.runs.append((command, opts.kwargs))
        stdout = []
        if command == "id -u":
            if self.image_uid is None:
                raise SandboxApiException("execd unavailable")
            stdout = [SimpleNamespace(text=f"{self.image_uid}\n")]
        return SimpleNamespace(logs=SimpleNamespace(stdout=stdout, stderr=[]), exit_code=0, error=None)

    def uids_sent(self, command: str) -> list[int | None]:
        return [opts.get("uid") for run, opts in self.runs if run == command]

    def probes(self) -> int:
        return sum(run == "id -u" for run, _ in self.runs)


def _fake_sdk(server: _Server) -> dict[str, Any]:
    class Sandbox:
        @staticmethod
        async def create(image: Any, **kwargs: Any) -> _Sandbox:
            server.creates.append(kwargs)
            sandbox_id = next(server._ids)
            server.sandboxes[sandbox_id] = dict(kwargs["metadata"])
            if server.create_failures:
                # The server created the sandbox, but the client never got a handle for it.
                raise server.create_failures.pop(0)
            return _Sandbox(server, sandbox_id, kwargs["network_policy"].payload)

    class SandboxManager:
        @staticmethod
        async def create(connection_config: Any) -> _Manager:
            return _Manager(server)

    return {
        "Sandbox": Sandbox,
        "RunCommandOpts": _RunCommandOpts,
        "ConnectionConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "NetworkPolicy": SimpleNamespace(model_validate=lambda payload: SimpleNamespace(payload=payload)),
        "SandboxManager": SandboxManager,
        "SandboxFilter": lambda metadata, page: SimpleNamespace(metadata=metadata, page=page),
    }


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    server = _Server()
    monkeypatch.setattr(harbor_opensandbox, "_HAS_OPENSANDBOX", True)
    monkeypatch.setattr(
        harbor_opensandbox.OpenSandboxEnvironment._create_sandbox.retry,  # ty: ignore[unresolved-attribute]  # tenacity
        "wait",
        wait_none(),
    )
    monkeypatch.setattr(NemoOpenSandboxEnvironment, "_load_opensandbox", lambda self: _fake_sdk(server))
    return server


def _environment(
    tmp_path: Path,
    mode: NetworkMode,
    allowed_hosts: list[str] | None = None,
    **kwargs: Any,
) -> NemoOpenSandboxEnvironment:
    return NemoOpenSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="hello-world",
        session_id="hello-world__abc__env",
        trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(docker_image="registry.example/hello:1"),
        network_policy=NetworkPolicy(network_mode=mode, allowed_hosts=allowed_hosts or []),
        domain="opensandbox.example",
        api_key="",
        metadata={"nemo-scaled-evals-run": "run-1"},
        trusted_allowed_hosts=TRUSTED,
        resolver_addresses=["10.96.0.10"],
        **kwargs,
    )


def test_harbor_private_hooks_still_exist() -> None:
    """The subclass overrides Harbor internals; an upgrade that moves them must fail here, not in a cluster."""
    import inspect
    from importlib.metadata import version

    base = harbor_opensandbox.OpenSandboxEnvironment
    assert version("harbor") == "0.20.0"
    assert list(inspect.signature(base._build_network_policy).parameters) == ["self", "sdk"]
    assert list(inspect.signature(base._create_sandbox).parameters) == ["self", "sdk"]
    assert "self._metadata" in inspect.getsource(base._create_sandbox)
    assert "self._build_network_policy(sdk)" in inspect.getsource(base._create_sandbox)
    assert hasattr(base, "_safe_kill")
    assert hasattr(base, "_build_connection_config")
    assert list(inspect.signature(base._resolve_uid).parameters) == ["self", "user"]
    assert "self._resolve_uid(" in inspect.getsource(base._build_run_command_opts)
    assert "self._sandbox.commands.run(" in inspect.getsource(base.exec)


def _policy_sent(server: _Server) -> dict[str, Any]:
    return server.creates[-1]["network_policy"].payload


async def test_no_network_is_deny_all(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)

    await environment._create_sandbox(environment._load_opensandbox())

    assert _policy_sent(server) == {"defaultAction": "deny", "egress": []}


async def test_public_gets_the_trusted_allowlist_not_unrestricted_access(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.PUBLIC)

    await environment._create_sandbox(environment._load_opensandbox())

    policy = _policy_sent(server)
    allowed = {rule["target"] for rule in policy["egress"] if rule["action"] == "allow"}
    denied = {rule["target"] for rule in policy["egress"] if rule["action"] == "deny"}
    assert policy["defaultAction"] == "deny"
    assert allowed == {*TRUSTED, "10.96.0.10"}
    assert "192.168.0.0/16" in denied
    assert "10.0.0.0/8" not in denied


async def test_records_the_verified_policy_for_the_supervisor(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.PUBLIC)

    sandbox = await environment._create_sandbox(environment._load_opensandbox())

    record = json.loads((tmp_path / APPLIED_EGRESS_FILENAME).read_text())
    canonical = json.dumps(record["policy"], sort_keys=True, separators=(",", ":"))
    assert record["sandbox_id"] == sandbox.id
    assert record["network_mode"] == "public"
    assert record["policy"] == _policy_sent(server)
    assert record["policy_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


async def test_allowlist_task_narrows_the_trusted_allowlist(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.ALLOWLIST, ["*.NVIDIA.com"])

    await environment._create_sandbox(environment._load_opensandbox())

    allowed = {rule["target"] for rule in _policy_sent(server)["egress"] if rule["action"] == "allow"}
    assert allowed == {"*.NVIDIA.com", "10.96.0.10"}


def test_allowlist_task_cannot_extend_the_trusted_allowlist(tmp_path: Path, server: _Server) -> None:
    with pytest.raises(ValueError, match=r"\['pypi.org'\].*does not include"):
        _environment(tmp_path, NetworkMode.ALLOWLIST, ["*.nvidia.com", "pypi.org"])

    assert server.creates == []


def test_advertises_static_allowlist_support(tmp_path: Path, server: _Server) -> None:
    capabilities = _environment(tmp_path, NetworkMode.PUBLIC).capabilities

    assert capabilities.network_allowlist
    assert capabilities.network_allowlist_wildcard_hostnames
    assert capabilities.network_allowlist_ipv6_cidrs
    assert not capabilities.dynamic_network_policy


async def test_labels_every_create_and_restores_caller_metadata(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)

    await environment._create_sandbox(environment._load_opensandbox())

    metadata = server.creates[-1]["metadata"]
    assert metadata["nemo-scaled-evals-run"] == "run-1"
    assert metadata["session_id"] == "hello-world__abc__env"
    assert metadata[MANAGED_BY_METADATA_KEY] == MANAGED_BY_METADATA_VALUE
    assert len(metadata[CREATE_ATTEMPT_METADATA_KEY]) == 32
    assert environment._metadata == {"nemo-scaled-evals-run": "run-1"}


async def test_kills_sandboxes_orphaned_by_a_retried_create(tmp_path: Path, server: _Server) -> None:
    server.create_failures.append(SandboxApiException("lost response"))
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)

    sandbox = await environment._create_sandbox(environment._load_opensandbox())

    assert len(server.creates) == 2
    assert (
        server.creates[0]["metadata"][CREATE_ATTEMPT_METADATA_KEY]
        == (server.creates[1]["metadata"][CREATE_ATTEMPT_METADATA_KEY])
    )
    assert server.live() == {sandbox.id}


async def test_failed_create_kills_everything_the_attempt_made(tmp_path: Path, server: _Server) -> None:
    server.create_failures.append(ValueError("bad request"))
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)

    with pytest.raises(ValueError, match="bad request"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert server.sandboxes and server.live() == set()
    assert environment._metadata == {"nemo-scaled-evals-run": "run-1"}


async def test_kills_the_sandbox_when_the_applied_default_is_wrong(tmp_path: Path, server: _Server) -> None:
    server.applied_default = "allow"
    environment = _environment(tmp_path, NetworkMode.PUBLIC)

    with pytest.raises(RuntimeError, match="applied egress default_action='allow'"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert server.live() == set()


async def test_kills_the_sandbox_when_the_policy_cannot_be_read_back(tmp_path: Path, server: _Server) -> None:
    server.readback = False
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)

    with pytest.raises(RuntimeError, match="cannot report its applied egress policy"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert server.live() == set()


async def test_strict_verification_rejects_a_dropped_rule(tmp_path: Path, server: _Server) -> None:
    environment = _environment(tmp_path, NetworkMode.PUBLIC, egress_verification="strict")
    sdk = environment._load_opensandbox()
    create = sdk["Sandbox"].create

    async def drop_rules(image: Any, **kwargs: Any) -> _Sandbox:
        sandbox = await create(image, **kwargs)
        sandbox._policy = {**sandbox._policy, "egress": sandbox._policy["egress"][1:]}
        return sandbox

    sdk["Sandbox"].create = drop_rules

    with pytest.raises(RuntimeError, match="did not apply 1 requested egress rule"):
        await environment._create_sandbox(sdk)

    assert server.live() == set()


def _started(tmp_path: Path, image_uid: int | None) -> tuple[NemoOpenSandboxEnvironment, _Commands]:
    environment = _environment(tmp_path, NetworkMode.NO_NETWORK)
    commands = _Commands(image_uid)
    environment._sandbox = SimpleNamespace(commands=commands)
    return environment, commands


async def test_root_commands_run_as_the_image_user_on_a_non_root_image(tmp_path: Path, server: _Server) -> None:
    environment, commands = _started(tmp_path, image_uid=10001)

    await environment.exec("tmux -V", user="root")
    await environment.exec("chmod 777 /logs/agent", user=0)

    assert commands.uids_sent("tmux -V") == [None]
    assert commands.uids_sent("chmod 777 /logs/agent") == [None]
    assert commands.probes() == 1


async def test_a_root_default_user_is_mapped_too(tmp_path: Path, server: _Server) -> None:
    environment, commands = _started(tmp_path, image_uid=10001)
    environment.default_user = "root"

    await environment.exec("mkdir -p /installed-agent")

    assert commands.uids_sent("mkdir -p /installed-agent") == [None]


async def test_root_images_keep_uid_0(tmp_path: Path, server: _Server) -> None:
    environment, commands = _started(tmp_path, image_uid=0)

    await environment.exec("apt-get install -y curl", user="root")

    assert commands.uids_sent("apt-get install -y curl") == [0]


async def test_non_root_requests_do_not_probe(tmp_path: Path, server: _Server) -> None:
    environment, commands = _started(tmp_path, image_uid=10001)

    await environment.exec("whoami")
    await environment.exec("whoami", user=1234)

    assert commands.uids_sent("whoami") == [None, 1234]
    assert commands.probes() == 0


async def test_a_failed_probe_keeps_harbor_behavior(tmp_path: Path, server: _Server) -> None:
    environment, commands = _started(tmp_path, image_uid=None)

    await environment.exec("tmux -V", user="root")
    await environment.exec("tmux -V", user="root")

    assert commands.uids_sent("tmux -V") == [0, 0]
    assert commands.probes() == 1
