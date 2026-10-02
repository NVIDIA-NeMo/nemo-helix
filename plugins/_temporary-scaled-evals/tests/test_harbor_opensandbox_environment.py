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
from scaled_evals.harbor_opensandbox_cleanup import APPLIED_EGRESS_GLOB, applied_egress_filename
from scaled_evals.harbor_opensandbox_environment import (
    CREATE_ATTEMPT_METADATA_KEY,
    MANAGED_BY_METADATA_KEY,
    MANAGED_BY_METADATA_VALUE,
    NemoOpenSandboxEnvironment,
    _sandbox_role,
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
    *,
    session_id: str = "hello-world__abc__env",
    **kwargs: Any,
) -> NemoOpenSandboxEnvironment:
    return NemoOpenSandboxEnvironment(
        environment_dir=tmp_path,
        environment_name="hello-world",
        session_id=session_id,
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

    record = json.loads((tmp_path / applied_egress_filename(sandbox.id)).read_text())
    canonical = json.dumps(record["policy"], sort_keys=True, separators=(",", ":"))
    assert record["sandbox_id"] == sandbox.id
    assert record["role"] == "agent"
    assert record["network_mode"] == "public"
    assert record["policy"] == _policy_sent(server)
    assert record["policy_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


async def test_agent_and_separate_verifier_each_keep_their_record(tmp_path: Path, server: _Server) -> None:
    agent = _environment(tmp_path, NetworkMode.PUBLIC, session_id="hello-world__abc__env")
    verifier = _environment(tmp_path, NetworkMode.NO_NETWORK, session_id="hello-world__abc__verifier__task")

    agent_sandbox = await agent._create_sandbox(agent._load_opensandbox())
    verifier_sandbox = await verifier._create_sandbox(verifier._load_opensandbox())

    records = {
        record["sandbox_id"]: record
        for record in (json.loads(path.read_text()) for path in tmp_path.glob(APPLIED_EGRESS_GLOB))
    }
    assert set(records) == {agent_sandbox.id, verifier_sandbox.id}
    assert records[agent_sandbox.id]["role"] == "agent"
    assert records[verifier_sandbox.id]["role"] == "verifier"
    assert records[verifier_sandbox.id]["network_mode"] == "no-network"


@pytest.mark.parametrize(
    ("session_id", "role"),
    [
        ("trial-1__env", "agent"),
        ("trial-1__verifier__task", "verifier"),
        ("trial-1__verifier__env", "verifier"),
        ("trial-1-with-a-long-name__verifier__step-o__1a2b3c4d", "verifier"),
    ],
)
def test_sandbox_role_from_harbor_session_id(session_id: str, role: str) -> None:
    assert _sandbox_role(session_id) == role


def test_harbor_session_naming_still_matches_sandbox_role() -> None:
    """``_sandbox_role`` relies on Harbor's session ID naming; an upgrade that changes it must fail here."""
    import inspect

    from harbor.trial.trial import Trial

    assert "}__verifier__{" in inspect.getsource(Trial._separate_verifier_session_id)
    assert '}__env"' in inspect.getsource(Trial)


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
