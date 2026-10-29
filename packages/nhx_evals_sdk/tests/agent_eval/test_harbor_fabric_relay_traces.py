# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Fabric agent with Relay brings Relay's OTLP trace back with its Harbor trial."""

import glob
import hashlib
import importlib.metadata
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from harbor.environments.base import BaseEnvironment
from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent
from nhx_evals_sdk.agent_eval.runtimes.fabric import otlp_receiver, tcp_forwarder
from nhx_evals_sdk.agent_eval.runtimes.harbor import fabric_agent
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent import (
    RELAY_CLI_DIR,
    RELAY_CLI_RELEASES_URL,
    RELAY_OTLP_PORT,
    RELAY_TRACES_DIR,
    DetachedHelper,
    NemoFabricAgent,
    detached_start_command,
)
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent import FabricInstalledAgent
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

RELAY_CLI = b"#!/bin/sh\necho nemo-relay 0.10.4\n"
RECEIVER_ENDPOINT = f"http://127.0.0.1:{RELAY_OTLP_PORT}/v1/traces"


def _config(*, relay: bool = True, opentelemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "metadata": {"name": "codex-agent"},
        "harness": {"adapter_id": "nvidia.fabric.codex"},
    }
    if relay:
        config["relay"] = {"observability": {"opentelemetry": opentelemetry} if opentelemetry else {}}
    return config


def _agent(tmp_path: Path, **config: Any) -> NemoFabricAgent:
    return NemoFabricAgent(logs_dir=tmp_path, fabric_config=_config(**config), fabric_workspace="/app")


class _Result:
    return_code = 0
    stdout = ""
    stderr = ""


class _Environment:
    """Records what the agent asks of the container, in order."""

    default_user = "agent"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def exec(self, command: str, **kwargs: Any) -> _Result:
        self.events.append(command)
        return _Result()

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploaded[target_path] = Path(source_path).read_text(encoding="utf-8")

    async def upload_dir(self, source_path: Path, target_path: str) -> None:
        pass


def _endpoints(agent: NemoFabricAgent) -> list[str]:
    relay = agent._build_config().relay
    assert relay is not None and relay.observability is not None and relay.observability.opentelemetry is not None
    assert relay.observability.opentelemetry.enabled
    return [endpoint.endpoint for endpoint in relay.observability.opentelemetry.endpoints]


def test_relay_exports_its_trace_to_the_receiver_beside_the_harness(tmp_path: Path) -> None:
    assert _endpoints(_agent(tmp_path)) == [RECEIVER_ENDPOINT]


def test_the_configs_own_trace_destinations_are_kept_only_when_it_enabled_them(tmp_path: Path) -> None:
    own = {"type": "full", "endpoint": "http://collector:4318/v1/traces"}

    assert _endpoints(_agent(tmp_path, opentelemetry={"enabled": True, "endpoints": [own]})) == [
        own["endpoint"],
        RECEIVER_ENDPOINT,
    ]
    assert _endpoints(_agent(tmp_path, opentelemetry={"enabled": False, "endpoints": [own]})) == [RECEIVER_ENDPOINT]


def test_a_config_without_relay_gets_no_trace_capture(tmp_path: Path) -> None:
    agent = _agent(tmp_path, relay=False)

    assert agent._build_config().relay is None
    assert RELAY_CLI_DIR not in (agent._runner_path_prefix or "")


async def test_setup_installs_the_relay_cli_and_the_receiver(tmp_path: Path) -> None:
    agent, environment = _agent(tmp_path), _Environment()

    await agent.setup(environment)

    assert environment.uploaded["/tmp/nemo-fabric-otlp-receiver.py"] == Path(otlp_receiver.__file__).read_text()
    [install] = [command for command in environment.events if RELAY_CLI_RELEASES_URL in command]
    assert install.startswith(agent._runner_python) and RELAY_CLI_DIR in install
    assert (agent._runner_path_prefix or "").split(":")[0] == RELAY_CLI_DIR


async def test_setup_without_relay_installs_neither(tmp_path: Path) -> None:
    environment = _Environment()

    await _agent(tmp_path, relay=False).setup(environment)

    assert environment.uploaded == {}
    assert not [command for command in environment.events if RELAY_CLI_RELEASES_URL in command]


async def test_the_installed_agent_installs_them_too(tmp_path: Path) -> None:
    agent = FabricInstalledAgent(
        logs_dir=tmp_path,
        fabric_config=_config(),
        fabric_package="nemo-fabric[relay]==0.5.0",
        fabric_workspace="/app",
        fabric_default_max_turns=None,
    )
    environment = _Environment()

    await agent.install(cast(BaseEnvironment, environment))

    assert "/tmp/nemo-fabric-otlp-receiver.py" in environment.uploaded


@pytest.mark.parametrize("fails", [False, True], ids=["succeeds", "fails"])
async def test_the_receiver_listens_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    async def fabric_run(self: Any, instruction: str, environment: _Environment, context: Any) -> None:
        environment.events.append("fabric run")
        if fails:
            raise RuntimeError("the harness failed")

    monkeypatch.setattr(FabricAgent, "run", fabric_run)
    environment = _Environment()

    if fails:
        with pytest.raises(RuntimeError):
            await _agent(tmp_path).run("say hello", environment, None)
    else:
        await _agent(tmp_path).run("say hello", environment, None)

    start, run, stop = environment.events
    assert f"/tmp/nemo-fabric-otlp-receiver.py {RELAY_TRACES_DIR} {RELAY_OTLP_PORT}" in start
    assert otlp_receiver.READY_FILENAME in start
    assert run == "fabric run"
    assert "kill" in stop and "/tmp/nemo-fabric-relay-" in stop


def test_after_the_run_the_exports_become_the_trials_otlp_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    export = ExportTraceServiceRequest(
        resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[Span(name="openai.responses")])])]
    )
    (traces / f"000001{otlp_receiver.EXPORT_SUFFIX}").write_bytes(export.SerializeToString())

    _agent(tmp_path).populate_context_post_run(_Context())

    assert "openai.responses" in (traces / "relay.jsonl").read_text()
    assert not list(traces.glob(f"*{otlp_receiver.EXPORT_SUFFIX}"))


class _Context:
    """Harbor's ``AgentContext``, as far as the post-run hook reads it."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return None


def _release(tmp_path: Path, *, checksum: str | None = None, binary: bytes = RELAY_CLI) -> tuple[str, Path]:
    """A local copy of a nemo-relay CLI release, laid out like GitHub's download URLs."""
    version = importlib.metadata.version("nemo-relay")
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine().lower()]
    libc = "musl" if glob.glob("/lib/ld-musl-*") else "gnu"
    asset = f"nemo-relay-cli-{arch}-unknown-linux-{libc}-{version}"
    release = tmp_path / "releases" / version
    release.mkdir(parents=True)
    (release / asset).write_bytes(binary)
    digest = checksum or hashlib.sha256(binary).hexdigest()
    (release / f"{asset}.sha256").write_text(f"{digest}  {asset}\n")
    return (tmp_path / "releases").as_uri(), tmp_path / "bin"


def _install(base_url: str, target: Path, path: str = "/nonexistent") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", fabric_agent._RELAY_CLI_INSTALL, base_url, str(target), "0,0"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": path},
        check=False,
    )


def test_the_cli_matching_the_installed_bindings_is_installed(tmp_path: Path) -> None:
    base_url, target = _release(tmp_path)

    result = _install(base_url, target)

    assert result.returncode == 0, result.stderr
    assert (target / "nemo-relay").read_bytes() == RELAY_CLI
    assert os.access(target / "nemo-relay", os.X_OK)


def _install_in_process(
    monkeypatch: pytest.MonkeyPatch, base_url: str, target: Path, urlopen: Callable[..., Any]
) -> object:
    """Run the install script here with ``urlopen`` replaced; the exit code, or 0 when it runs to the end."""
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sys, "argv", ["-c", base_url, str(target), "0,0"])
    try:
        exec(fabric_agent._RELAY_CLI_INSTALL, {"__name__": "__main__"})
    except SystemExit as exit_:
        return exit_.code
    return 0


def test_a_download_that_fails_transiently_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A setup sandbox's egress drops connections and DNS answers now and then; one failure mustn't end the trial."""
    base_url, target = _release(tmp_path)
    real_urlopen = urllib.request.urlopen
    failures = iter([urllib.error.URLError("[Errno -5] No address associated with hostname")])

    def flaky_urlopen(url: str, timeout: float) -> Any:
        if (failure := next(failures, None)) is not None:
            raise failure
        return real_urlopen(url, timeout=timeout)

    assert _install_in_process(monkeypatch, base_url, target, flaky_urlopen) == 0
    assert (target / "nemo-relay").read_bytes() == RELAY_CLI


def test_a_download_that_keeps_failing_gives_up_after_its_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url, target = _release(tmp_path)
    attempts: list[str] = []

    def unreachable(url: str, timeout: float) -> Any:
        attempts.append(url)
        raise urllib.error.URLError("[Errno 111] Connection refused")

    code = _install_in_process(monkeypatch, base_url, target, unreachable)

    assert isinstance(code, str) and "(attempts: 3)" in code and "Connection refused" in code
    assert len(attempts) == 3
    assert not (target / "nemo-relay").exists()


def test_a_missing_release_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 means no such release for this platform; retrying only delays the error."""
    base_url, target = _release(tmp_path)
    attempts: list[str] = []

    def not_found(url: str, timeout: float) -> Any:
        attempts.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)

    code = _install_in_process(monkeypatch, base_url, target, not_found)

    assert isinstance(code, str) and "(attempts: 1)" in code and "404" in code
    assert len(attempts) == 1


def test_a_cli_that_doesnt_match_its_checksum_is_refused(tmp_path: Path) -> None:
    base_url, target = _release(tmp_path, checksum="0" * 64)

    result = _install(base_url, target)

    assert result.returncode != 0 and "sha256" in result.stderr
    assert not (target / "nemo-relay").exists()


def test_a_cli_that_doesnt_run_in_the_image_is_refused(tmp_path: Path) -> None:
    """A build for the wrong libc downloads fine and fails only when Fabric launches it, with a vaguer error."""
    base_url, target = _release(tmp_path, binary=b"#!/bin/sh\necho missing loader >&2\nexit 127\n")

    result = _install(base_url, target)

    assert (
        result.returncode != 0 and "does not run in this image" in result.stderr and "missing loader" in result.stderr
    )
    assert not (target / "nemo-relay").exists()


def test_a_cli_already_on_path_is_kept(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "nemo-relay").write_text("#!/bin/sh\n")
    (existing / "nemo-relay").chmod(0o755)

    result = _install((tmp_path / "missing").as_uri(), tmp_path / "bin", path=str(existing))

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "bin").exists()


GATEWAY_PATH = "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"


def _gateway_agent(tmp_path: Path, base_url: str, *, relay: bool = True) -> NemoFabricAgent:
    config = _config(relay=relay)
    config["models"] = {"default": {"provider": "openai", "model": "default/recorded", "base_url": base_url}}
    return NemoFabricAgent(logs_dir=tmp_path, fabric_config=config, fabric_workspace="/app")


@pytest.mark.parametrize(
    ("base_url", "relay", "forwarded"),
    [
        pytest.param(f"http://host.docker.internal:8598{GATEWAY_PATH}", True, True, id="plain HTTP gateway"),
        pytest.param(f"https://helix.example.com{GATEWAY_PATH}", True, False, id="HTTPS gateway"),
        pytest.param(f"http://127.0.0.1:8080{GATEWAY_PATH}", True, False, id="loopback gateway"),
        pytest.param(f"http://host.docker.internal:8598{GATEWAY_PATH}", False, False, id="no Relay"),
        pytest.param("http://models.example.com/v1", True, False, id="not the gateway"),
    ],
)
def test_relay_reaches_a_plain_http_gateway_through_a_loopback_forwarder(
    tmp_path: Path, base_url: str, relay: bool, forwarded: bool
) -> None:
    """Relay sends model credentials only over HTTPS or to loopback."""
    agent = _gateway_agent(tmp_path, base_url, relay=relay)

    model = agent._build_config().models["default"]

    if forwarded:
        assert model.base_url == f"http://127.0.0.1:14320{GATEWAY_PATH}"
        assert agent._gateway_forwards() == {("host.docker.internal", 8598): 14320}
    else:
        assert model.base_url == base_url
        assert agent._gateway_forwards() == {}


async def test_the_forwarder_runs_beside_the_harness_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fabric_run(self: Any, instruction: str, environment: _Environment, context: Any) -> None:
        environment.events.append("fabric run")

    monkeypatch.setattr(FabricAgent, "run", fabric_run)
    environment = _Environment()

    await _gateway_agent(tmp_path, f"http://host.docker.internal:8598{GATEWAY_PATH}").run("hi", environment, None)

    start, run, stop = environment.events
    assert "/tmp/nemo-fabric-tcp-forwarder.py 14320 host.docker.internal 8598" in start
    assert (run, "kill" in stop) == ("fabric run", True)


def test_the_forwarder_copies_bytes_both_ways_until_each_side_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    client, client_end = socket.socketpair()
    upstream, upstream_end = socket.socketpair()
    monkeypatch.setattr(tcp_forwarder.socket, "create_connection", lambda address, timeout: upstream_end)
    serving = threading.Thread(target=tcp_forwarder._serve, args=(client_end, "gateway", 8080))
    serving.start()

    client.sendall(b"POST /v1/responses HTTP/1.1\r\ntraceparent: 00-a-b-01\r\n\r\n")
    client.shutdown(socket.SHUT_WR)
    assert _read_all(upstream) == b"POST /v1/responses HTTP/1.1\r\ntraceparent: 00-a-b-01\r\n\r\n"
    upstream.sendall(b"data: chunk-1\n\ndata: chunk-2\n\n")
    upstream.close()

    assert _read_all(client) == b"data: chunk-1\n\ndata: chunk-2\n\n"
    serving.join(timeout=5)
    assert not serving.is_alive()


def _read_all(sock: socket.socket) -> bytes:
    sock.settimeout(5)
    chunks = []
    while chunk := sock.recv(4096):
        chunks.append(chunk)
    return b"".join(chunks)


def test_each_helper_starts_detached_under_its_own_pid(tmp_path: Path) -> None:
    """The pid file names the helper itself, so stopping it stops it, and the start returns at once."""
    ready = tmp_path / "helper.ready"
    helper = DetachedHelper(name="sleeper", command=f"sh -c 'touch {ready}; exec sleep 30'", ready=str(ready))

    started = time.monotonic()
    subprocess.run(["sh", "-c", detached_start_command([helper], str(tmp_path), str(tmp_path))], check=True, timeout=10)
    assert time.monotonic() - started < 5

    pid = int((tmp_path / "nemo-fabric-relay-sleeper.pid").read_text())
    try:
        assert (
            "sleep" in subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
        )
    finally:
        os.kill(pid, signal.SIGTERM)


def test_an_https_gateway_on_a_forwarded_host_stays_https(tmp_path: Path) -> None:
    agent = _gateway_agent(tmp_path, f"http://helix.example{GATEWAY_PATH}")
    agent.fabric_config.models["judge"] = agent.fabric_config.models["default"].model_copy(
        update={"base_url": f"https://helix.example{GATEWAY_PATH}"}
    )

    models = agent._build_config().models

    assert models["default"].base_url == f"http://127.0.0.1:14320{GATEWAY_PATH}"
    assert models["judge"].base_url == f"https://helix.example{GATEWAY_PATH}"
