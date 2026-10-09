# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoOpenSandboxEnvironment with Compose services, against a fake SDK and a fake server extension."""

from __future__ import annotations

import base64
import io
import itertools
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("harbor.environments.opensandbox")
pytest.importorskip("nhx_sandbox")
pytest.importorskip("scaled_evals", reason="scaled-evals plugin not installed")

import harbor.environments.opensandbox as harbor_opensandbox
from harbor.environments.base import ExecResult, ServiceOperationsUnsupportedError
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from scaled_evals import harbor_opensandbox_environment as environment_module
from scaled_evals.harbor_opensandbox_environment import NemoOpenSandboxEnvironment
from scaled_evals.harbor_opensandbox_services import EXTENSION_KEY, ComposeServices, parse_compose_services
from tenacity import wait_none

API_KEY = "test-key"
COMPOSE_SERVICES: dict[str, Any] = {
    "volumes": ["shared"],
    "services": [
        {
            "name": "db",
            "image": "docker.io/library/postgres:16",
            "ports": [5432],
            "readiness": {"exec": ["pg_isready", "-U", "postgres"]},
        },
        {"name": "seed", "image": "registry.example.com/seed:1", "role": "run_once"},
    ],
    "main": {
        "entrypoint": ["/app/start.sh"],
        "env": {"DB_HOST": "db", "SHARED": "from-compose"},
        "volume_mounts": [{"name": "shared", "mount_path": "/shared"}],
        "add_capabilities": ["SYS_PTRACE"],
        "readiness": {"exec": ["curl", "-sf", "http://localhost:8000/health"], "timeout_sec": 5},
    },
}


class SandboxApiException(Exception):
    """Name-matched by Harbor as a transient SDK error."""


class _Extension:
    """The server extension's routes, answering from what a test queues.

    ``statuses`` and ``exec_results`` are consumed one per request, in order, so a test scripts
    the sequence the environment sees; every request is kept in ``requests``.
    """

    def __init__(self) -> None:
        self.healthy = True
        self.requests: list[httpx.Request] = []
        self.exec_results: list[httpx.Response] = []
        self.statuses: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """The ``httpx.MockTransport`` handler: route by path to health, services status, or exec."""
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/nemo-ext/health":
            if not self.healthy:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"status": "ok", "extension": "nemo-compose-services"})
        if path.endswith("/services"):
            return httpx.Response(200, json=self.statuses.pop(0))
        if path.endswith("/exec"):
            return self.exec_results.pop(0)
        return httpx.Response(404)

    def exec_bodies(self) -> list[dict[str, Any]]:
        """The JSON bodies of every exec request, in order."""
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/exec")]


class _Sandbox:
    """An SDK sandbox handle whose applied egress policy is the one it was created with."""

    def __init__(self, sandbox_id: str, policy: dict[str, Any]) -> None:
        self.id = sandbox_id
        self._policy = policy

    async def get_egress_policy(self) -> SimpleNamespace:
        rules = [SimpleNamespace(**rule) for rule in self._policy["egress"]]
        return SimpleNamespace(default_action=self._policy["defaultAction"], egress=rules)

    async def get_info(self) -> SimpleNamespace:
        return SimpleNamespace(status=SimpleNamespace(state="Running"))

    async def is_healthy(self) -> bool:
        return True

    async def kill(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _Server:
    """The sandbox API behind the fake SDK: records creates, and raises queued create failures."""

    def __init__(self) -> None:
        self.creates: list[dict[str, Any]] = []
        self.create_failures: list[BaseException] = []
        self._ids = (f"sbx-{n}" for n in itertools.count(1))


class _ConnectionConfig:
    """The SDK's ``ConnectionConfig``: what ``_ServerClient`` reads the server URL, key and headers from."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.headers = {"X-Client": "harbor"}

    def get_api_key(self) -> str:
        return self.kwargs["api_key"]

    def get_base_url(self) -> str:
        return f"{self.kwargs['protocol']}://{self.kwargs['domain']}/v1"


def _fake_sdk(server: _Server) -> dict[str, Any]:
    """The parts of the OpenSandbox SDK the environment loads, backed by ``server``.

    ``Sandbox.create`` records its kwargs and raises the next queued failure, if any; the manager
    lists no sandboxes, so cleanup finds nothing to delete.
    """

    class Sandbox:
        @staticmethod
        async def create(image: Any, **kwargs: Any) -> _Sandbox:
            server.creates.append(kwargs)
            if server.create_failures:
                raise server.create_failures.pop(0)
            return _Sandbox(next(server._ids), kwargs["network_policy"].payload)

    class _Manager:
        async def list_sandbox_infos(self, selector: Any) -> SimpleNamespace:
            return SimpleNamespace(sandbox_infos=[], pagination=SimpleNamespace(has_next_page=False))

        async def close(self) -> None:
            pass

    class SandboxManager:
        @staticmethod
        async def create(connection_config: Any) -> _Manager:
            return _Manager()

    return {
        "Sandbox": Sandbox,
        "ConnectionConfig": _ConnectionConfig,
        "NetworkPolicy": SimpleNamespace(model_validate=lambda payload: SimpleNamespace(payload=payload)),
        "SandboxManager": SandboxManager,
        "SandboxFilter": lambda metadata, page: SimpleNamespace(metadata=metadata, page=page),
    }


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    """Point the environment at the fake SDK, with Harbor's create retries not waiting."""
    server = _Server()
    monkeypatch.setattr(harbor_opensandbox, "_HAS_OPENSANDBOX", True)
    monkeypatch.setattr(
        harbor_opensandbox.OpenSandboxEnvironment._create_sandbox.retry,  # ty: ignore[unresolved-attribute]  # tenacity
        "wait",
        wait_none(),
    )
    monkeypatch.setattr(NemoOpenSandboxEnvironment, "_load_opensandbox", lambda self: _fake_sdk(server))
    monkeypatch.setattr(environment_module, "COMPOSE_READY_POLL_SEC", 0)
    return server


@pytest.fixture
def extension(monkeypatch: pytest.MonkeyPatch) -> _Extension:
    """Route the environment's direct server requests to a fake extension."""
    extension = _Extension()
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(extension.handle), **kwargs)

    monkeypatch.setattr(environment_module.httpx, "AsyncClient", client)
    return extension


def _task(tmp_path: Path, *, compose: bool = True) -> Path:
    """A task's ``environment/`` directory, with a Compose file for ``main``, ``db`` and ``seed`` unless ``compose`` is False."""
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    if compose:
        (environment_dir / "docker-compose.yaml").write_text("services: {main: {}, db: {}, seed: {}}\n")
    return environment_dir


def _environment(environment_dir: Path, **kwargs: Any) -> NemoOpenSandboxEnvironment:
    """The environment for ``environment_dir`` with the test's compose services; ``kwargs`` override any argument."""
    kwargs.setdefault("compose_services", COMPOSE_SERVICES)
    return NemoOpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="compose-task",
        session_id="compose-task__abc",
        trial_paths=TrialPaths(trial_dir=environment_dir.parent / "trial"),
        task_env_config=EnvironmentConfig(docker_image="registry.example/task:1", env={"SHARED": "from-task"}),
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        domain="opensandbox.example",
        api_key=API_KEY,
        protocol="http",
        trusted_allowed_hosts=["10.96.0.20"],
        resolver_addresses=["10.96.0.10"],
        **kwargs,
    )


async def test_create_sends_services_and_main_settings(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = _environment(_task(tmp_path))

    await environment._create_sandbox(environment._load_opensandbox())

    create = server.creates[-1]
    assert ComposeServices.model_validate_json(create["extensions"][EXTENSION_KEY]) == parse_compose_services(
        COMPOSE_SERVICES
    )
    assert create["entrypoint"] == ["/app/start.sh", "sh", "-c", "sleep infinity"]
    assert create["env"]["DB_HOST"] == "db"
    assert create["env"]["SHARED"] == "from-task"
    assert environment.capabilities.docker_compose is True
    health = extension.requests[0]
    assert str(health.url) == "http://opensandbox.example/v1/nemo-ext/health"
    assert health.headers["OPEN-SANDBOX-API-KEY"] == API_KEY
    assert health.headers["X-Client"] == "harbor"


async def test_verifier_environment_without_compose_file_gets_a_plain_sandbox(
    tmp_path: Path, server: _Server, extension: _Extension
) -> None:
    environment = _environment(_task(tmp_path, compose=False))

    await environment._create_sandbox(environment._load_opensandbox())

    assert EXTENSION_KEY not in server.creates[-1]["extensions"]
    assert server.creates[-1]["entrypoint"] is None
    assert environment.capabilities.docker_compose is False
    assert extension.requests == []


def test_compose_task_without_services_is_still_rejected(tmp_path: Path, server: _Server) -> None:
    with pytest.raises(ValueError, match="docker-compose"):
        _environment(_task(tmp_path), compose_services=None)


def test_compose_task_still_needs_a_prebuilt_image(tmp_path: Path, server: _Server) -> None:
    with pytest.raises(FileNotFoundError, match="docker_image"):
        NemoOpenSandboxEnvironment(
            environment_dir=_task(tmp_path),
            environment_name="compose-task",
            session_id="s",
            trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
            task_env_config=EnvironmentConfig(),
            compose_services=COMPOSE_SERVICES,
        )


async def test_server_without_the_extension_fails_before_creating(
    tmp_path: Path, server: _Server, extension: _Extension
) -> None:
    extension.healthy = False
    environment = _environment(_task(tmp_path))

    with pytest.raises(RuntimeError, match="does not run the NeMo compose services extension"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert server.creates == []


async def test_transient_create_errors_are_still_retried(
    tmp_path: Path, server: _Server, extension: _Extension
) -> None:
    server.create_failures = [SandboxApiException("HTTP 503")]
    environment = _environment(_task(tmp_path))

    await environment._create_sandbox(environment._load_opensandbox())

    assert len(server.creates) == 2


async def test_failed_service_ends_the_readiness_wait(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = _environment(_task(tmp_path))
    sandbox = _Sandbox("sbx-1", {})
    extension.statuses = [
        {"failure": None, "not_ready": ["db", "seed"]},
        {"failure": "service 'seed' exited with code 1 (Error). Last log lines:\nboom", "not_ready": ["seed"]},
    ]

    assert await environment._terminal_provisioning_failure(sandbox) is None
    failure = await environment._terminal_provisioning_failure(sandbox)

    assert failure == "OpenSandbox sandbox sbx-1: service 'seed' exited with code 1 (Error). Last log lines:\nboom"
    assert extension.requests[-1].url.path == "/v1/nemo-ext/sandboxes/sbx-1/services"


async def test_unreadable_service_status_keeps_waiting(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path))

    # As for a sandbox whose pod doesn't exist yet.
    monkeypatch.setattr(
        extension, "handle", lambda request: httpx.Response(404, json={"detail": {"code": "NEMO::NOT_FOUND"}})
    )

    assert await environment._terminal_provisioning_failure(_Sandbox("sbx-1", {})) is None


async def _started(environment: NemoOpenSandboxEnvironment) -> NemoOpenSandboxEnvironment:
    """``environment`` with a created sandbox, skipping Harbor's start waits."""
    environment._sandbox = await environment._create_sandbox(environment._load_opensandbox())
    return environment


async def test_start_waits_for_main_then_the_services(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    results = [ExecResult(stdout="", stderr="refused", return_code=7), ExecResult(stdout="ok", return_code=0)]
    commands: list[str] = []

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        commands.append(command)
        assert kwargs["user"] == "root"
        assert kwargs["cwd"] == "/"
        return results.pop(0)

    async def harbor_start(self: Any, force_build: bool) -> None:
        pass

    monkeypatch.setattr(harbor_opensandbox.OpenSandboxEnvironment, "start", harbor_start)
    monkeypatch.setattr(environment, "exec", exec_)
    extension.statuses = [{"failure": None, "not_ready": ["db"]}, {"failure": None, "not_ready": []}]

    await environment.start(force_build=False)

    assert commands == ["curl -sf http://localhost:8000/health"] * 2
    assert extension.statuses == []


async def test_main_readiness_timeout_reports_last_output(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path))

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        return ExecResult(stdout="", stderr="connection refused", return_code=7)

    monkeypatch.setattr(environment, "exec", exec_)

    with pytest.raises(RuntimeError, match="main was not ready.*connection refused"):
        await environment._wait_for_main(["false"], deadline=0)


async def test_services_wait_fails_on_a_failed_service(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    extension.statuses = [{"failure": "service 'db' exited with code 1 (Error)", "not_ready": ["db"]}]

    with pytest.raises(RuntimeError, match="service 'db' exited with code 1"):
        await environment._wait_for_services(deadline=float("inf"))


def _exec_ok(stdout: str = "", exit_code: int = 0) -> httpx.Response:
    """A 200 from the exec route: the command ran, with this output and exit code."""
    return httpx.Response(200, json={"exit_code": exit_code, "stdout": stdout, "stderr": ""})


async def test_service_exec_goes_through_the_exec_route(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    extension.exec_results = [_exec_ok("PONG\n")]

    result = await environment.service_exec(
        "redis-cli ping", service="db", cwd="/data", env={"A": "x y"}, timeout_sec=30
    )

    assert result == ExecResult(stdout="PONG\n", stderr="", return_code=0)
    request = extension.requests[-1]
    assert request.url.path == "/v1/nemo-ext/sandboxes/sbx-1/services/db/exec"
    assert request.headers["OPEN-SANDBOX-API-KEY"] == API_KEY
    assert extension.exec_bodies()[-1] == {
        "command": ["sh", "-c", "export A='x y'; cd /data && redis-cli ping"],
        "timeout": 30,
    }


async def test_service_exec_timeout_and_errors(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    extension.exec_results = [
        httpx.Response(504, json={"detail": {"code": "NEMO::EXEC_TIMEOUT", "message": "t"}}),
        httpx.Response(429, json={"detail": {"code": "NEMO::EXEC_BUSY", "message": "busy"}}),
    ]

    timed_out = await environment.service_exec("sleep 99", service="db", timeout_sec=5000)
    assert timed_out.return_code == 124
    assert extension.exec_bodies()[-1]["timeout"] == 3600
    with pytest.raises(RuntimeError, match="NEMO::EXEC_BUSY: busy"):
        await environment.service_exec("true", service="db")


async def test_unknown_service_and_stop_are_unsupported(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))

    with pytest.raises(ServiceOperationsUnsupportedError, match="'cache' is not one of"):
        await environment.service_exec("true", service="cache")
    with pytest.raises(ServiceOperationsUnsupportedError, match="can't be stopped"):
        await environment.stop_service("db")
    with pytest.raises(ServiceOperationsUnsupportedError, match="main is the sandbox itself"):
        await environment.stop_service("main")
    assert not extension.exec_bodies()


def _dd(path: str, skip: int, count: int = 8) -> list[str]:
    """The argv a download sends to read one chunk of ``path``, starting ``skip`` 1 MiB blocks in."""
    return ["sh", "-c", f"dd if={path} bs=1048576 skip={skip} count={count} 2>/dev/null | base64"]


async def test_downloads_decode_exec_output(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        info = tarfile.TarInfo("logs/app.log")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))
    extension.exec_results = [
        _exec_ok("7\n"),
        _exec_ok(base64.b64encode(b"\x00binary").decode()),
        _exec_ok("/tmp/tmp.abc\n"),
        _exec_ok(f"{len(archive.getvalue())}\n"),
        _exec_ok(base64.b64encode(archive.getvalue()).decode()),
        _exec_ok(),
    ]

    await environment.service_download_file("/data/dump.bin", tmp_path / "out" / "dump.bin", service="db")
    await environment.service_download_dir("/var/log", tmp_path / "out" / "logs", service="db")

    assert (tmp_path / "out" / "dump.bin").read_bytes() == b"\x00binary"
    assert (tmp_path / "out" / "logs" / "logs" / "app.log").read_text() == "hello"
    assert [body["command"] for body in extension.exec_bodies()] == [
        ["sh", "-c", "wc -c < /data/dump.bin"],
        _dd("/data/dump.bin", 0),
        ["sh", "-c", 't=$(mktemp) && tar -C /var/log -czf "$t" . && echo "$t"'],
        ["sh", "-c", "wc -c < /tmp/tmp.abc"],
        _dd("/tmp/tmp.abc", 0),
        ["rm", "-f", "/tmp/tmp.abc"],
    ]


async def test_large_downloads_are_read_in_chunks(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment_module, "SERVICE_DOWNLOAD_BLOCK_BYTES", 2)
    monkeypatch.setattr(environment_module, "SERVICE_DOWNLOAD_CHUNK_BLOCKS", 2)
    environment = await _started(_environment(_task(tmp_path)))
    data = b"0123456789"
    extension.exec_results = [
        _exec_ok("10\n"),
        *(_exec_ok(base64.b64encode(data[start : start + 4]).decode()) for start in (0, 4, 8)),
        _exec_ok("10\n"),
        _exec_ok(base64.b64encode(data[:4]).decode()),
        _exec_ok(),
        _exec_ok(),
    ]

    await environment.service_download_file("/data/big", tmp_path / "big", service="db")
    assert (tmp_path / "big").read_bytes() == data
    assert [body["command"] for body in extension.exec_bodies()][1:] == [
        ["sh", "-c", f"dd if=/data/big bs=2 skip={skip} count=2 2>/dev/null | base64"] for skip in (0, 2, 4)
    ]

    with pytest.raises(RuntimeError, match="got 4 of 10 bytes"):
        await environment.service_download_file("/data/big", tmp_path / "short", service="db")
