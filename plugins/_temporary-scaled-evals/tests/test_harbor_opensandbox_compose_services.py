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
from scaled_evals.harbor_opensandbox_environment import (
    SERVICES_REQUEST_TIMEOUT_FLOOR_SEC,
    ComposeServicesError,
    NemoOpenSandboxEnvironment,
)
from scaled_evals.harbor_opensandbox_services import SERVICES_EXTENSION_KEY, parse_compose_services
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
        {"name": "seed", "image": "registry.example.com/seed:1", "run_once": True},
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

    def __init__(self, status_code: int, body: bytes) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response_body = body


class _Extension:
    """The server's NeMo services extension routes."""

    def __init__(self) -> None:
        self.healthy = True
        self.requests: list[httpx.Request] = []
        self.exec_results: list[httpx.Response] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Record the request; answer health, and exec with the next queued result."""
        self.requests.append(request)
        if request.url.path == "/v1/nemo-ext/health":
            if not self.healthy:
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(200, json={"status": "ok", "extension": "nemo-services"})
        if request.url.path.endswith("/exec"):
            return self.exec_results.pop(0)
        return httpx.Response(404)

    def exec_bodies(self) -> list[dict[str, Any]]:
        """The JSON bodies of every exec request so far."""
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
    """The SDK's ``ConnectionConfig``: the server URL, API key and extra headers."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.headers = {"X-Client": "harbor"}

    def get_api_key(self) -> str:
        return self.kwargs["api_key"]

    def get_base_url(self) -> str:
        return f"{self.kwargs['protocol']}://{self.kwargs['domain']}/v1"


def _fake_sdk(server: _Server) -> dict[str, Any]:
    """The SDK classes the environment loads, backed by ``server``."""

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
    return server


@pytest.fixture
def extension(monkeypatch: pytest.MonkeyPatch) -> _Extension:
    """Route the environment's direct server requests to a fake services extension."""
    extension = _Extension()
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(extension.handle), **kwargs)

    monkeypatch.setattr(environment_module.httpx, "AsyncClient", client)
    return extension


def _task(tmp_path: Path, *, compose: bool = True) -> Path:
    """A task environment directory, with a Compose file defining ``db`` and ``seed`` unless ``compose`` is False."""
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    if compose:
        (environment_dir / "docker-compose.yaml").write_text("services: {main: {}, db: {}, seed: {}}\n")
    return environment_dir


def _environment(environment_dir: Path, **kwargs: Any) -> NemoOpenSandboxEnvironment:
    """A no-network environment for the task, with ``COMPOSE_SERVICES`` unless ``kwargs`` override it."""
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
    spec = json.loads(create["extensions"][SERVICES_EXTENSION_KEY])
    assert spec["version"] == 1
    assert [s["name"] for s in spec["services"]] == ["db", "seed"]
    assert spec["sandbox"] == {
        "volume_mounts": [{"name": "shared", "mount_path": "/shared", "read_only": False}],
        "add_capabilities": ["SYS_PTRACE"],
    }
    assert create["entrypoint"] == ["/app/start.sh", "sh", "-c", "sleep infinity"]
    assert create["env"]["DB_HOST"] == "db"
    assert create["env"]["SHARED"] == "from-task"
    assert environment.capabilities.docker_compose is True
    assert environment._request_timeout_sec == SERVICES_REQUEST_TIMEOUT_FLOOR_SEC
    health = extension.requests[0]
    assert str(health.url) == "http://opensandbox.example/v1/nemo-ext/health"
    assert health.headers["OPEN-SANDBOX-API-KEY"] == API_KEY
    assert health.headers["X-Client"] == "harbor"


def test_profile_spec_is_accepted_by_the_server_extension() -> None:
    server_spec = pytest.importorskip("nemo_ext_spec")
    every_field = {
        **COMPOSE_SERVICES,
        "services": [
            *COMPOSE_SERVICES["services"],
            {
                "name": "web",
                "image": "localhost:5000/web:1",
                "command": ["/bin/web"],
                "args": ["--port", "8080"],
                "env": {"MODE": "test"},
                "ports": [8080],
                "readiness": {"exec": ["curl", "-sf", "http://127.0.0.1:8080/healthz"]},
                "after_main": True,
                "resources": {"limits": {"memory": "512Mi"}, "requests": {"cpu": "250m"}},
                "volume_mounts": [{"name": "shared", "mount_path": "/shared", "read_only": True}],
                "shm_size": "1Gi",
            },
            {"name": "cache", "image": "docker.io/library/redis:7", "readiness": {"exec": ["redis-cli", "ping"]}},
        ],
    }

    spec = server_spec.parse_spec(parse_compose_services(every_field).server_spec())

    assert [s.name for s in spec.services] == ["db", "seed", "web", "cache"]
    assert spec.sandbox.add_capabilities == ["SYS_PTRACE"]


async def test_verifier_environment_without_compose_file_gets_a_plain_sandbox(
    tmp_path: Path, server: _Server, extension: _Extension
) -> None:
    environment = _environment(_task(tmp_path, compose=False))

    await environment._create_sandbox(environment._load_opensandbox())

    assert SERVICES_EXTENSION_KEY not in server.creates[-1]["extensions"]
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

    with pytest.raises(ComposeServicesError, match="does not run the NeMo services extension"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert server.creates == []


async def test_failed_service_is_not_retried(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    body = json.dumps({"detail": {"code": "NEMO::SERVICE_FAILED", "message": "db exited 1: bad config"}}).encode()
    server.create_failures = [SandboxApiException(422, body)]
    environment = _environment(_task(tmp_path))

    with pytest.raises(ComposeServicesError, match="bad config"):
        await environment._create_sandbox(environment._load_opensandbox())

    assert len(server.creates) == 1


async def test_transient_create_errors_are_still_retried(
    tmp_path: Path, server: _Server, extension: _Extension
) -> None:
    server.create_failures = [SandboxApiException(503, b"")]
    environment = _environment(_task(tmp_path))

    await environment._create_sandbox(environment._load_opensandbox())

    assert len(server.creates) == 2


async def _started(environment: NemoOpenSandboxEnvironment) -> NemoOpenSandboxEnvironment:
    """The environment with its sandbox created, as after ``start``."""
    environment._sandbox = await environment._create_sandbox(environment._load_opensandbox())
    return environment


def _exec_ok(stdout: str = "", exit_code: int = 0) -> httpx.Response:
    """A successful exec route response."""
    return httpx.Response(200, json={"exit_code": exit_code, "stdout": stdout, "stderr": ""})


async def test_service_exec_goes_through_the_exec_route(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    extension.exec_results = [_exec_ok("PONG\n")]

    result = await environment.service_exec(
        "redis-cli ping", service="db", cwd="/data", env={"A": "x y"}, timeout_sec=30
    )

    assert result == ExecResult(stdout="PONG\n", stderr="", return_code=0)
    request = extension.requests[-1]
    assert request.url.path == "/v1/sandboxes/sbx-1/containers/db/exec"
    assert request.headers["OPEN-SANDBOX-API-KEY"] == API_KEY
    assert extension.exec_bodies()[-1] == {
        "command": ["sh", "-c", "export A='x y'; cd /data && redis-cli ping"],
        "timeout": 30,
    }


async def test_service_exec_timeout_and_errors(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))
    extension.exec_results = [
        httpx.Response(504, json={"detail": {"code": "NEMO::EXEC_TIMEOUT", "message": "t"}}),
        httpx.Response(404, json={"detail": {"code": "NEMO::NOT_FOUND", "message": "no such container"}}),
    ]

    timed_out = await environment.service_exec("sleep 99", service="db", timeout_sec=5000)
    assert timed_out.return_code == 124
    assert extension.exec_bodies()[-1]["timeout"] == 3600
    with pytest.raises(RuntimeError, match="NEMO::NOT_FOUND: no such container"):
        await environment.service_exec("true", service="db")


async def test_unknown_service_and_stop_are_unsupported(tmp_path: Path, server: _Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path)))

    with pytest.raises(ServiceOperationsUnsupportedError, match="'cache' is not one of"):
        await environment.service_exec("true", service="cache")
    with pytest.raises(ServiceOperationsUnsupportedError, match="can't be stopped"):
        await environment.stop_service("db")
    with pytest.raises(ServiceOperationsUnsupportedError, match="'main' is the sandbox container itself"):
        await environment.stop_service("main")
    assert not extension.exec_bodies()


def _dd(path: str, skip: int, count: int = 32) -> list[str]:
    """The exec argv the transport sends to read one download chunk."""
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


async def test_waits_for_main_readiness(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path))
    results = [ExecResult(stdout="", stderr="refused", return_code=7), ExecResult(stdout="ok", return_code=0)]
    commands: list[str] = []

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        commands.append(command)
        assert kwargs["user"] == "root"
        assert kwargs["cwd"] == "/"
        return results.pop(0)

    monkeypatch.setattr(environment, "exec", exec_)
    monkeypatch.setattr(environment_module, "MAIN_READINESS_POLL_SEC", 0)

    await environment._wait_for_main(["curl", "-sf", "http://localhost:8000/health"], 5)

    assert commands == ["curl -sf http://localhost:8000/health"] * 2


async def test_main_readiness_timeout_reports_last_output(
    tmp_path: Path, server: _Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path))

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        return ExecResult(stdout="", stderr="connection refused", return_code=7)

    monkeypatch.setattr(environment, "exec", exec_)
    monkeypatch.setattr(environment_module, "MAIN_READINESS_POLL_SEC", 0.01)

    with pytest.raises(RuntimeError, match="main was not ready after 1s.*connection refused"):
        await environment._wait_for_main(["false"], 1)
