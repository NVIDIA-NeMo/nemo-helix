# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoOpenSandboxEnvironment with Compose services, against a mock OpenSandbox SDK and a fake server extension."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("harbor.environments.base")
pytest.importorskip("nhx_sandbox")

from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from nhx_sandbox import harbor_opensandbox_environment as environment_module
from nhx_sandbox.compose_env import ComposeEnvError
from nhx_sandbox.harbor_opensandbox_driver import CommandResult, OpenSandboxDriver
from nhx_sandbox.harbor_opensandbox_environment import NemoOpenSandboxEnvironment, ServiceOperationsUnsupportedError
from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_EXTENSION_KEY,
    EXTENSION_KEY,
    ComposeServices,
    SandboxAgentBundle,
    parse_compose_services,
)
from opensandbox_mocks import SandboxApiException, Server, mock_sdk

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


class _Extension:
    """The server extension's routes, answering from what a test queues.

    ``statuses`` and ``exec_results`` are consumed one per request, in order, so a test scripts
    the sequence the environment sees; every request is kept in ``requests``.
    """

    def __init__(self) -> None:
        self.healthy = True
        # What the health route reports; None leaves the key out, like an extension from before agent bundles.
        self.features: list[str] | None = ["compose-services", "agent-bundle"]
        self.requests: list[httpx.Request] = []
        self.exec_results: list[httpx.Response] = []
        # Answers for the services status route; once empty, every service is ready.
        self.statuses: list[dict[str, Any]] = []
        # Answers (or errors to raise) that come before the routes, for any request, one per request.
        self.preempt: list[httpx.Response | Exception] = []
        # The timeout each HTTP attempt was given.
        self.timeouts: list[float] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """The ``httpx.MockTransport`` handler: route by path to health, services status, or exec."""
        self.requests.append(request)
        if self.preempt:
            answer = self.preempt.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        path = request.url.path
        if path == "/v1/nemo-ext/health":
            if not self.healthy:
                return httpx.Response(404, json={"detail": "Not Found"})
            body: dict[str, Any] = {"status": "ok", "extension": "nemo-compose-services"}
            if self.features is not None:
                body["features"] = self.features
            return httpx.Response(200, json=body)
        if path.endswith("/services"):
            status = self.statuses.pop(0) if self.statuses else {"failure": None, "not_ready": []}
            return httpx.Response(200, json=status)
        if path.endswith("/exec"):
            return self.exec_results.pop(0)
        return httpx.Response(404)

    def exec_bodies(self) -> list[dict[str, Any]]:
        """The JSON bodies of every exec request, in order."""
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/exec")]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Server:
    """A fresh mock OpenSandbox server, with the environment's polls and retries not waiting."""
    monkeypatch.setattr(environment_module, "COMPOSE_READY_POLL_SEC", 0)
    monkeypatch.setattr(environment_module, "SERVER_RETRY_BASE_SEC", 0)
    return Server()


@pytest.fixture
def extension(monkeypatch: pytest.MonkeyPatch, server: Server) -> _Extension:
    """Route the environment's direct server requests to a fake extension, and the endpoints route to ``server``."""
    extension = _Extension()
    real_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        return server.endpoint_route(request) or extension.handle(request)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        extension.timeouts.append(kwargs["timeout"])
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(environment_module.httpx, "AsyncClient", client)
    return extension


def _task(tmp_path: Path, *, compose: bool = True) -> Path:
    """A task's ``environment/`` directory, with a Compose file for ``main``, ``db`` and ``seed`` unless ``compose`` is False."""
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    if compose:
        (environment_dir / "docker-compose.yaml").write_text("services: {main: {}, db: {}, seed: {}}\n")
    return environment_dir


def _environment(environment_dir: Path, server: Server, **kwargs: Any) -> NemoOpenSandboxEnvironment:
    """The environment for ``environment_dir`` on the mock server, with the test's compose services; ``kwargs`` override any argument."""
    kwargs.setdefault("compose_services", COMPOSE_SERVICES)
    kwargs.setdefault("session_id", "compose-task__abc")
    kwargs.setdefault("trial_paths", TrialPaths(trial_dir=environment_dir.parent / "trial"))
    environment = NemoOpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="compose-task",
        task_env_config=EnvironmentConfig(docker_image="registry.example/task:1", env={"SHARED": "from-task"}),
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        domain="opensandbox.example",
        api_key=API_KEY,
        protocol="http",
        trusted_allowed_hosts=["10.96.0.20"],
        resolver_addresses=["10.96.0.10"],
        health_check_poll_interval_sec=0,
        **kwargs,
    )
    environment._driver = OpenSandboxDriver(
        environment._connection, sdk=mock_sdk(server), logger=environment.logger, retry_wait=lambda attempt: 0
    )
    return environment


def test_environment_loads_on_a_harbor_without_per_service_operations() -> None:
    """Harbor 0.13.2 has no ServiceOperationsUnsupportedError; the module still imports and raises RuntimeError instead."""
    # Removed here to simulate 0.13.2 on a newer Harbor; on 0.13.2 itself it is already absent.
    script = (
        "import harbor.environments.base as base\n"
        "if hasattr(base, 'ServiceOperationsUnsupportedError'): del base.ServiceOperationsUnsupportedError\n"
        "import nhx_sandbox.harbor_opensandbox_environment as module\n"
        "assert module.ServiceOperationsUnsupportedError is RuntimeError\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


async def test_create_sends_services_and_main_settings(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = _environment(_task(tmp_path), server)

    await environment._create_sandbox()

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
    assert health.headers["X-Client"] == "opensandbox-sdk"


async def test_templated_env_values_reach_the_sandbox_and_the_services(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harbor persists the config, so it carries templates; the values come from this process's environment."""
    spec = {
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "${NEMO_COMPOSE_ENV_0}"},
            }
        ],
        "main": {"env": {"API_TOKEN": "${NEMO_COMPOSE_ENV_1}"}},
    }
    monkeypatch.setenv("NEMO_COMPOSE_ENV_0", "pg-pass")
    monkeypatch.setenv("NEMO_COMPOSE_ENV_1", "api-token")
    environment = _environment(_task(tmp_path), server, compose_services=spec)

    await environment._create_sandbox()

    create = server.creates[-1]
    sent = ComposeServices.model_validate_json(create["extensions"][EXTENSION_KEY])
    assert sent.services[0].env == {"POSTGRES_PASSWORD": "pg-pass"}
    assert create["env"]["API_TOKEN"] == "api-token"
    assert (environment._merge_env(None) or {})["API_TOKEN"] == "api-token"
    assert "NEMO_COMPOSE_ENV" not in str(create)


def test_a_template_without_its_variable_fails_naming_the_variable(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = {
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "${NEMO_COMPOSE_ENV_0}"},
            }
        ]
    }
    monkeypatch.delenv("NEMO_COMPOSE_ENV_0", raising=False)

    with pytest.raises(ComposeEnvError, match="NEMO_COMPOSE_ENV_0"):
        _environment(_task(tmp_path), server, compose_services=spec)


def test_a_separate_verifier_without_a_compose_file_needs_none_of_the_variables(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harbor builds the verifier's environment from the same config, but ``compose_services`` only applies with a Compose file."""
    spec = {
        "services": [
            {
                "name": "db",
                "image": "docker.io/library/postgres:16",
                "env": {"POSTGRES_PASSWORD": "${NEMO_COMPOSE_ENV_0}"},
            }
        ]
    }
    monkeypatch.delenv("NEMO_COMPOSE_ENV_0", raising=False)

    environment = _environment(_task(tmp_path, compose=False), server, compose_services=spec)

    assert environment.capabilities.docker_compose is False


async def test_verifier_environment_without_compose_file_gets_a_plain_sandbox(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = _environment(_task(tmp_path, compose=False), server)

    await environment._create_sandbox()

    assert EXTENSION_KEY not in server.creates[-1]["extensions"]
    assert server.creates[-1]["entrypoint"] is None
    assert environment.capabilities.docker_compose is False
    assert extension.requests == []


async def test_a_plain_sandbox_starts_with_its_image_s_own_entrypoint(tmp_path: Path, server: Server) -> None:
    entrypoints = {"registry.example/task:1": ["/bin/bash", "-l", "-c"], "registry.example/verifier:1": ["/verify"]}
    environment = _environment(_task(tmp_path, compose=False), server, image_entrypoints=entrypoints)

    await environment._create_sandbox()

    assert server.creates[-1]["entrypoint"] == ["/bin/bash", "-l", "-c", "sh", "-c", "sleep infinity"]


async def test_main_s_entrypoint_and_env_reach_the_container_as_written(tmp_path: Path, server: Server) -> None:
    """Kubernetes expands ``$(VAR)`` and ``$$`` at create, so both are escaped there; later commands get the env as is."""
    entrypoints = {"registry.example/task:1": ["/bin/sh", "-c", 'echo "$(HOME)" a$$b']}
    environment = _environment(_task(tmp_path, compose=False), server, image_entrypoints=entrypoints)
    environment._persistent_env["PASSWORD"] = "pa$$word"

    await environment._create_sandbox()

    sent = server.creates[-1]
    assert sent["entrypoint"][:3] == ["/bin/sh", "-c", 'echo "$$(HOME)" a$$$$b']
    assert sent["env"]["PASSWORD"] == "pa$$$$word"
    assert (environment._merge_env(None) or {})["PASSWORD"] == "pa$$word"


async def test_a_docker_runtime_gets_main_s_entrypoint_and_env_unescaped(tmp_path: Path, server: Server) -> None:
    """Docker never expands ``$`` in a container's command or env, so nothing is escaped for it."""
    entrypoints = {"registry.example/task:1": ["/bin/sh", "-c", 'echo "$(HOME)" a$$b']}
    environment = _environment(_task(tmp_path, compose=False), server, image_entrypoints=entrypoints, runtime="docker")
    environment._persistent_env["PASSWORD"] = "pa$$word"

    await environment._create_sandbox()

    sent = server.creates[-1]
    assert sent["entrypoint"][:3] == ["/bin/sh", "-c", 'echo "$(HOME)" a$$b']
    assert sent["env"]["PASSWORD"] == "pa$$word"


@pytest.mark.parametrize(
    "entrypoints",
    [{"registry.example/task:1": []}, {"registry.example/verifier:1": ["/verify"]}],
    ids=["image-has-none", "another-image-s"],
)
async def test_an_image_without_a_recorded_entrypoint_keeps_the_default(
    tmp_path: Path, server: Server, entrypoints: dict[str, list[str]]
) -> None:
    environment = _environment(_task(tmp_path, compose=False), server, image_entrypoints=entrypoints)

    await environment._create_sandbox()

    assert server.creates[-1]["entrypoint"] is None


async def test_compose_main_without_its_own_entrypoint_starts_with_its_image_s(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    main = {key: value for key, value in COMPOSE_SERVICES["main"].items() if key != "entrypoint"}
    environment = _environment(
        _task(tmp_path),
        server,
        compose_services={**COMPOSE_SERVICES, "main": main},
        image_entrypoints={"registry.example/task:1": ["/bin/bash", "-l", "-c"]},
    )

    await environment._create_sandbox()

    assert server.creates[-1]["entrypoint"] == ["/bin/bash", "-l", "-c", "sh", "-c", "sleep infinity"]


# A compose file that only changes `main`, as swe-lancer's do.
MAIN_ONLY: dict[str, Any] = {"main": {"command": ["/app/tests/run.sh"], "working_dir": "/app"}}


async def test_a_compose_file_that_only_changes_main_starts_it_as_harbor_would(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    extension.features = ["compose-services", "agent-bundle", "compose-main"]
    environment = _environment(
        _task(tmp_path),
        server,
        compose_services=MAIN_ONLY,
        image_entrypoints={"registry.example/task:1": ["/bin/bash", "-l", "-c"]},
    )

    await environment._create_sandbox()

    create = server.creates[-1]
    assert create["entrypoint"] == ["/bin/bash", "-l", "-c", "/app/tests/run.sh"]
    assert ComposeServices.model_validate_json(create["extensions"][EXTENSION_KEY]).main.working_dir == "/app"


async def test_an_extension_without_main_settings_is_refused_before_any_sandbox(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = _environment(_task(tmp_path), server, compose_services=MAIN_ONLY)

    with pytest.raises(RuntimeError, match="only change main"):
        await environment._create_sandbox()

    assert server.creates == []


# A compose file that restricts `main`, as the read-only, no-capability tasks do.
RESTRICTED_MAIN: dict[str, Any] = {
    "services": COMPOSE_SERVICES["services"],
    "main": {
        "security": {"run_as_user": 1000, "read_only_root_filesystem": True, "drop_capabilities": ["ALL"]},
        "tmpfs": [{"mount_path": "/tmp", "size_limit": "32Mi"}],
    },
}


async def test_main_s_restrictions_reach_the_extension(tmp_path: Path, server: Server, extension: _Extension) -> None:
    extension.features = ["compose-services", "agent-bundle", "compose-security"]
    environment = _environment(_task(tmp_path), server, compose_services=RESTRICTED_MAIN)

    await environment._create_sandbox()

    main = ComposeServices.model_validate_json(server.creates[-1]["extensions"][EXTENSION_KEY]).main
    assert main.security is not None and main.security.read_only_root_filesystem
    assert [mount.mount_path for mount in main.tmpfs] == ["/tmp"]


async def test_an_extension_without_security_settings_is_refused_before_any_sandbox(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = _environment(_task(tmp_path), server, compose_services=RESTRICTED_MAIN)

    with pytest.raises(RuntimeError, match="read_only, cap_drop, security_opt or tmpfs"):
        await environment._create_sandbox()

    assert server.creates == []


async def test_commands_run_in_main_s_working_dir_unless_the_task_names_one(
    tmp_path: Path, server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path), server, compose_services=MAIN_ONLY)
    environment._sandbox = SimpleNamespace(id="sbx-1")
    directories: list[str | None] = []

    async def run(sandbox: Any, command: str, **kwargs: Any) -> SimpleNamespace:
        directories.append(kwargs["cwd"])
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    monkeypatch.setattr(environment._driver, "run", run)
    await environment.exec("pwd")
    await environment.exec("pwd", cwd="/tmp")

    assert directories == ["/app", "/tmp"]


def test_compose_task_without_services_is_still_rejected(tmp_path: Path, server: Server) -> None:
    with pytest.raises(ValueError, match="docker-compose"):
        _environment(_task(tmp_path), server, compose_services=None)


def test_compose_task_still_needs_a_prebuilt_image(tmp_path: Path, server: Server) -> None:
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
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    extension.healthy = False
    environment = _environment(_task(tmp_path), server)

    with pytest.raises(RuntimeError, match="does not run the NeMo compose services extension"):
        await environment._create_sandbox()

    assert server.creates == []


async def test_transient_create_errors_are_still_retried(tmp_path: Path, server: Server, extension: _Extension) -> None:
    server.create_failures = [SandboxApiException("HTTP 503")]
    environment = _environment(_task(tmp_path), server)

    await environment._create_sandbox()

    assert len(server.creates) == 2


async def test_failed_service_ends_the_readiness_wait(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = _environment(_task(tmp_path), server)
    sandbox = SimpleNamespace(id="sbx-1")
    extension.statuses = [
        {"failure": None, "not_ready": ["db", "seed"]},
        {"failure": "service 'seed' exited with code 1 (Error). Last log lines:\nboom", "not_ready": ["seed"]},
    ]

    assert await environment._provisioning_failure(sandbox) is None
    failure = await environment._provisioning_failure(sandbox)

    assert failure == "OpenSandbox sandbox sbx-1: service 'seed' exited with code 1 (Error). Last log lines:\nboom"
    assert extension.requests[-1].url.path == "/v1/nemo-ext/sandboxes/sbx-1/services"


async def test_failed_service_fails_the_create_and_kills_the_sandbox(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    server.healthy = False  # main never starts while a service it waits for has failed
    extension.statuses = [{"failure": "service 'seed' exited with code 1 (Error)", "not_ready": ["seed"]}]
    environment = _environment(_task(tmp_path), server)

    with pytest.raises(RuntimeError, match="service 'seed' exited with code 1"):
        await environment._create_sandbox()

    # Not a transient error, so the create isn't retried, and nothing is left running.
    assert len(server.creates) == 1
    assert server.live() == set()


async def test_unreadable_service_status_keeps_waiting(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path), server)

    # As for a sandbox whose pod doesn't exist yet.
    monkeypatch.setattr(
        extension, "handle", lambda request: httpx.Response(404, json={"detail": {"code": "NEMO::NOT_FOUND"}})
    )

    assert await environment._provisioning_failure(SimpleNamespace(id="sbx-1")) is None


async def _started(environment: NemoOpenSandboxEnvironment) -> NemoOpenSandboxEnvironment:
    """``environment`` with a created sandbox, skipping the rest of ``start``."""
    environment._sandbox = await environment._create_sandbox()
    return environment


async def test_start_waits_for_main_then_the_services(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(_task(tmp_path), server)
    results = [
        CommandResult(stdout="", stderr="refused", exit_code=7),
        CommandResult(stdout="ok", stderr="", exit_code=0),
    ]
    commands: list[str] = []

    async def run(sandbox: Any, command: str, **kwargs: Any) -> CommandResult:
        # start() also opens up the log directories, as root; only main's readiness check is scripted.
        if command != "curl -sf http://localhost:8000/health":
            return CommandResult(stdout="0", stderr="", exit_code=0)
        commands.append(command)
        # Like a Docker healthcheck: the image's own user and WORKDIR, so neither is sent.
        assert kwargs.get("uid") is None
        assert kwargs.get("cwd") is None
        return results.pop(0)

    monkeypatch.setattr(environment._driver, "run", run)
    # The first status is read by the create's readiness wait, the rest by start's own wait.
    extension.statuses = [
        {"failure": None, "not_ready": ["db"]},
        {"failure": None, "not_ready": ["db"]},
        {"failure": None, "not_ready": []},
    ]

    await environment.start(force_build=False)

    assert commands == ["curl -sf http://localhost:8000/health"] * 2
    assert extension.statuses == []


@pytest.mark.parametrize(
    ("main", "services_timeout_sec", "expected_sec"),
    [
        # main's own timeout carries the wait, whatever the services' says.
        ({"readiness": {"exec": ["true"], "timeout_sec": 900}}, 660, 900),
        ({"readiness": {"exec": ["true"]}}, 660, 300),
        # Without a check on main, the services' timeout does.
        ({}, 660, 660),
        ({}, None, 300),
    ],
)
async def test_start_waits_as_long_as_the_spec_allows(
    tmp_path: Path,
    server: Server,
    extension: _Extension,
    monkeypatch: pytest.MonkeyPatch,
    main: dict[str, Any],
    services_timeout_sec: int | None,
    expected_sec: int,
) -> None:
    spec = {**COMPOSE_SERVICES, "main": main}
    if services_timeout_sec is not None:
        spec["services_timeout_sec"] = services_timeout_sec
    environment = _environment(_task(tmp_path), server, compose_services=spec)
    deadlines: list[float] = []

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        """A command that succeeds without running anything."""
        return ExecResult(stdout="", stderr="", return_code=0)

    async def wait_for_main(command: list[str], deadline: float) -> None:
        """Record the deadline ``start`` gives the wait for main."""
        deadlines.append(deadline)

    async def wait_for_services(deadline: float) -> None:
        """Record the deadline ``start`` gives the wait for the services."""
        deadlines.append(deadline)

    monkeypatch.setattr(environment, "exec", exec_)
    monkeypatch.setattr(environment, "_wait_for_main", wait_for_main)
    monkeypatch.setattr(environment, "_wait_for_services", wait_for_services)

    before = time.monotonic()
    await environment.start(force_build=False)
    after = time.monotonic()

    assert deadlines
    assert all(before + expected_sec <= deadline <= after + expected_sec for deadline in deadlines)


async def test_main_readiness_timeout_reports_last_output(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = await _started(_environment(_task(tmp_path), server))

    async def run(sandbox: Any, command: str, **kwargs: Any) -> CommandResult:
        return CommandResult(stdout="", stderr="connection refused", exit_code=7)

    monkeypatch.setattr(environment._driver, "run", run)

    with pytest.raises(RuntimeError, match="main was not ready.*connection refused"):
        await environment._wait_for_main(["false"], deadline=0)


async def test_services_wait_fails_on_a_failed_service(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    extension.statuses = [{"failure": "service 'db' exited with code 1 (Error)", "not_ready": ["db"]}]

    with pytest.raises(RuntimeError, match="service 'db' exited with code 1"):
        await environment._wait_for_services(deadline=float("inf"))


def _exec_ok(stdout: str = "", exit_code: int = 0) -> httpx.Response:
    """A 200 from the exec route: the command ran, with this output and exit code."""
    return httpx.Response(200, json={"exit_code": exit_code, "stdout": stdout, "stderr": ""})


async def test_service_exec_goes_through_the_exec_route(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
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


async def test_service_exec_timeout_and_errors(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    busy = httpx.Response(429, json={"detail": {"code": "NEMO::EXEC_BUSY", "message": "busy"}})
    extension.exec_results = [
        httpx.Response(504, json={"detail": {"code": "NEMO::EXEC_TIMEOUT", "message": "t"}}),
        *[busy] * environment_module.SERVER_RETRY_ATTEMPTS,
    ]

    timed_out = await environment.service_exec("sleep 99", service="db", timeout_sec=5000)
    assert timed_out.return_code == 124
    assert extension.exec_bodies()[-1]["timeout"] == 3600
    with pytest.raises(RuntimeError, match="NEMO::EXEC_BUSY: busy"):
        await environment.service_exec("true", service="db")
    # The busy answer was retried until the attempts ran out.
    assert len(extension.exec_bodies()) == 1 + environment_module.SERVER_RETRY_ATTEMPTS


async def test_service_exec_retries_a_busy_server(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    busy = httpx.Response(429, json={"detail": {"code": "NEMO::EXEC_BUSY", "message": "busy"}})
    extension.exec_results = [busy, busy, _exec_ok("PONG\n")]

    result = await environment.service_exec("redis-cli ping", service="db")

    assert result.stdout == "PONG\n"
    assert len(extension.exec_bodies()) == 3


async def test_gateway_504_is_not_a_command_timeout(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    extension.exec_results = [httpx.Response(504, text="upstream timed out")]

    with pytest.raises(RuntimeError, match=r"failed \(504\)"):
        await environment.service_exec("sleep 99", service="db")
    # The command may have run, so a POST that got a 504 isn't sent again.
    assert len(extension.exec_bodies()) == 1


async def test_service_exec_retries_only_connections_that_never_opened(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    started = len(extension.requests)
    extension.preempt = [httpx.ConnectError("refused")]
    extension.exec_results = [_exec_ok("ok")]

    assert (await environment.service_exec("true", service="db")).stdout == "ok"

    # A read timeout means the command may be running, so it isn't repeated.
    extension.preempt = [httpx.ReadTimeout("slow")]
    with pytest.raises(RuntimeError, match="exec in compose service 'db' failed"):
        await environment.service_exec("true", service="db")
    assert len(extension.requests) - started == 3  # the refused exec, its retry, the one that timed out


async def test_status_reads_retry_transient_errors(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    started = len(extension.requests)
    extension.preempt = [httpx.Response(503), httpx.ConnectError("reset")]
    extension.statuses = [{"failure": None, "not_ready": []}]

    await environment._wait_for_services(time.monotonic() + 30)

    # Two transient failures, then the answer.
    assert [r.url.path for r in extension.requests[started:]] == ["/v1/nemo-ext/sandboxes/sbx-1/services"] * 3


async def test_status_reads_stop_retrying_at_the_deadline(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    started = len(extension.requests)
    extension.preempt = [httpx.Response(503)] * environment_module.SERVER_RETRY_ATTEMPTS

    with pytest.raises(RuntimeError, match=r"reading compose services failed \(503\)"):
        await environment._wait_for_services(time.monotonic() - 1)

    # The deadline had passed, so the read got its one try and no retries.
    assert len(extension.requests) - started == 1


async def test_status_read_timeouts_are_capped_by_the_deadline(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    extension.timeouts.clear()
    extension.statuses = [{"failure": None, "not_ready": []}]

    await environment._wait_for_services(time.monotonic() + 5)

    assert extension.timeouts == [pytest.approx(5, abs=1)]  # not the usual 30


async def test_ready_status_is_accepted_after_the_deadline(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    extension.statuses = [{"failure": None, "not_ready": []}]

    await environment._wait_for_services(time.monotonic() - 1)  # returns, doesn't raise


async def test_status_reads_give_up_after_the_attempts(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
    extension.preempt = [httpx.Response(503)] * environment_module.SERVER_RETRY_ATTEMPTS

    with pytest.raises(RuntimeError, match=r"reading compose services failed \(503\)"):
        await environment._wait_for_services(time.monotonic() + 30)


async def test_unknown_service_and_stop_are_unsupported(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))

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


async def test_downloads_decode_exec_output(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = await _started(_environment(_task(tmp_path), server))
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
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment_module, "SERVICE_DOWNLOAD_BLOCK_BYTES", 2)
    monkeypatch.setattr(environment_module, "SERVICE_DOWNLOAD_CHUNK_BLOCKS", 2)
    environment = await _started(_environment(_task(tmp_path), server))
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


# Agent bundles go through the same extension, so they're tested here with the same fakes.

BUNDLE: dict[str, Any] = {
    "image": "registry.example.com/bundles/claude@sha256:" + "a" * 64,
    "installer": "/opt/agent-bundle/copy-agent",
    "entrypoint": "bin/claude",
}


def _bundle_environment(
    tmp_path: Path, server: Server, *, role: str = "agent", compose: bool = False
) -> NemoOpenSandboxEnvironment:
    """An environment with the test bundle, for the agent's sandbox or a separate verifier's."""
    environment_dir = _task(tmp_path, compose=compose)
    trial = "bundle-task__abc"
    return _environment(
        environment_dir,
        server,
        compose_services=COMPOSE_SERVICES if compose else None,
        agent_bundle=BUNDLE,
        session_id=f"{trial}__env" if role == "agent" else f"{trial}__verifier__tests",
        trial_paths=TrialPaths(trial_dir=tmp_path / trial),
    )


async def test_agent_sandbox_create_sends_the_bundle(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = _bundle_environment(tmp_path, server)

    await environment._create_sandbox()

    sent = server.creates[-1]["extensions"]
    assert SandboxAgentBundle.model_validate_json(
        sent[AGENT_BUNDLE_EXTENSION_KEY]
    ) == SandboxAgentBundle.model_validate(BUNDLE)
    assert EXTENSION_KEY not in sent
    assert extension.requests[0].url.path == "/v1/nemo-ext/health"


async def test_bundle_and_services_ride_on_the_same_create(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    environment = _bundle_environment(tmp_path, server, compose=True)

    await environment._create_sandbox()

    assert {AGENT_BUNDLE_EXTENSION_KEY, EXTENSION_KEY} <= server.creates[-1]["extensions"].keys()


async def test_separate_verifier_gets_no_bundle(tmp_path: Path, server: Server, extension: _Extension) -> None:
    environment = _bundle_environment(tmp_path, server, role="verifier")

    await environment._create_sandbox()

    assert AGENT_BUNDLE_EXTENSION_KEY not in (server.creates[-1].get("extensions") or {})
    # A plain sandbox needs no extension.
    assert extension.requests == []


async def test_extension_without_bundle_support_fails_before_creating(
    tmp_path: Path, server: Server, extension: _Extension
) -> None:
    extension.features = None
    environment = _bundle_environment(tmp_path, server)

    with pytest.raises(RuntimeError, match="doesn't support agent bundles"):
        await environment._create_sandbox()

    assert server.creates == []


@pytest.mark.parametrize(
    "image",
    [
        # Neither a digest nor a tag.
        "registry.example.com/bundles/claude",
        # No registry, which CRI-O won't resolve.
        "bundles/claude@sha256:" + "a" * 64,
    ],
)
def test_bundle_image_must_be_pinned_and_qualified(tmp_path: Path, server: Server, image: str) -> None:
    with pytest.raises(ValueError, match="image"):
        _environment(
            _task(tmp_path, compose=False),
            server,
            compose_services=None,
            agent_bundle={**BUNDLE, "image": image},
            session_id="bundle-task__abc__env",
            trial_paths=TrialPaths(trial_dir=tmp_path / "bundle-task__abc"),
        )


async def test_failed_bundle_fails_the_create(tmp_path: Path, server: Server, extension: _Extension) -> None:
    server.healthy = False  # main never starts while the bundle's init container has failed
    extension.statuses = [{"failure": "agent bundle exited with code 1 (Error)", "not_ready": []}]
    environment = _bundle_environment(tmp_path, server)

    # The create's readiness wait polls the extension, so it fails with the cause instead of timing out.
    with pytest.raises(RuntimeError, match="agent bundle exited with code 1"):
        await environment._create_sandbox()

    # Not a transient error, so the create isn't retried, and nothing is left running.
    assert len(server.creates) == 1
    assert server.live() == set()


def _scripted_exec(
    environment: NemoOpenSandboxEnvironment, monkeypatch: pytest.MonkeyPatch, results: dict[str, ExecResult]
) -> list[tuple[str, dict[str, Any]]]:
    """Answer ``environment.exec`` from ``results`` by command, recording each call."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def exec_(command: str, **kwargs: Any) -> ExecResult:
        # start() also opens up the log directories; only the bundle's commands are scripted.
        if command.startswith("chmod "):
            return ExecResult(stdout="", stderr="", return_code=0)
        calls.append((command, kwargs))
        return results[command]

    monkeypatch.setattr(environment, "exec", exec_)
    return calls


async def test_start_puts_the_bundle_first_on_the_image_path_and_checks_it(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _bundle_environment(tmp_path, server)
    calls = _scripted_exec(
        environment,
        monkeypatch,
        {
            'printf %s "$PATH"': ExecResult(stdout="/home/agent/.local/bin:/usr/bin:/bin", return_code=0),
            "/installed-agent/bin/claude --version": ExecResult(stdout="2.1.301 (Claude Code)\n", return_code=0),
        },
    )

    await environment.start(force_build=False)

    assert [command for command, _ in calls] == ['printf %s "$PATH"', "/installed-agent/bin/claude --version"]
    # Both run as the default user, like Harbor's own install check.
    assert all("user" not in kwargs for _, kwargs in calls)
    assert environment._persistent_env["PATH"] == "/installed-agent/bin:/home/agent/.local/bin:/usr/bin:/bin"


@pytest.mark.parametrize(
    "path_probe",
    [ExecResult(stdout="", stderr="sh: not found", return_code=127), ExecResult(stdout="", return_code=0)],
)
async def test_start_fails_when_the_path_cannot_be_read(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch, path_probe: ExecResult
) -> None:
    environment = _bundle_environment(tmp_path, server)
    calls = _scripted_exec(environment, monkeypatch, {'printf %s "$PATH"': path_probe})

    with pytest.raises(RuntimeError, match=r"could not read the sandbox's PATH"):
        await environment.start(force_build=False)

    # The agent's version check never runs, and PATH is left alone.
    assert [command for command, _ in calls] == ['printf %s "$PATH"']
    assert "PATH" not in environment._persistent_env


async def test_start_fails_when_the_bundle_does_not_run(
    tmp_path: Path, server: Server, extension: _Extension, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _bundle_environment(tmp_path, server)
    _scripted_exec(
        environment,
        monkeypatch,
        {
            'printf %s "$PATH"': ExecResult(stdout="/usr/bin:/bin", return_code=0),
            "/installed-agent/bin/claude --version": ExecResult(
                stderr="/installed-agent/bin/claude: not found", return_code=127
            ),
        },
    )

    with pytest.raises(RuntimeError, match=r"Agent bundle check failed: .* exited 127: .*not found"):
        await environment.start(force_build=False)
