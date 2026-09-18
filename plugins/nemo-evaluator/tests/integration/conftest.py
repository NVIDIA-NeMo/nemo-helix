# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared harness + fixtures for evaluator-plugin integration tests.

``running_platform`` is the one place the ``nemo services run`` lifecycle (launch on a chosen
port, wait for readiness, tear down) lives, so each suite contributes a thin session-scoped
fixture rather than re-rolling the boilerplate. Each platform binds its own port so independent
suites can coexist in a single test session (e.g. the agent-eval submit platform here vs. the
intake/publish platform on its own port).
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from nmp.testing import igw_mock_provider_mode

REPO_ROOT = Path(__file__).resolve().parents[4]

#: IGW mock-provider prefix, and the env var the in-process IGW config reads it from. Applied to
#: *this* (test) process via the ``_igw_mock_prefix`` fixture (not at import) so it never leaks into
#: unrelated unit suites that merely collect this conftest; passed to the platform subprocess via
#: each fixture's ``env_vars``.
MOCK_PROVIDER_PREFIX = "igw-mock-"
MOCK_PROVIDER_PREFIX_ENVVAR = "NMP_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX"
CLICKHOUSE_XDIST_GROUP = "nmp_intake_clickhouse"
CLICKHOUSE_XDIST_FIXTURE = "_clickhouse"

#: Base URL (and therefore port) for the agent-eval subprocess-backend platform. Distinct from
#: other integration platforms so both can run in the same session without a port clash.
AGENT_PLATFORM_BASE_URL = os.environ.get("NMP_AGENT_BASE_URL", "http://localhost:8090")

#: Base URL for the docker-backend platform — its own port so it can coexist with the subprocess one.
AGENT_DOCKER_PLATFORM_BASE_URL = os.environ.get("NMP_AGENT_DOCKER_BASE_URL", "http://localhost:8091")

#: Base URL for the auth-enabled subprocess platform (own port, coexists with the others).
AGENT_AUTH_PLATFORM_BASE_URL = os.environ.get("NMP_AGENT_AUTH_BASE_URL", "http://localhost:8092")

# xdist ``loadgroup`` does not infer shared fixtures; it only groups tests that
# carry the same ``xdist_group`` marker. Any evaluator test that depends on the
# ClickHouse fixture uses Intake's fixed local container, so keep those tests on
# one worker to avoid cross-worker startup/teardown races. ``tryfirst`` matters:
# xdist reads these markers during collection to build its scheduling groups.


def _docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _wait_for_ready(base_url: str, *, timeout: float, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"platform at {base_url} exited early (code {process.returncode}) before ready")
        try:
            with urllib.request.urlopen(f"{base_url}/health/ready", timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"platform at {base_url} not ready within {timeout}s")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if CLICKHOUSE_XDIST_FIXTURE in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.xdist_group(CLICKHOUSE_XDIST_GROUP))


@contextmanager
def running_platform(
    *,
    run_args: list[str],
    base_url: str,
    env_vars: dict[str, str] | None = None,
    ready_timeout: float = 180.0,
) -> Iterator[str]:
    """Run ``nemo services run`` bound to ``base_url``'s port, yield the URL, then tear it down.

    Fails loudly if the port is already taken: a stray platform would otherwise quietly serve the
    tests, masking whatever services/config they actually depend on.
    """
    split = urlsplit(base_url)
    host, port = split.hostname or "localhost", split.port
    if port is None:
        raise ValueError(f"base_url must include an explicit port: {base_url!r}")
    if _port_in_use(host, port):
        raise RuntimeError(f"{host}:{port} is already in use; stop other platform instances before running these tests")

    process = subprocess.Popen(
        ["uv", "run", "nemo", "services", "run", *run_args, "--port", str(port)],
        cwd=REPO_ROOT,
        env={**os.environ, "NMP_BASE_URL": base_url, **(env_vars or {})},
    )
    try:
        _wait_for_ready(base_url, timeout=ready_timeout, process=process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture(autouse=True)
def _igw_mock_prefix() -> Iterator[None]:
    """Configure the IGW mock-provider prefix in the *test* process for each integration test, so
    ``add_mock_provider`` can prefix and register mock providers.

    Function-scoped + autouse so it applies only to these integration tests — never the unit suite
    that merely collects this conftest (which would break the inference-gateway ``is_mock_provider``
    tests), and never another integration suite. Uses a config override via ``igw_mock_provider_mode``
    (an ``nmp.testing`` helper — a plugin must not import ``nmp-common`` directly) rather than an env
    var: the override is honored ahead of the cached/env config, so it works regardless of when the
    IGW config was first read, whereas a whole-repo run can read and cache that config before any
    plugin-local hook fires. The platform *subprocess* gets the same prefix via each fixture's
    ``env_vars``.
    """
    with igw_mock_provider_mode(MOCK_PROVIDER_PREFIX):
        yield


def _materialize_subprocess_config(work_root: Path, *, base_url: str, auth_enabled: bool = False) -> Path:
    """Write a self-contained subprocess-backend platform config under ``work_root``.

    Owned here rather than borrowed from ``e2e/configs`` (legacy, not run in CI). It pins
    ``platform.runtime: none`` with explicit subprocess job profiles and ABSOLUTE storage paths:
    the jobs service writes each step config under ``working_directory`` while the task subprocess
    resolves the same path against a different CWD, so a relative dir would make the task miss its
    config. Absolute paths keep the step-config path agreeing across both processes.

    ``auth_enabled`` turns on the PDP so the auth-forwarding test can prove a submitted task's
    ``X-NMP-*`` service-principal identity actually authenticates its IGW inference.
    """
    jobs_work_dir = str(work_root / "subprocess-jobs")
    subprocess_executor_config = {
        "working_directory": jobs_work_dir,
        "cleanup_completed_jobs_immediately": False,
        "ttl_seconds_before_active": 60,
        "ttl_seconds_active": 3600,
        "ttl_seconds_after_finished": 300,
    }
    config = {
        "platform": {"runtime": "none", "base_url": base_url},
        "auth": {"enabled": auth_enabled, "allow_unsigned_jwt": True},
        "jobs": {
            "executors": [
                {
                    "provider": "subprocess",
                    "profile": "default",
                    "backend": "subprocess",
                    "config": subprocess_executor_config,
                },
                {
                    "provider": "subprocess",
                    "profile": "harbor-test",
                    "backend": "subprocess",
                    "config": subprocess_executor_config,
                },
            ],
            "executor_defaults": {"subprocess": subprocess_executor_config},
        },
        "secrets": {"allow_key_creation": True},
        "files": {"default_storage_config": {"type": "local", "path": str(work_root / "files")}},
    }
    config_path = work_root / "subprocess-platform.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


#: The platform's entity store defaults to the *developer's* data directory
#: (``~/.local/share/nemo/nmp-platform.db``). These fixtures already point file storage at a
#: per-run temp dir, so leaving the database shared both writes test entities into a real local
#: platform and breaks reruns: metric bundles are content-addressed, so the second run finds last
#: run's bundle entity, skips the upload as a duplicate, and then fails to download a blob that
#: went away with the previous run's temp dir. Pointing NMP_DATA_DIR at the same work root keeps
#: entities and blobs with the same lifetime.
DATA_DIR_ENVVAR = "NMP_DATA_DIR"


@pytest.fixture(scope="session")
def subprocess_platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Session-scoped platform with the subprocess jobs backend + IGW mock-provider mode.

    Subprocess backend: the compiled task runs as a host process, so a runner target sees the host's
    agent toolchain. IGW mock mode (``NMP_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX``) lets
    Model/Agent-target tests register a mock provider returning a canned response — no real model or
    key.
    """
    work_root = tmp_path_factory.mktemp("platform")
    config_path = _materialize_subprocess_config(work_root, base_url=AGENT_PLATFORM_BASE_URL)
    with running_platform(
        run_args=["--service-group", "all", "--controller-group", "all"],
        base_url=AGENT_PLATFORM_BASE_URL,
        env_vars={
            "NMP_CONFIG_FILE_PATH": str(config_path),
            DATA_DIR_ENVVAR: str(work_root / "data"),
            MOCK_PROVIDER_PREFIX_ENVVAR: MOCK_PROVIDER_PREFIX,
        },
    ) as base_url:
        yield base_url


@pytest.fixture(scope="session")
def auth_subprocess_platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Session-scoped subprocess-backend platform with ``auth.enabled`` and IGW mock mode.

    Used to prove that a submitted job's forwarded ``X-NMP-*`` service-principal identity
    authenticates its online IGW inference under auth (no bearer). Test-side calls authenticate as an
    internal service principal (which the default PDP policy grants full permissions).
    """
    work_root = tmp_path_factory.mktemp("auth-platform")
    config_path = _materialize_subprocess_config(work_root, base_url=AGENT_AUTH_PLATFORM_BASE_URL, auth_enabled=True)
    with running_platform(
        run_args=["--service-group", "all", "--controller-group", "all"],
        base_url=AGENT_AUTH_PLATFORM_BASE_URL,
        env_vars={
            "NMP_CONFIG_FILE_PATH": str(config_path),
            DATA_DIR_ENVVAR: str(work_root / "data"),
            MOCK_PROVIDER_PREFIX_ENVVAR: MOCK_PROVIDER_PREFIX,
        },
    ) as base_url:
        yield base_url


def _materialize_docker_config(work_root: Path, *, base_url: str) -> Path:
    """Write a docker-backend platform config: ``cpu/default`` routes to the docker jobs backend.

    Critically registers NO subprocess executor: the jobs API's
    ``translate_cpu_container_steps_to_subprocess`` reroutes ``cpu/default`` container steps to the
    host subprocess backend whenever one is registered there, which would make a "docker" test
    silently run on subprocess. ``platform.runtime: docker`` + a ``cpu/default`` docker executor
    keeps the agent-eval step on the docker backend. The jobs-launcher binary is optional (the
    backend falls back to the container's own entrypoint when it's absent), so it isn't built here.

    ``enable_subprocess_executor: False`` is what makes that true, and is load-bearing — do not drop
    it as redundant with the ``executors`` list. Jobs auto-appends a ``subprocess/default`` profile
    for any non-Kubernetes runtime unless this is set (see
    ``get_default_executor_profiles_for_runtime``). The Harbor compiler deliberately selects any
    advertised subprocess profile, so leaving this unset would make the fixture subprocess-capable
    instead of Docker-only.
    """
    docker_executor_config = {
        "launcher_tool_path": str(REPO_ROOT / "services/core/jobs/jobs-launcher/jobs-launcher"),
        "cleanup_completed_jobs_immediately": False,
        "ttl_seconds_before_active": 60,
        "ttl_seconds_active": 3600,
        "ttl_seconds_after_finished": 300,
    }
    config = {
        "platform": {"runtime": "docker", "base_url": base_url},
        "auth": {"enabled": False, "allow_unsigned_jwt": True},
        "jobs": {
            "enable_subprocess_executor": False,
            "executors": [
                {"provider": "cpu", "profile": "default", "backend": "docker", "config": docker_executor_config},
            ],
            "executor_defaults": {"docker": docker_executor_config},
        },
        "secrets": {"allow_key_creation": True},
        "files": {"default_storage_config": {"type": "local", "path": str(work_root / "files")}},
    }
    config_path = work_root / "docker-platform.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture(scope="session")
def docker_platform(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Session-scoped platform with the docker jobs backend (for the docker-backend submit test).

    Gates on a reachable Docker daemon only — a runner executes *inside* the task container, not on
    the host, so the host toolchain is irrelevant here. The agent-eval step runs in the ``cpu-tasks``
    image; runner targets aren't expected to succeed there yet (the image carries no agent harness —
    see AALGO-301), which is why the test using this fixture is marked xfail.
    """
    if not _docker_available():
        pytest.skip("docker daemon not available")
    work_root = tmp_path_factory.mktemp("docker-platform")
    config_path = _materialize_docker_config(work_root, base_url=AGENT_DOCKER_PLATFORM_BASE_URL)
    with running_platform(
        run_args=["--service-group", "all", "--controller-group", "all"],
        base_url=AGENT_DOCKER_PLATFORM_BASE_URL,
        env_vars={"NMP_CONFIG_FILE_PATH": str(config_path), DATA_DIR_ENVVAR: str(work_root / "data")},
    ) as base_url:
        yield base_url
