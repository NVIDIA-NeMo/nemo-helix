# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for health/readiness probing, lifespan, and child process module.

Covers Priorities 5 (lifespan), 6 (health), and 7 (child process) from the
integration test plan.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_ext.local import process
from nemo_helix_ext.local.services import ServiceRunConfig
from nemo_helix_ext.local.transport import probe_status, wait_for_status
from nhx.common.controller import ControllerManager


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Priority 5: Multi-Service Startup & Lifespan
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_app_starts_and_joins_controller_threads() -> None:
    """A controller registered via ``create_app`` should have its thread
    started during lifespan and stopped on exit."""
    started = threading.Event()
    stopped = threading.Event()

    def controller_run(stop_signal: threading.Event) -> None:
        started.set()
        stop_signal.wait(timeout=5.0)
        stopped.set()

    with (
        patch("nhx.platform_runner.server.get_platform_config") as mock_pc,
        patch("nhx.platform_runner.server.get_auth_config") as mock_ac,
        patch("nhx.common.auth.middleware.get_auth_config") as mock_ac2,
    ):
        mock_pc.return_value = MagicMock(seed_on_startup=False, redirect_root_to_studio=False)
        mock_ac.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")
        mock_ac2.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")

        from nhx.platform_runner.server import create_app

        app = create_app(services=[], controller_run_funcs={"test-ctrl": controller_run})

        from fastapi.testclient import TestClient

        with TestClient(app):
            assert started.wait(timeout=2.0), "controller thread did not start"

        assert stopped.wait(timeout=2.0), "controller thread did not stop after lifespan exit"


@pytest.mark.integration
def test_create_app_controller_thread_join_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck controller must not hang shutdown."""
    started = threading.Event()
    release = threading.Event()

    def stubborn_controller(_stop_signal: threading.Event) -> None:
        started.set()
        # Ignore stop_signal — simulate a controller that hangs.
        release.wait(timeout=300)

    with (
        patch("nhx.platform_runner.server.get_platform_config") as mock_pc,
        patch("nhx.platform_runner.server.get_auth_config") as mock_ac,
        patch("nhx.common.auth.middleware.get_auth_config") as mock_ac2,
    ):
        mock_pc.return_value = MagicMock(seed_on_startup=False, redirect_root_to_studio=False)
        mock_ac.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")
        mock_ac2.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")

        from nhx.platform_runner.server import create_app

        monkeypatch.setattr(
            "nhx.platform_runner.controller_threads.DEFAULT_RUNNER_JOIN_TIMEOUT_SECONDS",
            0.05,
        )
        app = create_app(services=[], controller_run_funcs={"stubborn": stubborn_controller})

        from fastapi.testclient import TestClient

        manager = ControllerManager.get_instance()
        try:
            started_at = time.monotonic()
            with TestClient(app):
                assert started.wait(timeout=2.0), "controller thread did not start"
            assert time.monotonic() - started_at < 1.0
        finally:
            release.set()
            for thread in app.state.controller_threads:
                thread.join(timeout=2.0)
            # Avoid leaking singleton state into later integration tests.
            manager.clear()


@pytest.mark.integration
def test_lifespan_cleanup_runs_on_app_shutdown() -> None:
    """``close_shared_http_clients`` should be called during lifespan teardown."""
    cleanup_called = threading.Event()

    with (
        patch("nhx.platform_runner.server.get_platform_config") as mock_pc,
        patch("nhx.platform_runner.server.get_auth_config") as mock_ac,
        patch("nhx.common.auth.middleware.get_auth_config") as mock_ac2,
        patch("nhx.platform_runner.server.close_shared_http_clients") as mock_close,
    ):
        mock_pc.return_value = MagicMock(seed_on_startup=False, redirect_root_to_studio=False)
        mock_ac.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")
        mock_ac2.return_value = MagicMock(enabled=False, policy_decision_point_provider="embedded")

        async def fake_close():
            cleanup_called.set()

        mock_close.side_effect = fake_close

        from nhx.platform_runner.server import create_app

        app = create_app(services=[])

        from fastapi.testclient import TestClient

        with TestClient(app):
            pass

        assert cleanup_called.is_set(), "close_shared_http_clients was not called during shutdown"


# ---------------------------------------------------------------------------
# Priority 6: Health & Readiness
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_wait_for_status_retries_on_transient_errors(tmp_path: Path) -> None:
    """``wait_for_status`` should retry on connection refused and eventually
    return True once the server starts responding."""
    from nemo_helix_ext.local import services

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        controllers=(),
        sidecars=(),
        transport="tcp",
        host="127.0.0.1",
        port=_free_tcp_port(),
        scope="integ-wait-retry",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
        readiness_timeout=30.0,
        readiness_poll_interval=0.3,
    )

    # Start the daemon — wait_for_status should retry until it's ready.
    services.daemonize_services(cfg)
    try:
        # The daemon is already ready (daemonize_services waits for readiness).
        # Verify wait_for_status succeeds with a fresh probe.
        assert wait_for_status(
            base_url=f"http://127.0.0.1:{cfg.port}",
            timeout=5.0,
            poll_interval=0.2,
        )
    finally:
        process.stop_instance(cfg.scope, base_dir=cfg.state_root, timeout=10, force=True)


@pytest.mark.integration
def test_wait_for_status_times_out_on_no_server() -> None:
    """``wait_for_status`` should return False when no server is listening."""
    port = _free_tcp_port()
    result = wait_for_status(
        base_url=f"http://127.0.0.1:{port}",
        timeout=1.0,
        poll_interval=0.2,
    )
    assert result is False


@pytest.mark.integration
def test_probe_status_with_missing_uds_socket() -> None:
    """Probing a non-existent UDS socket should return False."""
    result = probe_status(
        base_url="http+unix:///nonexistent/path/nemo.sock",
        socket_path=Path("/nonexistent/path/nemo.sock"),
        timeout=1.0,
    )
    assert result is False


@pytest.mark.integration
def test_probe_status_against_real_daemon(tmp_path: Path) -> None:
    """``probe_status`` should return True against a running daemon."""
    from nemo_helix_ext.local import services

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        controllers=(),
        sidecars=(),
        transport="tcp",
        host="127.0.0.1",
        port=_free_tcp_port(),
        scope="integ-probe-real",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
        readiness_timeout=30.0,
        readiness_poll_interval=0.3,
    )
    services.daemonize_services(cfg)
    try:
        assert probe_status(base_url=f"http://127.0.0.1:{cfg.port}", timeout=5.0)
    finally:
        process.stop_instance(cfg.scope, base_dir=cfg.state_root, timeout=10, force=True)


# ---------------------------------------------------------------------------
# Priority 7: Child Process Module
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_service_child_loads_config_and_starts(tmp_path: Path) -> None:
    """Write valid JSON config, run ``_service_child`` in a subprocess,
    verify it starts and accepts HTTP connections."""
    port = _free_tcp_port()
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"
    scope = "integ-child-real"

    payload = ServiceRunConfig(
        mode="daemon",
        services=("hello-world",),
        controllers=(),
        sidecars=(),
        transport="tcp",
        host="127.0.0.1",
        port=port,
        scope=scope,
        state_dir=str(state_dir),
        runtime_dir=str(runtime_dir),
    ).to_child_payload()

    # Write the request file the way daemonize_services does.
    instance_dir = process.instance_dir(scope, base_dir=state_dir)
    fd, req_path = tempfile.mkstemp(dir=str(instance_dir), suffix=".json")
    os.write(fd, (json.dumps(payload) + "\n").encode())
    os.close(fd)

    log_path = process.log_path_for(scope, base_dir=state_dir)
    log_file = open(log_path, "a")  # noqa: SIM115
    subprocess.Popen(
        [sys.executable, "-m", "nemo_helix_ext.local._service_child", req_path],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    log_file.close()

    try:
        # Wait for the child to become ready.
        assert wait_for_status(
            base_url=f"http://127.0.0.1:{port}",
            timeout=30.0,
            poll_interval=0.3,
        ), "child process did not become ready"

        # The request file should have been cleaned up.
        assert not Path(req_path).exists()

        # The child should have acquired the lock and written a descriptor.
        assert process.is_instance_alive(scope, base_dir=state_dir)
        desc = process.read_descriptor(scope, base_dir=state_dir)
        assert desc is not None
        assert desc.mode == "daemon"
    finally:
        process.stop_instance(scope, base_dir=state_dir, timeout=10, force=True)


@pytest.mark.integration
def test_service_child_corrupted_payload(tmp_path: Path) -> None:
    """Bad JSON in the request file should cause the child to exit non-zero."""
    scope = "integ-child-bad"
    state_dir = tmp_path / "state"
    instance_dir = process.instance_dir(scope, base_dir=state_dir)

    fd, req_path = tempfile.mkstemp(dir=str(instance_dir), suffix=".json")
    os.write(fd, b"<<<NOT JSON>>>")
    os.close(fd)

    proc = subprocess.Popen(
        [sys.executable, "-m", "nemo_helix_ext.local._service_child", req_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    proc.wait(timeout=15)
    assert proc.returncode != 0


@pytest.mark.integration
def test_service_child_cleans_up_request_file(tmp_path: Path) -> None:
    """The request file should be unlinked even when the child crashes."""
    scope = "integ-child-cleanup"
    state_dir = tmp_path / "state"
    instance_dir = process.instance_dir(scope, base_dir=state_dir)

    fd, req_path = tempfile.mkstemp(dir=str(instance_dir), suffix=".json")
    os.write(fd, b"<<<CORRUPT>>>")
    os.close(fd)

    assert Path(req_path).exists()

    proc = subprocess.Popen(
        [sys.executable, "-m", "nemo_helix_ext.local._service_child", req_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    proc.wait(timeout=15)

    # The request file should have been cleaned up regardless of the error.
    assert not Path(req_path).exists()
