# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ``nemo services`` process lifecycle.

These tests spawn real subprocesses and use real flock/descriptor files to verify
the full lifecycle.  They complement the unit tests in ``test_services_process.py``
(which mock process operations) and the CLI tests in ``test_services.py``.

Coverage:
- flock-based liveness survives subprocess exit (SIGTERM, SIGKILL, clean exit)
- Descriptor cleanup on signal
- Multiple scopes coexist
- PID reuse detected via create_time
- Crash recovery (stale descriptor, released flock)
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.local.process import (
    InstanceDescriptor,
    PortConflict,
    acquire_lock,
    check_port_available_for_start,
    format_port_conflict,
    instance_dir,
    is_instance_alive,
    is_port_bindable,
    list_instances,
    prune_instances,
    read_descriptor,
    remove_instance,
    stop_instance,
    write_descriptor,
)
from nhx.platform_runner.config import HelixAppConfig
from typer.testing import CliRunner

_runner = CliRunner()

_STARTUP_TIMEOUT = 15
_SHUTDOWN_TIMEOUT = 5

# Script that acquires flock, writes descriptor, signals readiness, then sleeps.
_PLATFORM_SCRIPT = """\
import fcntl, json, os, signal, sys, time

base_dir = sys.argv[1]
scope = sys.argv[2]
ready_marker = sys.argv[3]

inst_dir = os.path.join(base_dir, "instances", scope)
os.makedirs(inst_dir, exist_ok=True)

# Acquire flock
lock_path = os.path.join(inst_dir, "services.lock")
lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

# Write descriptor
import psutil as _psutil
desc = {
    "pid": os.getpid(),
    "config": {
        "scope": scope,
        "host": "127.0.0.1",
        "port": 8080,
    },
    "mode": "background",
    "create_time": _psutil.Process(os.getpid()).create_time(),
    "started_at": "test",
    "log_path": None,
}
desc_path = os.path.join(inst_dir, "instance.json")
with open(desc_path, "w") as f:
    json.dump(desc, f, indent=2)

def cleanup(signum, _frame):
    try:
        os.unlink(desc_path)
    except FileNotFoundError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Signal readiness
with open(ready_marker, "w") as f:
    f.write(str(os.getpid()))

while True:
    time.sleep(0.1)
"""


def _spawn_platform(script_dir: Path, base_dir: Path, scope: str):
    """Spawn a fake platform that acquires flock and writes descriptor."""
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "fake_platform.py"
    script_path.write_text(_PLATFORM_SCRIPT)
    ready_marker = script_dir / "ready"

    proc = subprocess.Popen(
        [sys.executable, str(script_path), str(base_dir), scope, str(ready_marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if ready_marker.exists():
            return proc
        if proc.poll() is not None:
            pytest.fail(f"Platform process exited early: {proc.returncode}")
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=5)
    pytest.fail(f"Platform process did not become ready within {_STARTUP_TIMEOUT}s")


class TestFlockLifecycle:
    """Verify flock is released on process exit and descriptor is cleaned by signal handler."""

    def test_sigterm_releases_flock_and_cleans_descriptor(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "sigterm-test"
        proc = _spawn_platform(tmp_path / "scripts", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) is not None

            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)

            assert not is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) is None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_sigint_releases_flock_and_cleans_descriptor(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "sigint-test"
        proc = _spawn_platform(tmp_path / "scripts", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)

            os.kill(proc.pid, signal.SIGINT)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)

            assert not is_instance_alive(scope, base_dir=base_dir)
            assert read_descriptor(scope, base_dir=base_dir) is None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_sigkill_releases_flock_leaves_stale_descriptor(self, tmp_path: Path) -> None:
        """SIGKILL prevents cleanup handler from running, but flock is released by kernel."""
        base_dir = tmp_path / "state"
        scope = "sigkill-test"
        proc = _spawn_platform(tmp_path / "scripts", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)

            assert not is_instance_alive(scope, base_dir=base_dir)
            # Descriptor is stale but still on disk (signal handler didn't run)
            desc = read_descriptor(scope, base_dir=base_dir)
            assert desc is not None
            assert desc.pid == proc.pid
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


class TestMultipleScopes:
    """Verify instances in different scopes coexist independently."""

    def test_two_scopes_coexist(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        proc_a = _spawn_platform(tmp_path / "run-a", base_dir, "scope-a")
        proc_b = _spawn_platform(tmp_path / "run-b", base_dir, "scope-b")
        try:
            assert is_instance_alive("scope-a", base_dir=base_dir)
            assert is_instance_alive("scope-b", base_dir=base_dir)

            instances = list_instances(base_dir=base_dir)
            alive_scopes = {i.scope for i in instances if i.alive}
            assert alive_scopes == {"scope-a", "scope-b"}

            # Stop only scope-a
            os.kill(proc_a.pid, signal.SIGTERM)
            proc_a.wait(timeout=_SHUTDOWN_TIMEOUT)

            assert not is_instance_alive("scope-a", base_dir=base_dir)
            assert is_instance_alive("scope-b", base_dir=base_dir)
        finally:
            for p in (proc_a, proc_b):
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=5)


class TestStopInstance:
    """Integration: stop_instance with a real subprocess."""

    def test_stop_sends_sigterm(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "stop-real"
        proc = _spawn_platform(tmp_path / "scripts", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)

            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert proc.pid in result.stopped_pids

            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


class TestCrashRecovery:
    """After a crash (SIGKILL), a new instance should start cleanly."""

    def test_new_instance_after_crash(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "crash-recover"

        # Start and crash
        proc1 = _spawn_platform(tmp_path / "run1", base_dir, scope)
        os.kill(proc1.pid, signal.SIGKILL)
        proc1.wait(timeout=_SHUTDOWN_TIMEOUT)

        # Flock released by kernel, descriptor is stale
        assert not is_instance_alive(scope, base_dir=base_dir)
        assert read_descriptor(scope, base_dir=base_dir) is not None

        # New instance should start -- the stale descriptor doesn't prevent it
        # because the flock is what matters
        proc2 = _spawn_platform(tmp_path / "run2", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)
            desc = read_descriptor(scope, base_dir=base_dir)
            assert desc is not None
            assert desc.pid == proc2.pid
            assert desc.pid != proc1.pid
        finally:
            os.kill(proc2.pid, signal.SIGTERM)
            proc2.wait(timeout=_SHUTDOWN_TIMEOUT)


class TestPidReuseProtection:
    """Verify create_time validation prevents PID reuse false positives."""

    def test_stale_descriptor_with_reused_pid(self, tmp_path: Path) -> None:
        """A stale descriptor whose PID was reused by another process is detected."""
        base_dir = tmp_path / "state"
        scope = "reuse-test"

        # Create a real process to get a valid PID
        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # Write a descriptor with the sleeper's PID but wrong create_time
            desc = InstanceDescriptor(
                pid=sleeper.pid,
                config=HelixAppConfig(scope=scope),
                mode="background",
                create_time=0.0,  # intentionally wrong
            )
            write_descriptor(desc, base_dir=base_dir)

            # stop_instance should detect the mismatch and clean up
            result = stop_instance(scope, base_dir=base_dir)
            assert result.stopped_pids == []
            assert read_descriptor(scope, base_dir=base_dir) is None
        finally:
            sleeper.terminate()
            try:
                sleeper.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sleeper.kill()
                sleeper.wait(timeout=5)


class TestFullLifecycle:
    """P0: Full start -> detect -> stop -> detect-gone -> start-again cycle."""

    def test_start_detect_stop_restart(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "full-lifecycle"

        # Phase 1: Start
        proc1 = _spawn_platform(tmp_path / "run1", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)
            desc = read_descriptor(scope, base_dir=base_dir)
            assert desc is not None
            assert desc.pid == proc1.pid

            # Phase 2: Stop
            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert proc1.pid in result.stopped_pids
            proc1.wait(timeout=_SHUTDOWN_TIMEOUT)
        finally:
            if proc1.poll() is None:
                proc1.kill()
                proc1.wait(timeout=5)

        # Phase 3: Verify gone
        assert not is_instance_alive(scope, base_dir=base_dir)

        # Phase 4: Start again
        proc2 = _spawn_platform(tmp_path / "run2", base_dir, scope)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)
            desc = read_descriptor(scope, base_dir=base_dir)
            assert desc is not None
            assert desc.pid == proc2.pid
            assert desc.pid != proc1.pid
        finally:
            os.kill(proc2.pid, signal.SIGTERM)
            proc2.wait(timeout=_SHUTDOWN_TIMEOUT)


class TestLogSurvivesRestart:
    """Previous boot's logs survive a restart."""

    def test_log_preserved_across_restart(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "log-survive"

        d = instance_dir(scope, base_dir=base_dir)
        log = d / "services.log"
        log.write_text("first boot log content\n")

        from nemo_helix_ext.local.process import rotate_log

        new_log = rotate_log(scope, base_dir=base_dir)
        new_log.write_text("second boot log content\n")

        rotated = list(d.glob("services.log.*"))
        assert len(rotated) == 1
        assert rotated[0].read_text() == "first boot log content\n"
        assert new_log.read_text() == "second boot log content\n"


# ---------------------------------------------------------------------------
# End-to-end: health polling with a real HTTP server
# ---------------------------------------------------------------------------

_HEALTH_SERVER_SCRIPT = """\
\"\"\"Minimal platform stub: flock + descriptor + HTTP /health/ready.\"\"\"
import fcntl, http.server, json, os, signal, sys, threading

base_dir = sys.argv[1]
scope = sys.argv[2]
port = int(sys.argv[3])
ready_marker = sys.argv[4]

inst_dir = os.path.join(base_dir, "instances", scope)
os.makedirs(inst_dir, exist_ok=True)

# Acquire flock
lock_path = os.path.join(inst_dir, "services.lock")
lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

# Write descriptor
import psutil as _psutil
desc = {
    "pid": os.getpid(),
    "config": {
        "scope": scope,
        "host": "127.0.0.1",
        "port": port,
    },
    "mode": "background",
    "create_time": _psutil.Process(os.getpid()).create_time(),
    "started_at": "test",
    "log_path": None,
}
desc_path = os.path.join(inst_dir, "instance.json")
with open(desc_path, "w") as f:
    json.dump(desc, f, indent=2)

# HTTP server for /status
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass  # suppress request logs

server = http.server.HTTPServer(("127.0.0.1", port), Handler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()

def shutdown(signum, _frame):
    server.shutdown()
    try:
        os.unlink(desc_path)
    except FileNotFoundError:
        pass
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# Signal readiness
with open(ready_marker, "w") as f:
    f.write(str(os.getpid()))

# Block
import time
while True:
    time.sleep(1)
"""


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_health_server(script_dir: Path, base_dir: Path, scope: str, port: int):
    """Spawn a minimal platform stub with a real HTTP health endpoint."""
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "health_server.py"
    script_path.write_text(_HEALTH_SERVER_SCRIPT)
    ready_marker = script_dir / "ready"

    proc = subprocess.Popen(
        [sys.executable, str(script_path), str(base_dir), scope, str(port), str(ready_marker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if ready_marker.exists():
            return proc
        if proc.poll() is not None:
            pytest.fail(f"Health server exited early with code {proc.returncode}")
        time.sleep(0.1)
    proc.kill()
    proc.wait(timeout=5)
    pytest.fail(f"Health server did not become ready within {_STARTUP_TIMEOUT}s")


class TestEndToEndHealthPolling:
    """Integration: _wait_for_healthy against a real HTTP server."""

    def test_health_poll_succeeds_with_real_server(self, tmp_path: Path) -> None:
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        base_dir = tmp_path / "state"
        scope = "e2e-health"
        port = _find_free_port()

        proc = _spawn_health_server(tmp_path / "scripts", base_dir, scope, port)
        try:
            result = _wait_for_healthy("127.0.0.1", port, timeout=10, poll_interval=0.2)
            assert result is True
            assert is_instance_alive(scope, base_dir=base_dir)
        finally:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)

    def test_health_poll_fails_when_server_down(self, tmp_path: Path) -> None:
        from nemo_helix_ext.cli.commands.services.cli import _wait_for_healthy

        port = _find_free_port()
        result = _wait_for_healthy("127.0.0.1", port, timeout=0.5, poll_interval=0.1)
        assert result is False

    def test_stop_after_health_check(self, tmp_path: Path) -> None:
        """Full cycle: spawn health server, verify healthy, stop it, verify gone."""
        base_dir = tmp_path / "state"
        scope = "e2e-stop"
        port = _find_free_port()

        proc = _spawn_health_server(tmp_path / "scripts", base_dir, scope, port)
        try:
            assert is_instance_alive(scope, base_dir=base_dir)
            desc = read_descriptor(scope, base_dir=base_dir)
            assert desc is not None
            assert desc.config.port == port

            result = stop_instance(scope, base_dir=base_dir, timeout=5.0)
            assert proc.pid in result.stopped_pids
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
            assert not is_instance_alive(scope, base_dir=base_dir)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


class TestInstanceCleanup:
    """Integration tests for rm/prune and post-stop scope directories."""

    def test_stop_leaves_record_until_rm(self, tmp_path: Path, monkeypatch) -> None:
        base_dir = tmp_path / "state"
        scope = "stop-then-rm"
        monkeypatch.setenv("_NHX_STATE_DIR", str(base_dir))

        proc = _spawn_platform(tmp_path / "scripts", base_dir, scope)
        try:
            stop_instance(scope, base_dir=base_dir, timeout=5.0)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        scope_dir = instance_dir(scope, base_dir=base_dir)
        (scope_dir / "services.log").write_text("post-stop log\n")
        instances = list_instances(base_dir=base_dir)
        assert len(instances) == 1
        assert instances[0].alive is False

        assert remove_instance(scope, base_dir=base_dir) is True
        assert not scope_dir.exists()

    def test_stop_then_rm_via_cli(self, tmp_path: Path, monkeypatch) -> None:
        base_dir = tmp_path / "state"
        scope = "stop-then-rm-cli"
        monkeypatch.setenv("_NHX_STATE_DIR", str(base_dir))

        proc = _spawn_platform(tmp_path / "scripts-rm-cli", base_dir, scope)
        try:
            stop_instance(scope, base_dir=base_dir, timeout=5.0)
            proc.wait(timeout=_SHUTDOWN_TIMEOUT)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        scope_dir = instance_dir(scope, base_dir=base_dir)
        (scope_dir / "services.log").write_text("post-stop log\n")
        assert scope_dir.exists()

        result = _runner.invoke(app, ["services", "rm", scope])
        assert result.exit_code == 0
        assert f"Removed instance '{scope}'" in result.stdout
        assert not scope_dir.exists()

    def test_sigkill_crash_then_prune(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        scope = "crash-prune"
        proc = _spawn_platform(tmp_path / "run1", base_dir, scope)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=_SHUTDOWN_TIMEOUT)

        scope_dir = instance_dir(scope, base_dir=base_dir)
        (scope_dir / "services.log").write_text("crash log\n")

        removed = prune_instances(base_dir=base_dir)
        assert scope in removed
        assert not scope_dir.exists()

    def test_prune_skips_running(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        running = _spawn_platform(tmp_path / "run-a", base_dir, "scope-running")
        stopped_dir = instance_dir("scope-stopped", base_dir=base_dir)
        (stopped_dir / "services.log").write_text("x\n")
        try:
            removed = prune_instances(base_dir=base_dir)
            assert removed == ["scope-stopped"]
            assert is_instance_alive("scope-running", base_dir=base_dir)
            assert not stopped_dir.exists()
        finally:
            os.kill(running.pid, signal.SIGTERM)
            running.wait(timeout=_SHUTDOWN_TIMEOUT)

    def test_logs_available_before_rm_not_after(self, tmp_path: Path, monkeypatch) -> None:
        base_dir = tmp_path / "state"
        scope = "logs-rm"
        monkeypatch.setenv("_NHX_STATE_DIR", str(base_dir))
        d = instance_dir(scope, base_dir=base_dir)
        (d / "services.log").write_text("debug line\n")

        before = _runner.invoke(app, ["services", "logs", "--instance", scope])
        assert before.exit_code == 0
        assert "debug line" in before.stdout

        remove_instance(scope, base_dir=base_dir)

        after = _runner.invoke(app, ["services", "logs", "--instance", scope])
        assert after.exit_code == 0
        assert "No log file" in after.stdout


class TestPortAvailability:
    """Unit tests for port bind preflight helpers."""

    def test_is_port_bindable_false_when_port_in_use(self) -> None:
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            assert is_port_bindable("127.0.0.1", port) is False

    def test_check_port_returns_foreign_when_blocked_without_nemo_instance(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "state"
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            conflict = check_port_available_for_start(
                "127.0.0.1",
                port,
                f"scope-{port}",
                base_dir=base_dir,
            )
        assert conflict is not None
        assert conflict.kind == "foreign"
        assert conflict.port == port

    def test_check_port_returns_nemo_instance_when_lock_held_and_port_blocked(self, tmp_path: Path) -> None:
        """Held flock + occupied port → NeMo instance message, not foreign process."""
        base_dir = tmp_path / "state"
        scope = "nemo-lock-port-test"
        port = _find_free_port()
        fd = acquire_lock(scope, base_dir=base_dir)
        try:
            write_descriptor(
                InstanceDescriptor(
                    pid=os.getpid(),
                    config=HelixAppConfig(scope=scope, host="127.0.0.1", port=port),
                    mode="background",
                    create_time=1.0,
                ),
                base_dir=base_dir,
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                sock.listen(1)
                conflict = check_port_available_for_start(
                    "127.0.0.1",
                    port,
                    scope,
                    base_dir=base_dir,
                )
            assert conflict is not None
            assert conflict.kind == "nemo_instance"
            assert conflict.scope == scope
            assert conflict.port == port
        finally:
            os.close(fd)

    def test_check_port_returns_foreign_when_alive_instance_uses_different_port(self, tmp_path: Path) -> None:
        """Live instance on another port must not claim a foreign listener as NeMo-owned."""
        base_dir = tmp_path / "state"
        scope = "explicit-instance"
        nemo_port = _find_free_port()
        foreign_port = _find_free_port()
        fd = acquire_lock(scope, base_dir=base_dir)
        try:
            write_descriptor(
                InstanceDescriptor(
                    pid=os.getpid(),
                    config=HelixAppConfig(scope=scope, host="127.0.0.1", port=nemo_port),
                    mode="background",
                    create_time=1.0,
                ),
                base_dir=base_dir,
            )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", foreign_port))
                sock.listen(1)
                conflict = check_port_available_for_start(
                    "127.0.0.1",
                    foreign_port,
                    scope,
                    base_dir=base_dir,
                )
            assert conflict is not None
            assert conflict.kind == "foreign"
            assert conflict.port == foreign_port
        finally:
            os.close(fd)

    @pytest.mark.parametrize(
        ("conflict", "expected_substrings"),
        [
            (
                PortConflict(kind="foreign", port=8080),
                ("already in use", "EADDRINUSE", "lsof -i :8080"),
            ),
            (
                PortConflict(kind="nemo_instance", port=8080, scope="abc-8080"),
                ("NeMo Helix instance", "nemo services stop"),
            ),
        ],
    )
    def test_format_port_conflict(
        self,
        conflict: PortConflict,
        expected_substrings: tuple[str, ...],
    ) -> None:
        lines = format_port_conflict(conflict)
        for substring in expected_substrings:
            assert any(substring in line for line in lines)
