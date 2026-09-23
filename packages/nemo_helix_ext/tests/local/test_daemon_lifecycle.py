# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for daemon subprocess lifecycle.

These tests spawn REAL child processes via ``daemonize_services()``, exercise
real lock acquisition, descriptor file I/O, HTTP readiness probing, and
graceful shutdown via ``stop_instance()``.  Nothing is monkeypatched away —
the child runs a real uvicorn server with the ``hello-world`` service.

Requirements:
- All packages installed (``uv sync --all-packages``) so entry-point
  discovery finds hello-world.
- ``pyleak`` importable (from the ``[all]`` extra).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest
from nemo_helix_ext.local import process, services
from nemo_helix_ext.local.process import ForegroundInstanceError
from nemo_helix_ext.local.services import (
    ServiceRunConfig,
    ServicesAlreadyRunningError,
    ServicesStartupExitedError,
)
from nhx.platform_runner.config import HelixAppConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_tcp_port() -> int:
    """Bind to port 0, let the OS pick, then release and return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _daemon_config(
    tmp_path: Path,
    *,
    scope: str = "integ-daemon",
    port: int | None = None,
) -> ServiceRunConfig:
    """Build a ServiceRunConfig that is fully isolated under ``tmp_path``."""
    return ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        controllers=(),
        sidecars=(),
        transport="tcp",
        host="127.0.0.1",
        port=port or _free_tcp_port(),
        scope=scope,
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
        readiness_timeout=30.0,
        readiness_poll_interval=0.3,
    )


def _ensure_stopped(cfg: ServiceRunConfig) -> None:
    """Best-effort cleanup: stop any instance left running by a test."""
    try:
        process.stop_instance(cfg.scope, base_dir=cfg.state_root, timeout=10, force=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_daemonize_services_spawns_child_that_becomes_ready(tmp_path: Path) -> None:
    """Spawn a real daemon subprocess, verify readiness via HTTP, then
    gracefully shut down with ``stop_instance``."""
    cfg = _daemon_config(tmp_path)
    handle = None
    try:
        handle = services.daemonize_services(cfg)

        # -- The handle should report the child's PID and transport details.
        assert handle.pid is not None
        assert handle.port == cfg.port
        assert handle.transport == "tcp"

        # -- The lock file should be held by the child.
        assert process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)

        # -- The descriptor should have been written by the child.
        desc = process.read_descriptor(cfg.scope, base_dir=cfg.state_root)
        assert desc is not None
        assert desc.pid == handle.pid
        assert desc.mode == "daemon"
        assert "hello-world" in (desc.config.services or [])

        # -- The child should still be running and respond to /status.
        assert services.probe_status(base_url=f"http://127.0.0.1:{cfg.port}", timeout=5.0)

        # -- Graceful shutdown.
        result = process.stop_instance(cfg.scope, base_dir=cfg.state_root, timeout=15)
        assert handle.pid in result.stopped_pids

        # -- After stop, the lock should be released and the descriptor removed.
        assert not process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)
        assert process.read_descriptor(cfg.scope, base_dir=cfg.state_root) is None
    finally:
        _ensure_stopped(cfg)


@pytest.mark.integration
def test_daemonize_services_child_exit_before_readiness(tmp_path: Path) -> None:
    """When the child exits before becoming ready, ``daemonize_services``
    should raise ``ServicesStartupExitedError`` with the log path."""
    # Spawn a child that will exit immediately: give it a bogus service name
    # that will fail validation in resolve_run_configuration.
    bad_cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("nonexistent-service-xyz",),
        controllers=(),
        sidecars=(),
        transport="tcp",
        host="127.0.0.1",
        port=_free_tcp_port(),
        scope="integ-early-exit",
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "runtime",
        readiness_timeout=15.0,
        readiness_poll_interval=0.2,
    )
    with pytest.raises(ServicesStartupExitedError, match="exited with code"):
        services.daemonize_services(bad_cfg)

    # -- The lock should not be held after the failed startup.
    assert not process.is_instance_alive(bad_cfg.scope, base_dir=bad_cfg.state_root)


@pytest.mark.integration
def test_stale_socket_cleanup_after_process_crash(tmp_path: Path) -> None:
    """If a previous daemon crashed and left a UDS socket file, a new daemon
    startup should clean it up and succeed."""
    scope = "stale"
    # Use a short temp directory to stay within AF_UNIX path limits (103 bytes on macOS).
    short_tmp = Path(tempfile.mkdtemp(prefix="nemo-"))
    runtime_dir = short_tmp / "run"

    # Create a stale UDS socket file (no process listening).
    socket_dir = runtime_dir / scope
    socket_dir.mkdir(parents=True, exist_ok=True)
    stale_socket = socket_dir / "nemo-helix.sock"
    # Bind a real UDS socket to create the file, then close immediately.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.bind(str(stale_socket))
    assert stale_socket.exists()

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        controllers=(),
        sidecars=(),
        transport="uds",
        host="127.0.0.1",
        port=_free_tcp_port(),
        scope=scope,
        state_dir=short_tmp / "state",
        runtime_dir=runtime_dir,
        readiness_timeout=30.0,
        readiness_poll_interval=0.3,
    )
    try:
        services.daemonize_services(cfg)

        # -- The daemon should be ready.
        assert process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)

        # -- The stale socket should have been replaced with the new one.
        assert stale_socket.exists()
    finally:
        _ensure_stopped(cfg)
        import shutil

        shutil.rmtree(short_tmp, ignore_errors=True)


@pytest.mark.integration
def test_concurrent_daemonize_rejects_duplicate_instance(tmp_path: Path) -> None:
    """Starting a second daemon with the same instance scope should fail
    with ``ServicesAlreadyRunningError`` while the first is running."""
    cfg = _daemon_config(tmp_path, scope="integ-dup")
    try:
        services.daemonize_services(cfg)
        assert process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)

        # -- A second daemonize with the same scope should fail.
        dup_cfg = ServiceRunConfig(
            mode=services.ServiceMode.DAEMON,
            services=("hello-world",),
            controllers=(),
            sidecars=(),
            transport="tcp",
            host="127.0.0.1",
            port=_free_tcp_port(),  # Different port, same scope.
            scope="integ-dup",
            state_dir=tmp_path / "state",
            runtime_dir=tmp_path / "runtime",
            readiness_timeout=5.0,
            readiness_poll_interval=0.2,
        )
        with pytest.raises(ServicesAlreadyRunningError):
            services.daemonize_services(dup_cfg)

        # -- Original instance should still be alive.
        assert process.is_instance_alive(cfg.scope, base_dir=cfg.state_root)
    finally:
        _ensure_stopped(cfg)


@pytest.mark.integration
def test_stop_instance_escalates_sigterm_to_sigkill(tmp_path: Path) -> None:
    """If the daemon child ignores SIGTERM, ``stop_instance`` should escalate
    to SIGKILL after the timeout and successfully terminate the process."""
    # Instead of using daemonize_services (which starts a uvicorn server that
    # handles SIGTERM), we manually simulate a daemon process that ignores SIGTERM
    # using the process module primitives directly.
    scope = "integ-sigkill"
    state_dir = tmp_path / "state"

    # Spawn a child process that ignores SIGTERM.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open('/dev/null', 'w'); time.sleep(300)",
        ],
        start_new_session=True,
    )
    try:
        # Write a descriptor so stop_instance can find the process.
        desc = process.InstanceDescriptor(
            pid=child.pid,
            config=HelixAppConfig(scope=scope, host="127.0.0.1", port=0, state_root=state_dir),
            transport="tcp",
            mode="daemon",
            create_time=psutil.Process(child.pid).create_time(),
        )
        process.write_descriptor(desc, base_dir=state_dir)

        # Also create a lock file the process "holds" — but since it's a
        # different process, we simulate by NOT acquiring a real flock (the
        # test exercises PID-based stop, not flock-based liveness).

        # Stop with a very short timeout so it escalates quickly.
        result = process.stop_instance(scope, base_dir=state_dir, timeout=1.0, force=True)
        assert child.pid in result.stopped_pids

        # The child should be dead now.
        child.wait(timeout=5)
        assert child.returncode is not None
    finally:
        try:
            child.kill()
            child.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.integration
def test_daemonize_services_cleans_up_on_child_exception(tmp_path: Path) -> None:
    """When the child process crashes during init (e.g. corrupted request JSON),
    the parent detects the exit, raises, and the lock is not left held."""
    scope = "integ-crash"
    state_dir = tmp_path / "state"
    instance_dir = process.instance_dir(scope, base_dir=state_dir)

    # Write a corrupted request file that will make _service_child crash
    # during JSON deserialization.
    fd, tmp_req = tempfile.mkstemp(dir=str(instance_dir), suffix=".json")
    os.write(fd, b"NOT VALID JSON {{{")
    os.close(fd)

    log_path = process.log_path_for(scope, base_dir=state_dir)
    log_file = open(log_path, "a")  # noqa: SIM115
    child_module = "nemo_helix_ext.local._service_child"
    proc = subprocess.Popen(
        [sys.executable, "-m", child_module, tmp_req],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
    log_file.close()

    # Wait for the child to exit (it should crash quickly on bad JSON).
    proc.wait(timeout=10)
    assert proc.returncode != 0

    # The lock should not be held — the child never acquired it.
    assert not process.is_instance_alive(scope, base_dir=state_dir)

    # The request file should have been cleaned up by _service_child.
    assert not Path(tmp_req).exists()


# ---------------------------------------------------------------------------
# Priority 2: Process Lifecycle & Cleanup
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_stop_instance_sweeps_orphaned_children(tmp_path: Path) -> None:
    """When a daemon parent is stopped, any grandchild processes that survive
    should be swept by ``_sweep_orphans``."""
    scope = "integ-orphans"
    state_dir = tmp_path / "state"

    # Spawn a parent that spawns a long-lived grandchild, then sleeps.
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']); "
            "time.sleep(300)",
        ],
        start_new_session=True,
    )
    try:
        # Give the parent time to spawn the grandchild.
        time.sleep(0.5)
        grandchildren = psutil.Process(parent.pid).children(recursive=True)
        assert len(grandchildren) >= 1, "grandchild was not spawned"

        desc = process.InstanceDescriptor(
            pid=parent.pid,
            config=HelixAppConfig(scope=scope, host="127.0.0.1", port=0, state_root=state_dir),
            transport="tcp",
            mode="daemon",
            create_time=psutil.Process(parent.pid).create_time(),
        )
        process.write_descriptor(desc, base_dir=state_dir)

        result = process.stop_instance(scope, base_dir=state_dir, timeout=10, force=True)
        assert parent.pid in result.stopped_pids
        assert len(result.swept_children) >= 1

        # Both parent and grandchild should be dead.
        parent.wait(timeout=5)
        for gc in grandchildren:
            gc.wait(timeout=5)
    finally:
        try:
            parent.kill()
            parent.wait(timeout=3)
        except Exception:
            pass
        for gc in grandchildren:
            try:
                gc.kill()
                gc.wait(timeout=3)
            except Exception:
                pass


def test_wait_for_lock_release_blocks_until_holder_exits(tmp_path: Path) -> None:
    """``_wait_for_lock_release`` should return only once the flock is actually free."""
    scope = "unit-lock-release"
    state_dir = tmp_path / "state"
    lock_path = state_dir / "instances" / scope / process.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    # A holder that keeps the flock for ~1s, then exits and releases it.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time; "
            "fd = open(sys.argv[1], 'r+'); "
            "fcntl.flock(fd, fcntl.LOCK_EX); "
            "print('held', flush=True); "
            "time.sleep(1.0)",
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert process.is_instance_alive(scope, base_dir=state_dir)

        started = time.monotonic()
        assert process._wait_for_lock_release(scope, base_dir=state_dir, timeout=10.0)
        assert not process.is_instance_alive(scope, base_dir=state_dir)
        # It waited rather than returning eagerly on a still-held lock.
        assert time.monotonic() - started >= 0.5
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_wait_for_lock_release_times_out_while_held(tmp_path: Path) -> None:
    """A lock held for the whole window should report failure, not success."""
    scope = "unit-lock-held"
    state_dir = tmp_path / "state"
    lock_path = state_dir / "instances" / scope / process.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time; "
            "fd = open(sys.argv[1], 'r+'); "
            "fcntl.flock(fd, fcntl.LOCK_EX); "
            "print('held', flush=True); "
            "time.sleep(300)",
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert not process._wait_for_lock_release(scope, base_dir=state_dir, timeout=0.5)
    finally:
        holder.kill()
        holder.wait(timeout=5)


@pytest.mark.integration
def test_stop_instance_releases_lock_held_by_surviving_child(tmp_path: Path) -> None:
    """Regression: ``stop_instance`` must not report success while the scope's
    flock is still held.

    A process exiting and its flock being released are not the same instant — the
    kernel drops the lock while closing fds during teardown, and any process that
    inherited the fd keeps it held until *it* is gone too.  ``stop_instance`` used
    to return as soon as the descriptor PID was dead, so an immediate
    ``is_instance_alive`` probe raced that teardown and intermittently saw True.

    The holder here is deliberately outside the parent's process tree: that makes
    the window wide and fixed instead of scheduler-dependent, and it models the
    case the sweep can't reach (a lock-inheriting process spawned after the child
    snapshot was taken).
    """
    scope = "integ-lock-release"
    state_dir = tmp_path / "state"
    lock_path = state_dir / "instances" / scope / process.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()

    hold_seconds = 2.0
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time; "
            "fd = open(sys.argv[1], 'r+'); "
            "fcntl.flock(fd, fcntl.LOCK_EX); "
            "print('held', flush=True); "
            f"time.sleep({hold_seconds})",
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    # The descriptor PID: exits promptly on SIGTERM, and does not hold the lock.
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert process.is_instance_alive(scope, base_dir=state_dir)

        process.write_descriptor(
            process.InstanceDescriptor(
                pid=parent.pid,
                config=HelixAppConfig(scope=scope, host="127.0.0.1", port=0, state_root=state_dir),
                transport="tcp",
                mode="daemon",
                create_time=psutil.Process(parent.pid).create_time(),
            ),
            base_dir=state_dir,
        )

        started = time.monotonic()
        result = process.stop_instance(scope, base_dir=state_dir, timeout=10, force=True)
        elapsed = time.monotonic() - started
        assert parent.pid in result.stopped_pids

        # The post-condition callers rely on, checked with no grace period.
        assert not process.is_instance_alive(scope, base_dir=state_dir)
        assert process.read_descriptor(scope, base_dir=state_dir) is None
        # It blocked on the lock rather than returning the moment the PID died.
        assert elapsed >= hold_seconds / 2
    finally:
        for proc in (holder, parent):
            try:
                proc.kill()
                proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass


@pytest.mark.integration
def test_stop_instance_foreground_mode_requires_force(tmp_path: Path) -> None:
    """Stopping a foreground-mode instance without ``force=True`` should raise
    ``ForegroundInstanceError``.  With ``force=True`` it should proceed."""
    scope = "integ-foreground"
    state_dir = tmp_path / "state"

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        start_new_session=True,
    )
    try:
        desc = process.InstanceDescriptor(
            pid=child.pid,
            config=HelixAppConfig(scope=scope, host="127.0.0.1", port=0, state_root=state_dir),
            transport="tcp",
            mode="foreground",
            create_time=psutil.Process(child.pid).create_time(),
        )
        process.write_descriptor(desc, base_dir=state_dir)

        # Without force, should raise.
        with pytest.raises(ForegroundInstanceError):
            process.stop_instance(scope, base_dir=state_dir, timeout=5)

        # Process should still be alive after the rejected stop.
        assert child.poll() is None

        # With force, should succeed.
        result = process.stop_instance(scope, base_dir=state_dir, timeout=5, force=True)
        assert child.pid in result.stopped_pids
        child.wait(timeout=5)
    finally:
        try:
            child.kill()
            child.wait(timeout=3)
        except Exception:
            pass


@pytest.mark.integration
def test_is_instance_alive_with_stale_lock(tmp_path: Path) -> None:
    """If the lock file exists but no process holds the flock,
    ``is_instance_alive`` should return False."""
    scope = "integ-stale-lock"
    state_dir = tmp_path / "state"

    # Create the lock file without holding a flock on it.
    inst_dir = process.instance_dir(scope, base_dir=state_dir)
    lock_path = inst_dir / process.LOCK_FILENAME
    lock_path.touch()

    assert not process.is_instance_alive(scope, base_dir=state_dir)


@pytest.mark.integration
def test_is_instance_alive_with_held_lock(tmp_path: Path) -> None:
    """If a process holds the flock, ``is_instance_alive`` should return True."""
    scope = "integ-held-lock"
    state_dir = tmp_path / "state"

    fd = process.acquire_lock(scope, base_dir=state_dir)
    try:
        assert process.is_instance_alive(scope, base_dir=state_dir)
    finally:
        os.close(fd)

    # After releasing the fd (which releases the flock), should be false.
    assert not process.is_instance_alive(scope, base_dir=state_dir)


@pytest.mark.integration
def test_validate_pid_detects_recycled_process(tmp_path: Path) -> None:
    """After a process dies, ``validate_pid`` should return False if the PID is
    reused by a different process (detected via create_time mismatch)."""
    # Spawn and immediately kill a short-lived process to get a PID + create_time.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    pid = child.pid
    create_time = psutil.Process(pid).create_time()

    # The PID is alive and create_time matches.
    assert process.validate_pid(pid, create_time)

    # Kill it.
    child.kill()
    child.wait(timeout=5)

    # Now validate_pid should return False — the process is dead.
    assert not process.validate_pid(pid, create_time)

    # Even with a wildly wrong create_time, should be False for a dead PID.
    assert not process.validate_pid(pid, 0.0)


@pytest.mark.integration
def test_rotate_log_preserves_existing_content(tmp_path: Path) -> None:
    """``rotate_log`` should rename the existing log and return the path for
    the new (empty) log.  The old content must be preserved."""
    scope = "integ-rotate"
    state_dir = tmp_path / "state"

    # Write initial log content.
    log_path = process.log_path_for(scope, base_dir=state_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("original log content\n")

    # Rotate.
    new_log = process.rotate_log(scope, base_dir=state_dir)
    assert new_log == log_path
    assert not log_path.exists()  # Original was renamed.

    # Find the rotated file.
    rotated_files = [f for f in log_path.parent.iterdir() if f.name.startswith("services.log.")]
    assert len(rotated_files) == 1
    assert rotated_files[0].read_text() == "original log content\n"

    # Write new content, rotate again.
    log_path.write_text("second run\n")
    process.rotate_log(scope, base_dir=state_dir)

    rotated_files = sorted(f for f in log_path.parent.iterdir() if f.name.startswith("services.log."))
    assert len(rotated_files) == 2
    contents = {f.read_text() for f in rotated_files}
    assert "original log content\n" in contents
    assert "second run\n" in contents
