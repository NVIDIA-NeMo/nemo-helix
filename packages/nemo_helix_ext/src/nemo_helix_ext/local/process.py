# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local process lifecycle for ``nemo services``.

In this module, "instance" is a local services process/resource, and "scope"
is the stable key used for that instance's lock, descriptor, socket, and log
paths.  The CLI exposes this key as ``--instance`` for compatibility, but
internal code should use "scope" when referring to the key.

Uses per-scope directories under ``$XDG_STATE_HOME/nhx/instances/``
with flock-based liveness tracking. Each scope directory contains:

- ``services.lock`` -- exclusive flock held for the process lifetime
- ``instance.json``  -- descriptor with PID, port, services, etc.
- ``services.log``   -- stdout/stderr log (background mode)

The flock is the **source of truth** for whether an instance is alive.
The descriptor is metadata used by ``status``, ``ls``, and ``restart``.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

import psutil
from nhx.platform_runner.config import (
    DEFAULT_LOCAL_SERVICES_BIND_HOST,
    HelixAppConfig,
    default_state_root,
    validate_scope,
)
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

LOCK_FILENAME = "services.lock"
DESCRIPTOR_FILENAME = "instance.json"
LOG_FILENAME = "services.log"

SUGGESTED_ALT_PORT = 9090

_SIGTERM_POLL_INTERVAL = 0.25
_SIGKILL_WAIT_TIMEOUT = 5.0
_LOCK_RELEASE_WAIT_TIMEOUT = 5.0
_REAP_PID_TIMEOUT = 2.0
_DEFAULT_STOP_TIMEOUT = 30.0
# How long ``stop_instance`` waits for the flock to be released after the last process it
# signaled has exited.  The kernel drops the lock as part of closing the fd during process
# teardown, so this only has to cover the tail of that teardown, not a whole shutdown.
_LOCK_RELEASE_TIMEOUT = 5.0


def _pause(seconds: float) -> None:
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# State directory layout
# ---------------------------------------------------------------------------


def _base_state_dir() -> Path:
    return default_state_root()


def _instances_dir(*, base_dir: Path | None = None) -> Path:
    return (base_dir or _base_state_dir()) / "instances"


def _find_git_root() -> str:
    """Walk up from cwd looking for a ``.git`` directory. Falls back to cwd."""
    cur = Path.cwd().resolve()
    for parent in (cur, *cur.parents):
        if (parent / ".git").exists():
            return str(parent)
    return str(cur)


_scope_prefix_cache: str | None = None


def compute_scope(*, port: int, explicit_scope: str | None = None) -> str:
    """Compute the local services scope.

    The default scope is ``sha1(git_toplevel_or_cwd)[:8]-<port>``.  Including
    the port is intentional: it lets two local services instances from the same
    checkout use different TCP ports without sharing a lock, descriptor, or log
    directory.

    Explicit scopes are validated and returned as-is, so they do not encode the
    port.  Callers that pass an explicit scope own its uniqueness.
    """
    if explicit_scope:
        return validate_scope(explicit_scope)
    global _scope_prefix_cache  # noqa: PLW0603
    if _scope_prefix_cache is None:
        root = _find_git_root()
        _scope_prefix_cache = hashlib.sha1(root.encode()).hexdigest()[:8]  # noqa: S324
    return f"{_scope_prefix_cache}-{port}"


def instance_dir(scope: str, *, base_dir: Path | None = None) -> Path:
    """Return the state directory for *scope*, creating it if needed."""
    d = _instances_dir(base_dir=base_dir) / validate_scope(scope)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# flock-based liveness
# ---------------------------------------------------------------------------


def acquire_lock(scope: str, *, base_dir: Path | None = None) -> int:
    """Acquire an exclusive flock for *scope*.  Returns the open fd.

    Raises ``InstanceAlreadyRunningError`` if the lock is already held.
    The caller must keep the fd open for the process lifetime.
    """
    d = instance_dir(scope, base_dir=base_dir)
    lock_path = d / LOCK_FILENAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as err:
        os.close(fd)
        if err.errno in {errno.EACCES, errno.EAGAIN}:
            raise InstanceAlreadyRunningError(scope) from err
        raise
    return fd


def is_instance_alive(scope: str, *, base_dir: Path | None = None) -> bool:
    """Check if an instance is alive by probing its flock."""
    d = _instances_dir(base_dir=base_dir) / scope
    lock_path = d / LOCK_FILENAME
    if not lock_path.exists():
        return False
    fd = -1
    try:
        fd = os.open(str(lock_path), os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except FileNotFoundError:
        return False
    except OSError:
        return True
    finally:
        if fd >= 0:
            os.close(fd)


class InstanceAlreadyRunningError(Exception):
    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(f"Instance '{scope}' is already running")


class ForegroundInstanceError(Exception):
    """Raised when ``stop`` targets a foreground instance without ``--force``."""

    def __init__(self, scope: str, pid: int) -> None:
        self.scope = scope
        self.pid = pid
        super().__init__(
            f"Instance '{scope}' (pid {pid}) is running in the foreground. "
            "Use Ctrl-C in its terminal to stop it, or pass --force."
        )


class InstanceStillRunningError(Exception):
    """Raised when ``rm`` targets a live instance."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        super().__init__(
            f"Instance '{scope}' is still running. Stop it first with: nemo services stop --instance {scope}"
        )


# ---------------------------------------------------------------------------
# Port availability (preflight before bind)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortConflict:
    """Structured port conflict for terminal rendering by CLI callers."""

    kind: Literal["foreign", "nemo_instance"]
    port: int
    scope: str | None = None


def _normalize_bind_host(host: str) -> str:
    """Normalize bind hosts for descriptor comparison."""
    if host == "localhost":
        return "127.0.0.1"
    return host


def _instance_owns_listener(
    scope: str,
    host: str,
    port: int,
    *,
    base_dir: Path | None = None,
) -> bool:
    """Return True when a live instance for *scope* is bound to *host*:*port*."""
    if not is_instance_alive(scope, base_dir=base_dir):
        return False
    desc = read_descriptor(scope, base_dir=base_dir)
    if desc is None:
        return False
    return desc.config.port == port and _normalize_bind_host(desc.config.host) == _normalize_bind_host(host)


def is_port_bindable(host: str, port: int) -> bool:
    """Return True if *host*:*port* can be bound on at least one address family.

    Uses ``getaddrinfo`` so IPv4 and IPv6 hosts (for example ``::``) are probed
    with the correct socket family instead of always using ``AF_INET``.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except OSError:
        return False
    if not infos:
        return False
    for family, socktype, proto, _, sockaddr in infos:
        with contextlib.suppress(OSError):
            with socket.socket(family, socktype, proto) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)  # noqa: S104  # nosec B104
            return True
    return False


def check_port_available_for_start(
    host: str,
    port: int,
    scope: str,
    *,
    base_dir: Path | None = None,
) -> PortConflict | None:
    """Return conflict info when *port* cannot be bound, else None.

    Classifies conflicts as ``nemo_instance`` only when a live instance for
    *scope* is recorded on the same host and port. Otherwise reports ``foreign``.
    Does not log or print — callers render to the terminal.
    """
    if is_port_bindable(host, port):
        return None
    if _instance_owns_listener(scope, host, port, base_dir=base_dir):
        return PortConflict(kind="nemo_instance", port=port, scope=scope)
    return PortConflict(kind="foreign", port=port)


def format_port_conflict(err: PortConflict) -> list[str]:
    """Return actionable message lines for terminal display.

    Message text depends on ``err.kind`` (foreign process vs NeMo instance).
    """
    if err.kind == "nemo_instance":
        owner = f" '{err.scope}'" if err.scope else ""
        return [
            f"Port {err.port} is in use by NeMo Helix instance{owner}.",
            "Stop it first with: nemo services stop",
            "Or restart with:    nemo services restart",
        ]
    return [
        f"Port {err.port} is already in use by another process (EADDRINUSE).",
        "Free the port or choose a different one:",
        f"lsof -i :{err.port}          (see what's listening)",
        f"nemo services run --port {SUGGESTED_ALT_PORT}",
    ]


# ---------------------------------------------------------------------------
# Descriptor (instance.json)
# ---------------------------------------------------------------------------


class InstanceDescriptor(BaseModel):
    pid: int
    config: HelixAppConfig = Field(default_factory=HelixAppConfig)
    transport: Literal["tcp", "uds"] = "tcp"
    mode: Literal["foreground", "background", "daemon"] = "background"
    create_time: float = 0.0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def _validate_client_transport(self) -> Self:
        if self.transport == "uds" and self.config.socket_path is None:
            raise ValueError("UDS client transport requires config.socket_path")
        return self

    @classmethod
    def from_config(
        cls,
        config: HelixAppConfig,
        *,
        mode: Literal["foreground", "background", "daemon"],
        transport: Literal["uds", "tcp"] = "tcp",
        pid: int | None = None,
    ) -> Self:
        resolved_pid = os.getpid() if pid is None else pid
        return cls(
            pid=resolved_pid,
            config=config,
            transport=transport,
            mode=mode,
            create_time=get_create_time(resolved_pid),
        )


def write_descriptor(desc: InstanceDescriptor, *, base_dir: Path | None = None) -> Path:
    d = instance_dir(desc.config.scope, base_dir=base_dir)
    path = d / DESCRIPTOR_FILENAME
    payload = desc.model_dump()
    fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".tmp")
    try:
        os.write(fd, (json.dumps(payload, indent=2) + "\n").encode())
        os.close(fd)
        fd = -1
        os.replace(tmp, str(path))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_descriptor(scope: str, *, base_dir: Path | None = None) -> InstanceDescriptor | None:
    d = _instances_dir(base_dir=base_dir) / scope
    path = d / DESCRIPTOR_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        desc = InstanceDescriptor.model_validate(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.debug("Corrupt descriptor at %s, ignoring", path, exc_info=True)
        return None
    if desc.config.scope != scope:
        logger.debug(
            "Descriptor at %s has scope=%r but lives under %r, ignoring",
            path,
            desc.config.scope,
            scope,
        )
        return None
    return desc


def remove_descriptor(scope: str, *, base_dir: Path | None = None) -> None:
    d = _instances_dir(base_dir=base_dir) / scope
    path = d / DESCRIPTOR_FILENAME
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not remove descriptor %s", path, exc_info=True)


def _scope_dir(scope: str, *, base_dir: Path | None = None) -> Path:
    return _instances_dir(base_dir=base_dir) / validate_scope(scope)


def _is_log_file(path: Path) -> bool:
    return path.name == LOG_FILENAME or path.name.startswith(f"{LOG_FILENAME}.")


def _iter_log_files(scope_dir_path: Path):
    if not scope_dir_path.is_dir():
        return
    for path in scope_dir_path.iterdir():
        if path.is_file() and _is_log_file(path):
            yield path


def _has_preservable_logs(scope_dir_path: Path) -> bool:
    """Return True if *scope_dir_path* contains non-empty service log files."""
    return any(path.stat().st_size > 0 for path in _iter_log_files(scope_dir_path))


def is_removable_ghost(
    scope: str,
    *,
    base_dir: Path | None = None,
    descriptor: InstanceDescriptor | None = None,
) -> bool:
    """True when a dead scope directory has no descriptor and no non-empty logs."""
    if is_instance_alive(scope, base_dir=base_dir):
        return False
    if descriptor is not None:
        return False
    scope_dir_path = _scope_dir(scope, base_dir=base_dir)
    if not scope_dir_path.is_dir():
        return False
    if (scope_dir_path / DESCRIPTOR_FILENAME).exists():
        return False
    return not _has_preservable_logs(scope_dir_path)


# ---------------------------------------------------------------------------
# PID validation via psutil
# ---------------------------------------------------------------------------


def validate_pid(pid: int, expected_create_time: float, *, tolerance: float = 2.0) -> bool:
    """Check that *pid* is alive and its create_time matches the recorded value.

    The *tolerance* accounts for float precision across platforms.
    """
    try:
        proc = psutil.Process(pid)
        return abs(proc.create_time() - expected_create_time) < tolerance
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def get_create_time(pid: int) -> float:
    """Return the create_time for *pid*.  Raises if the process doesn't exist."""
    return psutil.Process(pid).create_time()


# ---------------------------------------------------------------------------
# Instance listing
# ---------------------------------------------------------------------------


@dataclass
class InstanceInfo:
    scope: str
    alive: bool
    descriptor: InstanceDescriptor | None


def list_instances(*, base_dir: Path | None = None) -> list[InstanceInfo]:
    """Scan all scope directories and return their status.

    Side effects:
    - Removes stale descriptors for dead instances.
    - Silently removes empty ghost directories (dead, no descriptor, no logs).
    """
    idir = _instances_dir(base_dir=base_dir)
    if not idir.exists():
        return []
    results: list[InstanceInfo] = []
    for child in sorted(idir.iterdir()):
        if not child.is_dir():
            continue
        scope = child.name
        alive = is_instance_alive(scope, base_dir=base_dir)
        desc = read_descriptor(scope, base_dir=base_dir)
        if not alive and desc is not None:
            remove_descriptor(scope, base_dir=base_dir)
            desc = None
        if is_removable_ghost(scope, base_dir=base_dir, descriptor=desc):
            try:
                shutil.rmtree(child)
            except OSError:
                logger.debug("Could not remove ghost scope directory %s", child, exc_info=True)
            else:
                continue
        results.append(InstanceInfo(scope=scope, alive=alive, descriptor=desc))
    return results


def remove_instance(scope: str, *, base_dir: Path | None = None) -> bool:
    """Remove a scope directory.

    Returns False if the scope directory did not exist or could not be removed.
    """
    scope = validate_scope(scope)
    if is_instance_alive(scope, base_dir=base_dir):
        raise InstanceStillRunningError(scope)
    scope_dir_path = _scope_dir(scope, base_dir=base_dir)
    if not scope_dir_path.is_dir():
        return False
    with contextlib.suppress(OSError):
        shutil.rmtree(scope_dir_path)
    return not scope_dir_path.is_dir()


def list_stopped_scopes(*, base_dir: Path | None = None) -> list[str]:
    """Return scopes for instances that are not alive."""
    return [info.scope for info in list_instances(base_dir=base_dir) if not info.alive]


def prune_instances(*, base_dir: Path | None = None) -> list[str]:
    """Remove all stopped scope directories.  Returns removed scopes."""
    removed: list[str] = []
    for scope in list_stopped_scopes(base_dir=base_dir):
        if remove_instance(scope, base_dir=base_dir):
            removed.append(scope)
    return removed


def instance_log_bytes(scope: str, *, base_dir: Path | None = None) -> int:
    """Total bytes across ``services.log`` and rotated logs for *scope*."""
    return sum(path.stat().st_size for path in _iter_log_files(_scope_dir(scope, base_dir=base_dir)))


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------


def rotate_log(scope: str, *, base_dir: Path | None = None) -> Path:
    """Rotate the existing log and return the path for the new one."""
    return rotate_log_path(log_path_for(scope, base_dir=base_dir))


def rotate_log_path(log_path: Path) -> Path:
    """Rotate the existing log at *log_path* and return the path for the new one."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        rotated = log_path.with_name(f"{log_path.name}.{ts}")
        log_path.rename(rotated)
    return log_path


def log_path_for(scope: str, *, base_dir: Path | None = None) -> Path:
    return instance_dir(scope, base_dir=base_dir) / LOG_FILENAME


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


@dataclass
class StopResult:
    stopped_pids: list[int]
    swept_children: list[int] = field(default_factory=list)


def _snapshot_children(pid: int) -> list[psutil.Process]:
    """Return all descendant processes of *pid*.

    Must be called while the parent is still alive; once it exits,
    children are reparented to init and won't appear in the tree.
    """
    try:
        return psutil.Process(pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def _sweep_orphans(children: list[psutil.Process], timeout: float = 5.0) -> list[int]:
    """Terminate any still-alive processes from a prior snapshot.

    Sends SIGTERM, waits up to *timeout*, then SIGKILL survivors and waits again.
    Returns PIDs that were signaled.  Handles ``NoSuchProcess`` gracefully
    since children may have already exited during graceful shutdown.

    The second wait matters: these children inherited the instance flock fd from the
    parent, and the kernel only releases the lock once the last holder's fd is closed.
    Returning while a SIGKILLed child is still being torn down leaves the lock held, so
    an ``is_instance_alive`` probe immediately after a "successful" stop reports True.
    """
    alive_children = [c for c in children if c.is_running()]
    if not alive_children:
        return []

    killed: list[int] = []
    for child in alive_children:
        try:
            child.terminate()
            killed.append(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Already exited or not owned by us — skip.

    _, still_alive = psutil.wait_procs(alive_children, timeout=timeout)
    kill_sent: list[psutil.Process] = []
    for child in still_alive:
        try:
            child.kill()
            kill_sent.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # Raced with exit or not owned — nothing to do.

    if kill_sent:
        _, unreaped = psutil.wait_procs(kill_sent, timeout=timeout)
        if unreaped:
            logger.warning("Orphaned child %s still alive after SIGKILL", [c.pid for c in unreaped])

    if killed:
        logger.info(
            "Swept %d orphaned child %s: %s", len(killed), "process" if len(killed) == 1 else "processes", killed
        )

    return killed


def stop_instance(
    scope: str,
    *,
    base_dir: Path | None = None,
    timeout: float = _DEFAULT_STOP_TIMEOUT,
    force: bool = False,
) -> StopResult:
    """Stop a running instance by scope.

    Uses the flock and descriptor to find the process.  If the flock is held,
    the instance is definitely alive.  If the flock is not held but a
    descriptor exists with a validated PID, we still attempt to stop (handles
    edge cases where the process outlives the flock probe window).

    Foreground instances (``mode="foreground"``) are protected: they must be
    stopped via Ctrl-C in their own terminal.  Pass *force=True* to override.

    Sends SIGTERM, waits up to *timeout* seconds, then escalates to SIGKILL.
    After the parent exits, any surviving child processes (e.g. agent
    deployments) are swept up via SIGTERM/SIGKILL.
    """
    desc = read_descriptor(scope, base_dir=base_dir)
    alive = is_instance_alive(scope, base_dir=base_dir)

    if not alive and desc is None:
        return StopResult(stopped_pids=[])

    if desc is None:
        return StopResult(stopped_pids=[])

    if desc.mode == "foreground" and not force:
        raise ForegroundInstanceError(scope, desc.pid)

    pid = desc.pid
    if not validate_pid(pid, desc.create_time):
        logger.debug("PID %d doesn't match recorded create_time, cleaning up descriptor", pid)
        remove_descriptor(scope, base_dir=base_dir)
        return StopResult(stopped_pids=[])

    # Snapshot child tree while parent is alive — children are reparented to
    # init once the parent exits, making them invisible to psutil afterwards.
    children = _snapshot_children(pid)

    try:
        os.kill(pid, signal.SIGTERM)
        logger.debug("Sent SIGTERM to pid %d", pid)
    except OSError:
        logger.debug("Failed to send SIGTERM to pid %d", pid, exc_info=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        _pause(_SIGTERM_POLL_INTERVAL)
    else:
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                logger.debug("Sent SIGKILL to pid %d", pid)
            except PermissionError:
                logger.warning("No permission to SIGKILL pid %d; process may still be running", pid)
                swept = _sweep_orphans(children) if children else []
                # Keep the descriptor so subsequent stop/restart can retry.
                return StopResult(stopped_pids=[], swept_children=swept)
            except OSError:
                logger.debug("Failed to send SIGKILL to pid %d", pid, exc_info=True)
            if not _wait_for_pid_exit(pid, timeout=_SIGKILL_WAIT_TIMEOUT):
                logger.warning("PID %d is still alive after SIGKILL; preserving descriptor", pid)
                swept = _sweep_orphans(children) if children else []
                return StopResult(stopped_pids=[], swept_children=swept)

    swept = _sweep_orphans(children) if children else []

    # A process exiting and its flock being released are not the same instant: the kernel
    # drops the lock while closing fds during teardown.  Callers treat a successful stop as
    # "the scope is free now" (and immediately probe with ``is_instance_alive``), so hold the
    # post-condition here rather than making every caller poll for it.
    if not _wait_for_lock_release(scope, base_dir=base_dir, timeout=_LOCK_RELEASE_TIMEOUT):
        logger.warning(
            "Instance '%s' flock still held %.0fs after pid %d exited; a child may have outlived the sweep",
            scope,
            _LOCK_RELEASE_TIMEOUT,
            pid,
        )
    _reap_stopped_pid(pid, timeout=min(_REAP_PID_TIMEOUT, timeout))
    lock_released = _wait_until_instance_lock_released(
        scope,
        base_dir=base_dir,
        timeout=min(_LOCK_RELEASE_WAIT_TIMEOUT, timeout),
    )
    if not lock_released:
        # A descendant may still hold the inherited flock, or a replacement
        # instance may have acquired it. Preserve its descriptor.
        return StopResult(stopped_pids=[pid], swept_children=swept)

    current_desc = read_descriptor(scope, base_dir=base_dir)
    if current_desc is not None and (current_desc.pid != desc.pid or current_desc.create_time != desc.create_time):
        logger.warning(
            "Instance %r descriptor changed during stop; preserving replacement descriptor",
            scope,
        )
    else:
        remove_descriptor(scope, base_dir=base_dir)
    return StopResult(stopped_pids=[pid], swept_children=swept)


def _wait_for_lock_release(scope: str, *, base_dir: Path | None, timeout: float) -> bool:
    """Poll until the instance flock is free, or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_instance_alive(scope, base_dir=base_dir):
            return True
        _pause(_SIGTERM_POLL_INTERVAL)
    return not is_instance_alive(scope, base_dir=base_dir)


def _wait_for_pid_exit(pid: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        _pause(_SIGTERM_POLL_INTERVAL)
    return not _pid_alive(pid)


def _reap_stopped_pid(pid: int, *, timeout: float) -> None:
    """Reap *pid* when this process is its parent so the kernel drops zombie state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            wpid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if wpid == pid:
            return
        if wpid == 0:
            _pause(min(_SIGTERM_POLL_INTERVAL, max(0.0, deadline - time.monotonic())))
            continue
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def _wait_until_instance_lock_released(
    scope: str,
    *,
    base_dir: Path | None,
    timeout: float,
) -> bool:
    """Wait until the scope flock is free (graceful shutdown may lag process exit)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_instance_alive(scope, base_dir=base_dir):
            return True
        _pause(_SIGTERM_POLL_INTERVAL)
    if not is_instance_alive(scope, base_dir=base_dir):
        return True
    logger.warning(
        "Instance %r lock still held after stop (waited %.1fs); preserving descriptor",
        scope,
        timeout,
    )
    return False


def _pid_alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Background start
# ---------------------------------------------------------------------------


def start_background(
    config: HelixAppConfig | None = None,
    *,
    data_dir: str | None = None,
) -> subprocess.Popen:
    """Launch ``nemo services run`` as a detached background subprocess.

    The child acquires the flock and writes its own descriptor.  The parent
    returns the ``Popen`` handle for health polling.
    """
    config = config or HelixAppConfig(host=DEFAULT_LOCAL_SERVICES_BIND_HOST)
    log_file_path = rotate_log_path(config.log_file_path())
    log_file = open(log_file_path, "a")  # noqa: SIM115

    nemo_bin = str(Path(sys.executable).parent / "nemo")
    args: list[str] = [nemo_bin, "services", "run"]
    if config.services:
        args += ["--services", ",".join(config.services)]
    if config.service_group:
        args += ["--service-group", config.service_group]
    if config.controllers:
        args += ["--controllers", ",".join(config.controllers)]
    if config.controller_group:
        args += ["--controller-group", config.controller_group]
    if config.sidecars:
        args += ["--sidecars", ",".join(config.sidecars)]
    if config.config_path:
        args += ["--config", config.config_path]
    args += ["--host", config.host, "--port", str(config.port)]
    args += ["--keep-alive-timeout-seconds", str(config.keep_alive_timeout_seconds)]
    args += ["--instance", config.scope]

    env = os.environ.copy()
    if data_dir and "NHX_DATA_DIR" not in env:
        env["NHX_DATA_DIR"] = data_dir
    if config.state_root is not None:
        env["_NHX_STATE_DIR"] = str(config.state_root)
    # Tell the child ``run`` process it was launched by ``start`` so it
    # records mode="background" in its descriptor.  This is internal
    # parent-to-child signaling -- not a public API surface -- following the
    # same convention as _NHX_STATE_DIR above.
    env["_NHX_LAUNCH_MODE"] = "background"

    try:
        proc = subprocess.Popen(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    except Exception:
        log_file.close()
        raise
    log_file.close()
    logger.debug("Started services (pid=%d), log=%s", proc.pid, log_file_path)
    return proc
