# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for TCP/UDS port and socket management.

These tests exercise real port binding, socket creation, and conflict
detection using actual OS resources.
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import pytest
from nemo_helix_ext.local import process, services
from nemo_helix_ext.local.services import (
    ServiceRunConfig,
    ServicesPortInUseError,
)
from nhx.platform_runner.config import HelixAppConfig


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tcp_port_conflict_with_foreign_process(tmp_path: Path) -> None:
    """When a foreign (non-NeMo) process holds a port, ``_check_tcp_available``
    should raise ``ServicesPortInUseError`` with a helpful suggestion."""
    # Bind a TCP port and hold it open.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        cfg = ServiceRunConfig(
            mode=services.ServiceMode.DAEMON,
            services=("hello-world",),
            transport="tcp",
            host="127.0.0.1",
            port=port,
            scope="integ-port-foreign",
            state_dir=tmp_path / "state",
            runtime_dir=tmp_path / "runtime",
        )

        with pytest.raises(ServicesPortInUseError, match="already in use by another process"):
            services._check_tcp_available(cfg)


@pytest.mark.integration
def test_tcp_port_conflict_with_nemo_instance(tmp_path: Path) -> None:
    """When a NeMo instance holds a port, the error should distinguish it
    from a foreign process."""
    scope = "integ-port-nemo"
    state_dir = tmp_path / "state"

    # Bind a port and also create a descriptor matching the scope/host/port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        # Create a live lock and descriptor so it looks like a NeMo instance.
        lock_fd = process.acquire_lock(scope, base_dir=state_dir)
        try:
            desc = process.InstanceDescriptor(
                pid=1,  # Dummy PID — the flock is what matters.
                config=HelixAppConfig(scope=scope, host="127.0.0.1", port=port),
                transport="tcp",
                mode="daemon",
                create_time=0.0,
            )
            process.write_descriptor(desc, base_dir=state_dir)

            conflict = process.check_port_available_for_start("127.0.0.1", port, scope, base_dir=state_dir)
            assert conflict is not None
            assert conflict.kind == "nemo_instance"
            assert conflict.port == port

            lines = process.format_port_conflict(conflict)
            assert any("NeMo Helix" in line for line in lines)
        finally:
            import os

            os.close(lock_fd)


@pytest.mark.integration
def test_tcp_port_available_when_free(tmp_path: Path) -> None:
    """When a port is free, ``check_port_available_for_start`` returns None."""
    port = _free_tcp_port()
    conflict = process.check_port_available_for_start("127.0.0.1", port, "integ-free", base_dir=tmp_path / "state")
    assert conflict is None


@pytest.mark.integration
def test_uds_socket_path_max_validation() -> None:
    """A socket path exceeding AF_UNIX_PATH_MAX should raise ValueError."""
    # Build a path that is exactly one byte over the limit.
    max_bytes = services._AF_UNIX_PATH_MAX_BYTES
    # Create a path that exceeds the limit.
    long_path = "/" + "x" * max_bytes  # len("/") + max_bytes > max_bytes
    assert len(long_path.encode()) > max_bytes

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        transport="uds",
        socket_path=long_path,
        scope="integ-long-sock",
        state_dir="/tmp/state",
        runtime_dir="/tmp/run",
    )
    # _validated_socket_path calls _validate_socket_path_length internally.
    with pytest.raises(ValueError, match="too long for AF_UNIX"):
        services._validated_socket_path(cfg)


@pytest.mark.integration
def test_prepare_socket_removes_stale_socket(tmp_path: Path) -> None:
    """``_prepare_socket`` should remove a stale (unreachable) socket file
    and allow a new daemon to bind."""
    scope = "stale2"
    short_tmp = Path(tempfile.mkdtemp(prefix="nemo-"))
    runtime_dir = short_tmp / "run"

    # Create a stale socket file.
    socket_dir = runtime_dir / scope
    socket_dir.mkdir(parents=True, exist_ok=True)
    stale_socket = socket_dir / "nemo-helix.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.bind(str(stale_socket))
    assert stale_socket.exists()

    cfg = ServiceRunConfig(
        mode=services.ServiceMode.DAEMON,
        services=("hello-world",),
        transport="uds",
        scope=scope,
        state_dir=short_tmp / "state",
        runtime_dir=runtime_dir,
    )

    # _prepare_socket should probe, find it stale, remove it, and return the path.
    result = services._prepare_socket(cfg)
    assert result is not None
    # The stale socket should have been removed (the new server hasn't bound yet).
    assert not stale_socket.exists()

    import shutil

    shutil.rmtree(short_tmp, ignore_errors=True)
