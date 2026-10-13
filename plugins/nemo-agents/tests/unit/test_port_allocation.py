# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InMemoryRunnerBackend port allocation logic."""

from __future__ import annotations

import errno
import socket
from unittest.mock import patch

import pytest
from nemo_agents_plugin.config import ControllerConfig
from nemo_agents_plugin.runner.in_memory import InMemoryRunnerBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend(start: int = 49152, end: int = 49161) -> InMemoryRunnerBackend:
    """Return a backend configured with a small, predictable port range."""
    cfg = ControllerConfig(port_range_start=start, port_range_end=end)
    return InMemoryRunnerBackend(cfg)


# ---------------------------------------------------------------------------
# ControllerConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("start,end", [(49152, 65535), (50000, 50000)])
def test_config_valid_range(start: int, end: int) -> None:
    cfg = ControllerConfig(port_range_start=start, port_range_end=end)
    assert (cfg.port_range_start, cfg.port_range_end) == (start, end)


def test_config_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="port_range_end"):
        ControllerConfig(port_range_start=9100, port_range_end=9001)


def test_config_defaults_are_dynamic_range() -> None:
    cfg = ControllerConfig()
    assert cfg.port_range_start == 49152
    assert cfg.port_range_end == 65535


# ---------------------------------------------------------------------------
# _is_port_free
# ---------------------------------------------------------------------------


def test_is_port_free_tracks_socket_lifetime() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert not InMemoryRunnerBackend._is_port_free(port)
    assert InMemoryRunnerBackend._is_port_free(port)


# ---------------------------------------------------------------------------
# allocate_port — scanning and wrap-around behaviour (mocked _is_port_free)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "end,availability,expected",
    [
        (49161, [True, True], [49152, 49153]),
        (49161, [False, False, True], [49154]),
        (49153, [True, True, True], [49152, 49153, 49152]),
        (49153, [False, True, True], [49153, 49152]),
    ],
    ids=["advances", "skips-occupied", "wraps", "reuses-freed-port"],
)
def test_allocate_port_scans_range(end: int, availability: list[bool], expected: list[int]) -> None:
    backend = _backend(start=49152, end=end)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", side_effect=availability):
        assert [backend.allocate_port() for _ in expected] == expected


def test_allocate_port_raises_when_range_exhausted() -> None:
    backend = _backend(start=49152, end=49153)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=False):
        with pytest.raises(RuntimeError, match=r"No free port available in range \[49152, 49153\]"):
            backend.allocate_port()


@pytest.mark.parametrize("serve_connection", [False, True], ids=["unused", "served-connection"])
def test_reservation_blocks_competing_backend_until_released(serve_connection: bool) -> None:
    backend = _backend(start=49152, end=65535)
    with backend.reserve_socket() as reserved:
        host, port = reserved.getsockname()
        assert host == "127.0.0.1"
        competitor = _backend(start=port, end=port)
        with pytest.raises(RuntimeError, match="No free port available"):
            with competitor.reserve_socket():
                pytest.fail("Reserved port was allocated twice")
        if serve_connection:
            reserved.listen()
            with socket.create_connection((host, port), timeout=2) as client:
                connection, _ = reserved.accept()
                # Server closes first so the accepted connection enters TIME_WAIT.
                connection.close()
                assert client.recv(1) == b""

    assert reserved.fileno() == -1
    with competitor.reserve_socket() as reused:
        assert reused.getsockname() == (host, port)
        assert competitor._next_port == port


def test_reservation_skips_occupied_port_when_revisited() -> None:
    backend = _backend(start=49152, end=65535)
    with backend.reserve_socket() as first:
        first_port = first.getsockname()[1]
        backend._next_port = first_port
        with backend.reserve_socket() as second:
            second_port = second.getsockname()[1]
            assert second_port != first_port
            assert backend._next_port == (second_port + 1 if second_port < 65535 else 49152)


def test_reservation_closes_on_caller_error() -> None:
    backend = _backend(start=49152, end=65535)
    with pytest.raises(ValueError, match="staging failed"):
        with backend.reserve_socket() as reserved:
            raise ValueError("staging failed")
    assert reserved.fileno() == -1


def test_reservation_propagates_permission_error_and_closes_socket() -> None:
    with patch("nemo_agents_plugin.runner.in_memory.socket.socket") as socket_cls:
        sock = socket_cls.return_value.__enter__.return_value
        sock.bind.side_effect = PermissionError(errno.EACCES, "Permission denied")
        with pytest.raises(PermissionError):
            with _backend().reserve_socket():
                pytest.fail("Reservation succeeded without bind permission")
        socket_cls.return_value.__exit__.assert_called_once()
