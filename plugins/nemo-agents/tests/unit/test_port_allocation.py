# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InMemoryRunnerBackend port allocation logic."""

from __future__ import annotations

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


def test_config_valid_range() -> None:
    cfg = ControllerConfig(port_range_start=49152, port_range_end=65535)
    assert cfg.port_range_start == 49152
    assert cfg.port_range_end == 65535


def test_config_single_port_range() -> None:
    cfg = ControllerConfig(port_range_start=50000, port_range_end=50000)
    assert cfg.port_range_start == cfg.port_range_end


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


def test_is_port_free_returns_true_for_available_port() -> None:
    # Bind to port 0 to let the OS assign a free port, then release and verify
    # our helper agrees it was free.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    # Port is now released — should be free.
    assert InMemoryRunnerBackend._is_port_free(free_port)


def test_is_port_free_returns_false_for_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen()
        occupied_port = s.getsockname()[1]
        # While s is listening, as a deployed server would be, the port should not be free.
        assert not InMemoryRunnerBackend._is_port_free(occupied_port)


# ---------------------------------------------------------------------------
# allocate_port — scanning and wrap-around behaviour (mocked _is_port_free)
# ---------------------------------------------------------------------------


def test_allocate_port_returns_first_free_port() -> None:
    backend = _backend(start=49152, end=49161)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=True):
        port = backend.allocate_port()
    assert port == 49152


def test_allocate_port_skips_occupied_ports() -> None:
    backend = _backend(start=49152, end=49161)
    # First two ports occupied, third is free.
    free_sequence = [False, False, True]
    with patch.object(InMemoryRunnerBackend, "_is_port_free", side_effect=free_sequence):
        port = backend.allocate_port()
    assert port == 49154


def test_allocate_port_advances_next_pointer() -> None:
    backend = _backend(start=49152, end=49161)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=True):
        p1 = backend.allocate_port()
        p2 = backend.allocate_port()
    assert p1 == 49152
    assert p2 == 49153


def test_allocate_port_wraps_around_at_range_end() -> None:
    backend = _backend(start=49152, end=49153)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=True):
        backend.allocate_port()  # 49152 → _next_port = 49153
        backend.allocate_port()  # 49153 → _next_port wraps to 49152
        port = backend.allocate_port()  # should be 49152 again
    assert port == 49152


def test_allocate_port_reuses_freed_port_via_wrap() -> None:
    # Simulate: 49152 is in use (not yet freed), 49153 is free.
    # After wrap, 49152 becomes free — should be returned next.
    backend = _backend(start=49152, end=49153)
    call_count = 0

    def free_except_first(port: int, host: str) -> bool:
        nonlocal call_count
        call_count += 1
        # First call probes 49152 (occupied), second probes 49153 (free).
        return port != 49152

    with patch.object(InMemoryRunnerBackend, "_is_port_free", side_effect=free_except_first):
        port = backend.allocate_port()
    assert port == 49153


def test_allocate_port_raises_when_range_exhausted() -> None:
    backend = _backend(start=49152, end=49153)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=False):
        with pytest.raises(RuntimeError, match="No free port available"):
            backend.allocate_port()


def test_allocate_port_error_message_includes_range() -> None:
    backend = _backend(start=49152, end=49153)
    with patch.object(InMemoryRunnerBackend, "_is_port_free", return_value=False):
        with pytest.raises(RuntimeError, match=r"\[49152, 49153\]"):
            backend.allocate_port()
