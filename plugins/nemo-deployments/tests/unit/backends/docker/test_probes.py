# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Docker readiness probes."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import pytest
from nemo_deployments_plugin.backends.docker.probes import check_readiness_probe
from nemo_deployments_plugin.entities import HTTPGetAction, Probe, TCPSocketAction


@pytest.mark.asyncio
async def test_http_probe_without_host_url_not_ready() -> None:
    container = MagicMock()
    probe = Probe(http_get=HTTPGetAction(path="/health", port=8080))

    ready, reason = await check_readiness_probe(
        container=container,
        probe=probe,
        host_url=None,
    )

    assert ready is False
    assert "no host_url available" in reason


@pytest.mark.asyncio
async def test_tcp_probe_without_host_url_not_ready() -> None:
    container = MagicMock()
    probe = Probe(tcp_socket=TCPSocketAction(port=8080))

    ready, reason = await check_readiness_probe(
        container=container,
        probe=probe,
        host_url=None,
    )

    assert ready is False
    assert "no host_url available" in reason


@pytest.mark.asyncio
async def test_no_probe_without_host_port_is_ready() -> None:
    # A portless workload has no socket to connect to, so running implies ready.
    ready, reason = await check_readiness_probe(
        container=MagicMock(),
        probe=None,
        host_url=None,
        host_ports={},
    )

    assert ready is True
    assert reason == "no readiness probe configured"


@pytest.mark.asyncio
async def test_no_probe_with_bound_host_port_is_ready() -> None:
    # No declared probe but the published port accepts a connection -> ready.
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        host_port = server.getsockname()[1]

        ready, reason = await check_readiness_probe(
            container=MagicMock(),
            probe=None,
            host_url=f"http://127.0.0.1:{host_port}",
            host_ports={8000: host_port},
        )

    assert ready is True
    assert "connected" in reason


@pytest.mark.asyncio
async def test_no_probe_with_unbound_host_port_not_ready() -> None:
    # No declared probe and nothing listening on the published port -> not ready, so
    # READY does not race the workload's bind(). Hold the port bound-but-not-listening
    # for the whole probe: that reserves it (a second bind gets EADDRINUSE, so nothing
    # else can grab it and start listening mid-test) while a connect() still gets
    # ECONNREFUSED, which is exactly the not-yet-bound state under test.
    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        host_port = probe_socket.getsockname()[1]

        ready, reason = await check_readiness_probe(
            container=MagicMock(),
            probe=None,
            host_url=f"http://127.0.0.1:{host_port}",
            host_ports={8000: host_port},
        )

    assert ready is False
    assert "not ready" in reason
