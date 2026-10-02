# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for OpenShellDeploymentBackend against a live gateway.

Mocks only the entities client (feeding a DeploymentConfig directly, like the docker
integration test) and drives the real backend against the OpenShell gateway. Skipped
unless the gateway is reachable.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import Container, ContainerPort, DeploymentConfig

GATEWAY = os.environ.get("OPENSHELL_GATEWAY_ENDPOINT", "http://127.0.0.1:17670")
IMAGE = os.environ.get("OPENSHELL_TEST_IMAGE", "ghcr.io/nvidia/openshell-community/sandboxes/base:latest")


def _gateway_reachable(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif("openshell" not in BACKEND_CLASSES, reason="OpenShell backend not registered"),
    pytest.mark.skipif(not _gateway_reachable(GATEWAY), reason=f"OpenShell gateway not reachable at {GATEWAY}"),
]


async def _poll_until_ready(backend, *, attempts: int = 60, delay: float = 2.0):
    """Drive read_status until the deployment leaves STARTING, mirroring the reconciler."""
    status = None
    for _ in range(attempts):
        status = await backend.read_status(workspace="itest", name="rt")
        if status.status != "STARTING":
            return status
        await asyncio.sleep(delay)
    return status


def _make_backend():
    mock_entities = AsyncMock()
    with (
        patch("nemo_deployments_plugin.backends.openshell.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.openshell.backend.NemoEntitiesClient", return_value=mock_entities),
    ):
        from nemo_deployments_plugin.backends.openshell.backend import OpenShellDeploymentBackend

        backend = OpenShellDeploymentBackend(MagicMock(), {"gateway_endpoint": GATEWAY})
    backend._entities = mock_entities
    return backend, mock_entities


async def test_openshell_roundtrip_serves_http() -> None:
    backend, mock_entities = _make_backend()
    mock_entities.get.return_value = DeploymentConfig(
        name="rt-cfg",
        workspace="itest",
        containers=[
            Container(
                name="web",
                image=IMAGE,
                command=["python3", "-m", "http.server"],
                args=["8000", "--bind", "0.0.0.0"],
                ports=[ContainerPort(containerPort=8000, name="http")],
            )
        ],
    )

    try:
        created = await backend.create_deployment(
            workspace="itest",
            name="rt",
            config_name="rt-cfg",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
        )
        assert created.status == "STARTING", created.status_message

        # Provisioning runs on read_status across polls, so the endpoint exists only at READY.
        status = await _poll_until_ready(backend)
        assert status.status == "READY", status.status_message
        assert status.endpoints, "expected an exposed endpoint"
        url = status.endpoints[0].url

        code = "000"
        for _ in range(20):
            proc = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "10", url],
                capture_output=True,
                text=True,
            )
            code = proc.stdout.strip()
            if code not in ("000", "502", "503", "504"):
                break
            time.sleep(1)
        assert code == "200", f"served endpoint {url} returned HTTP {code}"

        assert "itest/rt" in await backend.list_managed_deployment_names()
    finally:
        await backend.delete_deployment("itest", "rt")
        backend.shutdown()


async def test_openshell_readiness_holds_until_bind() -> None:
    # The workload sleeps before it binds, so READY must not be published until the curl
    # readiness probe can actually reach :8000. Regression guard (READY
    # published ahead of bind) that also exercises the real in-sandbox curl probe.
    backend, mock_entities = _make_backend()
    bind_delay = 15
    mock_entities.get.return_value = DeploymentConfig(
        name="slow-cfg",
        workspace="itest",
        containers=[
            Container(
                name="web",
                image=IMAGE,
                command=["/bin/sh", "-c"],
                args=[f"sleep {bind_delay}; exec python3 -m http.server 8000 --bind 0.0.0.0"],
                ports=[ContainerPort(containerPort=8000, name="http")],
            )
        ],
    )

    try:
        created = await backend.create_deployment(
            workspace="itest",
            name="slow",
            config_name="slow-cfg",
            labels={"managed-by": MANAGED_BY_LABEL},
            backend_config={},
        )
        assert created.status == "STARTING", created.status_message

        start = time.monotonic()
        saw_starting = False
        status = None
        for _ in range(90):
            status = await backend.read_status(workspace="itest", name="slow")
            if status.status != "STARTING":
                break
            saw_starting = True
            await asyncio.sleep(1.0)

        assert status is not None and status.status == "READY", status.status_message if status else "no status"
        assert saw_starting, "expected STARTING before READY"
        # Serve only starts once provisioning completes, so time-to-READY strictly exceeds the
        # sleep. A probe that failed open would publish READY during the sleep window.
        assert time.monotonic() - start >= bind_delay - 3, "READY published before the workload bound its port"
        assert status.endpoints, "expected an exposed endpoint"
    finally:
        await backend.delete_deployment("itest", "slow")
        backend.shutdown()
