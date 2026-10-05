# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local service transport helpers for TCP and Unix domain sockets."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, TypeAlias

import httpx
from fastapi.testclient import TestClient
from nhx.common.platform_endpoint import UDS_BASE_URL

HttpxTimeout: TypeAlias = float | httpx.Timeout | None
_DEFAULT_TIMEOUT: float = 5.0
EMBEDDED_BASE_URL = "http://nemo-helix.local"

__all__ = [
    "EMBEDDED_BASE_URL",
    "UDS_BASE_URL",
    "build_async_asgi_http_client",
    "build_async_http_client",
    "build_sync_asgi_http_client",
    "build_sync_http_client",
    "probe_status",
    "probe_status_async",
    "tcp_base_url",
    "wait_for_status",
    "wait_for_status_async",
]


def build_sync_asgi_http_client(app: Any, *, timeout: HttpxTimeout = _DEFAULT_TIMEOUT) -> Any:
    _ = timeout
    return TestClient(
        app,
        base_url=EMBEDDED_BASE_URL,
        follow_redirects=True,
    )


def build_async_asgi_http_client(app: Any, *, timeout: HttpxTimeout = _DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url=EMBEDDED_BASE_URL,
        follow_redirects=True,
        timeout=timeout,
    )


def build_sync_http_client(socket_path: Path, *, timeout: HttpxTimeout = _DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=str(socket_path)),
        follow_redirects=True,
        timeout=timeout,
    )


def build_async_http_client(socket_path: Path, *, timeout: HttpxTimeout = _DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=str(socket_path)),
        follow_redirects=True,
        timeout=timeout,
    )


def tcp_base_url(host: str, port: int) -> str:
    connect_host = "localhost" if host in {"0.0.0.0", "::"} else host  # noqa: S104
    normalized = connect_host.strip("[]")
    url_host = f"[{normalized}]" if ":" in normalized else normalized
    return str(httpx.URL(scheme="http", host=url_host, port=port))


def probe_status(
    *,
    base_url: str,
    socket_path: Path | None = None,
    timeout: float = 2.0,
) -> bool:
    client = (
        build_sync_http_client(socket_path, timeout=timeout)
        if socket_path is not None
        else httpx.Client(timeout=timeout)
    )
    try:
        response = client.get(f"{base_url.rstrip('/')}/status")
        return response.status_code == 200
    except httpx.RequestError:
        return False
    finally:
        client.close()


async def probe_status_async(
    *,
    base_url: str,
    socket_path: Path | None = None,
    timeout: float = 2.0,
) -> bool:
    client = (
        build_async_http_client(socket_path, timeout=timeout)
        if socket_path is not None
        else httpx.AsyncClient(timeout=timeout)
    )
    try:
        response = await client.get(f"{base_url.rstrip('/')}/status")
        return response.status_code == 200
    except httpx.RequestError:
        return False
    finally:
        await client.aclose()


def wait_for_status(
    *,
    base_url: str,
    socket_path: Path | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if probe_status(base_url=base_url, socket_path=socket_path, timeout=remaining):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(poll_interval, remaining))


async def wait_for_status_async(
    *,
    base_url: str,
    socket_path: Path | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if await probe_status_async(base_url=base_url, socket_path=socket_path, timeout=remaining):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(poll_interval, remaining))
