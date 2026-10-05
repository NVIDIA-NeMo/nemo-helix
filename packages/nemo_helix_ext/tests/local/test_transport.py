# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from nemo_helix_ext.local import transport


def _assert_timeout_values(timeout: httpx.Timeout, expected: float | None) -> None:
    assert timeout.connect == expected
    assert timeout.read == expected
    assert timeout.write == expected
    assert timeout.pool == expected


def test_build_sync_http_client_uses_finite_default_timeout(tmp_path) -> None:
    client = transport.build_sync_http_client(tmp_path / "nemo.sock")
    try:
        _assert_timeout_values(client.timeout, 5.0)
    finally:
        client.close()


def test_build_sync_http_client_preserves_explicit_timeout_values(tmp_path) -> None:
    no_timeout_client = transport.build_sync_http_client(tmp_path / "nemo.sock", timeout=None)
    finite_timeout_client = transport.build_sync_http_client(tmp_path / "nemo.sock", timeout=12.0)
    try:
        _assert_timeout_values(no_timeout_client.timeout, None)
        _assert_timeout_values(finite_timeout_client.timeout, 12.0)
    finally:
        no_timeout_client.close()
        finite_timeout_client.close()


@pytest.mark.asyncio
async def test_build_async_http_client_uses_finite_default_timeout(tmp_path) -> None:
    client = transport.build_async_http_client(tmp_path / "nemo.sock")
    try:
        _assert_timeout_values(client.timeout, 5.0)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_build_async_http_client_preserves_explicit_timeout_values(tmp_path) -> None:
    no_timeout_client = transport.build_async_http_client(tmp_path / "nemo.sock", timeout=None)
    finite_timeout_client = transport.build_async_http_client(tmp_path / "nemo.sock", timeout=12.0)
    try:
        _assert_timeout_values(no_timeout_client.timeout, None)
        _assert_timeout_values(finite_timeout_client.timeout, 12.0)
    finally:
        await no_timeout_client.aclose()
        await finite_timeout_client.aclose()


def test_build_sync_asgi_http_client_reaches_app() -> None:
    app = FastAPI()

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "healthy"}

    client = transport.build_sync_asgi_http_client(app)
    try:
        response = client.get("http://nemo-helix.local/status")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_build_async_asgi_http_client_reaches_app() -> None:
    app = FastAPI()

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "healthy"}

    client = transport.build_async_asgi_http_client(app)
    try:
        response = await client.get("http://nemo-helix.local/status")
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_wait_for_status_bounds_probe_and_sleep_by_remaining_deadline() -> None:
    with (
        patch("nemo_helix_ext.local.transport.probe_status", return_value=False) as probe_status,
        patch("nemo_helix_ext.local.transport.time.monotonic", side_effect=[0.0, 4.0, 4.5, 5.0]),
        patch("nemo_helix_ext.local.transport.time.sleep") as sleep,
    ):
        result = transport.wait_for_status(base_url="http://127.0.0.1:8080", timeout=5.0, poll_interval=10.0)

    assert result is False
    assert probe_status.call_args.kwargs["timeout"] == pytest.approx(1.0)
    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_wait_for_status_async_bounds_probe_and_sleep_by_remaining_deadline() -> None:
    with (
        patch("nemo_helix_ext.local.transport.probe_status_async", new=AsyncMock(return_value=False)) as probe_status,
        patch("nemo_helix_ext.local.transport.time.monotonic", side_effect=[0.0, 4.0, 4.5, 5.0]),
        patch("nemo_helix_ext.local.transport.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        result = await transport.wait_for_status_async(
            base_url="http://127.0.0.1:8080", timeout=5.0, poll_interval=10.0
        )

    assert result is False
    assert probe_status.await_args.kwargs["timeout"] == pytest.approx(1.0)
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(0.5)


def test_probe_status_returns_true_for_status_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:8080/status"
        return httpx.Response(200)

    with patch("nemo_helix_ext.local.transport.httpx.Client") as client_factory:
        client = client_factory.return_value
        client.get.side_effect = lambda url: handler(httpx.Request("GET", url))
        assert transport.probe_status(base_url="http://127.0.0.1:8080") is True
        client.close.assert_called_once_with()


def test_probe_status_returns_false_for_request_error() -> None:
    with patch("nemo_helix_ext.local.transport.httpx.Client") as client_factory:
        client = client_factory.return_value
        client.get.side_effect = httpx.ConnectError("boom")
        assert transport.probe_status(base_url="http://127.0.0.1:8080") is False
        client.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_probe_status_async_returns_false_for_request_error() -> None:
    with patch("nemo_helix_ext.local.transport.httpx.AsyncClient") as client_factory:
        client = client_factory.return_value
        client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
        client.aclose = AsyncMock()
        assert await transport.probe_status_async(base_url="http://127.0.0.1:8080") is False
        client.aclose.assert_awaited_once_with()


def test_wait_for_status_returns_true_without_sleep_when_probe_succeeds() -> None:
    with (
        patch("nemo_helix_ext.local.transport.probe_status", return_value=True) as probe_mock,
        patch("nemo_helix_ext.local.transport.time.sleep") as sleep,
    ):
        assert transport.wait_for_status(base_url="http://127.0.0.1:8080", timeout=5.0) is True

    probe_mock.assert_called_once()
    sleep.assert_not_called()
