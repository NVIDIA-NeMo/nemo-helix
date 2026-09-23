# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nhx.common.config import HelixConfig
from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nhx.common.service.api.health import (
    async_wait_for_dependencies,
    async_wait_for_service_ready,
    wait_for_service_ready,
)


@pytest.mark.asyncio
async def test_async_wait_for_service_ready_skips_service_absent_from_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"services": {"ready": ["entities"], "not_ready": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ready = await async_wait_for_service_ready(
            HelixConfig(base_url="http://platform.local"),
            "models",
            timeout=1.0,
            poll_interval=0,
            http_client=client,
        )

    assert ready is True
    assert len(requests) == 1
    assert str(requests[0].url) == "http://platform.local/status"
    for key, value in MARK_INTERNAL_REQUEST_HEADERS.items():
        assert requests[0].headers[key] == value


@pytest.mark.asyncio
async def test_async_wait_for_service_ready_waits_for_explicitly_not_ready_service() -> None:
    responses = [
        httpx.Response(200, json={"services": {"ready": ["entities"], "not_ready": [{"name": "models"}]}}),
        httpx.Response(200, json={"services": {"ready": ["entities", "models"], "not_ready": []}}),
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ready = await async_wait_for_service_ready(
            HelixConfig(base_url="http://platform.local"),
            "models",
            timeout=1.0,
            poll_interval=0,
            http_client=client,
        )

    assert ready is True
    assert responses == []


@pytest.mark.asyncio
async def test_async_wait_for_service_ready_retries_intermediate_failures() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection refused", request=request)
        if calls == 2:
            return httpx.Response(503)
        if calls == 3:
            return httpx.Response(200, content=b"{")
        if calls == 4:
            return httpx.Response(200, json=["not", "a", "mapping"])
        return httpx.Response(200, json={"services": {"ready": ["auth"], "not_ready": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ready = await async_wait_for_service_ready(
            HelixConfig(base_url="http://platform.local"),
            "auth",
            timeout=1.0,
            poll_interval=0,
            http_client=client,
        )

    assert ready is True
    assert calls == 5


@pytest.mark.asyncio
async def test_async_wait_for_dependencies_stops_after_unready_dependency() -> None:
    with patch(
        "nhx.common.service.api.health.async_wait_for_service_ready",
        side_effect=[True, False],
    ) as wait:
        ready = await async_wait_for_dependencies(
            HelixConfig(base_url="http://platform.local"),
            ["entities", "auth", "files"],
            timeout_per_service=0.01,
            poll_interval=0,
        )

    assert ready is False
    assert [call.args[1] for call in wait.await_args_list] == ["entities", "auth"]


def test_wait_for_service_ready_skips_service_absent_from_status() -> None:
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"services": {"ready": ["entities"], "not_ready": []}}
    client.get.return_value = response

    endpoint = MagicMock()
    endpoint.connect_base_url = "http://platform.local"
    endpoint.sync_http_client.return_value = client

    with patch("nhx.common.service.api.health.resolve_service_endpoint", return_value=endpoint):
        ready = wait_for_service_ready(
            HelixConfig(base_url="http://platform.local"),
            "models",
            threading.Event(),
            timeout=1.0,
            poll_interval=0,
        )

    assert ready is True
    client.get.assert_called_once_with("http://platform.local/status", headers=MARK_INTERNAL_REQUEST_HEADERS)
    client.close.assert_called_once_with()


def test_wait_for_service_ready_returns_false_when_stop_signal_is_set() -> None:
    stop_signal = threading.Event()
    stop_signal.set()
    client = MagicMock()
    endpoint = MagicMock()
    endpoint.connect_base_url = "http://platform.local"
    endpoint.sync_http_client.return_value = client

    with patch("nhx.common.service.api.health.resolve_service_endpoint", return_value=endpoint):
        ready = wait_for_service_ready(
            HelixConfig(base_url="http://platform.local"),
            "entities",
            stop_signal,
            timeout=1.0,
            poll_interval=0,
        )

    assert ready is False
    client.get.assert_not_called()
    client.close.assert_called_once_with()
