# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator job helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
from nemo_evaluator.jobs.utils import async_client_from_sync_client, run_with_isolated_async_client
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.files.client import AsyncFilesClient
from pytest_mock import MockerFixture

type AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]


def test_run_with_isolated_async_client_passes_cloned_client(mocker: MockerFixture) -> None:
    async_client = AsyncFilesClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=mocker.AsyncMock(spec=httpx.AsyncClient),
    )
    http_client = mocker.AsyncMock(name="http_client")
    http_client.__aenter__.return_value = http_client
    http_client.__aexit__.return_value = None
    mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", return_value=http_client)

    seen: list[AsyncFilesClient] = []

    async def fn(client: AsyncFilesClient) -> int:
        seen.append(client)
        return 42

    assert run_with_isolated_async_client(async_client, fn) == 42
    assert seen[0] is not async_client
    assert isinstance(seen[0], AsyncFilesClient)
    assert seen[0].base_url == async_client.base_url
    assert seen[0].workspace == async_client.workspace
    assert seen[0]._client is http_client


def test_run_with_isolated_async_client_preserves_request_event_hooks(mocker: MockerFixture) -> None:
    seen_authorization: list[str | None] = []

    async def refresh(request: httpx.Request) -> None:
        request.headers["Authorization"] = "Bearer refreshed"

    async def record(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers.get("Authorization"))
        return httpx.Response(200, request=request)

    source_http_client = httpx.AsyncClient(event_hooks={"request": [refresh]})
    async_client = AsyncFilesClient(
        base_url="http://platform.test",
        workspace="default",
        default_headers={"X-NMP-Principal-Id": "principal-123"},
        http_client=source_http_client,
    )
    async_client_cls = httpx.AsyncClient

    def make_isolated_http_client(event_hooks: dict[str, list[AsyncRequestHook]] | None = None) -> httpx.AsyncClient:
        return async_client_cls(
            transport=httpx.MockTransport(record),
            event_hooks=event_hooks,
        )

    mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", side_effect=make_isolated_http_client)

    async def fn(client: AsyncFilesClient) -> int:
        http_client = client._client
        assert isinstance(http_client, async_client_cls)
        assert client.default_headers == {"X-NMP-Principal-Id": "principal-123"}
        response = await http_client.get("http://platform.test/ping")
        return response.status_code

    assert run_with_isolated_async_client(async_client, fn) == 200
    assert seen_authorization == ["Bearer refreshed"]


def test_async_client_from_sync_client_preserves_headers_and_request_event_hooks(mocker: MockerFixture) -> None:
    seen_authorization: list[str | None] = []

    def refresh(request: httpx.Request) -> None:
        request.headers["Authorization"] = "Bearer refreshed"

    async def record(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers.get("Authorization"))
        return httpx.Response(200, request=request)

    source_http_client = httpx.Client(event_hooks={"request": [refresh]})
    sync_client = NemoClient(
        base_url="http://platform.test",
        workspace="default",
        default_headers={"X-NMP-Internal": "true"},
        http_client=source_http_client,
    )
    async_client_cls = httpx.AsyncClient

    def make_isolated_http_client(event_hooks: dict[str, list[AsyncRequestHook]] | None = None) -> httpx.AsyncClient:
        return async_client_cls(
            transport=httpx.MockTransport(record),
            event_hooks=event_hooks,
        )

    mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", side_effect=make_isolated_http_client)

    with async_client_from_sync_client(sync_client) as async_client:
        assert isinstance(async_client, AsyncNemoClient)
        assert async_client.default_headers == {"X-NMP-Internal": "true"}

        async def fn(client: AsyncNemoClient) -> int:
            http_client = client._client
            assert isinstance(http_client, async_client_cls)
            response = await http_client.get("http://platform.test/ping")
            return response.status_code

        assert run_with_isolated_async_client(async_client, fn) == 200

    assert seen_authorization == ["Bearer refreshed"]


def test_async_client_from_sync_client_ignores_missing_event_hook_mapping(mocker: MockerFixture) -> None:
    source_http_client = mocker.Mock(spec=httpx.Client)
    source_http_client.event_hooks = mocker.Mock()
    sync_client = NemoClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=source_http_client,
    )
    isolated_http_client = mocker.AsyncMock(spec=httpx.AsyncClient)
    async_client_ctor = mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", return_value=isolated_http_client)

    with async_client_from_sync_client(sync_client) as async_client:
        assert isinstance(async_client, AsyncNemoClient)
        assert async_client._client is isolated_http_client

    async_client_ctor.assert_called_once_with(event_hooks={})
