# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bearer token is resolved for every HTTP attempt, not once per logical request.

A token provider refreshes proactively, but that only helps if it is consulted
each time bytes go on the wire. Pages after the first and retries after a
backoff must therefore carry whatever the provider returns at that moment.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from nemo_platform_plugin.client.auth import TokenProviderAuth
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.types import BinaryContent, Paginated, RetryPolicy
from pydantic import BaseModel

BASE = "http://test"


class Item(BaseModel):
    name: str


@get("/apis/test/v2/items")
def list_items(*, query_params: dict[str, Any] | None = None) -> Paginated[Item]:
    raise NotImplementedError


@get("/apis/test/v2/items/{name}")
def get_item(*, name: str) -> Item:
    raise NotImplementedError


@get("/apis/test/v2/download")
def download() -> BinaryContent:
    raise NotImplementedError


class RotatingProvider:
    """Returns ``token-1``, ``token-2``, ... on successive calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_access_token(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"


class AsyncRotatingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def get_access_token(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"


def _page_body(page: int, total_pages: int) -> dict[str, Any]:
    return {
        "data": [{"name": f"item-{page}"}],
        "pagination": {
            "page": page,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": total_pages,
            "total_results": total_pages,
        },
    }


def _paging_handler(seen: list[str]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        page = int(parse_qs(request.url.query.decode()).get("page", ["1"])[0])
        return httpx.Response(200, json=_page_body(page, 3))

    return handler


def _retry_handler(seen: list[str], failures: int) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) <= failures:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json={"name": "alice"})

    return handler


NO_BACKOFF = RetryPolicy(max_retries=2, backoff_base=0.0)


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def test_each_page_uses_a_freshly_resolved_token() -> None:
    seen: list[str] = []
    client = NemoClient(
        base_url=BASE,
        auth=RotatingProvider(),
        http_client=httpx.Client(transport=httpx.MockTransport(_paging_handler(seen))),
    )

    names = [item.name for item in client.send(list_items()).items()]

    assert names == ["item-1", "item-2", "item-3"]
    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]


def test_each_retry_attempt_uses_a_freshly_resolved_token() -> None:
    seen: list[str] = []
    client = NemoClient(
        base_url=BASE,
        auth=RotatingProvider(),
        retry=NO_BACKOFF,
        http_client=httpx.Client(transport=httpx.MockTransport(_retry_handler(seen, failures=2))),
    )

    assert client.send(get_item(name="alice")).body.name == "alice"
    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]


def test_stream_retry_attempts_use_a_freshly_resolved_token() -> None:
    """Binary downloads go through the streaming path, which has its own attempt loop."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            raise httpx.ConnectError("no route to host", request=request)
        return httpx.Response(200, stream=httpx.ByteStream(b"payload"))

    client = NemoClient(
        base_url=BASE,
        auth=RotatingProvider(),
        retry=NO_BACKOFF,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with client.send(download()).stream() as chunks:
        assert b"".join(chunks) == b"payload"
    assert seen == ["Bearer token-1", "Bearer token-2"]


def test_explicit_authorization_header_is_not_overridden() -> None:
    seen: list[str] = []
    provider = RotatingProvider()
    client = NemoClient(
        base_url=BASE,
        auth=provider,
        http_client=httpx.Client(transport=httpx.MockTransport(_paging_handler(seen))),
    )

    list(client.send(list_items(), headers={"Authorization": "Bearer pinned"}).items())

    assert seen == ["Bearer pinned"] * 3
    assert provider.calls == 0


def test_transport_auth_hook_does_not_double_resolve() -> None:
    """Clients built with TokenProviderAuth on the transport still resolve once per attempt."""
    seen: list[str] = []
    provider = RotatingProvider()
    client = NemoClient(
        base_url=BASE,
        auth=provider,
        http_client=httpx.Client(
            transport=httpx.MockTransport(_paging_handler(seen)), auth=TokenProviderAuth(provider)
        ),
    )

    list(client.send(list_items()).items())

    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]
    assert provider.calls == 3


def test_sync_client_rejects_async_provider_at_send_time() -> None:
    client = NemoClient(
        base_url=BASE,
        auth=AsyncRotatingProvider(),  # type: ignore[arg-type]
        http_client=httpx.Client(transport=httpx.MockTransport(_paging_handler([]))),
    )

    with pytest.raises(TypeError, match="Async token provider"):
        client.send(get_item(name="alice"))


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_each_page_uses_a_freshly_resolved_token() -> None:
    seen: list[str] = []
    client = AsyncNemoClient(
        base_url=BASE,
        auth=AsyncRotatingProvider(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_paging_handler(seen))),
    )

    names = [item.name async for item in (await client.send(list_items())).items()]

    assert names == ["item-1", "item-2", "item-3"]
    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]


@pytest.mark.asyncio
async def test_async_each_retry_attempt_uses_a_freshly_resolved_token() -> None:
    seen: list[str] = []
    client = AsyncNemoClient(
        base_url=BASE,
        auth=AsyncRotatingProvider(),
        retry=NO_BACKOFF,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_retry_handler(seen, failures=2))),
    )

    assert (await client.send(get_item(name="alice"))).body.name == "alice"
    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]


@pytest.mark.asyncio
async def test_async_client_accepts_sync_provider() -> None:
    seen: list[str] = []
    client = AsyncNemoClient(
        base_url=BASE,
        auth=RotatingProvider(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_paging_handler(seen))),
    )

    [item async for item in (await client.send(list_items())).items()]

    assert seen == ["Bearer token-1", "Bearer token-2", "Bearer token-3"]
