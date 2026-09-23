# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example plugin SDK resources."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_example_plugin.sdk import AsyncExampleClient, ExampleClient
from nemo_example_plugin.types import endpoints
from nemo_example_plugin.types.payloads import (
    CountRequest,
    CreateExampleItemRequest,
    UpdateExampleItemRequest,
)
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError

BASE = "http://test:8000"
WS = "default"
ITEM_PAYLOAD = {
    "id": "default/my-item",
    "name": "my-item",
    "workspace": "default",
    "title": "My Item",
    "body": "",
    "tags": [],
    "entity_type": "example_item",
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-01T00:00:00Z",
}


def _resp(status: int, payload=None) -> httpx.Response:
    kwargs: dict = {"request": httpx.Request("GET", BASE)}
    if payload is not None:
        kwargs["json"] = payload
    else:
        kwargs["content"] = b""
    return httpx.Response(status, **kwargs)


def _sync_client() -> tuple[ExampleClient, MagicMock]:
    mock_http = MagicMock(spec=httpx.Client)
    client = ExampleClient(base_url=BASE, workspace=WS, http_client=mock_http)
    return client, mock_http


def _async_client() -> tuple[AsyncExampleClient, AsyncMock]:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    client = AsyncExampleClient(base_url=BASE, workspace=WS, http_client=mock_http)
    return client, mock_http


# ---------------------------------------------------------------------------
# hello — client.method() style
# ---------------------------------------------------------------------------


def test_sync_hello() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(200, {"message": "Hello, alice!"})
    resp = client.hello(name="alice")
    assert resp.data().message == "Hello, alice!"


@pytest.mark.asyncio
async def test_async_hello() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(200, {"message": "Hello, bob!"})
    resp = await client.hello(name="bob")
    assert resp.data().message == "Hello, bob!"


# ---------------------------------------------------------------------------
# Items CRUD — client.method() style (sync)
# ---------------------------------------------------------------------------


def test_sync_create_item() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(201, ITEM_PAYLOAD)

    resp = client.create_item(body=CreateExampleItemRequest(name="my-item", title="My Item"))
    item = resp.data()

    assert item.name == "my-item"
    assert item.title == "My Item"
    mock_http.request.assert_called_once()


def test_sync_create_item_explicit_workspace() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(201, ITEM_PAYLOAD)

    resp = client.create_item(workspace="other", body=CreateExampleItemRequest(name="my-item", title="My Item"))

    assert resp.data().name == "my-item"
    url_called = mock_http.request.call_args[0][1]
    assert "/workspaces/other/" in url_called


def test_sync_get_item() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(200, ITEM_PAYLOAD)

    resp = client.get_item(name="my-item")

    assert resp.data().name == "my-item"


def test_sync_list_items() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(
        200,
        {
            "data": [ITEM_PAYLOAD],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
            "sort": None,
            "filter": None,
        },
    )

    resp = client.list_items()
    page = resp.page()

    assert len(page.items) == 1
    assert page.items[0].name == "my-item"


def test_sync_update_item() -> None:
    client, mock_http = _sync_client()
    updated = {**ITEM_PAYLOAD, "title": "Updated"}
    mock_http.request.return_value = _resp(200, updated)

    resp = client.update_item(name="my-item", body=UpdateExampleItemRequest(title="Updated"))

    assert resp.data().title == "Updated"


def test_sync_delete_item() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(204)

    client.delete_item(name="my-item")

    mock_http.request.assert_called_once()


# ---------------------------------------------------------------------------
# Low-level: endpoints + client.send() still works
# ---------------------------------------------------------------------------


def test_send_with_endpoint_function() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = _resp(200, {"message": "Hello, alice!"})
    client = NemoClient(base_url=BASE, workspace=WS, http_client=mock_http)

    resp = client.send(endpoints.hello(name="alice"))

    assert resp.data().message == "Hello, alice!"


# ---------------------------------------------------------------------------
# Items CRUD — client.method() style (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_create_item() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(201, ITEM_PAYLOAD)

    resp = await client.create_item(body=CreateExampleItemRequest(name="my-item", title="My Item"))

    assert resp.data().name == "my-item"


@pytest.mark.asyncio
async def test_async_get_item() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(200, ITEM_PAYLOAD)

    resp = await client.get_item(name="my-item")

    assert resp.data().name == "my-item"


@pytest.mark.asyncio
async def test_async_list_items() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(
        200,
        {
            "data": [ITEM_PAYLOAD],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
            "sort": None,
            "filter": None,
        },
    )

    resp = await client.list_items()
    page = resp.page()

    assert len(page.items) == 1
    assert page.items[0].name == "my-item"


@pytest.mark.asyncio
async def test_async_update_item() -> None:
    client, mock_http = _async_client()
    updated = {**ITEM_PAYLOAD, "title": "Updated"}
    mock_http.request.return_value = _resp(200, updated)

    resp = await client.update_item(name="my-item", body=UpdateExampleItemRequest(title="Updated"))

    assert resp.data().title == "Updated"


@pytest.mark.asyncio
async def test_async_delete_item() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(204)

    await client.delete_item(name="my-item")

    mock_http.request.assert_awaited_once()


# ---------------------------------------------------------------------------
# Binary endpoints — upload_blob / download_blob
# ---------------------------------------------------------------------------


def _stream_ctx(resp: httpx.Response):
    """Create a sync context manager that yields *resp*."""

    @contextmanager
    def _ctx(*_args, **_kwargs):
        yield resp

    return _ctx


def _async_stream_ctx(resp: httpx.Response):
    """Create an async context manager that yields *resp*."""

    @asynccontextmanager
    async def _ctx(*_args, **_kwargs):
        yield resp

    return _ctx


def test_sync_upload_blob() -> None:
    client, mock_http = _sync_client()
    mock_http.request.return_value = _resp(200, {"name": "pic.png", "size": 42})

    resp = client.upload_blob(name="pic.png", content=b"\x89PNG")

    assert resp.data().name == "pic.png"
    assert resp.data().size == 42


def test_sync_download_blob_read() -> None:
    client, mock_http = _sync_client()
    raw = httpx.Response(200, content=b"file-bytes", request=httpx.Request("GET", BASE))
    mock_http.stream = _stream_ctx(raw)

    resp = client.download_blob(name="pic.png")
    data = resp.read()

    assert data == b"file-bytes"


def test_sync_download_blob_stream() -> None:
    client, mock_http = _sync_client()
    raw = httpx.Response(200, stream=httpx.ByteStream(b"chunk1chunk2"), request=httpx.Request("GET", BASE))
    mock_http.stream = _stream_ctx(raw)

    resp = client.download_blob(name="pic.png")
    with resp.stream() as chunks:
        result = b"".join(chunks)

    assert result == b"chunk1chunk2"


@pytest.mark.asyncio
async def test_async_upload_blob() -> None:
    client, mock_http = _async_client()
    mock_http.request.return_value = _resp(200, {"name": "pic.png", "size": 42})

    resp = await client.upload_blob(name="pic.png", content=b"\x89PNG")

    assert resp.data().name == "pic.png"


@pytest.mark.asyncio
async def test_async_download_blob_read() -> None:
    client, mock_http = _async_client()
    raw = httpx.Response(200, content=b"file-bytes", request=httpx.Request("GET", BASE))
    mock_http.stream = _async_stream_ctx(raw)

    resp = await client.download_blob(name="pic.png")
    data = await resp.read()

    assert data == b"file-bytes"


# ---------------------------------------------------------------------------
# Streaming endpoint — count
# ---------------------------------------------------------------------------


def test_sync_count_stream() -> None:
    client, mock_http = _sync_client()
    body = '{"kind":"tick","n":1}\n{"kind":"tick","n":2}\n{"kind":"done","n":null}\n'
    raw = httpx.Response(200, content=body.encode(), request=httpx.Request("POST", BASE))
    mock_http.stream = _stream_ctx(raw)

    resp = client.count(body=CountRequest(upto=2))
    with resp.stream() as ticks:
        items = list(ticks)

    assert len(items) == 3
    assert items[0].kind == "tick"
    assert items[0].n == 1
    assert items[2].kind == "done"


@pytest.mark.asyncio
async def test_async_count_stream() -> None:
    client, mock_http = _async_client()
    body = '{"kind":"tick","n":1}\n{"kind":"done","n":null}\n'
    raw = httpx.Response(200, content=body.encode(), request=httpx.Request("POST", BASE))
    mock_http.stream = _async_stream_ctx(raw)

    resp = await client.count(body=CountRequest(upto=1))
    async with resp.stream() as ticks:
        items = [t async for t in ticks]

    assert len(items) == 2
    assert items[0].kind == "tick"
    assert items[1].kind == "done"


# ---------------------------------------------------------------------------
# SSE framing support
# ---------------------------------------------------------------------------


def test_sync_stream_sse_framing() -> None:
    """SSE data: prefixes are stripped when Content-Type is text/event-stream."""
    client, mock_http = _sync_client()
    body = 'data: {"kind":"tick","n":1}\ndata: {"kind":"done","n":null}\n\n'
    raw = httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", BASE),
    )
    mock_http.stream = _stream_ctx(raw)

    resp = client.count(body=CountRequest(upto=1))
    with resp.stream() as ticks:
        items = list(ticks)

    assert len(items) == 2
    assert items[0].kind == "tick"
    assert items[1].kind == "done"


def test_sync_stream_sse_skips_non_data_fields() -> None:
    """SSE event:, id:, and comment lines are skipped."""
    client, mock_http = _sync_client()
    body = 'event: tick\ndata: {"kind":"tick","n":1}\n: comment\nid: 42\ndata: {"kind":"done","n":null}\n\n'
    raw = httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", BASE),
    )
    mock_http.stream = _stream_ctx(raw)

    resp = client.count(body=CountRequest(upto=1))
    with resp.stream() as ticks:
        items = list(ticks)

    assert len(items) == 2
    assert items[0].kind == "tick"
    assert items[1].kind == "done"


@pytest.mark.asyncio
async def test_async_stream_sse_framing() -> None:
    """Async SSE data: prefixes are stripped."""
    client, mock_http = _async_client()
    body = 'data: {"kind":"tick","n":1}\ndata: {"kind":"done","n":null}\n\n'
    raw = httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", BASE),
    )
    mock_http.stream = _async_stream_ctx(raw)

    resp = await client.count(body=CountRequest(upto=1))
    async with resp.stream() as ticks:
        items = [t async for t in ticks]

    assert len(items) == 2
    assert items[0].kind == "tick"
    assert items[1].kind == "done"


# ---------------------------------------------------------------------------
# Error detail extraction from streaming responses
# ---------------------------------------------------------------------------


def test_binary_read_error_has_detail() -> None:
    """Binary read() on error response should extract JSON detail."""
    client, mock_http = _sync_client()
    raw = httpx.Response(
        404,
        content=b'{"detail": "File not found"}',
        request=httpx.Request("GET", BASE),
    )
    mock_http.stream = _stream_ctx(raw)

    resp = client.download_blob(name="missing.png")
    with pytest.raises(NemoHTTPError) as exc_info:
        resp.read()

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "File not found"
