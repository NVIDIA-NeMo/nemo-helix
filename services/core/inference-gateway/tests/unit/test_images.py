# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving fileset image references in chat requests."""

import base64
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from nemo_helix_plugin.client.client import AsyncNemoClient
from nhx.core.inference_gateway.api.images import MAX_IMAGE_BYTES, resolve_fileset_images

IMAGE_BYTES = b"\x89PNG\r\n\x1a\nimage-bytes"


def _body(*urls: str) -> dict[str, Any]:
    return {
        "model": "default/super-vl",
        "messages": [
            {"role": "system", "content": "Describe images."},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}} for url in urls]
                + [{"type": "text", "text": "What is in these images?"}],
            },
        ],
    }


def _urls(body: dict[str, Any]) -> list[str]:
    return [part["image_url"]["url"] for part in body["messages"][1]["content"] if part["type"] == "image_url"]


async def _chunks(data: bytes, size: int) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


def _client(
    status_code: int = 200,
    content: bytes = IMAGE_BYTES,
    seen: list[httpx.Request] | None = None,
    unreachable: bool = False,
) -> AsyncNemoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if status_code != 200:
            return httpx.Response(status_code, json={"detail": "files said no"})
        return httpx.Response(200, content=_chunks(content, 1024 * 1024))

    return AsyncNemoClient(
        base_url="http://platform.test",
        default_headers={"x-test-caller": "caller"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected_path", "mime"),
    [
        ("fileset://img-ws/pics#cats/one.png", "/apis/files/v2/workspaces/img-ws/filesets/pics/-/cats/one.png", "png"),
        ("fileset://pics#dog.JPG", "/apis/files/v2/workspaces/req-ws/filesets/pics/-/dog.JPG", "jpeg"),
        ("fileset://pics#a.jpeg", "/apis/files/v2/workspaces/req-ws/filesets/pics/-/a.jpeg", "jpeg"),
        ("fileset://img-ws/pics#a.gif", "/apis/files/v2/workspaces/img-ws/filesets/pics/-/a.gif", "gif"),
        ("fileset://img-ws/pics#a.webp", "/apis/files/v2/workspaces/img-ws/filesets/pics/-/a.webp", "webp"),
    ],
)
async def test_resolve_fileset_images_inlines_data_url_with_caller_client(url: str, expected_path: str, mime: str):
    seen: list[httpx.Request] = []
    body = _body(url)

    await resolve_fileset_images(body, workspace="req-ws", request_nemo_client=_client(seen=seen))

    assert _urls(body) == [f"data:image/{mime};base64,{base64.b64encode(IMAGE_BYTES).decode()}"]
    assert [request.url.path for request in seen] == [expected_path]
    assert seen[0].headers["x-test-caller"] == "caller"


@pytest.mark.asyncio
async def test_resolve_fileset_images_leaves_inline_and_remote_urls_unchanged():
    urls = ("data:image/png;base64,iVBORw0KGgo", "https://example.com/cat.png", "http://example.com/cat.png")
    body = _body(*urls, "fileset://img-ws/pics#cat.png")

    await resolve_fileset_images(body, workspace="req-ws", request_nemo_client=_client())

    assert _urls(body)[:3] == list(urls)
    assert _urls(body)[3].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_resolve_fileset_images_without_fileset_refs_needs_no_client():
    body = _body("data:image/png;base64,iVBORw0KGgo")
    body["messages"].append({"role": "user", "content": "plain text"})

    await resolve_fileset_images(body, workspace="req-ws", request_nemo_client=None)

    assert _urls(body) == ["data:image/png;base64,iVBORw0KGgo"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "status_code", "client_kwargs"),
    [
        ("file:///etc/passwd", 400, {}),
        ("/home/me/cat.png", 400, {}),
        ("cat.png", 400, {}),
        ("C:\\images\\cat.png", 400, {}),
        ("fileset://img-ws/pics#notes.txt", 400, {}),
        ("fileset://img-ws/pics#noextension", 400, {}),
        ("fileset://img-ws/pics", 400, {}),
        ("fileset://img-ws/pics#../../etc/passwd png", 400, {}),
        ("fileset://img-ws/pics#cat.png", 413, {"content": b"x" * (MAX_IMAGE_BYTES + 1)}),
        ("fileset://img-ws/pics#cat.png", 404, {"status_code": 404}),
        ("fileset://img-ws/pics#cat.png", 403, {"status_code": 403}),
        ("fileset://img-ws/pics#cat.png", 502, {"status_code": 500}),
        ("fileset://img-ws/pics#cat.png", 502, {"unreachable": True}),
    ],
)
async def test_resolve_fileset_images_rejects(url: str, status_code: int, client_kwargs: dict[str, Any]):
    with pytest.raises(HTTPException) as exc_info:
        await resolve_fileset_images(_body(url), workspace="req-ws", request_nemo_client=_client(**client_kwargs))

    assert exc_info.value.status_code == status_code
    assert "base64" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_resolve_fileset_images_refuses_without_caller_client():
    with pytest.raises(HTTPException) as exc_info:
        await resolve_fileset_images(
            _body("fileset://img-ws/pics#cat.png"), workspace="req-ws", request_nemo_client=None
        )

    assert exc_info.value.status_code == 500
