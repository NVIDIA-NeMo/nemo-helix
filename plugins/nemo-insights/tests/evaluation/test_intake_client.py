# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The basic-auth intake client builder: path rewrite + basic auth."""

import base64

import httpx
import pytest
from evaluation import intake_client as intake_client_module
from evaluation.intake_client import build_basic_auth_intake_client, build_rewriting_http_client
from nemo_helix import AsyncNeMoHelix


async def test_rewrites_sdk_prefix_and_attaches_basic_auth() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    client = build_rewriting_http_client(
        username="intake",
        password="secret",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.get("https://agenthub.aire.nvidia.com/apis/intake/v2/workspaces/default/spans?page=1")
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert seen["url"] == "https://agenthub.aire.nvidia.com/api/intake/v2/workspaces/default/spans?page=1"
    assert seen["auth"] == f"Basic {base64.b64encode(b'intake:secret').decode()}"


async def test_rewriting_http_client_uses_60_second_timeout() -> None:
    client = build_rewriting_http_client(username="u", password="p")
    try:
        assert client.timeout == httpx.Timeout(60.0)
    finally:
        await client.aclose()


async def test_leaves_non_intake_paths_untouched() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    client = build_rewriting_http_client(
        username="u",
        password="p",
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.get("https://agenthub.aire.nvidia.com/other/path")
    finally:
        await client.aclose()

    assert seen["url"] == "https://agenthub.aire.nvidia.com/other/path"


async def test_build_basic_auth_intake_client_returns_sdk_client() -> None:
    client = build_basic_auth_intake_client(
        base_url="https://agenthub.aire.nvidia.com",
        username="u",
        password="p",
    )
    try:
        assert isinstance(client, AsyncNeMoHelix)
    finally:
        await client.close()


async def test_sdk_client_closes_its_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    monkeypatch.setattr(
        intake_client_module,
        "build_rewriting_http_client",
        lambda **kwargs: http_client,
    )
    client = build_basic_auth_intake_client(
        base_url="https://agenthub.aire.nvidia.com",
        username="u",
        password="p",
    )

    try:
        assert not http_client.is_closed
        await client.close()
        assert http_client.is_closed
    finally:
        if not http_client.is_closed:
            await http_client.aclose()
