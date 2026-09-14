# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx
import pytest
from nemo_platform_plugin.auth.client import AsyncAuthenticationClient, AuthenticationClient
from nemo_platform_plugin.client.errors import AuthenticationError

BASE = "http://test:8000"
AUTH_RESPONSE = {
    "principal": "alice@example.com",
    "email": "alice@example.com",
    "groups": ["engineering"],
    "scopes": ["read"],
    "jti": "token-id",
    "token_kind": "oidc_access_token",
    "on_behalf_of": None,
    "on_behalf_of_email": None,
    "on_behalf_of_groups": [],
}


def test_sync_authentication_client_dispatches_get_with_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=AUTH_RESPONSE, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AuthenticationClient(base_url=BASE, auth="tok", http_client=http_client)

    result = client.authenticate_bearer_token_get().data()

    assert result.principal == "alice@example.com"
    assert result.token_kind == "oidc_access_token"
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/apis/auth/authenticate"
    assert seen[0].headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_async_authentication_client_dispatches_post_with_auth_header() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=AUTH_RESPONSE, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncAuthenticationClient(base_url=BASE, auth="tok", http_client=http_client)

        response = await client.authenticate_bearer_token_post()

    result = response.data()
    assert result.principal == "alice@example.com"
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/apis/auth/authenticate"
    assert seen[0].headers["Authorization"] == "Bearer tok"


def test_authentication_client_raises_status_specific_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid bearer token"}, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = AuthenticationClient(base_url=BASE, auth="tok", http_client=http_client)

    with pytest.raises(AuthenticationError, match="Invalid bearer token"):
        client.authenticate_bearer_token_get()
