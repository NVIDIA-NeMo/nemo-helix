# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for IAMClient and AsyncIAMClient with mocked HTTP transports."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.client.types import BinaryContent
from nemo_platform_plugin.iam.client import AsyncIAMClient, IAMClient
from nemo_platform_plugin.iam.types import AuthzRequest, RoleBindingInput

BASE = "http://test:8000"


@get("/ordinary-binary")
def ORDINARY_BINARY() -> BinaryContent:
    raise NotImplementedError


BINDING = {
    "id": "id-1",
    "name": "rb-123",
    "principal": "user@example.com",
    "workspace": "default",
    "role": "Viewer",
    "granted_by": "service:test",
    "granted_at": "2026-01-01T00:00:00Z",
    "revoked_at": None,
}


def test_sync_role_binding_and_authz_responses() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("POST", f"{BASE}/apis/auth/v2/iam/role-bindings"), json=BINDING),
        httpx.Response(
            200,
            request=httpx.Request("POST", f"{BASE}/apis/auth/v2/authz/allow"),
            json={"result": {"allowed": True}},
        ),
    ]
    client = IAMClient(base_url=BASE, http_client=mock_http)

    created = client.create_role_binding(
        body=RoleBindingInput(principal="user@example.com", workspace="default", role="Viewer")
    ).data()
    decision = client.evaluate_authorization(
        entrypoint="allow", body=AuthzRequest(input={"principal_id": "user@example.com"})
    ).data()

    assert created.name == "rb-123"
    assert decision.result == {"allowed": True}
    assert mock_http.request.call_args_list[0].kwargs["params"] is None


def test_role_binding_pagination_dtos_and_filter_encoding() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/auth/v2/iam/role-bindings"),
        json={
            "data": [BINDING],
            "pagination": {
                "page": 1,
                "page_size": 10,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
            "sort": "created_at",
            "filter": {"principal": {"$like": "user"}},
        },
    )
    client = IAMClient(base_url=BASE, http_client=mock_http)

    response = client.list_role_bindings(query_params={"filter[principal][$like]": "user"})
    page = response.page()

    assert [binding.name for binding in page.items] == ["rb-123"]
    assert page.metadata["page_size"] == 10
    assert mock_http.request.call_args.kwargs["params"] == {"filter[principal][$like]": "user"}


def test_binary_bundle_read_preserves_headers_and_304() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("If-None-Match") == '"etag-1"':
            return httpx.Response(304, headers={"ETag": '"etag-1"'}, request=request)
        return httpx.Response(
            200,
            content=b"bundle-bytes",
            headers={"Content-Type": "application/gzip", "ETag": '"etag-1"'},
            request=request,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = IAMClient(base_url=BASE, http_client=http_client)

    response = client.get_opa_bundle()
    assert response.read() == b"bundle-bytes"
    assert response.http_response.status_code == 200
    assert response.http_response.headers["ETag"] == '"etag-1"'

    not_modified = client.with_headers({"If-None-Match": '"etag-1"'}).get_opa_bundle()
    assert not_modified.read() == b""
    assert not_modified.http_response.status_code == 304
    assert not_modified.http_response.headers["ETag"] == '"etag-1"'


@pytest.mark.parametrize("status_code", [304, 400, 500])
def test_sync_ordinary_binary_unexpected_status_raises(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "failed"}, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = IAMClient(base_url=BASE, http_client=http_client)

    with pytest.raises(NemoHTTPError) as exc_info:
        client.send(ORDINARY_BINARY()).read()

    assert exc_info.value.status_code == status_code


@pytest.mark.asyncio
async def test_async_iam_request_dispatch_parity() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.side_effect = [
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/auth/v2/iam/role-bindings"),
            json={
                "data": [BINDING],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
                "sort": "created_at",
                "filter": {},
            },
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", f"{BASE}/apis/auth/v2/iam/role-bindings"),
            json=BINDING,
        ),
        httpx.Response(
            200,
            request=httpx.Request("DELETE", f"{BASE}/apis/auth/v2/iam/role-bindings/rb-123"),
            json={"message": "Resource deleted successfully.", "id": "id-1"},
        ),
        httpx.Response(
            200,
            request=httpx.Request("POST", f"{BASE}/apis/auth/v2/authz/allow"),
            json={"result": {"allowed": True}},
        ),
    ]
    client = AsyncIAMClient(base_url=BASE, http_client=mock_http)
    body = RoleBindingInput(principal="user@example.com", workspace="default", role="Viewer")

    listed = await client.list_role_bindings(query_params={"filter[role]": "Viewer"})
    created = await client.create_role_binding(body=body)
    revoked = await client.revoke_role_binding(name="rb-123")
    decision = await client.evaluate_authorization(
        entrypoint="allow", body=AuthzRequest(input={"principal_id": "user@example.com"})
    )

    assert listed.page().items[0].name == "rb-123"
    assert created.data().name == "rb-123"
    assert revoked.data().id == "id-1"
    assert decision.data().result == {"allowed": True}

    calls = mock_http.request.call_args_list
    assert [(call.args[0], call.args[1]) for call in calls] == [
        ("GET", f"{BASE}/apis/auth/v2/iam/role-bindings"),
        ("POST", f"{BASE}/apis/auth/v2/iam/role-bindings"),
        ("DELETE", f"{BASE}/apis/auth/v2/iam/role-bindings/rb-123"),
        ("POST", f"{BASE}/apis/auth/v2/authz/allow"),
    ]
    assert calls[0].kwargs["params"] == {"filter[role]": "Viewer"}
    assert calls[1].kwargs["params"] is None
    assert json.loads(calls[1].kwargs["content"]) == body.model_dump()
    assert calls[2].kwargs["params"] is None
    assert json.loads(calls[3].kwargs["content"]) == {"input": {"principal_id": "user@example.com"}}


@pytest.mark.asyncio
async def test_async_binary_bundle_read_preserves_headers_and_304() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("If-None-Match") == '"etag-1"':
            return httpx.Response(304, headers={"ETag": '"etag-1"'}, request=request)
        return httpx.Response(
            200,
            content=b"bundle-bytes",
            headers={"Content-Type": "application/gzip", "ETag": '"etag-1"'},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIAMClient(base_url=BASE, http_client=http_client)

        response = await client.get_opa_bundle()
        assert await response.read() == b"bundle-bytes"
        assert response.http_response.status_code == 200
        assert response.http_response.headers["ETag"] == '"etag-1"'

        not_modified = await client.with_headers({"If-None-Match": '"etag-1"'}).get_opa_bundle()
        assert await not_modified.read() == b""
        assert not_modified.http_response.status_code == 304
        assert not_modified.http_response.headers["ETag"] == '"etag-1"'


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [304, 400, 500])
async def test_async_ordinary_binary_unexpected_status_raises(status_code: int) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": "failed"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncIAMClient(base_url=BASE, http_client=http_client)

        with pytest.raises(NemoHTTPError) as exc_info:
            response = await client.send(ORDINARY_BINARY())
            await response.read()

    assert exc_info.value.status_code == status_code
