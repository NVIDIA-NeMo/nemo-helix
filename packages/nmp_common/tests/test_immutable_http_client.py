# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType

import httpx
import pytest
from nmp.common.immutable_http_client import ImmutableHttpClientMixin


def _noop_request_hook(request: httpx.Request) -> None:
    pass


class _FrozenClient(ImmutableHttpClientMixin, httpx.Client):
    def __init__(self) -> None:
        super().__init__(
            headers={"X-Initial": "true"},
            event_hooks={"request": [_noop_request_hook]},
        )
        self._freeze_http_client()


class _FrozenAsyncClient(ImmutableHttpClientMixin, httpx.AsyncClient):
    def __init__(self) -> None:
        super().__init__(
            headers={"X-Initial": "true"},
            event_hooks={"request": [_noop_request_hook]},
        )
        self._freeze_http_client()


def test_immutable_sdk_client_still_builds_requests() -> None:
    with _FrozenClient() as client:
        request = client.build_request("GET", "http://nmp.example.test/health")

    assert request.url == "http://nmp.example.test/health"
    assert request.headers["X-Initial"] == "true"


def test_immutable_sdk_client_blocks_client_configuration_assignment() -> None:
    with _FrozenClient() as client:
        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client.headers = httpx.Headers({"Authorization": "Bearer stale"})

        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client.base_url = httpx.URL("http://other.example.test")

        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client.params = {"debug": "true"}


def test_immutable_sdk_client_blocks_transport_assignment() -> None:
    with _FrozenClient() as client:
        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client._transport = httpx.HTTPTransport()

        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client._mounts = {}

        assert isinstance(client._mounts, MappingProxyType)


@pytest.mark.asyncio
async def test_immutable_async_sdk_client_blocks_transport_assignment() -> None:
    async with _FrozenAsyncClient() as client:
        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client._transport = httpx.AsyncHTTPTransport()

        with pytest.raises(AttributeError, match="SDK HTTP clients are immutable"):
            client._mounts = {}

        assert isinstance(client._mounts, MappingProxyType)


def test_immutable_sdk_client_blocks_header_mutation() -> None:
    with _FrozenClient() as client:
        with pytest.raises(TypeError, match="SDK HTTP clients are immutable"):
            client.headers["Authorization"] = "Bearer stale"

        with pytest.raises(TypeError, match="SDK HTTP clients are immutable"):
            client.headers.update({"Authorization": "Bearer stale"})

        with pytest.raises(TypeError, match="SDK HTTP clients are immutable"):
            client.headers.pop("X-Initial")


def test_immutable_sdk_client_blocks_cookie_mutation_and_ignores_response_cookies() -> None:
    with _FrozenClient() as client:
        with pytest.raises(TypeError, match="SDK HTTP clients are immutable"):
            client.cookies["session"] = "stale"

        request = client.build_request("GET", "http://nmp.example.test/health")
        response = httpx.Response(200, headers={"Set-Cookie": "session=stale"}, request=request)

        client.cookies.extract_cookies(response)

        assert "session" not in client.cookies


def test_immutable_sdk_client_blocks_event_hook_mutation() -> None:
    with _FrozenClient() as client:
        with pytest.raises(AttributeError):
            client.event_hooks["request"].append(_noop_request_hook)

        with pytest.raises(TypeError):
            client.event_hooks["request"] = [_noop_request_hook]
