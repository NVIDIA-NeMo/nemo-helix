# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpcore
import httpx
import pytest
from fastapi import FastAPI
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.auth import AuthClient, Principal
from nhx.common.config import Configuration
from nhx.common.service import DependencyProvider
from nhx.common.service.dependencies import (
    get_effective_principal_id,
    get_entity_client,
    get_platform_config,
    get_sdk_client,
)


def test_get_http_client_caches_runtime_context_endpoint_client() -> None:
    provider = DependencyProvider()
    client = MagicMock()
    runtime_context = MagicMock()
    runtime_context.endpoint.async_sdk_http_client.return_value = client

    with patch(
        "nhx.common.service.base.build_platform_runtime_context",
        return_value=runtime_context,
    ) as build_runtime_context:
        first = provider.get_http_client()
        second = provider.get_http_client()

    assert first is client
    assert second is client
    build_runtime_context.assert_called_once_with(platform_config=provider.get_platform_config())
    runtime_context.endpoint.async_sdk_http_client.assert_called_once_with()


def _uds_of(client: httpx.AsyncClient) -> str | None:
    """Socket path bound to the client's transport pool, or None for TCP."""
    transport = client._transport
    if not isinstance(transport, httpx.AsyncHTTPTransport):
        return None
    pool = transport._pool
    if not isinstance(pool, httpcore.AsyncConnectionPool):
        return None
    return pool._uds


def test_get_http_client_binds_uds_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a unix:// endpoint the provider-owned client must be socket-bound.

    Regression: the provider used to build a plain TCP DefaultAsyncHttpxClient
    and inject it into the SDK/NemoClient factories, which then skipped their
    own UDS selection, so service-to-service calls went to http://nemo-helix
    .local over TCP (_uds=None) and failed.
    """
    monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
    Configuration.clear_cache()
    try:
        provider = DependencyProvider()
        assert _uds_of(provider.get_http_client()) == "/tmp/nemo-helix.sock"
    finally:
        Configuration.clear_cache()


def test_request_scoped_nemo_client_binds_uds_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")
    Configuration.clear_cache()
    try:
        provider = DependencyProvider()
        client = provider.get_request_scoped_nemo_client()
        assert _uds_of(client._http) == "/tmp/nemo-helix.sock"
    finally:
        Configuration.clear_cache()


def test_tcp_endpoint_client_is_not_socket_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://platform:8080")
    Configuration.clear_cache()
    try:
        provider = DependencyProvider()
        assert _uds_of(provider.get_http_client()) is None
    finally:
        Configuration.clear_cache()


def test_get_sdk_client_caches_request_sdk_and_creates_fresh_service_sdk() -> None:
    provider = DependencyProvider()
    request_sdk = MagicMock(name="request_sdk")
    service_sdk = MagicMock(name="service_sdk")

    with patch("nhx.common.sdk_factory.get_async_platform_sdk", side_effect=[request_sdk, service_sdk]) as factory:
        assert provider.get_sdk_client() is request_sdk
        assert provider.get_sdk_client() is request_sdk
        assert provider.get_service_sdk_client("jobs") is service_sdk

    # The provider now shares its pooled HTTP client with every SDK it builds.
    http_client = provider.get_http_client()
    assert factory.call_args_list[0].kwargs == {"http_client": http_client}
    assert factory.call_args_list[1].kwargs == {
        "as_service": "jobs",
        "internal": True,
        "http_client": http_client,
    }


def test_get_sync_sdk_client_caches_request_sdk_and_creates_fresh_service_sdk() -> None:
    provider = DependencyProvider()
    request_sdk = MagicMock(name="request_sdk")
    service_sdk = MagicMock(name="service_sdk")
    http_client = MagicMock(name="http_client")

    with (
        patch.object(provider, "get_sync_http_client", return_value=http_client) as get_sync_http_client,
        patch("nhx.common.sdk_factory.get_platform_sdk", side_effect=[request_sdk, service_sdk]) as factory,
    ):
        assert provider.get_sync_sdk_client() is request_sdk
        assert provider.get_sync_sdk_client() is request_sdk
        assert provider.get_service_sync_sdk_client("jobs") is service_sdk

    assert get_sync_http_client.call_count == 2
    assert factory.call_args_list[0].kwargs == {"http_client": http_client}
    assert factory.call_args_list[1].kwargs == {
        "as_service": "jobs",
        "internal": True,
        "http_client": http_client,
    }


def test_setup_dependencies_registers_fastapi_overrides() -> None:
    provider = DependencyProvider()
    app = FastAPI()
    service = MagicMock()
    service._service_config = None

    provider.setup_dependencies(app, service)

    assert app.dependency_overrides[get_sdk_client] == provider.get_request_scoped_sdk
    assert app.dependency_overrides[get_entity_client] == provider.get_entity_client
    assert app.dependency_overrides[get_effective_principal_id] == provider.get_effective_principal_id
    assert app.dependency_overrides[get_platform_config] == provider.get_platform_config


def test_get_effective_principal_id_uses_delegated_identity() -> None:
    provider = DependencyProvider()
    request = MagicMock()
    auth_client = MagicMock(spec=AuthClient)
    auth_client.principal = Principal(
        id="service:agents",
        email=None,
        on_behalf_of="session-owner",
        on_behalf_of_email=None,
    )

    with patch("nhx.common.auth.get_auth_client", return_value=auth_client):
        assert provider.get_effective_principal_id(request) == "session-owner"


@pytest.mark.asyncio
async def test_close_closes_managed_clients_and_clears_references() -> None:
    provider = DependencyProvider()
    http_client = MagicMock()
    http_client.aclose = AsyncMock()
    sdk = MagicMock()
    sdk.close = AsyncMock()
    provider._http_client = http_client
    provider._sdk_client = sdk

    await provider.close()

    # close() owns and closes the shared HTTP transport; the SDK borrows that
    # transport, so it is dropped without a separate sdk.close().
    http_client.aclose.assert_awaited_once_with()
    sdk.close.assert_not_awaited()
    assert provider._http_client is None
    assert provider._sdk_client is None


def test_get_service_entity_client_uses_service_nemo_client() -> None:
    provider = DependencyProvider()
    nemo_client = MagicMock(name="service_nemo_client")
    http_client = MagicMock(name="http_client")
    entities_client = MagicMock(name="entities_client")
    entity_client = MagicMock(name="entity_client")

    with (
        patch.object(provider, "get_http_client", return_value=http_client) as get_http_client,
        patch("nhx.common.client_factory.get_async_nemo_client", return_value=nemo_client) as get_nemo_client,
        patch(
            "nemo_helix_plugin.client.adapter.client_from_platform",
            return_value=entities_client,
        ) as adapter,
        patch("nhx.common.entities.client.EntityClient", return_value=entity_client) as client_factory,
    ):
        result = provider.get_service_entity_client("models")

    assert result is entity_client
    get_http_client.assert_called_once_with()
    get_nemo_client.assert_called_once_with(
        as_service="models",
        internal=True,
        on_behalf_of=None,
        http_client=http_client,
    )
    adapter.assert_called_once()
    assert adapter.call_args.args == (nemo_client, AsyncEntitiesClient)
    client_factory.assert_called_once_with(entities_client)
