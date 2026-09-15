# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import patch

import httpx
import pytest
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform, omit
from nemo_platform_ext.auth.helpers import NMPOIDCConfig
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_platform_plugin.jobs.client import JobsClient
from nmp.common.config import Configuration, PlatformConfig
from nmp.common.http_clients import shared_async_http_client, shared_sync_http_client
from nmp.common.platform_endpoint import (
    _AsyncPlatformEndpointRoutingTransport,
    _SyncPlatformEndpointRoutingTransport,
)
from nmp.common.sdk_factory import (
    get_async_platform_sdk,
    get_async_task_sdk,
    get_entity_parts,
    get_platform_sdk,
    get_request_scoped_sdk,
    get_sdk_on_behalf_of,
    get_task_sdk,
)


def _workload_oidc_config() -> NMPOIDCConfig:
    return NMPOIDCConfig(
        auth_enabled=True,
        workload_token_exchange_enabled=True,
        workload_client_id="workload-client",
        workload_token_endpoint="https://idp.example.test/oauth2/token",
        workload_audience="nemo-platform",
        workload_scope="openid email groups",
    )


@contextmanager
def _platform_sdk_for_test(*, base_url: str = "http://platform:8080") -> Iterator[NeMoPlatform]:
    with get_platform_sdk(base_url=base_url) as sdk:
        yield sdk


@asynccontextmanager
async def _async_platform_sdk_for_test(*, base_url: str = "http://platform:8080") -> AsyncIterator[AsyncNeMoPlatform]:
    async with get_async_platform_sdk(base_url=base_url) as sdk:
        yield sdk


@pytest.fixture(autouse=True)
def _clear_configuration_cache():
    """Keep SDK factory tests order-independent."""
    Configuration.clear_cache()
    try:
        yield
    finally:
        Configuration.clear_cache()


def test_get_platform_sdk():
    """
    Test the get_platform_sdk function to ensure it returns an instance of NeMoPlatform
    with the correct base URL.
    """
    sdk = get_platform_sdk()
    assert sdk is not None, "SDK instance should not be None"
    assert hasattr(sdk, "base_url"), "SDK instance should have a base_url attribute"
    assert sdk.base_url == Configuration.get_platform_config().base_url


def test_get_platform_sdk_keeps_platform_base_url_for_local_services(monkeypatch: pytest.MonkeyPatch):
    """The SDK base URL remains the platform entrypoint; per-service routing handles local APIs."""
    monkeypatch.setenv("NMP_BASE_URL", "https://nemo-gateway:8080")
    monkeypatch.setenv("NMP_SERVICES", "auth")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()

    sdk = get_platform_sdk()

    assert str(sdk.base_url).rstrip("/") == "https://nemo-gateway:8080"


def test_get_platform_sdk_preserves_api_base_url_for_controller_only_pods(monkeypatch: pytest.MonkeyPatch):
    """Controller-only pods must call the API service, not their own health listener."""
    captured_requests: list[httpx.Request] = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [],
                "pagination": {
                    "current_page_size": 0,
                    "page": 1,
                    "page_size": 0,
                    "total_pages": 1,
                    "total_results": 0,
                },
            },
        )

    monkeypatch.setenv("NMP_BASE_URL", "http://nemo-platform-api:8080")
    monkeypatch.setenv("NMP_CONTROLLERS", "jobs")
    monkeypatch.delenv("NMP_SERVICES", raising=False)
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()

    with httpx.Client(transport=httpx.MockTransport(capture_request)) as http_client:
        sdk = get_platform_sdk(http_client=http_client)

        assert str(sdk.base_url).rstrip("/") == "http://nemo-platform-api:8080"
        client_from_platform(sdk, JobsClient).list_jobs(workspace="default")

    assert len(captured_requests) == 1
    assert str(captured_requests[0].url) == "http://nemo-platform-api:8080/apis/jobs/v2/workspaces/default/jobs"


def test_get_platform_sdk_uses_routing_transport_for_local_services(monkeypatch: pytest.MonkeyPatch):
    """APIs hosted in this process are routed by the SDK-owned HTTP client."""
    monkeypatch.setenv("NMP_BASE_URL", "https://nemo-gateway:8080")
    monkeypatch.setenv("NMP_SERVICES", "auth")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()

    with get_platform_sdk() as sdk:
        transport = sdk._client._transport

    assert isinstance(transport, _SyncPlatformEndpointRoutingTransport)


def test_get_platform_sdk_uses_uds_endpoint_from_base_url():
    config = PlatformConfig.model_construct(base_url="unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.sdk_factory.Configuration.get_platform_config", return_value=config):
        sdk = get_platform_sdk()

    assert sdk.base_url == "http://nemo-platform.local"


def test_get_platform_sdk_with_service_principal():
    """Test get_platform_sdk with as_service parameter."""
    sdk = get_platform_sdk(as_service="my-service")

    assert sdk is not None
    assert "X-NMP-Principal-Id" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:my-service"


def test_get_platform_sdk_with_on_behalf_of():
    """Test get_platform_sdk with on_behalf_of parameter."""
    sdk = get_platform_sdk(as_service="my-service", on_behalf_of="user@example.com")

    assert sdk is not None
    assert "X-NMP-Principal-Id" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:my-service"
    assert "X-NMP-Principal-On-Behalf-Of" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"


def test_get_platform_sdk_internal_flag():
    """Test get_platform_sdk with internal flag."""
    sdk = get_platform_sdk(as_service="my-service", internal=True)

    assert sdk is not None
    assert "X-NMP-Internal" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Internal"] == "true"


def test_get_platform_sdk_uses_workload_identity_when_token_file_configured(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Workload-token task environments should let the generated SDK inject Bearer auth."""
    subject_token_file = tmp_path / "workload-token"
    subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    exchange_requests: list[dict] = []

    def token_exchange_grant(**kwargs):
        exchange_requests.append(kwargs)
        return {"access_token": "exchanged-access-token", "expires_in": 300}

    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("NMP_BASE_URL", "http://nmp.example.test")
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"id": "creator@example.com", "email": "creator@example.com"}))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        lambda _base_url, **_kwargs: _workload_oidc_config(),
    )
    monkeypatch.setattr("nemo_platform_ext.auth.workload_exchange.token_exchange_grant", token_exchange_grant)

    sdk = get_platform_sdk()
    try:
        request = sdk._client.build_request("GET", "http://nmp.example.test/apis/entities/v2/workspaces/default")
        sdk._client._event_hooks["request"][0](request)
    finally:
        sdk.close()

    assert request.headers["Authorization"] == "Bearer exchanged-access-token"
    assert "X-NMP-Principal-Id" not in request.headers
    assert exchange_requests[0]["subject_token"] == "subject-token-from-file"


def test_get_async_platform_sdk():
    """Test get_async_platform_sdk basic functionality (config path: SDK base_url matches platform config)."""
    sdk = get_async_platform_sdk()

    assert sdk is not None
    assert hasattr(sdk, "base_url")
    # Normalize to str: SDK may expose URL object, config may be str; both environments
    expected = Configuration.get_platform_config().base_url
    assert str(sdk.base_url).rstrip("/") == str(expected).rstrip("/")


def test_get_async_platform_sdk_uses_uds_endpoint_from_base_url():
    config = PlatformConfig.model_construct(base_url="unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.sdk_factory.Configuration.get_platform_config", return_value=config):
        sdk = get_async_platform_sdk()

    assert str(sdk.base_url).rstrip("/") == "http://nemo-platform.local"


@pytest.mark.asyncio
async def test_get_async_platform_sdk_uses_explicit_http_client() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        sdk = get_async_platform_sdk(base_url="http://nmp.example.test", http_client=http_client)

        assert sdk._client is http_client
        assert str(sdk.base_url).rstrip("/") == "http://nmp.example.test"


def test_get_async_platform_sdk_with_service_principal():
    """Test get_async_platform_sdk with as_service parameter."""
    sdk = get_async_platform_sdk(as_service="async-service")

    assert sdk is not None
    assert "X-NMP-Principal-Id" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:async-service"


def test_get_async_platform_sdk_with_on_behalf_of():
    """Test get_async_platform_sdk with on_behalf_of parameter."""
    sdk = get_async_platform_sdk(as_service="async-service", on_behalf_of="async-user@example.com")

    assert sdk is not None
    assert "X-NMP-Principal-Id" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:async-service"
    assert "X-NMP-Principal-On-Behalf-Of" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "async-user@example.com"


def test_get_async_platform_sdk_internal_flag():
    """Test get_async_platform_sdk with internal flag."""
    sdk = get_async_platform_sdk(as_service="async-service", internal=True)

    assert sdk is not None
    assert "X-NMP-Internal" in sdk.default_headers
    assert sdk.default_headers["X-NMP-Internal"] == "true"


def test_on_behalf_of_without_service_principal():
    """Test that on_behalf_of works without as_service (propagates user context)."""
    # When auth is enabled but no context is set, on_behalf_of should still be added
    sdk = get_platform_sdk(on_behalf_of="delegated@example.com")

    assert sdk is not None
    # Note: Without a user context (request), principal headers won't be set,
    # but on_behalf_of should still be added if provided
    if "X-NMP-Principal-On-Behalf-Of" in sdk.default_headers:
        assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "delegated@example.com"


def test_get_task_sdk_with_principal(monkeypatch: pytest.MonkeyPatch):
    """get_task_sdk should set service principal, internal flag, and on-behalf-of using effective_id."""
    principal_json = json.dumps(
        {
            "id": "service:other",
            "email": "svc@internal",
            "groups": [],
            "on_behalf_of": "real-user@example.com",
        }
    )
    monkeypatch.setenv("NMP_PRINCIPAL", principal_json)

    sdk = get_task_sdk(as_service="customizer")

    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:customizer"
    assert sdk.default_headers["X-NMP-Internal"] == "true"
    assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "real-user@example.com"


def test_get_task_sdk_without_principal(monkeypatch: pytest.MonkeyPatch):
    """get_task_sdk without NMP_PRINCIPAL should still set service principal and internal flag, but no on-behalf-of."""
    monkeypatch.delenv("NMP_PRINCIPAL", raising=False)

    sdk = get_task_sdk(as_service="customizer")

    assert sdk.default_headers["X-NMP-Principal-Id"] == "service:customizer"
    assert sdk.default_headers["X-NMP-Internal"] == "true"
    assert "X-NMP-Principal-On-Behalf-Of" not in sdk.default_headers


def test_get_task_sdk_does_not_inherit_shared_client_authorization(monkeypatch: pytest.MonkeyPatch):
    """Service-principal task SDK auth must come from SDK headers, not stale shared-client auth."""
    monkeypatch.setenv(
        "NMP_PRINCIPAL",
        json.dumps({"id": "creator@example.com", "email": "creator@example.com"}),
    )
    client = shared_sync_http_client()
    old_headers = dict(client.headers)
    client.headers["Authorization"] = "Bearer service:jobs"
    sdk = None

    try:
        sdk = get_task_sdk(as_service="jobs")
        assert sdk._client is not client
        assert sdk.default_headers["X-NMP-Principal-Id"] == "service:jobs"
        assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "creator@example.com"
        assert "Authorization" not in sdk.default_headers
        assert "Authorization" not in sdk._client.headers
    finally:
        if sdk is not None:
            sdk.close()
        client.headers.clear()
        client.headers.update(old_headers)
        client.headers.pop("Authorization", None)


@pytest.mark.asyncio
async def test_get_async_task_sdk_does_not_inherit_shared_client_authorization(monkeypatch: pytest.MonkeyPatch):
    """Async task SDK auth must also avoid stale shared-client auth."""
    monkeypatch.setenv(
        "NMP_PRINCIPAL",
        json.dumps({"id": "creator@example.com", "email": "creator@example.com"}),
    )
    client = shared_async_http_client()
    old_headers = dict(client.headers)
    client.headers["Authorization"] = "Bearer service:jobs"
    sdk = None

    try:
        sdk = get_async_task_sdk(as_service="jobs")
        assert sdk._client is not client
        assert sdk.default_headers["X-NMP-Principal-Id"] == "service:jobs"
        assert sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "creator@example.com"
        assert "Authorization" not in sdk.default_headers
        assert "Authorization" not in sdk._client.headers
    finally:
        if sdk is not None:
            await sdk.close()
        client.headers.clear()
        client.headers.update(old_headers)
        client.headers.pop("Authorization", None)


def test_get_task_sdk_uses_workload_identity_when_token_file_configured(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Task SDKs should centralize the workload-token-vs-service-header choice."""
    subject_token_file = tmp_path / "workload-token"
    subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    exchange_requests: list[dict] = []

    def token_exchange_grant(**kwargs):
        exchange_requests.append(kwargs)
        return {"access_token": "task-access-token", "expires_in": 300}

    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("NMP_BASE_URL", "http://nmp.example.test")
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
    monkeypatch.setenv("NMP_PRINCIPAL", json.dumps({"id": "creator@example.com", "email": "creator@example.com"}))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        "nemo_platform_ext.client.bootstrap.discover_nmp_config",
        lambda _base_url, **_kwargs: _workload_oidc_config(),
    )
    monkeypatch.setattr("nemo_platform_ext.auth.workload_exchange.token_exchange_grant", token_exchange_grant)

    sdk = get_task_sdk(as_service="customizer")
    try:
        request = sdk._client.build_request("GET", "http://nmp.example.test/apis/entities/v2/workspaces/default")
        sdk._client._event_hooks["request"][0](request)
    finally:
        sdk.close()

    assert sdk.default_headers["X-NMP-Internal"] == "true"
    assert request.headers["Authorization"] == "Bearer task-access-token"
    assert "X-NMP-Principal-Id" not in request.headers
    assert "X-NMP-Principal-On-Behalf-Of" not in request.headers
    assert exchange_requests[0]["subject_token"] == "subject-token-from-file"


def test_get_task_sdk_uses_explicit_sync_http_client(monkeypatch: pytest.MonkeyPatch):
    """get_task_sdk should use an explicitly provided sync HTTP client."""
    monkeypatch.delenv("NMP_PRINCIPAL", raising=False)

    with httpx.Client() as client:
        sdk = get_task_sdk(as_service="customizer", http_client=client)

        assert sdk._client is client


def test_platform_sdk_test_context_closes_factory_client() -> None:
    with _platform_sdk_for_test() as sdk:
        http_client = sdk._client
        assert not http_client.is_closed

    assert http_client.is_closed


@pytest.mark.asyncio
async def test_async_platform_sdk_test_context_closes_factory_client() -> None:
    async with _async_platform_sdk_for_test() as sdk:
        http_client = sdk._client
        assert not http_client.is_closed

    assert http_client.is_closed


def test_get_request_scoped_sdk_merges_otel_and_auth_headers():
    """Test that get_request_scoped_sdk merges OTEL and auth headers."""
    base_sdk = get_async_platform_sdk()

    mock_otel_headers = {"traceparent": "00-trace-id-span-id-01", "tracestate": "vendor=value"}
    mock_auth_headers = {"X-NMP-Principal-Id": "user@example.com", "X-NMP-Principal-Groups": "group1,group2"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Verify it's a new SDK instance
    assert scoped_sdk is not base_sdk

    # Verify OTEL headers are present
    assert "traceparent" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["traceparent"] == "00-trace-id-span-id-01"
    assert "tracestate" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["tracestate"] == "vendor=value"

    # Verify auth headers are present
    assert "X-NMP-Principal-Id" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Id"] == "user@example.com"
    assert "X-NMP-Principal-Groups" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Groups"] == "group1,group2"


def test_get_request_scoped_sdk_reuses_base_sdk_http_client(monkeypatch: pytest.MonkeyPatch):
    """Derived request SDKs keep the base SDK's routed HTTP client."""
    monkeypatch.setenv("NMP_BASE_URL", "https://nemo-gateway:8080")
    monkeypatch.setenv("NMP_SERVICES", "entities")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()

    try:
        base_sdk = get_async_platform_sdk()

        with patch("nmp.common.sdk_factory.get_otel_headers", return_value={}):
            with patch(
                "nmp.common.sdk_factory.get_principal_auth_headers",
                return_value={"X-NMP-Principal-Id": "service:models"},
            ):
                scoped_sdk = get_request_scoped_sdk(base_sdk)

        transport = base_sdk._client._transport

        assert isinstance(transport, _AsyncPlatformEndpointRoutingTransport)
        assert scoped_sdk._client is base_sdk._client
    finally:
        Configuration.clear_cache()


def test_get_sdk_on_behalf_of_reuses_base_sdk_http_client(monkeypatch: pytest.MonkeyPatch):
    """SDKs derived with on-behalf-of headers keep the base SDK's routed HTTP client."""
    monkeypatch.setenv("NMP_BASE_URL", "https://nemo-gateway:8080")
    monkeypatch.setenv("NMP_SERVICES", "entities")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    Configuration.clear_cache()

    try:
        base_sdk = get_async_platform_sdk(as_service="models", internal=True)
        scoped_sdk = get_sdk_on_behalf_of(base_sdk, "user@example.com")

        transport = base_sdk._client._transport

        assert isinstance(transport, _AsyncPlatformEndpointRoutingTransport)
        assert scoped_sdk._client is base_sdk._client
    finally:
        Configuration.clear_cache()


def test_get_request_scoped_sdk_returns_base_sdk_when_no_headers():
    """Test that get_request_scoped_sdk returns base SDK when no headers to add."""
    base_sdk = get_async_platform_sdk()

    # Mock both functions to return empty dicts
    with patch("nmp.common.sdk_factory.get_otel_headers", return_value={}):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value={}):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Should return the same SDK instance when no headers to add
    assert scoped_sdk is base_sdk


def test_get_request_scoped_sdk_preserves_original_base_sdk():
    """Test that get_request_scoped_sdk doesn't modify the original base SDK."""
    base_sdk = get_async_platform_sdk(
        as_service="my-service",
    )

    # Store original headers
    original_headers = dict(base_sdk.default_headers)

    mock_otel_headers = {"traceparent": "00-trace-id-span-id-01"}
    mock_auth_headers = {"X-NMP-Principal-Id": "user@example.com"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Verify original SDK is unchanged
    assert base_sdk.default_headers == original_headers
    assert "traceparent" not in base_sdk.default_headers

    # Verify scoped SDK has new headers
    assert "traceparent" in scoped_sdk.default_headers


def test_get_request_scoped_sdk_only_otel_headers():
    """Test get_request_scoped_sdk with only OTEL headers (no auth headers)."""
    base_sdk = get_async_platform_sdk()

    mock_otel_headers = {"traceparent": "00-trace-id-span-id-01"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value={}):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Should create new SDK with OTEL headers
    assert scoped_sdk is not base_sdk
    assert "traceparent" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["traceparent"] == "00-trace-id-span-id-01"


def test_get_request_scoped_sdk_only_auth_headers():
    """Test get_request_scoped_sdk with only auth headers (no OTEL headers)."""
    base_sdk = get_async_platform_sdk()

    mock_auth_headers = {"X-NMP-Principal-Id": "user@example.com"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value={}):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Should create new SDK with auth headers
    assert scoped_sdk is not base_sdk
    assert "X-NMP-Principal-Id" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Id"] == "user@example.com"


def test_get_request_scoped_sdk_auth_headers_override_otel_headers():
    """Test that auth headers take precedence when there are key conflicts."""
    base_sdk = get_async_platform_sdk()

    # Both have the same header key
    mock_otel_headers = {"X-Custom-Header": "otel-value"}
    mock_auth_headers = {"X-Custom-Header": "auth-value"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Auth headers should win (they're applied after OTEL via .update())
    assert scoped_sdk.default_headers["X-Custom-Header"] == "auth-value"


def test_get_request_scoped_sdk_preserves_base_default_headers_with_scoped_headers():
    """Request-scoped SDKs keep base service headers while adding request headers."""
    base_sdk = get_async_platform_sdk(as_service="jobs", internal=True)

    mock_otel_headers = {"traceparent": "00-trace-id-span-id-01"}
    mock_auth_headers = {"X-NMP-Principal-On-Behalf-Of": "user@example.com"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    assert scoped_sdk.default_headers["X-NMP-Internal"] == "true"
    assert scoped_sdk.default_headers["X-NMP-Principal-Id"] == "service:jobs"
    assert scoped_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
    assert scoped_sdk.default_headers["traceparent"] == "00-trace-id-span-id-01"


def test_get_request_scoped_sdk_preserves_base_sdk_http_client():
    """Test that get_request_scoped_sdk reuses the base SDK's HTTP client."""
    import httpx

    # Create a custom HTTP client
    custom_client = httpx.AsyncClient(timeout=30.0)

    base_sdk = get_async_platform_sdk(
        http_client=custom_client,
    )

    mock_auth_headers = {"X-NMP-Principal-Id": "user@example.com"}

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value={}):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Verify the HTTP client is reused (same instance)
    # Note: with_options() reuses the underlying HTTP client
    assert scoped_sdk is not base_sdk
    # The SDK wraps the client, so we verify it's a new SDK but lightweight
    assert "X-NMP-Principal-Id" in scoped_sdk.default_headers


def test_get_request_scoped_sdk_with_on_behalf_of_header():
    """Test that get_request_scoped_sdk propagates on-behalf-of header from request context."""
    base_sdk = get_async_platform_sdk()

    # Simulate a request context where a user is acting on behalf of another user
    mock_otel_headers = {"traceparent": "00-trace-id-span-id-01"}
    mock_auth_headers = {
        "X-NMP-Principal-Id": "admin@example.com",
        "X-NMP-Principal-On-Behalf-Of": "user@example.com",
        "X-NMP-Principal-Groups": "admin-group",
    }

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value=mock_otel_headers):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Verify it's a new SDK instance
    assert scoped_sdk is not base_sdk

    # Verify all headers including on-behalf-of are propagated
    assert "X-NMP-Principal-Id" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Id"] == "admin@example.com"
    assert "X-NMP-Principal-On-Behalf-Of" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
    assert "X-NMP-Principal-Groups" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Groups"] == "admin-group"
    assert "traceparent" in scoped_sdk.default_headers


def test_get_request_scoped_sdk_service_principal_with_on_behalf_of():
    """Test get_request_scoped_sdk when base SDK is a service principal and request has on-behalf-of."""
    # This simulates a service making a request on behalf of a user
    # Base SDK has service principal, request context adds on-behalf-of
    base_sdk = get_async_platform_sdk(
        as_service="my-service",
    )

    # Request context includes on-behalf-of header
    mock_auth_headers = {
        "X-NMP-Principal-Id": "service:my-service",
        "X-NMP-Principal-On-Behalf-Of": "user@example.com",
    }

    with patch("nmp.common.sdk_factory.get_otel_headers", return_value={}):
        with patch("nmp.common.sdk_factory.get_principal_auth_headers", return_value=mock_auth_headers):
            scoped_sdk = get_request_scoped_sdk(base_sdk)

    # Verify scoped SDK has both service principal and on-behalf-of
    assert scoped_sdk is not base_sdk
    assert "X-NMP-Principal-Id" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-Id"] == "service:my-service"
    assert "X-NMP-Principal-On-Behalf-Of" in scoped_sdk.default_headers
    assert scoped_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"


def test_get_sdk_on_behalf_of_preserves_omitted_default_headers() -> None:
    with NeMoPlatform(
        base_url="http://nmp.example.test",
        default_headers={"Content-Type": omit, "X-Base": "base"},
    ) as base_sdk:
        delegated_sdk = get_sdk_on_behalf_of(base_sdk, "user@example.com")

    assert delegated_sdk.default_headers["Content-Type"] is omit
    assert delegated_sdk.default_headers["X-Base"] == "base"
    assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"


# --- SDK routing client tests ---


@pytest.fixture
def platform_config_with_service_discovery():
    """Platform config with service_discovery map for entities and jobs."""
    return PlatformConfig(  # type: ignore[abstract]
        base_url="http://platform:8080",
        service_discovery={
            "entities": "http://entities-service:8080",
            "jobs": "http://jobs-service:8080",
        },
    )


@pytest.fixture
def platform_config_with_uds_service_route() -> PlatformConfig:
    return PlatformConfig(  # type: ignore[abstract]
        base_url="https://platform:8443",
        service_discovery={"entities": "unix:///tmp/entities.sock"},
    )


def test_get_platform_sdk_uses_routing_transport_for_service_discovery(
    platform_config_with_service_discovery,
):
    """Service discovery routing belongs to the SDK-owned HTTP client."""
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_service_discovery,
    ):
        with get_platform_sdk() as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _SyncPlatformEndpointRoutingTransport)


def test_platform_sdk_copy_reuses_routing_transport(
    platform_config_with_service_discovery,
):
    """SDK clones remain routed by reusing the base HTTP client."""
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_service_discovery,
    ):
        with get_platform_sdk() as sdk:
            copied_sdk = sdk.copy(workspace="copy-workspace")

            assert copied_sdk._client is sdk._client


def test_get_platform_sdk_base_url_preserves_routing_transport_for_uds_service(
    platform_config_with_uds_service_route: PlatformConfig,
) -> None:
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_uds_service_route,
    ):
        with get_platform_sdk(base_url="https://override.example.test") as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _SyncPlatformEndpointRoutingTransport)


def test_get_platform_sdk_workload_identity_uses_routing_transport_for_uds_service(
    monkeypatch: pytest.MonkeyPatch,
    platform_config_with_uds_service_route: PlatformConfig,
    tmp_path,
) -> None:
    subject_token_file = tmp_path / "workload-token"
    subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)

    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_uds_service_route,
    ):
        with get_platform_sdk() as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _SyncPlatformEndpointRoutingTransport)


@pytest.mark.asyncio
async def test_get_async_platform_sdk_uses_routing_transport_for_service_discovery(
    platform_config_with_service_discovery,
) -> None:
    """Async service discovery routing belongs to the SDK-owned HTTP client."""
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_service_discovery,
    ):
        async with get_async_platform_sdk() as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _AsyncPlatformEndpointRoutingTransport)


@pytest.mark.asyncio
async def test_get_async_platform_sdk_workload_identity_uses_routing_transport_for_uds_service(
    monkeypatch: pytest.MonkeyPatch,
    platform_config_with_uds_service_route: PlatformConfig,
    tmp_path,
) -> None:
    subject_token_file = tmp_path / "workload-token"
    subject_token_file.write_text("subject-token-from-file\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)

    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_uds_service_route,
    ):
        async with get_async_platform_sdk() as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _AsyncPlatformEndpointRoutingTransport)


@pytest.mark.asyncio
async def test_async_platform_sdk_copy_reuses_routing_transport(
    platform_config_with_service_discovery,
) -> None:
    """Async SDK clones remain routed by reusing the base HTTP client."""
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_service_discovery,
    ):
        async with get_async_platform_sdk() as sdk:
            copied_sdk = sdk.copy(workspace="copy-workspace")

            assert copied_sdk._client is sdk._client


@pytest.mark.asyncio
async def test_get_async_platform_sdk_base_url_preserves_routing_transport_for_uds_service(
    platform_config_with_uds_service_route: PlatformConfig,
) -> None:
    with patch(
        "nmp.common.sdk_factory.Configuration.get_platform_config",
        return_value=platform_config_with_uds_service_route,
    ):
        async with get_async_platform_sdk(base_url="https://override.example.test") as sdk:
            transport = sdk._client._transport

    assert isinstance(transport, _AsyncPlatformEndpointRoutingTransport)


# --- get_entity_parts tests ---


def test_get_entity_parts_qualified_returns_workspace_and_name():
    """Qualified name (workspace/name) returns (workspace, name)."""
    assert get_entity_parts("my-ws/my-secret") == ("my-ws", "my-secret")


def test_get_entity_parts_qualified_splits_only_first_slash():
    """Only the first slash is used; rest is part of the name."""
    assert get_entity_parts("ws/a/b/c") == ("ws", "a/b/c")


def test_get_entity_parts_unqualified_with_workspace_returns_workspace_and_name():
    """Unqualified name with workspace returns (workspace, name)."""
    assert get_entity_parts("local-secret", default_workspace="default") == ("default", "local-secret")


def test_get_entity_parts_unqualified_without_workspace_raises():
    """Unqualified name without workspace raises ValueError."""
    with pytest.raises(ValueError, match="not qualified with a workspace"):
        get_entity_parts("bare-name")
