# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from nmp.common.config import Configuration, PlatformConfig, get_common_service_config
from nmp.common.platform_endpoint import (
    UDS_BASE_URL,
    PlatformEndpoint,
    _AsyncPlatformEndpointRoutingTransport,
    _SyncPlatformEndpointRoutingTransport,
    parse_platform_endpoint,
    require_authorization_header_endpoint,
    resolve_platform_endpoint,
    resolve_service_endpoint,
)


def _assert_log_record_extra(record: logging.LogRecord, expected: dict[str, str | int]) -> None:
    fields = vars(record)
    for name, value in expected.items():
        assert fields[name] == value


@pytest.fixture(autouse=True)
def clear_platform_url_env_vars(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    Configuration.clear_cache()
    for key in tuple(os.environ):
        if key.startswith("NMP_") and key.endswith("_URL"):
            monkeypatch.delenv(key, raising=False)
    try:
        yield
    finally:
        Configuration.clear_cache()


def test_parse_tcp_endpoint() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080/")

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "http://127.0.0.1:8080"
    assert endpoint.socket_path is None


@pytest.mark.parametrize("url", ["http://", "https://"])
def test_parse_rejects_hostless_http_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match="must include a host"):
        parse_platform_endpoint(url)


def test_parse_https_endpoint_preserves_normalized_connect_base_url() -> None:
    endpoint = parse_platform_endpoint("https://platform.example.com/api/")

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "https://platform.example.com/api"
    assert endpoint.socket_path is None


def test_parse_uds_endpoint() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    assert endpoint.transport == "uds"
    assert endpoint.connect_base_url == UDS_BASE_URL
    assert endpoint.socket_path == Path("/tmp/nemo-platform.sock")


def test_parse_rejects_raw_socket_path() -> None:
    with pytest.raises(ValueError, match="use unix:///tmp/nemo-platform.sock"):
        parse_platform_endpoint("/tmp/nemo-platform.sock")


def test_parse_rejects_relative_uds_socket_path() -> None:
    with pytest.raises(ValueError, match="absolute socket path"):
        parse_platform_endpoint("unix://relative.sock")


@pytest.mark.parametrize(
    "url",
    [
        "https://platform.example.com",
        "unix:///tmp/nemo-platform.sock",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
    ],
)
def test_authorization_header_endpoint_allows_secure_and_local_transports(url: str) -> None:
    require_authorization_header_endpoint(parse_platform_endpoint(url), purpose="test call")


def test_authorization_header_endpoint_rejects_remote_cleartext_http() -> None:
    with pytest.raises(ValueError, match="cleartext remote endpoint"):
        require_authorization_header_endpoint(
            parse_platform_endpoint("http://platform.example.com"), purpose="test call"
        )


def test_resolve_service_endpoint_uses_service_specific_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NMP_SECRETS_URL", "unix:///tmp/secrets.sock")
    config = PlatformConfig(base_url="http://platform:8080")

    endpoint = resolve_service_endpoint("secrets", config)

    assert endpoint.transport == "uds"
    assert endpoint.socket_path == Path("/tmp/secrets.sock")


def test_resolve_service_endpoint_falls_back_to_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NMP_SECRETS_URL", raising=False)
    config = PlatformConfig(base_url="http://platform:8080")

    endpoint = resolve_service_endpoint("secrets", config)

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "http://platform:8080"


def test_resolve_service_endpoint_skips_invalid_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(base_url="http://platform:8080")
    monkeypatch.setenv("NMP_SECRETS_URL", "not-a-url")

    endpoint = resolve_service_endpoint("secrets", config)

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "http://platform:8080"


def test_resolve_service_endpoint_skips_invalid_env_override_and_uses_configured_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PlatformConfig(
        base_url="http://platform:8080",
        service_discovery={"secrets": "http://secrets-service:8080"},
    )
    monkeypatch.setenv("NMP_SECRETS_URL", "not-a-url")

    endpoint = resolve_service_endpoint("secrets", config)

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "http://secrets-service:8080"


def test_endpoint_env_family_is_not_part_of_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NMP_SECRETS_ENDPOINT", "unix:///tmp/secrets.sock")
    monkeypatch.delenv("NMP_SECRETS_URL", raising=False)
    config = PlatformConfig(base_url="http://platform:8080")

    endpoint = resolve_service_endpoint("secrets", config)

    assert endpoint.transport == "tcp"
    assert endpoint.connect_base_url == "http://platform:8080"


@pytest.fixture
def platform_config_with_service_discovery() -> PlatformConfig:
    return PlatformConfig(  # type: ignore[abstract]
        base_url="http://platform:8080",
        service_discovery={
            "entities": "http://entities-service:8080",
            "jobs": "http://jobs-service:8080",
        },
    )


def test_resolve_platform_endpoint_carries_service_routes(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    assert endpoint.connect_base_url == "http://platform:8080"
    assert endpoint.service_endpoints["entities"].connect_base_url == "http://entities-service:8080"
    assert endpoint.service_endpoints["jobs"].connect_base_url == "http://jobs-service:8080"


def test_resolve_platform_endpoint_rejects_malformed_service_route() -> None:
    config = PlatformConfig(  # type: ignore[abstract]
        base_url="http://platform:8080",
        service_discovery={
            "entities": "http://entities-service:8080",
            "jobs": "not-a-url",
        },
    )

    with pytest.raises(ValueError, match="Unsupported platform endpoint URL 'not-a-url'"):
        resolve_platform_endpoint(config)


def test_resolve_platform_endpoint_rejects_malformed_base_url() -> None:
    config = PlatformConfig(  # type: ignore[abstract]
        base_url="not-a-url",
        service_discovery={"entities": "http://entities-service:8080"},
    )

    with pytest.raises(ValueError, match="Unsupported platform endpoint URL 'not-a-url'"):
        resolve_platform_endpoint(config)


def test_platform_endpoint_routes_api_path_to_service_url(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces?limit=10")

    assert routed.endpoint.connect_base_url == "http://entities-service:8080"
    assert routed.url.scheme == "http"
    assert routed.url.host == "entities-service"
    assert routed.url.port == 8080
    assert routed.url.path == "/apis/entities/v2/workspaces"
    assert routed.url.query == b"limit=10"


def test_platform_endpoint_preserves_request_path_when_service_url_has_path_prefix() -> None:
    config = PlatformConfig(  # type: ignore[abstract]
        base_url="http://platform:8080",
        service_discovery={"entities": "http://entities-service:8080/entities-prefix"},
    )
    endpoint = resolve_platform_endpoint(config)

    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces?limit=10")

    assert routed.endpoint.connect_base_url == "http://entities-service:8080/entities-prefix"
    assert routed.url.scheme == "http"
    assert routed.url.host == "entities-service"
    assert routed.url.port == 8080
    assert routed.url.path == "/apis/entities/v2/workspaces"
    assert routed.url.query == b"limit=10"


def test_platform_endpoint_routes_service_discovery_with_underscore_configured_name() -> None:
    config = PlatformConfig(
        base_url="http://platform:8080",
        service_discovery={"my_service": "http://my-service:8080"},
    )

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/my-service/v2/workspaces/default/items")

    assert routed.endpoint.connect_base_url == "http://my-service:8080"
    assert str(routed.url) == "http://my-service:8080/apis/my-service/v2/workspaces/default/items"


def test_platform_endpoint_uses_env_override(
    monkeypatch: pytest.MonkeyPatch,
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    monkeypatch.setenv("NMP_ENTITIES_URL", "http://entities-env:9090")
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces")

    assert routed.endpoint.connect_base_url == "http://entities-env:9090"
    assert routed.url.host == "entities-env"
    assert routed.url.port == 9090


def test_platform_endpoint_routes_env_only_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(base_url="http://platform:8080")  # type: ignore[abstract]
    monkeypatch.setenv("NMP_ENTITIES_URL", "http://entities-env:9090")

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces/default?limit=1")

    assert "entities" in endpoint.service_endpoints
    assert "entities" not in config.service_discovery
    assert routed.endpoint.connect_base_url == "http://entities-env:9090"
    assert str(routed.url) == "http://entities-env:9090/apis/entities/v2/workspaces/default?limit=1"


@pytest.mark.parametrize("service_url", ["not-a-url", "http://", "unix://relative.sock"])
def test_platform_endpoint_skips_invalid_env_only_service_url(
    monkeypatch: pytest.MonkeyPatch,
    service_url: str,
) -> None:
    monkeypatch.setenv("NMP_FILES_URL", service_url)
    config = PlatformConfig(base_url="http://platform:8080")

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/files/v2/workspaces/default/filesets")

    assert "files" not in endpoint.service_endpoints
    assert str(routed.url) == "http://platform:8080/apis/files/v2/workspaces/default/filesets"


def test_platform_endpoint_skips_invalid_unrelated_service_url_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NMP_NOT_A_SERVICE_URL", "not-a-url")
    config = PlatformConfig(base_url="http://platform:8080")

    endpoint = resolve_platform_endpoint(config)

    assert "not-a-service" not in endpoint.service_endpoints


def test_platform_endpoint_keeps_configured_local_service_on_local_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(base_url="http://platform:8080", services="hello-world")  # type: ignore[abstract]
    monkeypatch.setenv("NMP_HELLO_WORLD_URL", "http://hello-world-service:8080")
    endpoint = resolve_platform_endpoint(config)

    routed = endpoint.route_request_url("http://platform:8080/apis/hello-world/v2/workspaces/default/hello")

    assert routed.endpoint.connect_base_url == get_common_service_config().get_host_url()
    assert routed.url.host == "127.0.0.1"


def test_platform_endpoint_keeps_local_service_with_underscore_configured_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PlatformConfig(base_url="http://platform:8080", services="my_service")
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/my-service/v2/workspaces/default/items")

    assert routed.endpoint.connect_base_url == "http://127.0.0.1:8080"
    assert str(routed.url) == "http://127.0.0.1:8080/apis/my-service/v2/workspaces/default/items"


def test_platform_endpoint_keeps_local_service_when_env_override_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(base_url="http://platform:8080", services="hello-world")  # type: ignore[abstract]
    monkeypatch.setenv("NMP_SERVICE_HOST", "127.0.0.1")
    monkeypatch.setenv("NMP_SERVICE_PORT", "8080")
    monkeypatch.setenv("NMP_HELLO_WORLD_URL", "http://hello-world-env:9090")

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/hello-world/v2/workspaces/default/hello")

    assert routed.endpoint.connect_base_url == "http://127.0.0.1:8080"
    assert str(routed.url) == "http://127.0.0.1:8080/apis/hello-world/v2/workspaces/default/hello"


def test_platform_endpoint_ignores_unrelated_service_url_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    config = PlatformConfig(base_url="http://platform:8080")  # type: ignore[abstract]
    monkeypatch.setenv("NMP_NOT_A_SERVICE_URL", "http://not-a-service:8080")

    endpoint = resolve_platform_endpoint(config)
    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces/default")

    assert routed.endpoint.connect_base_url == "http://platform:8080"
    assert routed.url.host == "platform"


def test_platform_endpoint_routes_uds_service() -> None:
    config = PlatformConfig(  # type: ignore[abstract]
        base_url="http://platform:8080",
        service_discovery={"entities": "unix:///tmp/entities.sock"},
    )
    endpoint = resolve_platform_endpoint(config)

    routed = endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces")

    assert routed.endpoint.transport == "uds"
    assert routed.endpoint.socket_path == Path("/tmp/entities.sock")
    assert str(routed.url) == "http://nemo-platform.local/apis/entities/v2/workspaces"


def test_platform_endpoint_keeps_non_api_path_on_default_endpoint(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    routed = endpoint.route_request_url("http://platform:8080/health/ready")

    assert routed.endpoint.connect_base_url == "http://platform:8080"
    assert str(routed.url) == "http://platform:8080/health/ready"


def test_platform_endpoint_keeps_non_api_service_url(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    routed = endpoint.route_request_url("http://entities-service:8080/status")

    assert routed.endpoint.connect_base_url == "http://platform:8080"
    assert str(routed.url) == "http://entities-service:8080/status"


def test_platform_endpoint_logs_path_without_raw_url(
    caplog: pytest.LogCaptureFixture,
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    caplog.set_level(logging.DEBUG, logger="nmp.common.platform_endpoint")
    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)

    endpoint.route_request_url("http://platform:8080/health/ready?token=secret")
    endpoint.route_request_url("http://platform:8080/apis/entities/v2/workspaces?token=secret")

    default_record = next(
        record for record in caplog.records if record.message == "Routing SDK URL to default endpoint"
    )
    service_record = next(
        record for record in caplog.records if record.message == "Routing SDK URL to service endpoint"
    )

    assert "url" not in vars(default_record)
    _assert_log_record_extra(
        default_record,
        {
            "service": "unknown",
            "path": "/health/ready",
        },
    )

    assert "url" not in vars(service_record)
    _assert_log_record_extra(
        service_record,
        {
            "service": "entities",
            "path": "/apis/entities/v2/workspaces",
            "host": "entities-service",
            "port": 8080,
        },
    )

    for record in (default_record, service_record):
        assert "token=secret" not in str(vars(record))


def test_sync_http_client_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.httpx.Client") as client:
        endpoint.sync_http_client()

    client.assert_called_once_with(follow_redirects=True)


def test_sync_http_client_passes_explicit_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.httpx.Client") as client:
        endpoint.sync_http_client(timeout=2.0)

    client.assert_called_once_with(follow_redirects=True, timeout=2.0)


def test_sync_sdk_http_client_uses_sdk_default_for_tcp() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultHttpxClient") as client:
        endpoint.sync_sdk_http_client()

    client.assert_called_once_with()


def test_sync_sdk_http_client_passes_explicit_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultHttpxClient") as client:
        endpoint.sync_sdk_http_client(timeout=2.0)

    client.assert_called_once_with(timeout=2.0)


def test_sync_sdk_http_client_can_disable_redirects() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultHttpxClient") as client:
        endpoint.sync_sdk_http_client(follow_redirects=False)

    client.assert_called_once_with(follow_redirects=False)


def test_uds_sync_http_client_keeps_transport_and_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.platform_endpoint.httpx.Client") as client:
        endpoint.sync_http_client()

    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is True
    assert "transport" in kwargs
    assert "timeout" not in kwargs


def test_uds_sync_sdk_http_client_keeps_transport_and_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with (
        patch("nmp.common.platform_endpoint.ImmutableDefaultHttpxClient") as sdk_client,
        patch("nmp.common.platform_endpoint.ImmutableHttpxClient") as client,
    ):
        endpoint.sync_sdk_http_client()

    sdk_client.assert_not_called()
    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is True
    assert "transport" in kwargs
    assert "timeout" not in kwargs


def test_uds_sync_sdk_http_client_can_disable_redirects() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.platform_endpoint.ImmutableHttpxClient") as client:
        endpoint.sync_sdk_http_client(follow_redirects=False)

    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is False


def test_sync_routing_transport_prebuilds_uds_service_transports() -> None:
    service_endpoint = parse_platform_endpoint("unix:///tmp/entities.sock")
    constructed_transports: list[object] = []
    closed_transports: list[object] = []

    class FakeHTTPTransport:
        def __init__(self, *, uds: str | None = None, **kwargs: object) -> None:
            self.uds = uds
            if uds is not None:
                constructed_transports.append(self)

        def close(self) -> None:
            if self.uds is not None:
                closed_transports.append(self)

    with patch("nmp.common.platform_endpoint.httpx.HTTPTransport", FakeHTTPTransport):
        routing_transport = _SyncPlatformEndpointRoutingTransport(
            endpoint=PlatformEndpoint(
                connect_base_url="http://platform:8080",
                socket_path=None,
                transport="tcp",
                service_endpoints={"entities": service_endpoint},
            )
        )
        returned_transport = routing_transport._transport_for_endpoint(service_endpoint)

    routing_transport.close()

    assert len(constructed_transports) == 1
    assert returned_transport is constructed_transports[0]
    assert closed_transports == constructed_transports


def test_sync_routing_transport_routes_api_path_to_service_url(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    captured_urls: list[str] = []
    captured_hosts: list[str] = []

    class FakeHTTPTransport:
        def __init__(self, **kwargs: object) -> None:
            pass

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            captured_hosts.append(request.headers["host"])
            return httpx.Response(200, request=request)

        def close(self) -> None:
            pass

    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)
    with patch("nmp.common.platform_endpoint.httpx.HTTPTransport", FakeHTTPTransport):
        routing_transport = _SyncPlatformEndpointRoutingTransport(endpoint=endpoint)
        request = httpx.Request("GET", "http://platform:8080/apis/entities/v2/workspaces")
        routing_transport.handle_request(request)

    assert captured_urls == ["http://entities-service:8080/apis/entities/v2/workspaces"]
    assert captured_hosts == ["entities-service:8080"]


def test_sync_routing_transport_preserves_non_api_service_status_url(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    captured_urls: list[str] = []

    class FakeHTTPTransport:
        def __init__(self, **kwargs: object) -> None:
            pass

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(200, request=request)

        def close(self) -> None:
            pass

    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)
    with patch("nmp.common.platform_endpoint.httpx.HTTPTransport", FakeHTTPTransport):
        routing_transport = _SyncPlatformEndpointRoutingTransport(endpoint=endpoint)
        request = httpx.Request("GET", "http://entities-service:8080/status")
        routing_transport.handle_request(request)

    assert captured_urls == ["http://entities-service:8080/status"]


def test_sync_sdk_http_client_routes_explicit_client_without_owning_it(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    captured_requests: list[httpx.Request] = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, request=request)

    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)
    with httpx.Client(
        transport=httpx.MockTransport(capture_request),
        follow_redirects=False,
        headers={"x-test": "sync"},
        max_redirects=7,
        params={"debug": "true"},
    ) as explicit_client:
        with endpoint.sync_sdk_http_client(http_client=explicit_client) as routed_client:
            assert routed_client.follow_redirects is False
            assert routed_client.max_redirects == 7
            routed_client.get("http://platform:8080/apis/entities/v2/workspaces")

        assert not explicit_client.is_closed

    assert len(captured_requests) == 1
    assert str(captured_requests[0].url) == "http://entities-service:8080/apis/entities/v2/workspaces?debug=true"
    assert captured_requests[0].headers["x-test"] == "sync"


@pytest.mark.asyncio
async def test_async_sdk_http_client_routes_explicit_client_without_owning_it(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    captured_requests: list[httpx.Request] = []

    async def capture_request(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, request=request)

    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)
    transport = httpx.MockTransport(capture_request)
    async with httpx.AsyncClient(
        transport=transport,
        follow_redirects=False,
        headers={"x-test": "async"},
        max_redirects=7,
        params={"debug": "true"},
    ) as explicit_client:
        async with endpoint.async_sdk_http_client(http_client=explicit_client) as routed_client:
            assert routed_client.follow_redirects is False
            assert routed_client.max_redirects == 7
            await routed_client.get("http://platform:8080/apis/jobs/v2/workspaces/default/jobs")

        assert not explicit_client.is_closed

    assert len(captured_requests) == 1
    assert str(captured_requests[0].url) == "http://jobs-service:8080/apis/jobs/v2/workspaces/default/jobs?debug=true"
    assert captured_requests[0].headers["x-test"] == "async"


@pytest.mark.asyncio
async def test_async_routing_transport_routes_api_path_to_service_url(
    platform_config_with_service_discovery: PlatformConfig,
) -> None:
    captured_urls: list[str] = []
    captured_hosts: list[str] = []

    class FakeAsyncHTTPTransport:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            captured_hosts.append(request.headers["host"])
            return httpx.Response(200, request=request)

        async def aclose(self) -> None:
            pass

    endpoint = resolve_platform_endpoint(platform_config_with_service_discovery)
    with patch("nmp.common.platform_endpoint.httpx.AsyncHTTPTransport", FakeAsyncHTTPTransport):
        routing_transport = _AsyncPlatformEndpointRoutingTransport(endpoint=endpoint)
        request = httpx.Request("GET", "http://platform:8080/apis/jobs/v2/workspaces/default/jobs")
        await routing_transport.handle_async_request(request)

    assert captured_urls == ["http://jobs-service:8080/apis/jobs/v2/workspaces/default/jobs"]
    assert captured_hosts == ["jobs-service:8080"]


def test_routing_transports_pass_custom_ca_to_tcp_transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cert_file = tmp_path / "ca.pem"
    cert_file.write_text("certificate", encoding="utf-8")
    monkeypatch.setenv("NMP_CLIENT_SSL_CERT_FILE", str(cert_file))
    sync_kwargs: list[dict[str, object]] = []
    async_kwargs: list[dict[str, object]] = []

    class FakeHTTPTransport:
        def __init__(self, **kwargs: object) -> None:
            sync_kwargs.append(kwargs)

        def close(self) -> None:
            pass

    class FakeAsyncHTTPTransport:
        def __init__(self, **kwargs: object) -> None:
            async_kwargs.append(kwargs)

        async def aclose(self) -> None:
            pass

    endpoint = resolve_platform_endpoint(
        PlatformConfig(  # type: ignore[abstract]
            base_url="https://platform:8443",
            service_discovery={"entities": "https://entities-service:9443"},
        )
    )
    with (
        patch("nmp.common.platform_endpoint.httpx.HTTPTransport", FakeHTTPTransport),
        patch("nmp.common.platform_endpoint.httpx.AsyncHTTPTransport", FakeAsyncHTTPTransport),
    ):
        _SyncPlatformEndpointRoutingTransport(endpoint=endpoint)
        _AsyncPlatformEndpointRoutingTransport(endpoint=endpoint)

    assert sync_kwargs[0] == {"verify": str(cert_file)}
    assert async_kwargs[0] == {"verify": str(cert_file)}


def test_async_http_client_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.httpx.AsyncClient") as client:
        endpoint.async_http_client()

    client.assert_called_once_with(follow_redirects=True)


def test_async_http_client_passes_explicit_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.httpx.AsyncClient") as client:
        endpoint.async_http_client(timeout=2.0)

    client.assert_called_once_with(follow_redirects=True, timeout=2.0)


def test_async_sdk_http_client_uses_sdk_default_for_tcp() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultAsyncHttpxClient") as client:
        endpoint.async_sdk_http_client()

    client.assert_called_once_with()


def test_async_sdk_http_client_passes_explicit_timeout() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultAsyncHttpxClient") as client:
        endpoint.async_sdk_http_client(timeout=2.0)

    client.assert_called_once_with(timeout=2.0)


def test_async_sdk_http_client_can_disable_redirects() -> None:
    endpoint = parse_platform_endpoint("http://127.0.0.1:8080")

    with patch("nmp.common.platform_endpoint.ImmutableDefaultAsyncHttpxClient") as client:
        endpoint.async_sdk_http_client(follow_redirects=False)

    client.assert_called_once_with(follow_redirects=False)


def test_uds_async_http_client_keeps_transport_and_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.platform_endpoint.httpx.AsyncClient") as client:
        endpoint.async_http_client()

    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is True
    assert "transport" in kwargs
    assert "timeout" not in kwargs


def test_uds_async_sdk_http_client_keeps_transport_and_omits_unset_timeout() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with (
        patch("nmp.common.platform_endpoint.ImmutableDefaultAsyncHttpxClient") as sdk_client,
        patch("nmp.common.platform_endpoint.ImmutableAsyncHttpxClient") as client,
    ):
        endpoint.async_sdk_http_client()

    sdk_client.assert_not_called()
    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is True
    assert "transport" in kwargs
    assert "timeout" not in kwargs


def test_uds_async_sdk_http_client_can_disable_redirects() -> None:
    endpoint = parse_platform_endpoint("unix:///tmp/nemo-platform.sock")

    with patch("nmp.common.platform_endpoint.ImmutableAsyncHttpxClient") as client:
        endpoint.async_sdk_http_client(follow_redirects=False)

    kwargs = client.call_args.kwargs
    assert kwargs["follow_redirects"] is False
