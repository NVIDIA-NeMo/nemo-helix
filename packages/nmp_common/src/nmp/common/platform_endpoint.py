# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed platform endpoint resolution for HTTP(S) and Unix domain sockets."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypedDict

import httpx
from httpx._types import TimeoutTypes
from nemo_platform_plugin.client.tls import client_verify_from_env
from nmp.common.config import Configuration, PlatformConfig
from nmp.common.immutable_http_client import (
    ImmutableAsyncHttpxClient,
    ImmutableDefaultAsyncHttpxClient,
    ImmutableDefaultHttpxClient,
    ImmutableHttpxClient,
)

UDS_BASE_URL = "http://nemo-platform.local"
logger = logging.getLogger(__name__)
_SyncRequestHook = Callable[[httpx.Request], None]
_AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]
_SyncEventHooks = Mapping[str, list[_SyncRequestHook]]
_AsyncEventHooks = Mapping[str, list[_AsyncRequestHook]]


class _SyncClientKwargs(TypedDict, total=False):
    event_hooks: _SyncEventHooks
    follow_redirects: bool
    timeout: TimeoutTypes
    transport: httpx.BaseTransport
    verify: str | bool


class _AsyncClientKwargs(TypedDict, total=False):
    event_hooks: _AsyncEventHooks
    follow_redirects: bool
    timeout: TimeoutTypes
    transport: httpx.AsyncBaseTransport
    verify: str | bool


def _empty_service_endpoints() -> Mapping[str, "PlatformEndpoint"]:
    return MappingProxyType({})


def _sync_event_hooks(request_hooks: Iterable[_SyncRequestHook] | None) -> _SyncEventHooks | None:
    hooks = list(request_hooks) if request_hooks is not None else []
    return {"request": hooks, "response": []} if hooks else None


def _async_event_hooks(request_hooks: Iterable[_AsyncRequestHook] | None) -> _AsyncEventHooks | None:
    hooks = list(request_hooks) if request_hooks is not None else []
    return {"request": hooks, "response": []} if hooks else None


def _sync_default_sdk_client(
    *,
    timeout: TimeoutTypes | None,
    event_hooks: _SyncEventHooks | None,
    verify: str | bool | None = None,
    transport: httpx.BaseTransport | None = None,
    follow_redirects: bool | None = None,
) -> httpx.Client:
    kwargs: _SyncClientKwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks
    if verify is not None:
        kwargs["verify"] = verify
    if transport is not None:
        kwargs["transport"] = transport
    if follow_redirects is not None:
        kwargs["follow_redirects"] = follow_redirects
    return ImmutableDefaultHttpxClient(**kwargs)


def _sync_uds_sdk_client(
    *,
    transport: httpx.BaseTransport,
    timeout: TimeoutTypes | None,
    event_hooks: _SyncEventHooks | None,
    follow_redirects: bool,
) -> httpx.Client:
    kwargs: _SyncClientKwargs = {"transport": transport, "follow_redirects": follow_redirects}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks
    return ImmutableHttpxClient(**kwargs)


def _async_default_sdk_client(
    *,
    timeout: TimeoutTypes | None,
    event_hooks: _AsyncEventHooks | None,
    verify: str | bool | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    follow_redirects: bool | None = None,
) -> httpx.AsyncClient:
    kwargs: _AsyncClientKwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks
    if verify is not None:
        kwargs["verify"] = verify
    if transport is not None:
        kwargs["transport"] = transport
    if follow_redirects is not None:
        kwargs["follow_redirects"] = follow_redirects
    return ImmutableDefaultAsyncHttpxClient(**kwargs)


def _async_uds_sdk_client(
    *,
    transport: httpx.AsyncBaseTransport,
    timeout: TimeoutTypes | None,
    event_hooks: _AsyncEventHooks | None,
    follow_redirects: bool,
) -> httpx.AsyncClient:
    kwargs: _AsyncClientKwargs = {"transport": transport, "follow_redirects": follow_redirects}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if event_hooks is not None:
        kwargs["event_hooks"] = event_hooks
    return ImmutableAsyncHttpxClient(**kwargs)


def _get_platform_config() -> PlatformConfig:
    platform_config = Configuration.get_platform_config()
    if not isinstance(platform_config, PlatformConfig):
        raise TypeError("Expected PlatformConfig from Configuration.get_platform_config()")
    return platform_config


@dataclass(frozen=True)
class PlatformEndpoint:
    connect_base_url: str
    socket_path: Path | None
    transport: Literal["tcp", "uds"]
    service_pattern: re.Pattern[str] | None = field(default=None, repr=False, compare=False)
    service_endpoints: Mapping[str, "PlatformEndpoint"] = field(
        default_factory=_empty_service_endpoints,
        repr=False,
        compare=False,
    )

    def sync_http_client(self, *, timeout: TimeoutTypes | None = None) -> httpx.Client:
        if self.transport == "uds":
            if self.socket_path is None:
                raise ValueError("UDS endpoint is missing a socket path")
            transport = httpx.HTTPTransport(uds=str(self.socket_path))
            if timeout is None:
                return httpx.Client(transport=transport, follow_redirects=True)
            return httpx.Client(transport=transport, follow_redirects=True, timeout=timeout)
        if timeout is None:
            return httpx.Client(follow_redirects=True)
        return httpx.Client(follow_redirects=True, timeout=timeout)

    def async_http_client(self, *, timeout: TimeoutTypes | None = None) -> httpx.AsyncClient:
        if self.transport == "uds":
            if self.socket_path is None:
                raise ValueError("UDS endpoint is missing a socket path")
            transport = httpx.AsyncHTTPTransport(uds=str(self.socket_path))
            if timeout is None:
                return httpx.AsyncClient(transport=transport, follow_redirects=True)
            return httpx.AsyncClient(transport=transport, follow_redirects=True, timeout=timeout)
        if timeout is None:
            return httpx.AsyncClient(follow_redirects=True)
        return httpx.AsyncClient(follow_redirects=True, timeout=timeout)

    def sync_sdk_http_client(
        self,
        *,
        timeout: TimeoutTypes | None = None,
        http_client: httpx.Client | None = None,
        request_hooks: Iterable[_SyncRequestHook] | None = None,
        verify: str | bool | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Client:
        if http_client is not None:
            if not self.service_endpoints:
                return http_client
            return ImmutableDefaultHttpxClient(
                base_url=http_client.base_url,
                cookies=http_client.cookies,
                follow_redirects=http_client.follow_redirects,
                headers=http_client.headers,
                max_redirects=http_client.max_redirects,
                params=http_client.params,
                timeout=http_client.timeout,
                transport=_SyncExplicitClientRoutingTransport(endpoint=self, http_client=http_client),
            )
        event_hooks = _sync_event_hooks(request_hooks)
        if self.service_endpoints:
            transport = _SyncPlatformEndpointRoutingTransport(endpoint=self, verify=verify)
            return _sync_default_sdk_client(
                transport=transport,
                timeout=timeout,
                event_hooks=event_hooks,
                follow_redirects=follow_redirects,
            )
        if self.transport == "uds":
            if self.socket_path is None:
                raise ValueError("UDS endpoint is missing a socket path")
            transport = httpx.HTTPTransport(uds=str(self.socket_path))
            return _sync_uds_sdk_client(
                transport=transport,
                timeout=timeout,
                event_hooks=event_hooks,
                follow_redirects=True if follow_redirects is None else follow_redirects,
            )
        return _sync_default_sdk_client(
            timeout=timeout,
            event_hooks=event_hooks,
            verify=verify,
            follow_redirects=follow_redirects,
        )

    def async_sdk_http_client(
        self,
        *,
        timeout: TimeoutTypes | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_hooks: Iterable[_AsyncRequestHook] | None = None,
        verify: str | bool | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.AsyncClient:
        if http_client is not None:
            if not self.service_endpoints:
                return http_client
            return ImmutableDefaultAsyncHttpxClient(
                base_url=http_client.base_url,
                cookies=http_client.cookies,
                follow_redirects=http_client.follow_redirects,
                headers=http_client.headers,
                max_redirects=http_client.max_redirects,
                params=http_client.params,
                timeout=http_client.timeout,
                transport=_AsyncExplicitClientRoutingTransport(endpoint=self, http_client=http_client),
            )
        event_hooks = _async_event_hooks(request_hooks)
        if self.service_endpoints:
            transport = _AsyncPlatformEndpointRoutingTransport(endpoint=self, verify=verify)
            return _async_default_sdk_client(
                transport=transport,
                timeout=timeout,
                event_hooks=event_hooks,
                follow_redirects=follow_redirects,
            )
        if self.transport == "uds":
            if self.socket_path is None:
                raise ValueError("UDS endpoint is missing a socket path")
            transport = httpx.AsyncHTTPTransport(uds=str(self.socket_path))
            return _async_uds_sdk_client(
                transport=transport,
                timeout=timeout,
                event_hooks=event_hooks,
                follow_redirects=True if follow_redirects is None else follow_redirects,
            )
        return _async_default_sdk_client(
            timeout=timeout,
            event_hooks=event_hooks,
            verify=verify,
            follow_redirects=follow_redirects,
        )

    def route_request_url(self, url: str | httpx.URL) -> "RoutedPlatformEndpointRequest":
        """Resolve one outgoing SDK URL using this endpoint's fixed routing table."""

        request_url = httpx.URL(url)
        endpoint = self
        service_name = "unknown"

        match = self.service_pattern.search(request_url.path) if self.service_pattern is not None else None
        if match is not None:
            service_name = match.group(1)
            endpoint = self.service_endpoints.get(service_name) or self
            routed_url = _url_for_endpoint(request_url, endpoint)
        else:
            routed_url = request_url

        logger.debug(
            "Routing SDK URL to service endpoint"
            if service_name != "unknown"
            else "Routing SDK URL to default endpoint",
            extra={
                "service": service_name,
                "path": request_url.path,
                "host": routed_url.host,
                "port": routed_url.port,
                "transport": endpoint.transport,
            },
        )
        return RoutedPlatformEndpointRequest(url=routed_url, endpoint=endpoint)


@dataclass(frozen=True)
class RoutedPlatformEndpointRequest:
    url: httpx.URL
    endpoint: PlatformEndpoint


def resolve_platform_endpoint(platform_config: PlatformConfig | None = None) -> PlatformEndpoint:
    """Resolve the default platform endpoint from ``NMP_BASE_URL`` / config."""

    if platform_config is None:
        platform_config = _get_platform_config()
    default_endpoint = parse_platform_endpoint(platform_config.base_url)
    service_endpoints = {
        service_name: resolve_service_endpoint(service_name, platform_config)
        for service_name in sorted(_service_route_names(platform_config, os.environ))
    }
    return PlatformEndpoint(
        connect_base_url=default_endpoint.connect_base_url,
        socket_path=default_endpoint.socket_path,
        transport=default_endpoint.transport,
        service_pattern=platform_config.create_service_pattern(),
        service_endpoints=MappingProxyType(service_endpoints),
    )


def resolve_service_endpoint(service_name: str, platform_config: PlatformConfig | None = None) -> PlatformEndpoint:
    """Resolve a service endpoint using ``NMP_<SERVICE>_URL`` before ``NMP_BASE_URL``."""

    if platform_config is None:
        platform_config = _get_platform_config()
    normalized_name = _normalize_service_name(service_name)
    local_service_name = _matching_service_name(normalized_name, platform_config.get_services())
    if local_service_name is not None:
        return parse_platform_endpoint(platform_config.get_service_url(local_service_name))

    env_name = _service_url_env_var_name(service_name)
    env_endpoint = os.environ.get(env_name)
    parsed_env_endpoint = _parse_optional_platform_endpoint(env_endpoint)
    if parsed_env_endpoint is not None:
        return parsed_env_endpoint

    configured_service_name = _matching_service_name(normalized_name, platform_config.service_discovery)
    if configured_service_name is not None:
        configured_endpoint = platform_config.get_service_url(configured_service_name)
        if configured_endpoint != env_endpoint:
            return parse_platform_endpoint(configured_endpoint)

    endpoint = platform_config.base_url
    return parse_platform_endpoint(endpoint)


def parse_platform_endpoint(endpoint: str) -> PlatformEndpoint:
    """Parse an HTTP(S) or ``unix://`` endpoint into a typed transport model."""

    if endpoint.startswith(("http://", "https://")):
        try:
            parsed = httpx.URL(endpoint)
        except httpx.InvalidURL as error:
            raise ValueError(f"Invalid platform endpoint URL {endpoint!r}") from error
        if not parsed.host:
            raise ValueError(f"HTTP(S) platform endpoint must include a host, got {endpoint!r}")
        return PlatformEndpoint(connect_base_url=endpoint.rstrip("/"), socket_path=None, transport="tcp")
    if endpoint.startswith("unix://"):
        socket_path = _parse_unix_socket_path(endpoint)
        return PlatformEndpoint(connect_base_url=UDS_BASE_URL, socket_path=socket_path, transport="uds")
    if endpoint.startswith("/"):
        raise ValueError(f"Raw socket paths are not valid endpoint URLs; use unix://{endpoint}")
    raise ValueError(f"Unsupported platform endpoint URL {endpoint!r}; expected http://, https://, or unix://")


def require_authorization_header_endpoint(endpoint: PlatformEndpoint, *, purpose: str) -> None:
    """Require an endpoint safe enough for requests carrying Authorization."""
    if endpoint.transport == "uds":
        return

    parsed = httpx.URL(endpoint.connect_base_url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and _is_loopback_host(parsed.host):
        return

    raise ValueError(
        f"{purpose} cannot send Authorization to cleartext remote endpoint "
        f"{endpoint.connect_base_url!r}; use https://, unix://, or loopback HTTP for local development"
    )


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_unix_socket_path(endpoint: str) -> Path:
    raw_path = endpoint.removeprefix("unix://")
    if not raw_path.startswith("/"):
        raise ValueError(f"UDS endpoint must use an absolute socket path, got {endpoint!r}")
    return Path(raw_path)


def _service_route_names(platform_config: PlatformConfig, env: Mapping[str, str]) -> set[str]:
    names: set[str] = set()
    for service_name, endpoint in platform_config.service_discovery.items():
        normalized_name = _normalize_service_name(service_name)
        if _is_invalid_env_service_url(normalized_name, endpoint, env):
            continue
        names.add(normalized_name)
    names.update(_normalize_service_name(service_name) for service_name in platform_config.get_services())
    names.update(_service_route_names_from_env(env))
    names.discard("")
    return names


def _service_route_names_from_env(env: Mapping[str, str]) -> set[str]:
    names: set[str] = set()
    for env_name, endpoint in env.items():
        if not env_name.startswith("NMP_") or not env_name.endswith("_URL"):
            continue
        raw_service_name = env_name.removeprefix("NMP_").removesuffix("_URL")
        if raw_service_name == "BASE":
            continue
        if _parse_optional_platform_endpoint(endpoint) is None:
            continue
        names.add(_normalize_service_name(raw_service_name))
    return names


def _service_url_env_var_name(service_name: str) -> str:
    return f"NMP_{_normalize_service_name(service_name).upper().replace('-', '_')}_URL"


def _matching_service_name(normalized_name: str, configured_names: Iterable[str]) -> str | None:
    for configured_name in configured_names:
        if _normalize_service_name(configured_name) == normalized_name:
            return configured_name
    return None


def _is_invalid_env_service_url(service_name: str, endpoint: str, env: Mapping[str, str]) -> bool:
    env_endpoint = env.get(_service_url_env_var_name(service_name))
    return env_endpoint == endpoint and _parse_optional_platform_endpoint(endpoint) is None


def _parse_optional_platform_endpoint(endpoint: str | None) -> PlatformEndpoint | None:
    if not endpoint:
        return None
    try:
        return parse_platform_endpoint(endpoint)
    except ValueError:
        return None


def _normalize_service_name(service_name: str) -> str:
    return service_name.strip().lower().replace("_", "-")


def _url_for_endpoint(url: httpx.URL, endpoint: PlatformEndpoint) -> httpx.URL:
    if endpoint.transport == "uds":
        return url.copy_with(scheme="http", host=httpx.URL(UDS_BASE_URL).host, port=None)
    endpoint_url = httpx.URL(endpoint.connect_base_url)
    return url.copy_with(scheme=endpoint_url.scheme, host=endpoint_url.host, port=endpoint_url.port)


def _set_request_url(request: httpx.Request, url: httpx.URL) -> None:
    if request.url == url:
        return
    request.url = url
    if url.host:
        request.headers["Host"] = url.netloc.decode("ascii")


def _uds_socket_paths(endpoint: PlatformEndpoint) -> frozenset[Path]:
    paths: set[Path] = set()
    for candidate in (endpoint, *endpoint.service_endpoints.values()):
        if candidate.transport != "uds":
            continue
        if candidate.socket_path is None:
            raise ValueError("UDS endpoint is missing a socket path")
        paths.add(candidate.socket_path)
    return frozenset(paths)


class _SyncPlatformEndpointRoutingTransport(httpx.BaseTransport):
    def __init__(self, *, endpoint: PlatformEndpoint, verify: str | bool | None = None) -> None:
        self._endpoint = endpoint
        client_verify = client_verify_from_env() if verify is None else verify
        self._tcp_transport = (
            httpx.HTTPTransport() if client_verify is True else httpx.HTTPTransport(verify=client_verify)
        )
        self._uds_transports = {
            socket_path: httpx.HTTPTransport(uds=str(socket_path)) for socket_path in _uds_socket_paths(endpoint)
        }

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        routed = self._endpoint.route_request_url(request.url)
        _set_request_url(request, routed.url)
        return self._transport_for_endpoint(routed.endpoint).handle_request(request)

    def close(self) -> None:
        self._tcp_transport.close()
        for transport in self._uds_transports.values():
            transport.close()

    def _transport_for_endpoint(self, endpoint: PlatformEndpoint) -> httpx.HTTPTransport:
        if endpoint.transport == "tcp":
            return self._tcp_transport
        if endpoint.socket_path is None:
            raise ValueError("UDS endpoint is missing a socket path")
        return self._uds_transports[endpoint.socket_path]


class _SyncExplicitClientRoutingTransport(httpx.BaseTransport):
    def __init__(self, *, endpoint: PlatformEndpoint, http_client: httpx.Client) -> None:
        self._endpoint = endpoint
        self._http_client = http_client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        routed = self._endpoint.route_request_url(request.url)
        _set_request_url(request, routed.url)
        return self._http_client.send(request, stream=True)

    def close(self) -> None:
        # The wrapped explicit client is caller-owned.
        pass


class _AsyncPlatformEndpointRoutingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, endpoint: PlatformEndpoint, verify: str | bool | None = None) -> None:
        self._endpoint = endpoint
        client_verify = client_verify_from_env() if verify is None else verify
        self._tcp_transport = (
            httpx.AsyncHTTPTransport() if client_verify is True else httpx.AsyncHTTPTransport(verify=client_verify)
        )
        self._uds_transports = {
            socket_path: httpx.AsyncHTTPTransport(uds=str(socket_path)) for socket_path in _uds_socket_paths(endpoint)
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        routed = self._endpoint.route_request_url(request.url)
        _set_request_url(request, routed.url)
        return await self._transport_for_endpoint(routed.endpoint).handle_async_request(request)

    async def aclose(self) -> None:
        await self._tcp_transport.aclose()
        for transport in self._uds_transports.values():
            await transport.aclose()

    def _transport_for_endpoint(self, endpoint: PlatformEndpoint) -> httpx.AsyncHTTPTransport:
        if endpoint.transport == "tcp":
            return self._tcp_transport
        if endpoint.socket_path is None:
            raise ValueError("UDS endpoint is missing a socket path")
        return self._uds_transports[endpoint.socket_path]


class _AsyncExplicitClientRoutingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, endpoint: PlatformEndpoint, http_client: httpx.AsyncClient) -> None:
        self._endpoint = endpoint
        self._http_client = http_client

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        routed = self._endpoint.route_request_url(request.url)
        _set_request_url(request, routed.url)
        return await self._http_client.send(request, stream=True)

    async def aclose(self) -> None:
        # The wrapped explicit client is caller-owned.
        pass
