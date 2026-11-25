# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint transport for sandbox ports behind the OpenSandbox ingress gateway.

Advertised URL: ``http://<sandbox id>-<port>.osb.helix.internal``.
Dialed URL: the gateway, with the advertised URL's path and the advertised host in ``Host``.
Request headers: ``OpenSandbox-Ingress-To: <sandbox id>-<port>`` (``header`` route mode) and those the
OpenSandbox server returns for the port, fetched on first use.

The header lookup needs the ``opensandbox`` extra.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import httpx
from nemo_deployments_plugin.endpoint_transport import EndpointTransport

logger = logging.getLogger(__name__)

ENDPOINT_HOST_SUFFIX = "osb.helix.internal"
INGRESS_HEADER = "OpenSandbox-Ingress-To"
# Sandbox ports of execd and of the egress sidecar's policy API.
RESERVED_PORTS = frozenset({44772, 18080})

# The ingress gateway reads this header when ``OpenSandbox-Ingress-To`` is absent.
_LEGACY_INGRESS_HEADER = "OPEN-SANDBOX-INGRESS"
_ROUTE_LABEL = re.compile(r"(?P<sandbox>[a-z0-9][a-z0-9-]*)-(?P<port>[0-9]{1,5})")
_MAX_HOST_LABEL_LENGTH = 63

_LOOKUP_TIMEOUT_SECONDS = 10.0
_CACHE_SIZE = 1024

# (server base URL, sandbox id, port)
EndpointKey = tuple[str, str, int]
# Process-wide cache of endpoint headers.
_endpoint_headers: dict[EndpointKey, dict[str, str]] = {}
_endpoint_headers_lock = threading.Lock()


async def endpoint_headers(key: EndpointKey, api_key: str) -> dict[str, str]:
    """Headers the OpenSandbox server returns for a sandbox port. Cached for the process."""
    with _endpoint_headers_lock:
        cached = _endpoint_headers.get(key)
    if cached is not None:
        return cached
    from opensandbox.adapters.factory import AdapterFactory
    from opensandbox.config import ConnectionConfig
    from opensandbox.transport import RetryPolicy

    server_base, sandbox_id, port = key
    connection = ConnectionConfig(
        api_key=api_key,
        domain=server_base,
        request_timeout=timedelta(seconds=_LOOKUP_TIMEOUT_SECONDS),
        retry_policy=RetryPolicy.disabled(),
        endpoint_cache_disabled=True,
    ).with_transport_if_missing()
    try:
        endpoint = await AdapterFactory(connection).create_sandbox_service().get_sandbox_endpoint(sandbox_id, port)
    finally:
        await connection.close_transport_if_owned()
    headers = dict(endpoint.headers)
    with _endpoint_headers_lock:
        if len(_endpoint_headers) >= _CACHE_SIZE:
            _endpoint_headers.pop(next(iter(_endpoint_headers)))
        _endpoint_headers[key] = headers
    return headers


def forget_endpoint_headers(key: EndpointKey) -> None:
    """Drop the cached headers for a sandbox port, so the next use fetches them again."""
    with _endpoint_headers_lock:
        _endpoint_headers.pop(key, None)


def route_label(sandbox_id: str, port: int) -> str:
    """``<sandbox id>-<port>``: the gateway's name for a sandbox port, and the advertised host's first label."""
    label = f"{sandbox_id}-{port}"
    if len(label) > _MAX_HOST_LABEL_LENGTH or _ROUTE_LABEL.fullmatch(label) is None:
        raise ValueError(f"sandbox id {sandbox_id!r} and port {port!r} don't form a host name")
    return label


def parse_route_label(label: str) -> tuple[str, int] | None:
    """Sandbox id and port of a route label, or None when ``label`` is not one."""
    route = _ROUTE_LABEL.fullmatch(label)
    return (route["sandbox"], int(route["port"])) if route else None


def sandbox_endpoint_host(sandbox_id: str, port: int) -> str:
    """Advertised host for ``port`` of sandbox ``sandbox_id``."""
    return f"{route_label(sandbox_id, port)}.{ENDPOINT_HOST_SUFFIX}"


def sandbox_endpoint_url(sandbox_id: str, port: int) -> str:
    """Advertised URL for ``port`` of sandbox ``sandbox_id``."""
    return f"http://{sandbox_endpoint_host(sandbox_id, port)}"


def sandbox_route(host: str) -> tuple[str, int] | None:
    """Sandbox id and port named by an advertised host, or None when ``host`` is not one."""
    label, _, suffix = host.partition(".")
    return parse_route_label(label) if suffix == ENDPOINT_HOST_SUFFIX else None


class SandboxEndpointAuth(httpx.Auth):
    """Routes and authorizes requests that carry an advertised sandbox host in ``Host``.

    Sets the gateway's routing header from that host and adds the headers the OpenSandbox server
    returns for the sandbox port. The request path takes no part in routing.

    Raises ``httpx.ConnectError`` without sending when the request is not for the gateway, its
    host is not an advertised sandbox host, the port is reserved, or the header lookup fails.
    """

    def __init__(self, *, server_base: str, gateway_base: str, api_key_env: str) -> None:
        self._server_base = server_base.rstrip("/")
        gateway = httpx.URL(gateway_base)
        self._gateway_origin = (gateway.scheme, gateway.host, gateway.port)
        self._api_key_env = api_key_env

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        raise RuntimeError("SandboxEndpointAuth supports async clients only")

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        try:
            sandbox_id, port = self._route(request)
            key = (self._server_base, sandbox_id, port)
            api_key = os.environ.get(self._api_key_env)
            if not api_key:
                raise RuntimeError(f"{self._api_key_env} is not set")
            headers = await endpoint_headers(key, api_key)
        except Exception as exc:
            logger.warning("Could not authorize a request to a sandbox endpoint.", exc_info=True)
            # The message is returned to API clients.
            raise httpx.ConnectError("sandbox endpoint authorization is unavailable", request=request) from exc
        for name, value in headers.items():
            request.headers[name] = value
        request.headers.pop(_LEGACY_INGRESS_HEADER, None)
        request.headers[INGRESS_HEADER] = route_label(sandbox_id, port)
        response = yield request
        if response.status_code == 401:
            forget_endpoint_headers(key)

    def _route(self, request: httpx.Request) -> tuple[str, int]:
        url = request.url
        route = sandbox_route(request.headers.get("host", ""))
        if route is None or (url.scheme, url.host, url.port) != self._gateway_origin:
            raise ValueError("the request is not for a sandbox endpoint on the OpenSandbox gateway")
        if route[1] in RESERVED_PORTS:
            raise ValueError(f"port {route[1]} is reserved by OpenSandbox")
        return route


@dataclass(frozen=True)
class OpenSandboxEndpointTransport(EndpointTransport):
    """Transport for advertised sandbox endpoints. ``connect_base`` is the ingress gateway."""

    server_base: str = ""
    api_key_env: str = "OPEN_SANDBOX_API_KEY"

    def target(self, url: str) -> tuple[str, dict[str, str]]:
        """Dial an advertised sandbox URL at the gateway, keeping its host. Other URLs are returned unchanged."""
        if self.connect_base is None or sandbox_route(urlparse(url).netloc) is None:
            return url, {}
        return super().target(url)

    def client_kwargs(self) -> dict[str, Any]:
        return {
            **super().client_kwargs(),
            "auth": SandboxEndpointAuth(
                server_base=self.server_base, gateway_base=self.connect_base or "", api_key_env=self.api_key_env
            ),
        }
