# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint transport for sandbox ports behind the OpenSandbox ingress gateway.

Advertised URL: ``http://<sandbox id>-<port>.osb.helix.internal``.
Dialed URL: ``<gateway>/<sandbox id>/<port>/<path>`` (``uri`` route mode).
Request headers: those the OpenSandbox server returns for the port, fetched on first use.

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
from urllib.parse import urlparse, urlunparse

import httpx
from nemo_deployments_plugin.endpoint_transport import EndpointTransport

logger = logging.getLogger(__name__)

ENDPOINT_HOST_SUFFIX = "osb.helix.internal"

_SANDBOX_ID = r"[a-z0-9][a-z0-9-]*"
_PORT = r"[0-9]{1,5}"
_HOST_LABEL = re.compile(rf"(?P<sandbox>{_SANDBOX_ID})-(?P<port>{_PORT})")
_GATEWAY_ROUTE = re.compile(rf"/(?P<sandbox>{_SANDBOX_ID})/(?P<port>{_PORT})(?:/|$)")
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


def gateway_route(path: str) -> tuple[str, int] | None:
    """Sandbox id and port of a gateway route path, or None when ``path`` is not one."""
    route = _GATEWAY_ROUTE.match(path)
    return (route["sandbox"], int(route["port"])) if route else None


def sandbox_endpoint_url(sandbox_id: str, port: int) -> str:
    """Advertised URL for ``port`` of sandbox ``sandbox_id``."""
    label = f"{sandbox_id}-{port}"
    if len(label) > _MAX_HOST_LABEL_LENGTH or _HOST_LABEL.fullmatch(label) is None:
        raise ValueError(f"sandbox id {sandbox_id!r} and port {port!r} don't form a host name")
    return f"http://{label}.{ENDPOINT_HOST_SUFFIX}"


class SandboxEndpointAuth(httpx.Auth):
    """Sets the OpenSandbox endpoint headers on requests to a gateway sandbox route.

    Raises ``httpx.ConnectError`` without sending when the request targets anything else or the
    header lookup fails.
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
            key = self._cache_key(request)
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
        response = yield request
        if response.status_code == 401:
            forget_endpoint_headers(key)

    def _cache_key(self, request: httpx.Request) -> EndpointKey:
        url = request.url
        route = gateway_route(url.path)
        if route is None or (url.scheme, url.host, url.port) != self._gateway_origin:
            raise ValueError("the request is not for a sandbox route on the OpenSandbox gateway")
        return self._server_base, *route


@dataclass(frozen=True)
class OpenSandboxEndpointTransport(EndpointTransport):
    """Transport for advertised sandbox endpoints. ``connect_base`` is the ingress gateway."""

    server_base: str = ""
    api_key_env: str = "OPEN_SANDBOX_API_KEY"

    def target(self, url: str) -> tuple[str, dict[str, str]]:
        """Map an advertised sandbox URL to its gateway route. Other URLs are returned unchanged."""
        parsed = urlparse(url)
        label, _, suffix = (parsed.hostname or "").partition(".")
        route = _HOST_LABEL.fullmatch(label)
        if self.connect_base is None or route is None or suffix != ENDPOINT_HOST_SUFFIX:
            return url, {}
        gateway = urlparse(self.connect_base)
        path = f"/{route['sandbox']}/{route['port']}{parsed.path or '/'}"
        return urlunparse(parsed._replace(scheme=gateway.scheme, netloc=gateway.netloc, path=path)), {}

    def client_kwargs(self) -> dict[str, Any]:
        return {
            **super().client_kwargs(),
            "auth": SandboxEndpointAuth(
                server_base=self.server_base, gateway_base=self.connect_base or "", api_key_env=self.api_key_env
            ),
        }
