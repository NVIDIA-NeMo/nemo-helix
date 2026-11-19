# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming reverse proxy from ``/apis/observability/ui/<connection>`` to the upstream UI."""

from __future__ import annotations

import logging
from http.cookies import CookieError, SimpleCookie
from urllib.parse import unquote

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from nemo_observability_plugin.adapters.base import ProviderAdapter
from nemo_observability_plugin.config import ObservabilityConnection
from nemo_observability_plugin.session import COOKIE_NAME
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

UI_PREFIX = "/apis/observability/ui"

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
# The caller's platform credentials must never reach a third-party service.
_REQUEST_DROP = _HOP_BY_HOP | {"host", "authorization", "cookie", "content-length", "x-api-key"}
_RESPONSE_DROP = _HOP_BY_HOP | {"content-length"}


def connection_prefix(connection: ObservabilityConnection) -> str:
    return f"{UI_PREFIX}/{connection.name}"


def upstream_url(request: Request, connection: ObservabilityConnection, adapter: ProviderAdapter) -> str:
    """Upstream URL for *request*, built from the raw (still percent-encoded) request path."""
    prefix = connection_prefix(connection)
    raw_path = request.scope.get("raw_path", request.url.path.encode()).decode("latin-1")
    if raw_path != prefix and not raw_path.startswith(prefix + "/"):
        raise HTTPException(status_code=404, detail="Not found")
    rest = raw_path[len(prefix) :]
    segments = unquote(rest).replace("\\", "/").split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise HTTPException(status_code=400, detail="Invalid path")
    path = rest if adapter.strip_prefix else prefix + rest
    url = connection.base_url + (path or "/")
    query = request.scope.get("query_string", b"")
    return f"{url}?{query.decode('latin-1')}" if query else url


def forward_headers(
    request: Request, connection: ObservabilityConnection, adapter: ProviderAdapter, credential: str | None
) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _REQUEST_DROP and not key.lower().startswith("x-nhx-")
    }
    if cookie := _upstream_cookies(request.headers.get("cookie")):
        headers["cookie"] = cookie
    # Upstream CSRF checks compare Origin to their own origin; the browser saw Studio's.
    if headers.get("origin") == _request_origin(request):
        headers["origin"] = connection.base_url
    headers.update(adapter.upstream_headers(credential))
    return headers


async def proxy(
    request: Request,
    client: httpx.AsyncClient,
    connection: ObservabilityConnection,
    adapter: ProviderAdapter,
    credential: str | None,
) -> StreamingResponse:
    url = upstream_url(request, connection, adapter)
    upstream_request = client.build_request(
        request.method,
        url,
        headers=forward_headers(request, connection, adapter, credential),
        content=request.stream() if request.method not in ("GET", "HEAD", "OPTIONS") else None,
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        logger.warning("Observability connection %r unreachable: %s", connection.name, exc)
        raise HTTPException(status_code=502, detail=f"Trace explorer {connection.name!r} is unreachable.") from exc
    headers = [
        (key, _rewrite_location(value, connection) if key.lower() == "location" else value)
        for key, value in upstream.headers.multi_items()
        if key.lower() not in _RESPONSE_DROP
    ]
    response = StreamingResponse(
        upstream.aiter_raw(), status_code=upstream.status_code, background=BackgroundTask(upstream.aclose)
    )
    # Raw bytes keep their content-encoding; multi_items keeps repeated Set-Cookie headers.
    response.raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers]
    return response


def _upstream_cookies(cookie_header: str | None) -> str:
    """The Cookie header minus the plugin's session cookie and platform cookies."""
    if not cookie_header:
        return ""
    jar = SimpleCookie()
    try:
        jar.load(cookie_header)
    except CookieError:
        return ""
    return "; ".join(
        f"{name}={morsel.coded_value}"
        for name, morsel in jar.items()
        if name != COOKIE_NAME and not name.startswith("nhx_")
    )


def _request_origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def _rewrite_location(location: str, connection: ObservabilityConnection) -> str:
    """Keep redirects on the Studio origin instead of sending the frame to the upstream host."""
    if location == connection.base_url or location.startswith(connection.base_url + "/"):
        return location[len(connection.base_url) :] or "/"
    return location
