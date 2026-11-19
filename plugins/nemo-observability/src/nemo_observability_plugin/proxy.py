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
# The caller's platform credentials must never reach a third-party service. Content-Length is
# kept: without it the streamed body goes out chunked, which WSGI upstreams (MLflow) read as empty.
_REQUEST_DROP = _HOP_BY_HOP | {"host", "authorization", "cookie", "x-api-key"}
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
    # Bodies pass through still encoded, so only offer the encodings the caller accepts;
    # otherwise httpx's default Accept-Encoding gets a gzip body to a client that can't read it.
    if not headers.get("accept-encoding"):
        headers["accept-encoding"] = "identity"
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
        logger.warning("Observability connection unreachable", extra={"connection": connection.name, "error": str(exc)})
        raise HTTPException(status_code=502, detail=f"Trace explorer {connection.name!r} is unreachable.") from exc
    headers: list[tuple[str, str]] = []
    for key, value in upstream.headers.multi_items():
        name = key.lower()
        if name == "location":
            headers.append((key, _rewrite_location(value, connection, adapter)))
        elif name == "set-cookie":
            if (cookie := _confine_set_cookie(value, connection)) is not None:
                headers.append((key, cookie))
        elif name not in _RESPONSE_DROP:
            headers.append((key, value))
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


def _confine_set_cookie(set_cookie: str, connection: ObservabilityConnection) -> str | None:
    """Scope an upstream cookie to the connection's prefix on the Studio origin, or drop it.

    The browser files upstream cookies under Studio's origin, so a broad ``Path`` or a
    ``Domain`` would reach other Studio routes, and an ``nhx_*`` name could overwrite a platform cookie.
    """
    name_value, *attributes = (part.strip() for part in set_cookie.split(";"))
    cookie_name = name_value.split("=", 1)[0].strip()
    if not cookie_name or cookie_name == COOKIE_NAME or cookie_name.startswith("nhx_"):
        return None
    prefix = connection_prefix(connection)
    path = prefix
    kept = []
    for attribute in attributes:
        key, _, value = attribute.partition("=")
        key = key.strip().lower()
        if key == "path":
            path = value.strip() if value.strip().startswith(prefix + "/") else prefix
        elif key != "domain":
            kept.append(attribute)
    return "; ".join([name_value, *kept, f"Path={path}"])


def _request_origin(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def _rewrite_location(location: str, connection: ObservabilityConnection, adapter: ProviderAdapter) -> str:
    """Keep redirects on the Studio origin and under the connection's prefix (the inverse of ``upstream_url``)."""
    base_url = connection.base_url
    if location == base_url or location.startswith(base_url + "/"):
        path = location[len(base_url) :] or "/"
    elif location.startswith("/") and not location.startswith("//"):
        path = location
    else:
        return location
    prefix = connection_prefix(connection)
    if adapter.strip_prefix and path != prefix and not path.startswith(prefix + "/"):
        return prefix + path
    return path
