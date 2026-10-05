# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP forwarding shared by deployed workloads and plugin job runtimes."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

_READ_TIMEOUT_ENVVAR = "NHX_AUTH_PROXY_READ_TIMEOUT"
_PRINCIPAL_ID_HEADER = "x-nhx-principal-id"
_PRINCIPAL_ACCOUNT_ID_HEADER = "x-nhx-actor-account-id"
_PRINCIPAL_EMAIL_HEADER = "x-nhx-principal-email"
_PRINCIPAL_GROUPS_HEADER = "x-nhx-principal-groups"
_PRINCIPAL_ALIASES_HEADER = "x-nhx-actor-aliases"
_ON_BEHALF_OF_HEADER = "x-nhx-principal-on-behalf-of"
# Companion metadata for the on-behalf-of principal. The platform derives the
# delegated user's groups/email from these (Principal.from_headers -> effective_*),
# and effective_groups/effective_email feed the PDP authorization input. We stamp
# only the OBO id here, so any inbound companion headers are untrusted and must be
# dropped — otherwise a colocated workload could pair our stamped OBO id with
# attacker-chosen groups/email and be evaluated with those, defeating the scoping.
_ON_BEHALF_OF_EMAIL_HEADER = "x-nhx-principal-on-behalf-of-email"
_ON_BEHALF_OF_GROUPS_HEADER = "x-nhx-principal-on-behalf-of-groups"
_ON_BEHALF_OF_ACCOUNT_ID_HEADER = "x-nhx-subject-account-id"
_ON_BEHALF_OF_ALIASES_HEADER = "x-nhx-subject-aliases"
_SCOPES_HEADER = "x-nhx-scopes"

# Request-header sanitization drops identity, framing, and hop-by-hop metadata:
# - the workload's own credential / principal / on-behalf-of headers (we set the
#   identity), so they can't be spoofed or conflict with what we stamp;
# - host, which httpx recomputes for the upstream request.
# Preserve Content-Length for unchanged request bodies so known-length uploads
# do not become chunked on their way to the gateway.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_STRIP_REQUEST_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    {
        "host",
        "authorization",
        _PRINCIPAL_ID_HEADER,
        _PRINCIPAL_ACCOUNT_ID_HEADER,
        _PRINCIPAL_EMAIL_HEADER,
        _PRINCIPAL_GROUPS_HEADER,
        _PRINCIPAL_ALIASES_HEADER,
        _ON_BEHALF_OF_HEADER,
        _ON_BEHALF_OF_EMAIL_HEADER,
        _ON_BEHALF_OF_GROUPS_HEADER,
        _ON_BEHALF_OF_ACCOUNT_ID_HEADER,
        _ON_BEHALF_OF_ALIASES_HEADER,
        _SCOPES_HEADER,
    }
)
# We stream the response, so the upstream's framing headers no longer apply.
_STRIP_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | frozenset({"content-length"})
_MULTI_VALUE_RESPONSE_HEADERS = frozenset({"set-cookie"})


def _sanitized_headers(headers: Mapping[str, str], *, strip: frozenset[str]) -> dict[str, str]:
    """Remove fixed and Connection-nominated hop-by-hop headers."""
    connection_tokens = {
        token.strip().lower()
        for key, value in headers.items()
        if key.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    blocked = strip | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def build_auth_proxy_app(
    *,
    base_url: str,
    headers: Mapping[str, str] | None = None,
    auth: httpx.Auth | None = None,
) -> FastAPI:
    """Forward with trusted identity headers or per-request transport auth.

    Caller-supplied identity is stripped before either mode is applied. The
    upstream is fixed; redirects are returned to the caller, never followed.
    """
    if (headers is not None) == (auth is not None):
        raise ValueError("Provide exactly one of headers or auth")
    identity_headers = {name.lower(): value for name, value in (headers or {}).items()}
    read_timeout = float(os.environ.get(_READ_TIMEOUT_ENVVAR, "300"))
    timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=10.0)
    client = httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=False, auth=auth)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="nhx-auth-proxy", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def forward(request: Request, path: str) -> StreamingResponse:
        headers = _sanitized_headers(request.headers, strip=_STRIP_REQUEST_HEADERS)
        # Transfer-Encoding takes precedence over Content-Length on ingress.
        # Let httpx frame such streams instead of forwarding a conflicting size.
        if "transfer-encoding" in request.headers:
            headers.pop("content-length", None)
        headers.update(identity_headers)
        url = httpx.URL(path="/" + path, query=request.url.query.encode("utf-8"))
        # Stream request bodies to keep a co-located caller from forcing the
        # privileged proxy to buffer an unbounded payload in memory.
        has_body = "content-length" in request.headers or "transfer-encoding" in request.headers
        upstream = client.build_request(
            request.method, url, headers=headers, content=request.stream() if has_body else None
        )
        response = await client.send(upstream, stream=True)

        async def _body() -> AsyncIterator[bytes]:
            # The finally runs on normal completion, exception, and client
            # disconnect (Starlette closes the generator), so this is the only
            # cleanup the response needs.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        resp_headers = _sanitized_headers(
            response.headers,
            strip=_STRIP_RESPONSE_HEADERS | _MULTI_VALUE_RESPONSE_HEADERS,
        )
        proxy_response = StreamingResponse(
            _body(),
            status_code=response.status_code,
            headers=resp_headers,
        )
        for cookie in response.headers.get_list("set-cookie"):
            proxy_response.headers.append("set-cookie", cookie)
        return proxy_response

    return app
