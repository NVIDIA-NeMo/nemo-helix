# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability plugin service, mounted at ``/apis/observability``.

Authorization note: the ``/ui`` proxy routes are authenticated by the signed
session cookie, checked here. The platform's auth middleware accepts only bearer
tokens or trusted principal headers, so with auth enabled a frame's cookie-only
requests are rejected before reaching this handler. The POC targets local mode
(auth disabled); supporting auth-enabled deployments needs a platform-level way
for a plugin to declare a cookie-authenticated prefix.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.service import NemoService, RouterSpec
from nemo_observability_plugin import proxy
from nemo_observability_plugin.adapters.registry import adapter_for
from nemo_observability_plugin.authz import ConnectionPerms, UIPerms, scope
from nemo_observability_plugin.config import ObservabilityConfig, ObservabilityConnection
from nemo_observability_plugin.session import COOKIE_NAME, SessionSigner
from pydantic import BaseModel

_PROXY_READ_METHODS = ["GET", "HEAD", "OPTIONS"]
_PROXY_WRITE_METHODS = ["POST", "PUT", "PATCH", "DELETE"]


class ConnectionSummary(BaseModel):
    name: str
    provider: str
    display_name: str
    ui_prefix: str
    frame_url: str


class ConnectionList(BaseModel):
    data: list[ConnectionSummary]


class UISessionResponse(BaseModel):
    frame_url: str
    expires_at: int


class OTLPSnippet(BaseModel):
    """Paste ``yaml`` into an agent's ``agent.yaml``; ``upstream_setup`` lists what the upstream needs."""

    yaml: str
    telemetry: dict[str, Any]
    upstream_setup: list[str]


class ObservabilityService(NemoService):
    name: ClassVar[str] = "observability"
    dependencies: ClassVar[list[str]] = []

    def __init__(self) -> None:
        super().__init__()
        self._client: httpx.AsyncClient | None = None
        self._signer: SessionSigner | None = None

    async def on_startup(self) -> None:
        config = ObservabilityConfig.get()
        self._client = httpx.AsyncClient(
            # Never follow redirects: the browser does, against the rewritten Location.
            follow_redirects=False,
            timeout=httpx.Timeout(connect=10.0, read=config.upstream_timeout_seconds, write=60.0, pool=10.0),
        )

    async def on_shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def signer(self) -> SessionSigner:
        if self._signer is None:
            secret = ObservabilityConfig.get().session_secret
            self._signer = SessionSigner(secret.get_secret_value().encode() if secret else None)
        return self._signer

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise HTTPException(status_code=503, detail="Observability proxy is not started.")
        return self._client

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                self._build_connections_router(),
                tag="Observability",
                description="External trace explorer connections embedded in Studio.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                self._build_ui_router(),
                tag="Observability UI Proxy",
                description="Same-origin proxy to an external trace explorer, authorized by a session cookie.",
            ),
        ]

    def _build_connections_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/connections", response_model=ConnectionList)
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ConnectionPerms.LIST])
        async def list_connections(workspace: str) -> ConnectionList:
            """List configured trace explorers. Never includes upstream URLs or credentials."""
            return ConnectionList(data=[_summary(c) for c in ObservabilityConfig.get().connections])

        @router.post("/connections/{name}/session", response_model=UISessionResponse)
        @scope.write
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ConnectionPerms.READ])
        async def create_ui_session(
            workspace: str,
            name: str,
            request: Request,
            response: Response,
            principal_id: str = Depends(get_effective_principal_id),
        ) -> UISessionResponse:
            """Set the cookie that lets a Studio frame load this connection's proxied UI."""
            connection = _connection_or_404(name)
            config = ObservabilityConfig.get()
            token, session = self.signer.mint(principal_id, workspace, name, config.session_ttl_seconds)
            response.set_cookie(
                COOKIE_NAME,
                token,
                max_age=config.session_ttl_seconds,
                path=proxy.connection_prefix(connection),
                httponly=True,
                samesite="strict",
                secure=_is_https(request),
            )
            return UISessionResponse(frame_url=_summary(connection).frame_url, expires_at=session.expires_at)

        @router.get("/connections/{name}/otlp-snippet", response_model=OTLPSnippet)
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ConnectionPerms.READ])
        async def get_otlp_snippet(workspace: str, name: str) -> OTLPSnippet:
            """``telemetry`` stanza that exports an agent's traces straight to this connection."""
            connection = _connection_or_404(name)
            adapter = adapter_for(connection.provider)
            endpoint = adapter.otlp_endpoint(connection)
            telemetry: dict[str, Any] = {"enabled": True, "opentelemetry": {"enabled": True, "endpoints": [endpoint]}}
            if insecure_host := _insecure_http_host(endpoint["endpoint"]):
                telemetry["insecure_http_hosts"] = [insecure_host]
            return OTLPSnippet(
                yaml=yaml.safe_dump({"telemetry": telemetry}, sort_keys=False),
                telemetry=telemetry,
                upstream_setup=adapter.upstream_setup(connection, proxy.connection_prefix(connection)),
            )

        return router

    def _build_ui_router(self) -> APIRouter:
        router = APIRouter()

        async def _serve(request: Request, name: str) -> StreamingResponse:
            connection = _connection_or_404(name)
            if self.signer.verify(request.cookies.get(COOKIE_NAME), name) is None:
                raise HTTPException(status_code=401, detail="Trace explorer session missing or expired.")
            credential = os.environ.get(connection.secret_env) if connection.secret_env else None
            return await proxy.proxy(request, self.client, connection, adapter_for(connection.provider), credential)

        @router.api_route("/ui/{name}", methods=_PROXY_READ_METHODS, include_in_schema=False)
        @router.api_route("/ui/{name}/{path:path}", methods=_PROXY_READ_METHODS, include_in_schema=False)
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[UIPerms.READ])
        async def proxy_read(request: Request, name: str, path: str = "") -> StreamingResponse:
            return await _serve(request, name)

        @router.api_route("/ui/{name}", methods=_PROXY_WRITE_METHODS, include_in_schema=False)
        @router.api_route("/ui/{name}/{path:path}", methods=_PROXY_WRITE_METHODS, include_in_schema=False)
        @scope.write
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[UIPerms.READ])
        async def proxy_write(request: Request, name: str, path: str = "") -> StreamingResponse:
            return await _serve(request, name)

        return router


def _connection_or_404(name: str) -> ObservabilityConnection:
    connection = ObservabilityConfig.get().connection(name)
    if connection is None:
        raise HTTPException(status_code=404, detail=f"Observability connection {name!r} not found.")
    return connection


def _summary(connection: ObservabilityConnection) -> ConnectionSummary:
    adapter = adapter_for(connection.provider)
    prefix = proxy.connection_prefix(connection)
    return ConnectionSummary(
        name=connection.name,
        provider=connection.provider,
        display_name=connection.display_name or adapter.display_name,
        ui_prefix=prefix,
        frame_url=prefix + adapter.landing_path(connection),
    )


def _is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return (forwarded or request.url.scheme) == "https"


def _insecure_http_host(endpoint: str) -> str | None:
    """Host to list in ``insecure_http_hosts``: plain HTTP to anything but loopback needs opting in."""
    parsed = httpx.URL(endpoint)
    if parsed.scheme != "http" or parsed.host in {"localhost", "127.0.0.1", "::1"}:
        return None
    return parsed.host
