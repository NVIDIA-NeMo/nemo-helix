# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ephemeral caller-context broker for the bundled Studio stdio tool process.

Only a revocable, invocation-scoped capability traverses Fabric's adapter
protocol (which can be persisted in runtime artifacts). Caller credentials stay
in this serving process until the trusted stdio peer redeems the capability.
"""

import secrets
from collections.abc import Callable
from ipaddress import ip_address
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

CONTEXT_KEY = "nhx_tool_authority"
BROKER_PATH = "/_internal/studio-tool-authority"


class ToolAuthorityBroker:
    def __init__(self, app: ASGIApp, enabled: Callable[[], bool]) -> None:
        self.app = app
        self.enabled = enabled
        self._active: dict[str, dict[str, Any]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if scope["path"] == BROKER_PATH:
            token = headers.get("authorization", "").removeprefix("Bearer ")
            value = self._active.get(token)
            peer = scope.get("client")
            try:
                local_peer = peer is not None and ip_address(peer[0]).is_loopback
            except ValueError:
                local_peer = False
            # A persisted adapter artifact must not turn into a public credential
            # exchange. Only the local stdio peer may redeem, never the gateway
            # (which can itself dial this server over loopback in subprocess mode).
            allowed = (
                value is not None and scope["method"] == "GET" and local_peer and "x-nhx-tool-workspace" not in headers
            )
            response = JSONResponse(
                value if allowed else {"detail": "invalid tool capability"},
                status_code=200 if allowed else 401,
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        if not self.enabled() or scope["path"] != "/v1/chat/completions" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        server = scope.get("server")
        port = server[1] if server else None
        token = secrets.token_urlsafe(32)
        self._active[token] = {
            "workspace": headers.get("x-nhx-tool-workspace"),
            "headers": {key: headers[key] for key in ("authorization", "cookie") if key in headers},
            "studio_session": headers.get("x-nhx-studio-session"),
            "studio_capability": headers.get("x-nhx-studio-capability"),
        }
        scope.setdefault("state", {})[CONTEXT_KEY] = {"broker_port": port, "capability": token}
        try:
            # ASGI completion includes the full streaming response and disconnect
            # cleanup, unlike BaseHTTPMiddleware's call_next boundary.
            await self.app(scope, receive, send)
        finally:
            self._active.pop(token, None)
