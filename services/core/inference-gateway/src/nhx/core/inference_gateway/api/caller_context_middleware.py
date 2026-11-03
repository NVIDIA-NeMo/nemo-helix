# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Puts a router's caller back in place on each call it makes back through the gateway, before platform auth runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from fastapi import HTTPException
from nhx.core.inference_gateway.api import dependencies
from nhx.core.inference_gateway.api.caller_context import (
    CALLER_CONTEXT_HEADER,
    CallerContextSeal,
    caller_headers,
    gateway_seal,
)
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

GATEWAY_PATH_PREFIX = "/apis/inference-gateway/"


def _router_keys() -> Mapping[str, str | None]:
    model_cache = dependencies._MODEL_CACHE
    return model_cache.router_keys() if model_cache is not None else {}


class CallerContextMiddleware:
    """Authenticates a router's call back as the caller it carries the context of.

    Installed outside the platform's authorization middleware, so a call back is held to the caller's credential:
    its scopes, whether it is still live, and its permissions.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        router_keys: Callable[[], Mapping[str, str | None]] = _router_keys,
        seal: Callable[[], CallerContextSeal] = gateway_seal,
    ) -> None:
        self._app = app
        self._router_keys = router_keys
        self._seal = seal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(GATEWAY_PATH_PREFIX):
            await self._app(scope, receive, send)
            return
        raw: list[tuple[bytes, bytes]] = scope["headers"]
        headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in raw}
        router_keys = self._router_keys()
        if CALLER_CONTEXT_HEADER not in headers and not router_keys:
            await self._app(scope, receive, send)
            return
        try:
            acting = caller_headers(headers, router_keys, self._seal())
        except HTTPException as exc:
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(scope, receive, send)
            return
        if CALLER_CONTEXT_HEADER in headers:
            credentials = ("authorization", "x-api-key")
            kept = [
                (name, value)
                for name, value in raw
                if name.decode("latin-1").lower() not in {CALLER_CONTEXT_HEADER, *credentials}
            ]
            kept += [(name.encode("latin-1"), acting[name].encode("latin-1")) for name in credentials if name in acting]
            scope = {**scope, "headers": kept}
        await self._app(scope, receive, send)
