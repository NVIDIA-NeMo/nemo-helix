# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP request logging middleware."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .otel import INTERNAL_REQUEST_HEADER, otel_headers_context


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Custom request logging middleware that logs HTTP requests using structured logging.

    Requests with the X-NHX-Internal header are marked with an `internal` flag,
    allowing internal controller-to-service calls to be filtered out from access logs.
    The header is also propagated to downstream service calls via otel_headers_context.
    """

    def __init__(self, app, logger_name: str = "nhx.request"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        internal_header_value = request.headers.get(INTERNAL_REQUEST_HEADER)
        is_internal = bool(internal_header_value)

        token = None
        if is_internal:
            token = otel_headers_context.set({INTERNAL_REQUEST_HEADER: internal_header_value})

        try:
            response = await call_next(request)
        finally:
            if token is not None:
                otel_headers_context.reset(token)

        duration = time.time() - start_time

        service = getattr(request.state, "service", None)
        workspace = getattr(request.state, "workspace", None)

        extra = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": round(duration, 3),
            "service": service,
            "workspace": workspace,
            "client": request.client.host if request.client else "-",
            "port": request.client.port if request.client else "-",
            "http_version": request.scope.get("http_version", "1.1"),
        }
        if is_internal:
            extra["internal"] = True

        if (
            not request.url.path.startswith("/health")
            and not request.url.path.startswith("/metrics")
            and request.url.path != "/status"
        ):
            self.logger.info("Request completed", extra=extra)
        return response
