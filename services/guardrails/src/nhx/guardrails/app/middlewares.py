# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from opentelemetry import trace

logger = logging.getLogger(__name__)


tracer = trace.get_tracer(__name__)


def is_health_check(request: Request) -> bool:
    return request.method == "GET" and str(request.url).endswith("/health")


async def capture_trace_id(request: Request, call_next: Callable) -> Response:
    trace_id = trace.get_current_span().get_span_context().trace_id
    request.state.trace_id = trace_id

    request_identifier = f"{id(request)}-{request.url.path}"
    logger.debug(f"{request_identifier} - Middleware before call_next: trace_id = {format(trace_id, 'x')}")

    response = await call_next(request)

    logger.debug(f"{request_identifier} - Middleware after call_next: trace_id = {format(trace_id, 'x')}")
    response.headers["X-Trace-Id"] = format(trace_id, "x")

    return response


async def add_request_id_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id")
    if not request_id:
        request_id = f"req_{uuid.uuid4()}"
    request.state.request_id = request_id

    request_id_short = request_id.split("-")[0]
    request_identifier = f"{id(request)}-{request.url.path}"

    logger.debug(f"{request_identifier} - Middleware before call_next: request_id = {request_id_short}")
    response = await call_next(request)
    logger.debug(f"{request_identifier} - Middleware after call_next: request_id = {request_id_short}")
    response.headers["X-Request-Id"] = request_id
    return response
