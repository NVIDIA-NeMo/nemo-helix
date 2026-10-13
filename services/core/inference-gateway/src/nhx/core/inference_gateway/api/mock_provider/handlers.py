# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock request handling for mock provider mode."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from nhx.core.inference_gateway.api.mock_provider.responses import (
    MOCK_RESPONSE_HEADER,
    MOCK_RESPONSE_MAP_HEADER,
    get_call_tracker,
    get_mock_response_config,
)
from nhx.core.inference_gateway.api.mock_provider.streaming import convert_to_streaming_chunks

logger = logging.getLogger(__name__)


async def handle_mock_request(
    request: Request,
    trailing_uri: str,
    default_extra_headers: dict[str, str] | None = None,
    request_body: dict[str, Any] | None = None,
) -> JSONResponse | StreamingResponse:
    """Handle a request in mock provider mode.

    Returns a mock response instead of proxying to the real backend.
    Supports both streaming and non-streaming responses based on the `stream` parameter
    in the request body.

    Args:
        request: The incoming FastAPI request
        trailing_uri: The path suffix (e.g., "v1/chat/completions")
        default_extra_headers: Provider's default extra headers (may contain mock config)
        request_body: The middleware-modified request body. When provided this is used
            directly instead of re-reading from ``request.json()``, ensuring that body
            mutations made by request middleware are visible to the mock handler — the
            same behaviour as the real-backend proxy path. Callers should pass the
            ``json_body`` dict at the point of the mock check (before the model-name
            rewrite), so ``body["model"]`` still holds the entity ID used as the
            mock-response-map key.

    Returns:
        JSONResponse for non-streaming requests, StreamingResponse for streaming requests

    Raises:
        HTTPException: 400 if no mock response is configured and no smart default applies
    """
    logger.debug(f"Mock provider: handling {request.method} {trailing_uri}")

    # Use the caller-supplied body when available (carries middleware mutations).
    # Fall back to request.json() for call sites that don't provide a body (e.g.
    # GET /v1/models where there is no JSON body).
    if request_body is None:
        try:
            request_body = await request.json()
        except Exception:
            request_body = {}

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Mock provider entry: method=%s trailing_uri=%s body_model=%r default_extra_headers_keys=%s",
            request.method,
            trailing_uri,
            request_body.get("model"),
            sorted((default_extra_headers or {}).keys()),
        )

    # Get or create the call tracker. This is used when the mock request is configured to return
    # a dynamic response. The call tracker tracks the index of the response to return per-model.
    call_tracker = get_call_tracker(request.app.state)

    # Get mock response config (handles all response selection logic)
    try:
        mock_config = get_mock_response_config(
            request=request,
            trailing_uri=trailing_uri,
            default_extra_headers=default_extra_headers,
            request_body=request_body,
            call_tracker=call_tracker,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if mock_config is None:
        normalized_uri = trailing_uri.lstrip("/")
        raise HTTPException(
            status_code=400,
            detail=f"Mock provider mode is enabled but no mock response is configured for {request.method} {normalized_uri}. "
            f"Provide a response via the '{MOCK_RESPONSE_HEADER}' or '{MOCK_RESPONSE_MAP_HEADER}' header, "
            "or configure it on the ModelProvider's default_extra_headers.",
        )

    # Check if this is a streaming request
    is_streaming = request_body.get("stream") is True

    logger.debug(
        f"Mock provider: returning mock response with status {mock_config.status_code} (streaming={is_streaming})"
    )

    if is_streaming:
        return await _create_streaming_response(mock_config.body, mock_config.status_code)
    else:
        return JSONResponse(content=mock_config.body, status_code=mock_config.status_code)


async def _create_streaming_response(response_body: dict, status_code: int) -> StreamingResponse:
    """Create a StreamingResponse from a non-streaming response body.

    Args:
        response_body: The non-streaming chat completion response body
        status_code: HTTP status code for the response

    Returns:
        StreamingResponse with SSE-formatted chunks
    """

    async def stream_generator() -> AsyncIterator[bytes]:
        async for chunk in convert_to_streaming_chunks(response_body):
            yield chunk

    return StreamingResponse(
        stream_generator(),
        status_code=status_code,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
