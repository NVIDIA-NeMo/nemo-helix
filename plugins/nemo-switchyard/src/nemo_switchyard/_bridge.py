# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge between IGW's typed envelope fields and Switchyard's ChatRequest/ChatResponse types.

Provides helpers in two directions:

Request side (Switchyard pipeline output → InferenceRequest):
  ``write_back_request``  — apply a processed ChatRequest back into an InferenceRequest.

Response side (InferenceResponse typed_body ↔ Switchyard):
  ``_wrap_streaming``     — wrap a TypedResponseStream into a Switchyard streaming response.
  ``_wrap_non_streaming`` — wrap a non-streaming pydantic model into a Switchyard response.
  ``write_back_response`` — write a pipeline-processed ChatResponse back into an InferenceResponse.

All translation (streaming and non-streaming) goes through the Switchyard response pipeline.
``write_back_response`` handles the output of that pipeline for both shapes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import cast

import anthropic.types as anthropic_types
import openai.types.chat as openai_chat_types
from nemo_helix_plugin.inference_middleware import (
    BackendFormat,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    TypedResponseResult,
)
from nemo_switchyard._processors import CTX_PATH_UPDATE
from nhx.core.inference_gateway.api.typed_response import TypedResponseStream
from switchyard.lib.chat_request.base import ChatRequest
from switchyard.lib.chat_response.anthropic import (
    AnthropicChatResponse,
    AnthropicResponseStream,
    AnthropicStreamingChatResponse,
)
from switchyard.lib.chat_response.base import ChatResponse
from switchyard.lib.chat_response.openai_chat import (
    CompletionChatResponse,
    ResponseStream,
    StreamingChatResponse,
)
from switchyard.lib.chat_response.openai_responses import ResponsesApiStreamingChatResponse
from switchyard.lib.proxy_context import ProxyContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


def write_back_request(
    request: InferenceRequest,
    processed: ChatRequest,
    sy_context: ProxyContext,
) -> None:
    """Apply Switchyard pipeline output back to an InferenceRequest.

    Sets both ``body`` and ``typed_body`` to the same dict object from the processed
    ``ChatRequest`` — keeping them in sync with no possibility of drift — then applies
    any path update stamped by ``PathUpdateProcessor``.
    """
    request.body = processed.body
    request.typed_body = processed.body  # same object — always in sync
    path_update = sy_context.metadata.get(CTX_PATH_UPDATE)
    if path_update:
        request.path = path_update


# ---------------------------------------------------------------------------
# Response side
# ---------------------------------------------------------------------------


def _wrap_streaming(
    typed_body: TypedResponseStream,
    backend_format: BackendFormat,
) -> StreamingChatResponse | AnthropicStreamingChatResponse | None:
    """Wrap a ``TypedResponseStream`` into the matching Switchyard streaming wrapper.

    Dispatches on ``backend_format`` (from ``ctx.backend_format``). Returns ``None``
    for unrecognised formats so the caller can pass through unchanged.
    """
    if backend_format == BackendFormat.OPENAI_CHAT:
        return StreamingChatResponse(
            ResponseStream(cast(AsyncIterator[openai_chat_types.ChatCompletionChunk], typed_body))
        )
    if backend_format == BackendFormat.ANTHROPIC_MESSAGES:
        return AnthropicStreamingChatResponse(
            AnthropicResponseStream(cast(AsyncIterator[anthropic_types.RawMessageStreamEvent], typed_body))
        )
    logger.debug("_wrap_streaming: unhandled backend_format %s", backend_format)
    return None


def _wrap_non_streaming(typed_body: TypedResponseResult) -> CompletionChatResponse | AnthropicChatResponse:
    """Wrap a non-streaming pydantic model into the matching Switchyard response wrapper.

    The Switchyard response pipeline expects a ``ChatResponse`` as input. This is the
    non-streaming equivalent of ``_wrap_streaming``: both produce a ``ChatResponse``
    that the pipeline's ``FormatTranslateResponseProcessor`` can operate on.

    Raises ``InferenceMiddlewareError`` (500) for unexpected types so any future
    gap fails loudly.
    """
    if isinstance(typed_body, openai_chat_types.ChatCompletion):
        return CompletionChatResponse(typed_body)
    if isinstance(typed_body, anthropic_types.Message):
        return AnthropicChatResponse(typed_body)
    raise InferenceMiddlewareError(
        f"Unexpected non-streaming typed_body type: {type(typed_body).__name__}",
        status_code=500,
    )


def write_back_response(response: InferenceResponse, processed: ChatResponse) -> None:
    """Write Switchyard's pipeline-processed ChatResponse back into an InferenceResponse.

    Non-streaming (``CompletionChatResponse`` / ``AnthropicChatResponse``):
        ``response.typed_body`` is updated to the translated pydantic model.

    Streaming (``StreamingChatResponse`` / ``AnthropicStreamingChatResponse`` /
    ``ResponsesApiStreamingChatResponse``):
        ``response.result`` is replaced with ``processed.stream`` directly. IGW's
        ``_sse_gen`` already calls ``_json_ready_payload(chunk)`` per item so no
        wrapping is needed. ``typed_body`` is cleared to ``None`` — there is no
        typed-iterator wrapper for translated streams yet.

    Raises ``InferenceMiddlewareError`` (500) for unexpected types so any future
    gap fails loudly.
    """
    if isinstance(processed, (CompletionChatResponse, AnthropicChatResponse)):
        response.typed_body = processed.body
        # Keep result in sync so downstream plugins that read result directly
        # (e.g. guardrails) see the translated payload, not the pre-translation dict.
        response.result = processed.body.model_dump(mode="json")
        return

    if isinstance(
        processed,
        (StreamingChatResponse, AnthropicStreamingChatResponse, ResponsesApiStreamingChatResponse),
    ):
        response.result = processed.stream
        response.typed_body = None  # intentional: no typed wrapper for translated streams
        return

    raise InferenceMiddlewareError(
        f"Unexpected ChatResponse type from pipeline: {type(processed).__name__}",
        status_code=500,
    )
