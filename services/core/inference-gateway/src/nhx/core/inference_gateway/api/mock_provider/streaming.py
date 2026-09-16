# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming response conversion for mock provider mode."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


async def convert_to_streaming_chunks(response_body: dict) -> AsyncIterator[bytes]:
    """Convert a non-streaming chat completion response to SSE streaming chunks.

    Args:
        response_body: The non-streaming chat completion response body (dict)

    Yields:
        SSE-formatted chunks as bytes: "data: {json}\\n\\n"
    """
    # Extract chunk metadata from response
    chat_id = response_body.get("id", f"chatcmpl-{uuid.uuid4()}")
    model = response_body.get("model", "mock-model")
    created = response_body.get("created", int(time.time()))

    # We'll create chunks from the message content
    choices = response_body.get("choices", [])
    if not choices:
        # No choices - log warning and return empty stream
        logger.warning("No choices in mock response body, returning empty stream")
        return

    message = choices[0].get("message", {})
    content = message.get("content", "")

    # Initial chunk with role and empty content
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n".encode()

    # Create chunks for each word to simulate streaming behavior
    words = content.split() if content else []
    for i, word in enumerate(words):
        # Add space after word except for the last word to avoid trailing space
        word_content = f"{word} " if i < len(words) - 1 else word
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": word_content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

    # Final chunk with finish_reason
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n".encode()

    # Terminal marker
    yield b"data: [DONE]\n\n"
