# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mock_provider streaming module."""

import json

import pytest
from nhx.core.inference_gateway.api.mock_provider.streaming import convert_to_streaming_chunks


@pytest.mark.asyncio
async def test_convert_to_streaming_chunks_basic():
    """Test basic conversion of non-streaming response to SSE chunks."""
    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
    }

    chunks = []
    async for chunk in convert_to_streaming_chunks(response_body):
        chunks.append(chunk)

    # Should have: initial chunk, content chunks (2 words), final chunk, [DONE]
    assert len(chunks) == 5

    # Decode and verify format
    stream_content = b"".join(chunks).decode("utf-8")
    lines = [line for line in stream_content.strip().split("\n\n") if line and line.startswith("data: ")]

    # First chunk: role and empty content
    first_data = lines[0].replace("data: ", "")
    first_chunk = json.loads(first_data)
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_chunk["id"] == "chatcmpl-test"
    assert first_chunk["model"] == "test-model"
    assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
    assert first_chunk["choices"][0]["delta"]["content"] == ""  # Empty content in first chunk
    assert first_chunk["choices"][0]["finish_reason"] is None

    # Content chunks: "Hello " and "world"
    content_lines = [line for line in lines[1:-2] if line and not line.startswith("data: [DONE]")]
    assert len(content_lines) == 2

    hello_chunk = json.loads(content_lines[0].replace("data: ", ""))
    assert hello_chunk["choices"][0]["delta"]["content"] == "Hello "
    assert hello_chunk["choices"][0]["finish_reason"] is None

    world_chunk = json.loads(content_lines[1].replace("data: ", ""))
    assert world_chunk["choices"][0]["delta"]["content"] == "world"

    # Final chunk: finish_reason
    final_data = lines[-2].replace("data: ", "")
    final_chunk = json.loads(final_data)
    assert final_chunk["choices"][0]["finish_reason"] == "stop"
    assert final_chunk["choices"][0]["delta"] == {}

    # Terminal marker
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_convert_to_streaming_chunks_empty_content():
    """Test conversion with empty content produces exactly 3 chunks with no content deltas."""
    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
    }

    chunks = []
    async for chunk in convert_to_streaming_chunks(response_body):
        chunks.append(chunk)

    stream_content = b"".join(chunks).decode("utf-8")
    lines = [line for line in stream_content.strip().split("\n\n") if line and line.startswith("data: ")]

    # Should have exactly: initial chunk, final chunk, [DONE] (no content chunks)
    assert len(lines) == 3

    # Chunk 1: Initial chunk with role and empty content
    initial_chunk = json.loads(lines[0].replace("data: ", ""))
    assert initial_chunk["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert initial_chunk["choices"][0]["finish_reason"] is None

    # Chunk 2: Final chunk with empty delta and finish_reason
    final_chunk = json.loads(lines[1].replace("data: ", ""))
    assert final_chunk["choices"][0]["delta"] == {}
    assert final_chunk["choices"][0]["finish_reason"] == "stop"

    # Chunk 3: [DONE] marker
    assert lines[2] == "data: [DONE]"


@pytest.mark.asyncio
async def test_convert_to_streaming_chunks_no_choices():
    """Test conversion with no choices (should return empty stream)."""
    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "test-model",
        "choices": [],
    }

    chunks = []
    async for chunk in convert_to_streaming_chunks(response_body):
        chunks.append(chunk)

    # Should return empty stream
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_convert_to_streaming_chunks_preserves_metadata():
    """Test that metadata from response is preserved in all chunks."""
    response_body = {
        "id": "chatcmpl-custom-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "custom-model-name",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Test"}, "finish_reason": "stop"}],
    }

    chunks = []
    async for chunk in convert_to_streaming_chunks(response_body):
        chunks.append(chunk)

    stream_content = b"".join(chunks).decode("utf-8")
    lines = [
        line
        for line in stream_content.strip().split("\n\n")
        if line and line.startswith("data: ") and not line.startswith("data: [DONE]")
    ]

    # Should have: initial, "Test", final
    assert len(lines) == 3

    # Verify all chunks preserve the custom metadata
    expected_metadata = {
        "id": "chatcmpl-custom-id",
        "model": "custom-model-name",
        "created": 1234567890,
    }

    for i, line in enumerate(lines):
        chunk = json.loads(line.replace("data: ", ""))
        assert chunk["id"] == expected_metadata["id"], f"Chunk {i} has wrong id"
        assert chunk["model"] == expected_metadata["model"], f"Chunk {i} has wrong model"
        assert chunk["created"] == expected_metadata["created"], f"Chunk {i} has wrong created timestamp"

    # Verify content chunk has expected content
    content_chunk = json.loads(lines[1].replace("data: ", ""))
    assert content_chunk["choices"][0]["delta"]["content"] == "Test"
