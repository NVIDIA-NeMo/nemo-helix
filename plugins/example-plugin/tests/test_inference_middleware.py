# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ExampleInferenceMiddleware and related helpers.

Tests run entirely without a platform — entity client calls are mocked.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_example_plugin.middleware import (  # pyright: ignore[reportMissingImports]
    ExampleInferenceMiddleware,
    ExampleMiddlewareConfigData,
    _extract_message_text,
    _find_keyword,
    _redact_keywords,
)
from nemo_example_plugin.middleware_config import ExampleMiddlewareConfig  # pyright: ignore[reportMissingImports]
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareCacheAccessor,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    InferenceResponse,
    ResponseResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin() -> ExampleInferenceMiddleware:
    plugin = ExampleInferenceMiddleware()
    cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
    cache.list_model_entities_for_workspace.return_value = ["ws/llama-3b"]
    plugin._inject_cache(cache)
    return plugin


def _make_config(**kwargs) -> ExampleMiddlewareConfigData:
    return ExampleMiddlewareConfigData(**kwargs)


def _chat_body(*messages: str) -> dict:
    return {
        "model": "ws/llama-3b",
        "messages": [{"role": "user", "content": m} for m in messages],
    }


def _make_ctx() -> InferenceMiddlewareContext:
    return InferenceMiddlewareContext(
        request_id="test-request-id",
        virtual_model_name="test-vm",
        workspace="ws",
        original_request=InferenceRequest(
            body={"model": "ws/llama-3b", "messages": []},
            headers={},
            path="v1/chat/completions",
        ),
    )


def _make_request(body: dict) -> InferenceRequest:
    return InferenceRequest(body=body, headers={}, path="v1/chat/completions")


def _make_response(result: ResponseResult) -> InferenceResponse:
    return InferenceResponse(result=result, headers={})


@pytest.mark.asyncio
async def test_startup_adapts_sdk_to_typed_entities_client() -> None:
    plugin = _make_plugin()
    sdk = MagicMock()
    typed_client = MagicMock()
    entity_client = MagicMock()

    with (
        patch("nemo_helix_plugin.sdk_provider.get_async_platform_sdk", return_value=sdk),
        patch(
            "nemo_example_plugin.middleware.client_from_platform",
            return_value=typed_client,
        ) as mock_adapter,
        patch(
            "nemo_example_plugin.middleware.NemoEntitiesClient",
            return_value=entity_client,
        ) as mock_entity_client,
    ):
        await plugin.on_startup()

    mock_adapter.assert_called_once_with(sdk, AsyncEntitiesClient)
    mock_entity_client.assert_called_once_with(typed_client)
    assert plugin._entity_client is entity_client


# ---------------------------------------------------------------------------
# validate_middleware_config
# ---------------------------------------------------------------------------


class TestValidateMiddlewareConfig:
    @pytest.mark.asyncio
    async def test_accepts_entity_instance(self):
        """Entity fetched from store is normalised to ExampleMiddlewareConfigData."""
        plugin = _make_plugin()
        entity = ExampleMiddlewareConfig(name="filter", workspace="ws", blocked_keywords=["bad"], block_message="No.")
        result = await plugin.validate_middleware_config("example_middleware_config", entity)
        assert isinstance(result, ExampleMiddlewareConfigData)
        assert result.blocked_keywords == ["bad"]

    @pytest.mark.asyncio
    async def test_coerces_inline_dict(self):
        """Inline dict is validated against ExampleMiddlewareConfigData."""
        plugin = _make_plugin()
        raw = {"blocked_keywords": ["bad"], "block_message": "Nope."}
        result = await plugin.validate_middleware_config("example_middleware_config", raw)
        assert isinstance(result, ExampleMiddlewareConfigData)
        assert result.blocked_keywords == ["bad"]

    @pytest.mark.asyncio
    async def test_raises_for_unknown_config_type(self):
        plugin = _make_plugin()
        with pytest.raises(InferenceMiddlewareError, match="Unknown config_type"):
            await plugin.validate_middleware_config("unknown_type", {})

    @pytest.mark.asyncio
    async def test_raises_for_invalid_dict(self):
        plugin = _make_plugin()
        with pytest.raises(Exception):
            await plugin.validate_middleware_config("example_middleware_config", {"blocked_keywords": "not-a-list"})


# ---------------------------------------------------------------------------
# get_middleware_config
# ---------------------------------------------------------------------------


def _make_plugin_with_entity_client(mock_entity_client) -> ExampleInferenceMiddleware:
    """Make a plugin with a pre-injected mock entity client (bypassing on_startup)."""
    plugin = _make_plugin()
    plugin._entity_client = mock_entity_client
    return plugin


class TestGetMiddlewareConfig:
    @pytest.mark.asyncio
    async def test_fetches_entity_from_store(self):
        expected = ExampleMiddlewareConfig(name="test-filter", workspace="ws", blocked_keywords=["bad"])
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=expected)

        plugin = _make_plugin_with_entity_client(mock_client)
        result = await plugin.get_middleware_config("example_middleware_config", "ws/test-filter")

        mock_client.get.assert_awaited_once_with(ExampleMiddlewareConfig, name="test-filter", workspace="ws")
        assert result is expected

    @pytest.mark.asyncio
    async def test_raises_when_entity_client_not_initialised(self):
        plugin = _make_plugin()
        # _entity_client is None — on_startup() not called
        with pytest.raises(InferenceMiddlewareError, match="not initialised"):
            await plugin.get_middleware_config("example_middleware_config", "ws/cfg")

    @pytest.mark.asyncio
    async def test_raises_for_unknown_config_type(self):
        plugin = _make_plugin_with_entity_client(AsyncMock())
        with pytest.raises(InferenceMiddlewareError, match="does not support config_type"):
            await plugin.get_middleware_config("wrong_type", "ws/cfg")

    @pytest.mark.asyncio
    async def test_wraps_fetch_errors(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("entity store down"))
        plugin = _make_plugin_with_entity_client(mock_client)

        with pytest.raises(InferenceMiddlewareError, match="Could not fetch"):
            await plugin.get_middleware_config("example_middleware_config", "ws/cfg")


# ---------------------------------------------------------------------------
# process_request
# ---------------------------------------------------------------------------


class TestProcessRequest:
    @pytest.mark.asyncio
    async def test_passes_through_when_no_keywords(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=[])
        body = _chat_body("Tell me about history.")
        request = _make_request(body)
        ctx = _make_ctx()
        result = await plugin.process_request(ctx, request, cfg)
        assert result is request

    @pytest.mark.asyncio
    async def test_passes_through_when_no_match(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["violence"])
        body = _chat_body("Tell me about flowers.")
        request = _make_request(body)
        ctx = _make_ctx()
        result = await plugin.process_request(ctx, request, cfg)
        assert result is request

    @pytest.mark.asyncio
    async def test_blocks_matching_request(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["violence"], block_message="Blocked.")
        body = _chat_body("Tell me about violence.")
        request = _make_request(body)
        ctx = _make_ctx()
        result = await plugin.process_request(ctx, request, cfg)
        assert isinstance(result, ImmediateResponse)
        data = result.data
        if isinstance(data, dict):
            # ty loses generic params when narrowing ResponseResult via isinstance;
            # bind to Any so subscript access type-checks cleanly.
            d: dict = cast("dict", data)
            assert d["choices"][0]["message"]["content"] == "Blocked."
            assert d["choices"][0]["finish_reason"] == "content_filter"
        else:
            raise AssertionError("Expected ImmediateResponse.data to be a dict")

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["VIOLENCE"])
        body = _chat_body("Tell me about violence.")
        request = _make_request(body)
        ctx = _make_ctx()
        result = await plugin.process_request(ctx, request, cfg)
        assert isinstance(result, ImmediateResponse)

    @pytest.mark.asyncio
    async def test_first_match_in_list_triggers_block(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["hate", "violence"], block_message="No.")
        body = _chat_body("This message contains violence.")
        request = _make_request(body)
        ctx = _make_ctx()
        result = await plugin.process_request(ctx, request, cfg)
        assert isinstance(result, ImmediateResponse)


# ---------------------------------------------------------------------------
# process_response
# ---------------------------------------------------------------------------


class TestProcessResponse:
    @pytest.mark.asyncio
    async def test_passes_through_when_no_keywords(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=[])
        result_dict = {"choices": [{"message": {"content": "Hello!"}}]}
        response = _make_response(result_dict)
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        assert result is response

    @pytest.mark.asyncio
    async def test_redacts_matching_keyword(self):
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["secret"])
        result_dict = {"choices": [{"message": {"content": "The secret is out."}}]}
        response = _make_response(result_dict)
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        assert isinstance(result, InferenceResponse)
        if isinstance(result.result, dict):
            d: dict = cast("dict", result.result)
            content: str = d["choices"][0]["message"]["content"]
            assert "secret" not in content
            assert "[REDACTED]" in content
        else:
            raise AssertionError("Expected dict result from process_response")

    @pytest.mark.asyncio
    async def test_redacts_keywords_in_streaming_response(self):
        """Streaming responses have keywords redacted from delta.content."""
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["secret"])

        async def _stream():
            yield {"choices": [{"index": 0, "delta": {"content": "The "}}]}
            yield {"choices": [{"index": 0, "delta": {"content": "secret is out."}}]}

        response = _make_response(_stream())
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        stream_result = cast(AsyncIterator[dict[str, Any]], result.result)
        chunks = [c async for c in stream_result]
        full_text = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content", "") for c in chunks)
        assert "secret" not in full_text.lower()
        assert "[REDACTED]" in full_text

    @pytest.mark.asyncio
    async def test_streaming_redacts_keyword_split_across_chunks(self):
        """Keywords split across chunk boundaries are still detected and redacted."""
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["drugs"])

        async def _stream():
            # "drugs" is split: "dr" in chunk 1, "ugs" in chunk 2
            yield {"choices": [{"index": 0, "delta": {"content": "dr"}}]}
            yield {"choices": [{"index": 0, "delta": {"content": "ugs are harmful"}}]}

        response = _make_response(_stream())
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        stream_result = cast(AsyncIterator[dict[str, Any]], result.result)
        chunks = [c async for c in stream_result]
        full_text = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content", "") for c in chunks)
        assert "drugs" not in full_text.lower()
        assert "[REDACTED]" in full_text

    @pytest.mark.asyncio
    async def test_streaming_passes_through_when_no_keywords(self):
        """Stream is returned as-is when blocked_keywords is empty."""
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=[])

        async def _stream():
            yield {"choices": [{"index": 0, "delta": {"content": "Hello!"}}]}

        stream = _stream()
        response = InferenceResponse(result=stream, headers={})
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        assert isinstance(result, InferenceResponse)
        assert result.result is stream  # identical iterator — not touched

    @pytest.mark.asyncio
    async def test_response_headers_preserved(self):
        """Headers on the InferenceResponse pass through unchanged."""
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=[])
        response = InferenceResponse(result={"choices": []}, headers={"x-custom": "value"})
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        assert result.headers == {"x-custom": "value"}

    @pytest.mark.asyncio
    async def test_streaming_preserves_non_content_chunks(self):
        """Role announcement and finish_reason chunks pass through unmodified."""
        plugin = _make_plugin()
        cfg = _make_config(blocked_keywords=["secret"])

        role_chunk = {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        finish_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

        async def _stream():
            yield role_chunk
            yield {"choices": [{"index": 0, "delta": {"content": "No banned words here."}}]}
            yield finish_chunk

        response = _make_response(_stream())
        ctx = _make_ctx()
        result = await plugin.process_response(ctx, response, cfg)
        stream_result = cast(AsyncIterator[dict[str, Any]], result.result)
        chunks = [c async for c in stream_result]
        # First chunk has no content — passes through
        assert chunks[0] is role_chunk


# ---------------------------------------------------------------------------
# Private helper unit tests
# ---------------------------------------------------------------------------


def test_extract_message_text_single():
    body = {"messages": [{"role": "user", "content": "hello world"}]}
    assert _extract_message_text(body) == "hello world"


def test_extract_message_text_multi():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
        ]
    }
    assert "You are helpful." in _extract_message_text(body)
    expected_question = "What is AI?"
    assert expected_question in _extract_message_text(body)


def test_extract_message_text_multimodal():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "http://..."}},
                ],
            }
        ]
    }
    assert _extract_message_text(body) == "Describe this image"


def test_find_keyword_hit():
    expected = "violence"
    assert _find_keyword("There is violence here", [expected]) == expected


def test_find_keyword_miss():
    assert _find_keyword("There is peace here", ["violence"]) is None


def test_find_keyword_case_insensitive():
    assert _find_keyword("VIOLENCE occurred", ["violence"]) == "violence"


def test_redact_keywords_replaces():
    response = {"choices": [{"message": {"content": "The secret password is 123."}}]}
    result = _redact_keywords(response, ["secret", "password"])
    if not isinstance(result, dict):
        raise AssertionError("Expected dict from _redact_keywords")
    d: dict = cast("dict", result)
    content = d["choices"][0]["message"]["content"]
    assert "secret" not in content
    assert "password" not in content
    assert "[REDACTED]" in content


def test_redact_keywords_no_match_returns_original():
    response = {"choices": [{"message": {"content": "Safe content."}}]}
    result = _redact_keywords(response, ["violence"])
    assert result is response  # same object — no copy made
