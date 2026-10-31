# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token usage and the served model are read from every inference API's responses and stream chunks."""

from __future__ import annotations

from typing import Any

import pytest
from nemo_evals.gateway.usage import CallApi, TokenUsage, model_from_payload, usage_from_payload


@pytest.mark.parametrize(
    ("api", "payloads", "expected"),
    [
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [{"usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}],
            TokenUsage(uncached_input_tokens=40, cached_input_tokens=60, output_tokens=7),
            id="chat-body",
        ),
        pytest.param(
            CallApi.CHAT_COMPLETIONS,
            [
                {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            ],
            TokenUsage(uncached_input_tokens=12, output_tokens=3),
            id="chat-stream-include-usage",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [{"usage": {"input_tokens": 50, "output_tokens": 9, "input_tokens_details": {"cached_tokens": 20}}}],
            TokenUsage(uncached_input_tokens=30, cached_input_tokens=20, output_tokens=9),
            id="responses-body",
        ),
        pytest.param(
            CallApi.RESPONSES,
            [
                {"type": "response.output_text.delta", "delta": "Hi"},
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 40, "output_tokens": 4, "input_tokens_details": {}}},
                },
            ],
            TokenUsage(uncached_input_tokens=40, output_tokens=4),
            id="responses-stream",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 30,
                        "cache_creation_input_tokens": 5,
                        "output_tokens": 8,
                    }
                }
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8),
            id="messages-body",
        ),
        pytest.param(
            CallApi.MESSAGES,
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 30, "output_tokens": 1}},
                },
                {"type": "content_block_delta", "delta": {"text": "Hi"}},
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 10, "cache_read_input_tokens": None, "output_tokens": 15},
                },
            ],
            TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, output_tokens=15),
            id="messages-stream",
        ),
    ],
)
def test_usage_is_read_from_every_api_and_stream_shape(
    api: CallApi, payloads: list[dict[str, Any]], expected: TokenUsage
) -> None:
    """Every API's counts land in the same classes; a later chunk's null count keeps the earlier value."""
    usage = TokenUsage()
    for payload in payloads:
        if (reported := usage_from_payload(api, payload)) is not None:
            usage.merge(reported)
    assert usage == expected


@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param([{"model": "served", "choices": []}], id="body"),
        pytest.param([{"model": "served", "choices": [{"delta": {}}]}, {"choices": [], "usage": {}}], id="chat-stream"),
        pytest.param(
            [
                {"type": "response.created", "response": {"model": "served"}},
                {"type": "response.output_text.delta", "delta": "Hi"},
            ],
            id="responses-stream",
        ),
        pytest.param(
            [{"type": "message_start", "message": {"model": "served"}}, {"type": "message_delta", "usage": {}}],
            id="messages-stream",
        ),
    ],
)
def test_the_served_model_is_read_from_every_api_and_stream_shape(payloads: list[dict[str, Any]]) -> None:
    """A routing provider answers with the model it picked, which can differ from the one the caller asked for."""
    models = [model for payload in payloads if (model := model_from_payload(payload)) is not None]
    assert models == ["served"]


def test_input_total_counts_every_class_once() -> None:
    usage = TokenUsage(uncached_input_tokens=10, cached_input_tokens=30, cache_write_input_tokens=5, output_tokens=8)
    assert usage.input_tokens == 45
    assert usage.attributes() == {
        "gen_ai.usage.input_tokens": 45,
        "nemo.usage.uncached_input_tokens": 10,
        "gen_ai.usage.cached_tokens": 30,
        "llm.token_count.prompt_details.cache_write": 5,
        "gen_ai.usage.output_tokens": 8,
    }


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("v1/chat/completions", CallApi.CHAT_COMPLETIONS),
        ("/v1/responses", CallApi.RESPONSES),
        ("v1/messages", CallApi.MESSAGES),
        ("v1/embeddings", CallApi.OTHER),
    ],
)
def test_api_is_read_from_the_path(path: str, api: CallApi) -> None:
    assert CallApi.from_path(path) is api
