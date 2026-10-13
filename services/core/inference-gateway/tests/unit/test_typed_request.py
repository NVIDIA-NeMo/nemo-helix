# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for typed inbound request parsing."""

from __future__ import annotations

from typing import Any

from nemo_helix_plugin.inference_middleware import InferenceRequest
from nhx.core.inference_gateway.api.typed_request import build_inference_request, parse_typed_request


def _openai_chat_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


def _anthropic_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "claude-3",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


def _responses_body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "gpt-4o",
        "input": "hello",
        **extra,
    }


# ---------------------------------------------------------------------------
# parse_typed_request — known paths succeed
# ---------------------------------------------------------------------------


def test_parse_openai_chat_completions_returns_dict():
    result = parse_typed_request("v1/chat/completions", _openai_chat_body())

    assert result is not None
    assert isinstance(result, dict)
    assert result["model"] == "gpt-4"


def test_parse_openai_chat_completions_with_leading_slash():
    """Path with a leading slash should be normalised correctly."""
    result = parse_typed_request("/v1/chat/completions", _openai_chat_body())

    assert result is not None
    assert result["model"] == "gpt-4"


def test_parse_anthropic_messages_returns_dict():
    result = parse_typed_request("v1/messages", _anthropic_body())

    assert result is not None
    assert isinstance(result, dict)
    assert result["model"] == "claude-3"


def test_parse_responses_returns_dict():
    result = parse_typed_request("v1/responses", _responses_body())

    assert result is not None
    assert isinstance(result, dict)
    assert result["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# parse_typed_request — validation failures return None
# ---------------------------------------------------------------------------


def test_parse_invalid_openai_body_returns_none():
    """A body missing required fields should not raise; returns None instead."""
    result = parse_typed_request("v1/chat/completions", {"bad": "body"})

    assert result is None


def test_parse_unknown_path_returns_none():
    result = parse_typed_request("v1/unknown", {})

    assert result is None


def test_parse_empty_path_returns_none():
    result = parse_typed_request("", {})

    assert result is None


# ---------------------------------------------------------------------------
# build_inference_request
# ---------------------------------------------------------------------------


def test_build_inference_request_known_path_populates_typed_body():
    req = build_inference_request(
        body=_openai_chat_body(),
        headers={"content-type": "application/json"},
        path="v1/chat/completions",
    )

    assert isinstance(req, InferenceRequest)
    assert req.typed_body is not None
    assert isinstance(req.typed_body, dict)  # TypedDict is plain dict at runtime
    assert req.typed_body["model"] == "gpt-4"


def test_build_inference_request_unknown_path_typed_body_is_none():
    req = build_inference_request(
        body={"model": "some-model"},
        headers={},
        path="v1/completions",
    )

    assert isinstance(req, InferenceRequest)
    assert req.typed_body is None
    assert req.body["model"] == "some-model"


def test_build_inference_request_anthropic_path():
    req = build_inference_request(
        body=_anthropic_body(),
        headers={},
        path="v1/messages",
    )

    assert req.typed_body is not None
    assert req.typed_body["model"] == "claude-3"


def test_build_inference_request_responses_path():
    req = build_inference_request(
        body=_responses_body(),
        headers={},
        path="v1/responses",
    )

    assert req.typed_body is not None
    assert req.typed_body["model"] == "gpt-4o"


def test_build_inference_request_preserves_body_and_headers():
    body = _openai_chat_body()
    headers = {"x-request-id": "abc123"}
    req = build_inference_request(body=body, headers=headers, path="v1/chat/completions")

    assert req.body is body
    assert req.headers is headers
    assert req.path == "v1/chat/completions"
