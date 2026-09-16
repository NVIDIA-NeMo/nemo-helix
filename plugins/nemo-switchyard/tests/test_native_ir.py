# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_switchyard._native_ir import (
    apply_llm_request_to_openai_body,
    llm_request_to_openai_chat,
    openai_chat_to_agg,
    openai_chat_to_llm_request,
)


def test_openai_tools_and_null_content_become_llm_request() -> None:
    body = {
        "model": "ws/router",
        "messages": [
            {"role": "system", "content": "route"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "look up",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "max_tokens": 16,
    }
    ir = openai_chat_to_llm_request(body)
    assert ir["instructions"][0]["content"][0]["text"] == "route"
    assert ir["messages"][0]["content"][0]["type"] == "text"
    assert ir["messages"][1]["content"][0]["type"] == "tool_call"
    assert ir["messages"][1]["content"][0]["name"] == "search"
    assert ir["tools"][0]["name"] == "search"
    assert "function" not in ir["tools"][0]
    assert ir["output"]["max_output_tokens"] == 16

    chat = llm_request_to_openai_chat(ir, model="judge-model")
    assert chat["model"] == "judge-model"
    assert chat["messages"][0]["role"] == "system"
    assert chat["tools"][0]["type"] == "function"
    assert "instructions" not in chat
    assert chat["max_tokens"] == 16


def test_image_url_becomes_image_block() -> None:
    ir = openai_chat_to_llm_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://ex/a.png", "detail": "low"}}],
                }
            ]
        }
    )
    assert ir["messages"][0]["content"][0]["type"] == "image"


def test_openai_choices_become_agg_outputs() -> None:
    agg = openai_chat_to_agg(
        {
            "id": "cmpl",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"p_solve":0.9}'},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    assert agg["outputs"][0]["content"][0]["text"] == '{"p_solve":0.9}'
    assert agg["usage"]["input_tokens"] == 3
    assert "choices" not in agg


def test_outcome_request_overlay_rewrites_messages() -> None:
    original = {"model": "ws/router", "messages": [{"role": "user", "content": "hi"}]}
    ir = {
        "instructions": [{"role": "system", "content": [{"type": "text", "text": "handoff"}]}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    merged = apply_llm_request_to_openai_body(original, ir)
    assert merged["messages"][0]["role"] == "system"
    assert merged["messages"][0]["content"] == "handoff"
