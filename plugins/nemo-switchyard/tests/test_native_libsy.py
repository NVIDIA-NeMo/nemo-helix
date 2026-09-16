# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tests that need an isolated venv with switchyard_rust (no May vendor)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.switchyard_native

libsy = pytest.importorskip("switchyard_rust.libsy")

from nemo_platform_plugin.inference_middleware import InferenceRequest  # noqa: E402
from nemo_switchyard._native_config import map_random_routing_config  # noqa: E402
from nemo_switchyard._native_host import native_request_dict, run_native_stream  # noqa: E402


class _NoHttp:
    async def complete(self, model_entity_id: str, body: dict, headers: dict) -> dict:
        raise AssertionError("random routing must not CallModel")


class _CapabilityJudge:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def complete(self, model_entity_id: str, body: dict, headers: dict) -> dict:
        self.bodies.append(body)
        assert "instructions" not in body
        assert body["messages"]
        return {
            "id": "judge",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"crux":"bounded task","primary_rule":"SUP-1",'
                            '"capability_boundary":"supported","p_solve":0.9}'
                        ),
                    },
                }
            ],
        }


def _openai_request(body: dict) -> InferenceRequest:
    return InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)


@pytest.mark.asyncio
async def test_native_random_run_stream_selects_strong() -> None:
    weights, seed, models = map_random_routing_config(
        {
            "strong": {"model": "workspace/llama-3-70b"},
            "weak": {"model": "workspace/llama-3-8b"},
            "strong_probability": 1.0,
            "rng_seed": 1,
        }
    )
    algorithm = libsy.random(weights=weights, seed=seed)
    body = {
        "model": "workspace/router",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    }
    request = _openai_request(body)
    out = await run_native_stream(
        algorithm=algorithm,
        request=request,
        models=models,
        headers={},
        transport=_NoHttp(),
    )
    assert out is request
    assert request.body["model"] == "workspace/llama-3-70b"
    coerced = native_request_dict({"messages": [{"role": "user", "content": "hi"}]})
    assert coerced["messages"][0]["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_native_stage_router_without_classifier() -> None:
    algorithm = libsy.stage_router(picker="efficient_first", confidence_threshold=0.5)
    request = _openai_request(
        {
            "model": "workspace/router",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    out = await run_native_stream(
        algorithm=algorithm,
        request=request,
        models={
            "capable": ["workspace/strong"],
            "efficient": ["workspace/weak"],
            "any": ["workspace/strong", "workspace/weak"],
        },
        headers={},
        transport=_NoHttp(),
    )
    assert out is request
    assert request.body["model"] in {"workspace/strong", "workspace/weak"}


@pytest.mark.asyncio
async def test_native_llm_classifier_judge_uses_openai_wire() -> None:
    algorithm = libsy.llm_classifier(libsy.LlmClassifierConfig.capability(config=libsy.TaskClassifierConfig(0.5)))
    transport = _CapabilityJudge()
    request = _openai_request(
        {
            "model": "workspace/router",
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        }
    )
    out = await run_native_stream(
        algorithm=algorithm,
        request=request,
        models={
            "judge": ["workspace/judge"],
            "capable": ["workspace/strong"],
            "efficient": ["workspace/weak"],
            "any": ["workspace/strong", "workspace/weak"],
        },
        headers={},
        transport=transport,
    )
    assert out is request
    assert transport.bodies
    assert request.body["model"] in {"workspace/strong", "workspace/weak"}
