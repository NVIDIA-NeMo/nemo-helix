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
    request = InferenceRequest(body=body, headers={}, path="v1/chat/completions", typed_body=body)
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
