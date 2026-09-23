# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_optimization.backends.ga.transform import ModelPromptTransformer, PromptTransformError


@dataclass
class _Recorder:
    response: object
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.response)


def _sdk(response: object) -> tuple[NemoClient, _Recorder]:
    recorder = _Recorder(response)
    sdk = NemoClient(
        base_url="http://test",
        workspace="default",
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
    )
    return sdk, recorder


def _payload() -> dict[str, Any]:
    return {
        "models": {
            "prompt_optimizer": {
                "provider": "openai",
                "model": "models/gpt-test",
                "settings": {"temperature": 0.2, "max_tokens": 77, "timeout_s": 12.0},
            }
        }
    }


def test_model_prompt_transformer_uses_platform_inference_gateway() -> None:
    sdk, recorder = _sdk(
        {"choices": [{"finish_reason": "stop", "message": {"content": "```text\nImproved prompt\n```"}}]}
    )
    transformer = ModelPromptTransformer(
        sdk=sdk,
        workspace="default",
        payload=_payload(),
        model_name="prompt_optimizer",
    )

    result = transformer.mutate(
        prompt_name="system_prompt",
        prompt="Base prompt.",
        purpose="Answer accurately.",
        prompt_format=None,
        feedback=None,
    )

    assert result == "Improved prompt"
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == "/apis/inference-gateway/v2/workspaces/models/model/gpt-test/-/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "models/gpt-test",
        "messages": [
            {
                "role": "system",
                "content": "Return only the revised prompt text, without commentary or markdown fences.",
            },
            {
                "role": "user",
                "content": (
                    "Prompt dimension: system_prompt\n\nPurpose: Answer accurately.\n\n"
                    "Required format: Preserve the current prompt format.\n\nCurrent prompt:\n\n"
                    "Base prompt.\n\nRewrite the prompt while preserving its variables, role, and constraints."
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 77,
    }
    assert request.extensions["timeout"]["read"] == 12.0


def test_model_prompt_transformer_requires_helix_client() -> None:
    with pytest.raises(PromptTransformError, match="requires a Helix client"):
        ModelPromptTransformer(
            sdk=None,
            workspace="default",
            payload=_payload(),
            model_name="prompt_optimizer",
        )


def test_model_prompt_transformer_rejects_malformed_response() -> None:
    sdk, _ = _sdk([])
    transformer = ModelPromptTransformer(
        sdk=sdk,
        workspace="default",
        payload=_payload(),
        model_name="prompt_optimizer",
    )

    with pytest.raises(PromptTransformError, match="non-object response"):
        transformer.mutate(
            prompt_name="system_prompt",
            prompt="Base prompt.",
            purpose="Answer accurately.",
            prompt_format=None,
            feedback=None,
        )


@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter", "tool_calls"])
def test_model_prompt_transformer_rejects_incomplete_completion(finish_reason: str | None) -> None:
    sdk, _ = _sdk({"choices": [{"finish_reason": finish_reason, "message": {"content": "Partial prompt"}}]})
    transformer = ModelPromptTransformer(
        sdk=sdk,
        workspace="default",
        payload=_payload(),
        model_name="prompt_optimizer",
    )

    with pytest.raises(PromptTransformError, match="did not complete normally"):
        transformer.mutate(
            prompt_name="system_prompt",
            prompt="Base prompt.",
            purpose="Answer accurately.",
            prompt_format=None,
            feedback=None,
        )
