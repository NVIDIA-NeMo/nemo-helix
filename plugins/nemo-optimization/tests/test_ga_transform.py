# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from nemo_optimization.backends.ga.transform import ModelPromptTransformer, PromptTransformError


class _GatewayModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, trailing_uri: str, **kwargs: Any) -> object:
        self.calls.append({"trailing_uri": trailing_uri, **kwargs})
        return self.response


def _sdk(response: object) -> tuple[Any, _GatewayModel]:
    model = _GatewayModel(response)
    sdk = SimpleNamespace(inference=SimpleNamespace(gateway=SimpleNamespace(model=model)))
    return sdk, model


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
    sdk, model = _sdk({"choices": [{"message": {"content": "```text\nImproved prompt\n```"}}]})
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
    assert model.calls == [
        {
            "trailing_uri": "v1/chat/completions",
            "workspace": "models",
            "name": "gpt-test",
            "body": {
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
            },
            "timeout": 12.0,
        }
    ]


def test_model_prompt_transformer_requires_platform_client() -> None:
    with pytest.raises(PromptTransformError, match="requires a NeMo Platform SDK client"):
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
