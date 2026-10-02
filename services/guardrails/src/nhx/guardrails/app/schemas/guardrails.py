# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Union

from nhx.guardrails.entities.values.common import GuardrailsDataInput
from pydantic import Field, model_validator


class GuardrailsChatCompletionRequest(GuardrailsDataInput):
    messages: Union[str, List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]] = Field(
        ..., description="A list of messages comprising the conversation so far"
    )

    @model_validator(mode="before")
    @classmethod
    def inject_context(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("context"):
                messages = data.get("messages")
                if messages:
                    messages.insert(0, {"role": "context", "content": str(data["context"])})
            return data


class GuardrailsCompletionRequest(GuardrailsDataInput):
    prompt: Union[List[int], List[List[int]], str, List[str]] = Field(
        ...,
        min_length=1,
        description="User prompt or list of token ids.",
    )
