# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answers UserSim's simulated-user, judge and summary calls through the inference gateway."""

from dataclasses import dataclass
from typing import Any

import openai

_SAMPLING_KEYS = ("max_tokens", "temperature", "top_p")


@dataclass(frozen=True)
class RoleModels:
    user: str
    judge: str
    summary: str

    def for_role(self, role: str) -> str:
        return {"user": self.user, "judge": self.judge, "summary": self.summary}[role]


class RoleAnswerer:
    """Turns one UserSim activation into the payload ``ProbeEpisodeRuntime.advance`` records.

    Transient failures (connection errors, 429s, 5xx) are retried by the OpenAI client with
    exponential backoff, bounded by ``max_retries`` and the per-call ``timeout``.
    """

    def __init__(self, models: RoleModels, client: openai.AsyncOpenAI, *, max_retries: int = 4, timeout: float = 120):
        self.models = models
        self._client = client.with_options(max_retries=max_retries, timeout=timeout)

    async def answer(self, event: Any) -> dict:
        params = {k: v for k, v in (event.parameters or {}).items() if k in _SAMPLING_KEYS and v is not None}
        tools = event.tools if event.tools_enabled and event.tools else openai.omit
        completion = await self._client.chat.completions.create(
            model=self.models.for_role(event.role), messages=event.messages, tools=tools, **params
        )
        message = completion.choices[0].message
        usage = completion.usage
        return {
            "activation_id": event.activation_id,
            "response": {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [call.model_dump() for call in message.tool_calls] if message.tool_calls else None,
            },
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        }
