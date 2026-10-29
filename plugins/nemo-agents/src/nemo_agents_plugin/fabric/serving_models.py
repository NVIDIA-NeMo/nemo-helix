# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP request and response models for Fabric-backed agent serving."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatCompletionMessage(BaseModel):
    """Supported OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="allow")

    role: Literal["assistant", "developer", "system", "tool", "user"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Supported subset of an OpenAI chat-completions request."""

    model_config = ConfigDict(extra="allow")

    messages: list[ChatCompletionMessage] = Field(min_length=1)
    stream: bool = False

    @model_validator(mode="after")
    def validate_current_turn(self) -> ChatCompletionRequest:
        if self.messages[-1].role != "user":
            raise ValueError("The final chat message must have role 'user'.")
        return self


class ChatCompletionResponseMessage(BaseModel):
    """OpenAI-compatible assistant response message."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat-completion choice."""

    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response for one Fabric runtime invocation."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    model: str = "unknown-model"
    choices: list[ChatCompletionChoice]
    usage: dict[str, Any] | None = None


class ChatCompletionStreamDelta(BaseModel):
    """OpenAI-compatible streaming response delta."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    """OpenAI-compatible streaming chat-completion choice."""

    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: Literal["stop"] | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk for one Fabric runtime invocation."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "unknown-model"
    choices: list[ChatCompletionStreamChoice]


class ChatCompletionStreamError(BaseModel):
    """OpenAI-compatible streaming error payload."""

    message: str
    type: str


class ChatCompletionStreamErrorResponse(BaseModel):
    """OpenAI-compatible streaming error frame payload."""

    error: ChatCompletionStreamError
