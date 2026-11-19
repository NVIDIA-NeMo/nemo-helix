# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-backed type aliases for inference middleware typed request/response bodies."""

from __future__ import annotations

from typing import Annotated, Any, AsyncIterator, Literal, NotRequired, TypeAlias, TypedDict, Union

import anthropic.types as anthropic_types
import anthropic.types.message_create_params as anthropic_params
import openai.types.chat as openai_chat_types
import openai.types.chat.completion_create_params as openai_chat_params
import openai.types.responses.response_create_params as openai_responses_params
from pydantic import BaseModel, ConfigDict, Field

SYSTEMONE_PATH = "v1/systemone"
"""Backend-relative path of the System One decision API."""


class SystemOneQuestion(TypedDict):
    """One question in a System One request, keyed by a caller-chosen id.

    ``type`` is the answer kind. Known kinds: ``noul`` (probability the
    instruction holds, 0..1), ``choice`` (pick one ``criteria`` key),
    ``score``, and ``box``. Kept as ``str`` so a new kind does not fail typing.
    """

    type: str
    instructions: str
    criteria: NotRequired[dict[str, str] | list[Any]]


class SystemOneRequest(TypedDict):
    """Body of ``POST /v1/systemone``."""

    model: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, SystemOneQuestion]


class SystemOneNoulAnswer(BaseModel):
    """Answer to a ``noul`` question: the probability that the instruction holds."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    noul: float


class SystemOneChoiceAnswer(BaseModel):
    """Answer to a ``choice`` question: the picked option and every option's probability."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class SystemOneOtherAnswer(BaseModel):
    """Answer of a kind not modeled above. Its fields are kept as extras."""

    model_config = ConfigDict(extra="allow")

    type: str


SystemOneAnswer: TypeAlias = Annotated[
    SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneOtherAnswer,
    Field(union_mode="left_to_right"),
]


class SystemOneUsage(BaseModel):
    """Token usage of a System One call. Backend-specific counters are kept as extras."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    """Response of ``POST /v1/systemone``. ``metrics`` and other top-level fields are kept as extras."""

    model_config = ConfigDict(extra="allow")

    model: str
    answers: dict[str, SystemOneAnswer]
    usage: SystemOneUsage


TypedResponse: TypeAlias = Union[openai_chat_types.ChatCompletion, anthropic_types.Message, SystemOneResponse]
OpenAIResponseChunk: TypeAlias = openai_chat_types.ChatCompletionChunk
AnthropicResponseChunk: TypeAlias = anthropic_types.RawMessageStreamEvent
TypedResponseChunk: TypeAlias = Union[OpenAIResponseChunk, AnthropicResponseChunk]
TypedResponseResult: TypeAlias = Union[TypedResponse, AsyncIterator[TypedResponseChunk]]

TypedRequest: TypeAlias = Union[
    openai_chat_params.CompletionCreateParamsBase,
    anthropic_params.MessageCreateParamsBase,
    openai_responses_params.ResponseCreateParamsBase,
    SystemOneRequest,
]
"""Union of the TypedDict param types for each inbound API format.

All are TypedDicts — plain dicts at runtime. This alias exists for
static type checking. Plugins that need path-based dispatch should use
``request.path``, not ``isinstance``, since isinstance on TypedDict types
just checks ``isinstance(x, dict)``.
"""
