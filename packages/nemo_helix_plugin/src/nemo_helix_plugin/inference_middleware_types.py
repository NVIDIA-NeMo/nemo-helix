# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK-backed type aliases for inference middleware typed request/response bodies."""

from __future__ import annotations

from typing import AsyncIterator, TypeAlias, Union

import anthropic.types as anthropic_types
import anthropic.types.message_create_params as anthropic_params
import openai.types.chat as openai_chat_types
import openai.types.chat.completion_create_params as openai_chat_params
import openai.types.responses.response_create_params as openai_responses_params

TypedResponse: TypeAlias = Union[openai_chat_types.ChatCompletion, anthropic_types.Message]
OpenAIResponseChunk: TypeAlias = openai_chat_types.ChatCompletionChunk
AnthropicResponseChunk: TypeAlias = anthropic_types.RawMessageStreamEvent
TypedResponseChunk: TypeAlias = Union[OpenAIResponseChunk, AnthropicResponseChunk]
TypedResponseResult: TypeAlias = Union[TypedResponse, AsyncIterator[TypedResponseChunk]]

TypedRequest: TypeAlias = Union[
    openai_chat_params.CompletionCreateParamsBase,
    anthropic_params.MessageCreateParamsBase,
    openai_responses_params.ResponseCreateParamsBase,
]
"""Union of the SDK TypedDict param types for each inbound API format.

All three are TypedDicts — plain dicts at runtime. This alias exists for
static type checking. Plugins that need path-based dispatch should use
``request.path``, not ``isinstance``, since isinstance on TypedDict types
just checks ``isinstance(x, dict)``.
"""
