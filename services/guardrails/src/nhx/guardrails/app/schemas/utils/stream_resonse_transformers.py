# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from nhx.guardrails.api.schemas import (
    ChatCompletionResponseStreamChoice,
    CompletionResponseStreamChoice,
    DeltaMessage,
)
from nhx.guardrails.entities.values.chat import (
    GuardrailChatCompletionStreamResponse,
)
from nhx.guardrails.entities.values.completions import (
    GuardrailCompletionStreamResponse,
)


def create_guardrail_chat_completion_stream_response_from_chunk(
    index: int,
    chunk: str,
    role: str,
    finish_reason: Optional[str],
    model_name: Optional[str],
    **kwargs,
) -> GuardrailChatCompletionStreamResponse:
    """
    Create a GuardrailChatCompletionStreamResponse from chunk data.

    Args:
        index (int): The index of the chunk.
        chunk (str): The content chunk.
        role (str): The role associated with the message.
        finish_reason (Optional[str]): The reason for finishing.
        model_name (Optional[str]): The name of the model.

    Returns:
        GuardrailChatCompletionStreamResponse: The stream response object.
    """
    choice_data = ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(role=role, content=chunk),
        finish_reason=finish_reason,
    )
    instance = GuardrailChatCompletionStreamResponse.model_construct(
        model=model_name,
        choices=[choice_data],
    )

    return instance


# file: guardrail_completion_stream_response.py


def create_guardrail_completion_stream_response_from_chunk(
    index: int,
    chunk: str,
    model_name: Optional[str],
    **kwargs,
) -> GuardrailCompletionStreamResponse:
    """
    Create a GuardrailCompletionStreamResponse from chunk data.

    Args:
        index (int): The index of the chunk.
        chunk (str): The content chunk.
        model_name (Optional[str]): The name of the model.

    Returns:
        GuardrailCompletionStreamResponse: The stream response object.
    """
    choices = [
        CompletionResponseStreamChoice(
            index=index,
            text=chunk,
            logprobs=None,  # TODO: Implement logprobs if available
            finish_reason=None,  # Provide appropriate finish reason if available
        )
    ]

    instance = GuardrailCompletionStreamResponse.model_construct(
        model=model_name,
        choices=choices,
    )

    return instance
