# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.guardrails.api.schemas import BaseRequest
from nhx.guardrails.app.constants import NIM_PROVIDER
from nhx.guardrails.app.schemas.guardrails import (
    GuardrailsChatCompletionRequest,
    GuardrailsCompletionRequest,
)
from nhx.guardrails.config import settings
from nhx.guardrails.entities.values._private import GenerationOptions
from nhx.guardrails.entities.values.chat import GuardrailChatCompletionRequest
from nhx.guardrails.entities.values.completions import GuardrailCompletionRequest

from . import add_custom_init_to_base_request

add_custom_init_to_base_request()


def convert_chat_completion_request_to_guardrails(
    request: GuardrailChatCompletionRequest,
) -> GuardrailsChatCompletionRequest:
    """
    Convert a ChatCompletionRequest to GuardrailsChatCompletionRequest format.

    Args:
        request (GuardrailChatCompletionRequest): The original chat completion request.

    Returns:
        GuardrailsChatCompletionRequest: The converted guardrails request.
    """
    openai_request = BaseRequest(**request.model_dump(exclude_unset=True))
    options: GenerationOptions = request.guardrails.options
    options.llm_params = openai_request.model_dump(exclude_unset=True)
    options.llm_output = True
    stream = openai_request.streaming
    # NIM's openai compatible API expects it to be called "stream" and not "streaming"
    if settings.default_llm_provider == NIM_PROVIDER:
        options.llm_params["stream"] = options.llm_params.pop("streaming", False)

    # Convert ChatMessage objects to dictionaries
    messages = [msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg for msg in request.messages]

    return GuardrailsChatCompletionRequest(
        messages=messages,
        config_ids=request.guardrails.config_ids,
        context=request.guardrails.context,
        stream=stream,
        options=options,
    )


def convert_completion_request_to_guardrails(
    request: GuardrailCompletionRequest,
) -> GuardrailsCompletionRequest:
    """
    Convert a CompletionRequest to GuardrailCompletionRequest format.

    Args:
        request (GuardrailCompletionRequest): The original completion request.

    Returns:
        GuardrailsCompletionRequest: The converted guardrails request.
    """
    openai_request = BaseRequest(**request.model_dump(exclude_unset=True))
    options: GenerationOptions = request.guardrails.options
    options.llm_params = openai_request.model_dump(exclude_unset=True)
    options.llm_output = True
    stream = openai_request.streaming
    prompt = request.prompt

    return GuardrailsCompletionRequest(
        prompt=prompt,
        config_ids=request.guardrails.config_ids,
        context=request.guardrails.context,
        stream=stream,
        options=options,
    )


convert_check_request_to_guardrails = convert_chat_completion_request_to_guardrails
