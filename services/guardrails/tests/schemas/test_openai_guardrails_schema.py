# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemoguardrails.rails.llm.options import GenerationResponse
from nhx.guardrails.api.schemas import BaseRequest
from nhx.guardrails.app.schemas.utils.request_converters import (
    convert_chat_completion_request_to_guardrails,
)
from nhx.guardrails.entities.values.chat import (
    GuardrailChatCompletionRequest,
    GuardrailChatCompletionResponse,
)
from nhx.guardrails.entities.values.common import GuardrailsDataInput


def test_guardrails_openai_chat_completion_request_to_guardrails():
    # Setup a mock request with necessary attributes
    guardrails_request = GuardrailsDataInput(config_id="config123")
    chat_request = GuardrailChatCompletionRequest(
        model="gpt-3.5",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        guardrails=guardrails_request,
    )

    assert chat_request.model_dump(exclude_none=True)["streaming"] == chat_request.streaming
    # Call to_guardrails method
    transformed_request = convert_chat_completion_request_to_guardrails(chat_request)
    # transformed_request = chat_request.to_guardrails()

    # Validate transformation logic
    assert transformed_request.messages[0] == chat_request.messages[0].model_dump(exclude_none=True)
    assert transformed_request.config_ids == [guardrails_request.config_id]
    assert transformed_request.context == guardrails_request.context
    # assert transformed_request.stream == chat_request.stream

    # Ensure options are correctly set
    transformed_request.options.llm_params.pop("stream")
    assert transformed_request.options.llm_params == BaseRequest(
        **chat_request.model_dump(exclude_unset=True, exclude={"streaming"})
    ).model_dump(exclude_unset=True)
    assert transformed_request.options.llm_output


@pytest.mark.skip(reason="Skipping this test temporarily")
def test_guardrails_openai_chat_completion_response_from_response():
    # Simulate a GenerationResponse from a backend system

    response = GenerationResponse(
        response=[
            {
                "id": "cmpl-f94e4d70b7bf46e0af1234d7b334a181",
                "object": "text_completion",
                "created": 1718712476,
                "model_name": "meta/llama3-8b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "text": "bot says hi",
                        "logprobs": None,
                        "finish_reason": "length",
                        "stop_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "total_tokens": 36,
                    "completion_tokens": 32,
                },
                "role": "system",  # Add role
                "content": "bot says hi",  # Add content
            }
        ],
        llm_output={
            "id": "cmpl-f94e4d70b7bf46e0af1234d7b334a181",
            "object": "text_completion",
            "created": 1718712476,
            "model_name": "meta/llama3-8b-instruct",
            "choices": [
                {
                    "index": 0,
                    "text": "bot says hi",
                    "logprobs": None,
                    "finish_reason": "length",
                    "stop_reason": None,
                }
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 36, "completion_tokens": 32},
        },
        output_data={"extra": "data"},
        log=None,
    )

    # Convert to GuardrailsOpenAIChatCompletionResponse
    guardrails_response = GuardrailChatCompletionResponse.from_response(response=response)

    # Validate response transformation
    assert guardrails_response.model == "meta/llama3-8b-instruct"
    # / TODO: uncomment or add once fixed this in the model
    assert guardrails_response.choices[0]["text"] == "bot says hi"
    assert guardrails_response.usage["prompt_tokens"] == 4
