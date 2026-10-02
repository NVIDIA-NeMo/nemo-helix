# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Union

from fastapi import APIRouter, Body, Request, Response, status
from nhx.guardrails.api.dependencies import RailsServiceDep
from nhx.guardrails.app.handlers.completion import (
    CompletionRequestHandler,
)
from nhx.guardrails.app.utils.context_utils import (
    get_x_model_response_headers_from_context,
)
from nhx.guardrails.entities.values.chat import (
    GuardrailChatCompletionRequest,
    GuardrailChatCompletionResponse,
    GuardrailChatCompletionStreamResponse,
)

router = APIRouter()

EXAMPLE_MODEL = "meta/llama3-70b-instruct"

examples = {
    "openai_api_default_config": {
        "summary": "Example using the default guardrail configuration.",
        "description": "",
        "value": {
            "model": EXAMPLE_MODEL,
            "messages": [{"role": "user", "content": "what can you do for me?"}],
            "max_tokens": 16,
            "stream": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "stop": [],
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
    "openai_api": {
        "summary": "Example using specific guardrail configuration.",
        "description": "",
        "value": {
            "model": EXAMPLE_MODEL,
            "messages": [{"role": "user", "content": "what can you do for me?"}],
            "guardrails": {"config_id": "self-check"},
            "max_tokens": 16,
            "stream": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "stop": [],
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    },
}


@router.post(
    "/v2/workspaces/{workspace}/chat/completions",
    summary="Send chat completion requests",
    responses={
        status.HTTP_200_OK: {
            "description": "Successful Response",
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid Request Body",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Validation Error",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal Server Error",
        },
    },
    response_model=Union[
        GuardrailChatCompletionResponse,
        GuardrailChatCompletionStreamResponse,
    ],
    response_model_exclude_none=True,
)
async def chat_completion(
    workspace: str,
    request_body: Annotated[
        GuardrailChatCompletionRequest,
        Body(
            openapi_examples=examples,
        ),
    ],
    request: Request,
    response: Response,
    rails_service: RailsServiceDep,
):
    """Chat completion for the provided conversation."""

    # We extract the raw body as well, as we need to know exactly
    # what the user specified, and what came from the default values.
    request_handler = CompletionRequestHandler(
        rails_service=rails_service,
        request=request,
        request_body=request_body,
        normal_response_model=GuardrailChatCompletionResponse,
        streaming_response_model=GuardrailChatCompletionStreamResponse,
        workspace=workspace,
    )

    result = await request_handler.handle_request()

    response_headers = get_x_model_response_headers_from_context()

    if response_headers:
        response.headers["X-Model-Response-Headers"] = response_headers

    return result
