# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed inbound request parsing for inference middleware."""

from __future__ import annotations

import logging
from typing import Any

import anthropic.types.message_create_params as anthropic_params
import openai.types.chat.completion_create_params as openai_chat_params
import openai.types.responses.response_create_params as openai_responses_params
from nemo_helix_plugin.inference_middleware import InferenceRequest
from nemo_helix_plugin.inference_middleware_types import TypedRequest
from pydantic import TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

_REQUEST_ADAPTERS: dict[str, TypeAdapter] = {
    "v1/chat/completions": TypeAdapter(openai_chat_params.CompletionCreateParamsBase),
    "v1/messages": TypeAdapter(anthropic_params.MessageCreateParamsBase),
    "v1/responses": TypeAdapter(openai_responses_params.ResponseCreateParamsBase),
}


def parse_typed_request(path: str, body: dict[str, Any]) -> TypedRequest | None:
    """Validate a request body against the TypedDict schema for the given path.

    Returns a validated dict (TypedDict == dict at runtime), or ``None`` when
    the path is unknown or the body fails validation.

    Parsing is intentionally non-fatal: IGW falls back to ``typed_body=None``
    when the path is unknown or the body doesn't conform. Plugins should treat
    ``None`` as "format unknown / unvalidated".
    """
    adapter = _REQUEST_ADAPTERS.get(path.lstrip("/"))
    if adapter is None:
        return None
    try:
        return adapter.validate_python(body)
    except ValidationError:
        logger.debug(
            "Failed to validate request body for path %r against TypedDict schema",
            path,
            exc_info=True,
        )
        return None


def build_inference_request(
    body: dict[str, Any],
    headers: dict[str, str],
    path: str,
) -> InferenceRequest:
    """Construct an InferenceRequest with typed_body populated when possible."""
    return InferenceRequest(
        body=body,
        headers=headers,
        path=path,
        typed_body=parse_typed_request(path, body),
    )
