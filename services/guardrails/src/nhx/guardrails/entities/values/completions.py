# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completions-related value objects for the Guardrails service."""

from typing import Optional

from nhx.guardrails.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
)
from pydantic import ConfigDict, Field

from .common import GuardrailsDataInput, GuardrailsDataOutput


class GuardrailCompletionRequest(CompletionRequest):
    """Guardrails OpenAI compatible interface"""

    model_config = ConfigDict(extra="allow")
    guardrails: GuardrailsDataInput = Field(
        default_factory=GuardrailsDataInput,
        description="Guardrails specific options for the request.",
    )


class GuardrailCompletionResponse(CompletionResponse):
    """A class to represent the response from the Guardrails API that is compatible with the OpenAI API."""

    guardrails_data: Optional[GuardrailsDataOutput] = Field(
        default=None, description="The guardrails specific output data."
    )


class GuardrailCompletionStreamResponse(CompletionStreamResponse):
    """A class to represent the response from the Guardrails API that is compatible with the OpenAI API."""
