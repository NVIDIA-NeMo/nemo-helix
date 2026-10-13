# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check-related value objects for the Guardrails service."""

from typing import Dict, Optional

from pydantic import BaseModel, Field

from ..enums import StatusEnum
from .chat import GuardrailChatCompletionRequest
from .common import GuardrailsDataOutput


class GuardrailCheckRequest(GuardrailChatCompletionRequest):
    """Currently only inherits, in the future we might add new fields."""

    ...


class RailStatus(BaseModel):
    status: StatusEnum = Field(..., description="Status of the individual rail.")


class GuardrailCheckResponse(BaseModel):
    status: StatusEnum = Field(
        ...,
        description="Overall status indicating if all rails passed or if any failed.",
    )
    rails_status: Dict[str, RailStatus] = Field(..., description="Dictionary mapping each rail to its status.")
    guardrails_data: Optional[GuardrailsDataOutput] = Field(None, description="Additional data related to guardrails.")
