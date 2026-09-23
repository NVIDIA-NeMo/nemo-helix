# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrails service entities package.

This package contains all entity definitions, value types, and enums for the Guardrails service.
"""

# Entity classes
from .entities import GuardrailConfig

# Enums
from .enums import RoleEnum, StatusEnum

# Value types - private (internal configuration)
from .values._private import (
    GenerationOptions,
    Model,
    RailsConfig,
    TracingConfig,
)

# Value types - chat
from .values.chat import (
    GuardrailChatCompletionRequest,
    GuardrailChatCompletionResponse,
    GuardrailChatCompletionStreamResponse,
)

# Value types - check
from .values.check import (
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    RailStatus,
)

# Value types - common
from .values.common import (
    CheckResponseItem,
    GuardrailsDataInput,
    GuardrailsDataOutput,
)

# Value types - completions
from .values.completions import (
    GuardrailCompletionRequest,
    GuardrailCompletionResponse,
    GuardrailCompletionStreamResponse,
)

__all__ = [
    # Entity classes
    "GuardrailConfig",
    # Enums
    "RoleEnum",
    "StatusEnum",
    # Value types - common
    "CheckResponseItem",
    "GuardrailsDataInput",
    "GuardrailsDataOutput",
    # Value types - chat
    "GuardrailChatCompletionRequest",
    "GuardrailChatCompletionResponse",
    "GuardrailChatCompletionStreamResponse",
    # Value types - check
    "GuardrailCheckRequest",
    "GuardrailCheckResponse",
    "RailStatus",
    # Value types - completions
    "GuardrailCompletionRequest",
    "GuardrailCompletionResponse",
    "GuardrailCompletionStreamResponse",
    # Value types - private
    "GenerationOptions",
    "Model",
    "RailsConfig",
    "TracingConfig",
]
