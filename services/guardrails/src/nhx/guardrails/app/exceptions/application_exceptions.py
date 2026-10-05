# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nemoguardrails.actions.llm.utils import LLMCallException

logger = logging.getLogger(__name__)


class MissingEnvironmentVariableError(Exception):
    def __init__(self, message: str = "Missing environment variable."):
        super().__init__(message)


class CustomHTTPException(Exception):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class GuardrailConfigurationNotFoundError(Exception):
    """Raised when a requested guardrail config ID cannot be found."""

    def __init__(self, config_id: str, message: str | None = None):
        self.config_id = config_id
        super().__init__(message or f"Guardrail config not found: {config_id}.")


__all__ = [
    "MissingEnvironmentVariableError",
    "CustomHTTPException",
    "GuardrailConfigurationNotFoundError",
    "LLMCallException",
]
