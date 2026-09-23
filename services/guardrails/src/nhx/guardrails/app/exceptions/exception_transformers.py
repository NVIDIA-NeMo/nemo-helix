# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

# Generic `ModelInitializationError` prefix prepended to errors when `nemoguardrails` fails to initialize a model
MODEL_INITIALIZATION_ERROR_PREFIX = "Failed to initialize model"

# Pattern to match URL connection errors when invoking models
# Example: "NameResolutionError("HTTPConnection(host='localhost', port=8000): Failed to resolve 'localhost'"
URL_CONNECTION_ERROR_PATTERN = re.compile(
    r"""NameResolutionError\("HTTPConnection\(host='(?P<host>[^']+)', port=(?P<port>\d+)\):""",
    re.IGNORECASE,
)
URL_CONNECTION_EXCEPTION_PREFIX = "NameResolutionError"

# Pattern to match retry exhaustion, indicating URL connection error.
MAX_RETRIES_EXCEEDED_SUBSTRING = "Max retries exceeded with url"

logger = logging.getLogger(__name__)


def normalize_error_message(error_message: str) -> str:
    """
    Normalize an error message for matching purposes.

    Removes escaped characters and converts to lowercase to enable
    case-insensitive matching.
    """
    return error_message.replace("\\", "").lower()


@dataclass
class ErrorTransformer:
    """Defines a transformer for a specific error message."""

    # Function that determines if the transformer should be applied to the error message
    matcher: Callable[[str], bool]
    # Function that transforms the error message
    transformer: Callable[[str], str]


def matches_model_initialization_error(error_message: str) -> bool:
    normalized = normalize_error_message(error_message)
    return MODEL_INITIALIZATION_ERROR_PREFIX.lower() in normalized and ":" in error_message


def transform_model_initialization_error(error_message: str) -> str:
    """
    Returns the message with the
    "Failed to initialize model 'X' with provider 'Y' in 'Z' mode:" prefix removed.

    We do this because this hardcoded prefix is arguably more confusing than helpful to the user.
    For example, if a model's engine is `nim`, we transform it to `nimchat` or `nimllm`, which appears
    as the provider in the error message. This is an internal implementation detail that may lead the user to believe
    their config is incorrect. In reality, the rest of the error message is more likely to point to the issue.

     Example:
        error_message = "Failed to initialize model 'meta/llama-3.3-70b-instruct' with provider 'nimchat' in 'chat' mode: Invalid API key for model 'meta/llama-3.3-70b-instruct'"
        clean_model_initialization_error(error_message)
        -> "Invalid API key for model 'meta/llama-3.3-70b-instruct'"
    """
    # Split on the first colon to extract the actual error after the prefix
    parts = error_message.split(":", 1)
    if len(parts) > 1:
        cleaned_message = parts[1].strip()
        if cleaned_message:
            return cleaned_message

    return error_message


FAILED_TO_INITIALIZE_MODEL_TRANSFORMER = ErrorTransformer(
    matcher=matches_model_initialization_error,
    transformer=transform_model_initialization_error,
)


def matches_model_not_found_error(error_message: str) -> bool:
    normalized = normalize_error_message(error_message)
    return "404" in normalized or "not found" in normalized


def transform_model_not_found_error(_error_message: str) -> str:
    return "Model not found. Please check if the model exists at this endpoint."


MODEL_NOT_FOUND_TRANSFORMER = ErrorTransformer(
    matcher=matches_model_not_found_error,
    transformer=transform_model_not_found_error,
)


def matches_authentication_error(error_message: str) -> bool:
    normalized = normalize_error_message(error_message)
    return "401" in normalized or "unauthorized" in normalized


def transform_authentication_error(_error_message: str) -> str:
    return "Authentication failed. Please check your API key or provider credentials."


AUTHENTICATION_ERROR_TRANSFORMER = ErrorTransformer(
    matcher=matches_authentication_error,
    transformer=transform_authentication_error,
)


def matches_connection_error(error_message: str) -> bool:
    normalized = normalize_error_message(error_message)
    return (
        URL_CONNECTION_ERROR_PATTERN.search(error_message) is not None
        or URL_CONNECTION_EXCEPTION_PREFIX.lower() in normalized
        or MAX_RETRIES_EXCEEDED_SUBSTRING.lower() in normalized
    )


def transform_connection_error(error_message: str) -> str:
    match = URL_CONNECTION_ERROR_PATTERN.search(error_message)
    if match:
        host = match.group("host")
        port = match.group("port")
        return f"Failed to connect to '{host}:{port}'. Please check the URL and network connectivity."

    return "Failed to connect to the model endpoint. Please check the URL and network connectivity."


CONNECTION_ERROR_TRANSFORMER = ErrorTransformer(
    matcher=matches_connection_error,
    transformer=transform_connection_error,
)

# Error transformers for `ModelInitializationError` error messages
MODEL_INTILIZATION_ERROR_TRANSFORMERS: list[ErrorTransformer] = [
    CONNECTION_ERROR_TRANSFORMER,
    FAILED_TO_INITIALIZE_MODEL_TRANSFORMER,
]

# Error transformers for `LLMCallException` error messages
LLM_CALL_ERROR_TRANSFORMERS: list[ErrorTransformer] = [
    CONNECTION_ERROR_TRANSFORMER,
    MODEL_NOT_FOUND_TRANSFORMER,
    AUTHENTICATION_ERROR_TRANSFORMER,
]


def clean_error_message(error_message: str, transformers: list[ErrorTransformer]) -> str:
    """Apply the first matching error transform to the message."""
    if not error_message:
        return error_message

    # Remove escaped characters before matching/transforming
    unescaped_message = error_message.replace("\\", "")
    for transform in transformers:
        if transform.matcher(unescaped_message):
            return transform.transformer(unescaped_message)

    return error_message
