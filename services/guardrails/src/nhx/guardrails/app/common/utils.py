# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Dict, TypeVar

from nhx.common.entities import make_filter_obj_dep
from nhx.guardrails.app.exceptions.exception_transformers import (
    LLM_CALL_ERROR_TRANSFORMERS,
    MODEL_INITIALIZATION_ERROR_PREFIX,
    MODEL_INTILIZATION_ERROR_TRANSFORMERS,
    clean_error_message,
)
from starlette.datastructures import QueryParams


def filter_match(item: Dict, filter_dict: Dict):
    """Helper function with generic filtering on dicts."""
    for key, filter_value in filter_dict.items():
        if filter_value is None:
            continue

        if key not in item:
            if filter_value is True:
                # Key must exist but doesn't
                return False
            elif filter_value is False:
                # Key must not exist and doesn't
                continue
            else:
                # Key doesn't exist, but filter expects a specific value
                return False
        else:
            item_value = item[key]
            if filter_value is True:
                # Key exists, which is expected
                continue
            elif filter_value is False:
                # Key exists but shouldn't
                return False
            elif isinstance(filter_value, dict) and isinstance(item_value, dict):
                # Recursively match nested dictionaries
                if not filter_match(item_value, filter_value):
                    return False
            else:
                # Check for equality with the filter value
                if isinstance(filter_value, str) and isinstance(item_value, dict):
                    if item_value.get("name") != filter_value:
                        return False
                elif item_value != filter_value:
                    return False
    return True


def parse_deep_object(name: str, params: QueryParams) -> Dict:
    """ "Helper function to parse 'deepObject'-like query parameters."""
    result = {}
    for key, value in params.items():
        # Split key by '[' and ']', handling nested keys like 'sub_item[name]'
        keys = key.replace("]", "").split("[")
        current = result
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Depending on how the encoding is done, this could be an encoded JSON
        if value.startswith("{"):
            value = json.loads(value)
        current[keys[-1]] = value
    return result.get(name)


def clean_model_initialization_error(error_message: str) -> str:
    """
    Given the error message for a `ModelInitializationError`, apply the first matching transformer to the message,
    or returns the original message if no transformer is detected.

    Args:
        error_message: The raw error message string

    Returns:
        The cleaned error message, or the original if no transformer is detected
    """
    return clean_error_message(error_message, MODEL_INTILIZATION_ERROR_TRANSFORMERS)


_IMAGE_URL_HINT = "Please verify that any image URLs in the request are accessible."


def clean_llm_call_error(error_message: str, has_image_urls: bool = False) -> str:
    """
    Given the error message for an `LLMCallException`, apply the first matching transformer to the message,
    or returns the original message if no transformer is detected.

    If `has_image_urls` is True, a hint is appended to the message suggesting to verify
    that any image URLs in the request are accessible.

    Args:
        error_message: The raw error message string
        has_image_urls: Whether the original request messages contained image URLs

    Returns:
        The cleaned error message, or the original if no transformer is detected
    """
    cleaned = clean_error_message(error_message, LLM_CALL_ERROR_TRANSFORMERS)
    if has_image_urls:
        cleaned += f" {_IMAGE_URL_HINT}"

    return cleaned


# Helper to build a FastAPI dependency that parses a deepObject filter into a dict
FilterType = TypeVar("FilterType")


__all__ = [
    "filter_match",
    "parse_deep_object",
    "make_filter_obj_dep",
    "clean_model_initialization_error",
    "MODEL_INITIALIZATION_ERROR_PREFIX",
]
