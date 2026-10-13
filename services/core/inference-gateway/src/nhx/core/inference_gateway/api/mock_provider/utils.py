# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for mock provider mode."""

from fastapi import Request
from nhx.common.config import Configuration
from nhx.core.inference_gateway.config import InferenceGatewayConfig

from .responses import MOCK_RESPONSE_HEADER


def _get_mock_provider_prefix() -> str | None:
    """Get the mock_provider_prefix from config (supports test overrides)."""
    config = Configuration.get_service_config(InferenceGatewayConfig)
    return config.mock_provider_prefix


def is_mock_mode_enabled() -> bool:
    """Check if mock provider mode is enabled globally."""
    return _get_mock_provider_prefix() is not None


def is_mock_provider(provider_name: str) -> bool:
    """Check if a provider should return mock responses based on prefix matching.

    Args:
        provider_name: The name of the provider to check.

    Returns:
        True if mock_provider_prefix is set and provider_name starts with it.
    """
    return is_mock_mode_enabled() and provider_name.startswith(_get_mock_provider_prefix())


def is_mock_request(request: Request) -> bool:
    """Check if a request should return a mock response (early return path).

    This checks if mock mode is enabled AND the request has an X-Mock-Response header.
    Used for the early return path that bypasses provider/model lookup entirely.

    Args:
        request: The FastAPI request object.

    Returns:
        True if mock mode is enabled and request has X-Mock-Response header.
    """
    return is_mock_mode_enabled() and MOCK_RESPONSE_HEADER in request.headers
