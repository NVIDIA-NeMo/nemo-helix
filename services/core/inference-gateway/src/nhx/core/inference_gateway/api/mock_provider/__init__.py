# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock provider mode for Inference Gateway.

This module provides mock response handling for testing and development
without real inference backends.
"""

from nhx.core.inference_gateway.api.mock_provider.handlers import handle_mock_request
from nhx.core.inference_gateway.api.mock_provider.responses import (
    MOCK_RESPONSE_HEADER,
    MOCK_RESPONSE_MAP_HEADER,
    MOCK_SERVED_MODELS_HEADER,
    MOCK_STATUS_HEADER,
    MockCallTracker,
    get_call_tracker,
    reset_call_counts,
)
from nhx.core.inference_gateway.api.mock_provider.utils import (
    is_mock_mode_enabled,
    is_mock_provider,
    is_mock_request,
)

__all__ = [
    "handle_mock_request",
    "MockCallTracker",
    "get_call_tracker",
    "reset_call_counts",
    "MOCK_RESPONSE_HEADER",
    "MOCK_RESPONSE_MAP_HEADER",
    "MOCK_SERVED_MODELS_HEADER",
    "MOCK_STATUS_HEADER",
    "is_mock_mode_enabled",
    "is_mock_provider",
    "is_mock_request",
]
