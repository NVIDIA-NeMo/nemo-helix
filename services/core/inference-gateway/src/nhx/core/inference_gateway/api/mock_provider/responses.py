# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock response definitions and builders for mock provider mode."""

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from starlette.datastructures import State

logger = logging.getLogger(__name__)

MOCK_RESPONSE_HEADER = "X-Mock-Response"
MOCK_STATUS_HEADER = "X-Mock-Status"
MOCK_RESPONSE_MAP_HEADER = "X-Mock-Response-Map"
MOCK_SERVED_MODELS_HEADER = "X-Mock-Served-Models"

SMART_DEFAULTS: dict[tuple[str, str], dict[str, str]] = {
    ("GET", "v1/health/ready"): {"status": "ready"},
    ("GET", "v1/health/live"): {"status": "live"},
    ("GET", "health/ready"): {"status": "ready"},
    ("GET", "health/live"): {"status": "live"},
}

DEFAULT_MODELS_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"id": "mock-model", "object": "model", "created": 0, "owned_by": "mock-provider"}],
}


class MockCallTracker:
    """Tracks call counts for mock provider responses.

    Stores counts keyed by model name (which includes workspace for isolation).
    This allows parallel tests with different workspaces to have isolated state.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def get_and_increment(self, model: str) -> int:
        """Get current call index for a model and increment for next call.

        Args:
            model: The model name (e.g., "e2e-abc123/content-safety")

        Returns:
            The current call index (0-based) before incrementing.
        """
        idx = self._counts.get(model, 0)
        self._counts[model] = idx + 1
        return idx

    def reset(self) -> None:
        """Reset all call counts."""
        self._counts.clear()

    def get_count(self, model: str) -> int:
        """Get the current call count for a model (for testing/debugging)."""
        return self._counts.get(model, 0)


def get_call_tracker(app_state: State) -> MockCallTracker:
    """Get the MockCallTracker from the FastAPI app.state, creating one if it doesn't exist.

    Args:
        app_state: The FastAPI app.state object.

    Returns:
        The MockCallTracker instance.
    """
    if not hasattr(app_state, "mock_call_tracker"):
        app_state.mock_call_tracker = MockCallTracker()
    return app_state.mock_call_tracker


def reset_call_counts(app_state: State) -> None:
    """Reset call counts on the app's tracker.

    Args:
        app_state: The FastAPI app.state object.
    """
    if hasattr(app_state, "mock_call_tracker"):
        app_state.mock_call_tracker.reset()


@dataclass
class MockResponseConfig:
    """Configuration for a mock response."""

    body: dict[str, Any]
    status_code: int = 200


def get_mock_response_config(
    request: Request,
    trailing_uri: str,
    default_extra_headers: dict[str, str] | None = None,
    request_body: dict[str, Any] | None = None,
    call_tracker: MockCallTracker | None = None,
) -> MockResponseConfig | None:
    """Extract mock response configuration from headers or smart defaults.

    Priority for requests to /v1/models:
    1. If X-Mock-Response-Map header exists in provider's default_extra_headers, derives models from its keys
    2. Smart defaults

    Priority for other requests:
    1. X-Mock-Response header on incoming request (highest priority)
    2. X-Mock-Response in provider's default_extra_headers
    3. X-Mock-Response-Map header in provider's default_extra_headers
    4. Smart defaults for health endpoints
    5. None if no response available

    Args:
        request: The incoming FastAPI request
        trailing_uri: The path suffix (e.g., "v1/chat/completions")
        default_extra_headers: Provider's default extra headers
        request_body: Parsed request body (needed for dynamic responses to get model)
        call_tracker: Optional MockCallTracker for dynamic per-model responses.
            If provided and dynamic responses are configured, tracks call counts per model.

    Returns:
        MockResponseConfig if a mock response is available, None otherwise

    Raises:
        ValueError: If X-Mock-Response header contains invalid JSON
    """
    normalized_uri = trailing_uri.lstrip("/")

    # Handle /v1/models endpoint.
    # Priority: X-Mock-Served-Models > X-Mock-Response-Map > smart default
    if request.method == "GET" and normalized_uri in ("v1/models", "models"):
        if default_extra_headers and MOCK_SERVED_MODELS_HEADER in default_extra_headers:
            return _parse_mock_served_models_response(default_extra_headers[MOCK_SERVED_MODELS_HEADER])
        if default_extra_headers and MOCK_RESPONSE_MAP_HEADER in default_extra_headers:
            return _parse_mock_dynamic_models_response(default_extra_headers[MOCK_RESPONSE_MAP_HEADER])
        logger.debug("Using smart default for models endpoint")
        return MockResponseConfig(body=DEFAULT_MODELS_RESPONSE)

    # Priority 1: X-Mock-Response header on incoming request
    if mock_response_header := request.headers.get(MOCK_RESPONSE_HEADER):
        return _parse_mock_response(mock_response_header, request.headers.get(MOCK_STATUS_HEADER))

    # Priority 2: X-Mock-Response in provider's default_extra_headers
    if default_extra_headers and MOCK_RESPONSE_HEADER in default_extra_headers:
        return _parse_mock_response(
            default_extra_headers[MOCK_RESPONSE_HEADER],
            default_extra_headers.get(MOCK_STATUS_HEADER),
        )

    # Priority 3: X-Mock-Response-Map in provider's default_extra_headers that store dynamic
    # responses per-model
    if default_extra_headers and MOCK_RESPONSE_MAP_HEADER in default_extra_headers:
        model = (request_body or {}).get("model", "*")
        return _parse_mock_dynamic_response(
            default_extra_headers[MOCK_RESPONSE_MAP_HEADER],
            model,
            call_tracker,
        )

    # Priority 4: Smart defaults for health endpoints
    smart_default_key = (request.method, normalized_uri)
    if smart_default_key in SMART_DEFAULTS:
        logger.debug(f"Using smart default for {smart_default_key}")
        return MockResponseConfig(body=SMART_DEFAULTS[smart_default_key])

    # Priority 5: No response available
    return None


def _parse_mock_response(body_json: str, status_header: str | None) -> MockResponseConfig:
    """Parse mock response from JSON string and optional status header."""
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {MOCK_RESPONSE_HEADER} header: {e}") from e

    status_code = 200
    if status_header:
        try:
            status_code = int(status_header)
        except ValueError as e:
            raise ValueError(f"Invalid status code in {MOCK_STATUS_HEADER} header: {status_header}") from e

    return MockResponseConfig(body=body, status_code=status_code)


def _models_list_response(model_ids: list[str]) -> MockResponseConfig:
    return MockResponseConfig(
        body={
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "mock-provider"} for model_id in model_ids
            ],
        }
    )


def _parse_mock_served_models_response(served_models_json: str) -> MockResponseConfig:
    """Parse mock served models list from JSON string."""
    try:
        model_ids = json.loads(served_models_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {MOCK_SERVED_MODELS_HEADER} header: {e}") from e
    return _models_list_response(model_ids)


def _parse_mock_dynamic_models_response(body_json: str) -> MockResponseConfig:
    """Parse mock dynamic models response from JSON string."""
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {MOCK_RESPONSE_MAP_HEADER} header: {e}") from e

    # Extract the model IDs (without workspace) from the response map
    model_keys = [key for key in body.keys() if key != "*"]
    model_ids = [key.split("/")[-1] if "/" in key else key for key in model_keys]
    return _models_list_response(model_ids)


def _parse_mock_dynamic_response(
    body_json: str,
    model: str,
    call_tracker: MockCallTracker | None,
) -> MockResponseConfig | None:
    """Parse mock dynamic per-model response from JSON string.

    The header value is a map of model name to list of responses:
    {"workspace/model": [response1, response2], ...}

    Returns sequential responses per model, clamping to last response if exhausted.
    The `call_tracker` tracks the index of the response to return per-model.

    If provided, falls back to wildcard "*" response if exact model match not found.

    Raises:
        ValueError: If JSON is invalid.
    """
    try:
        response_map = json.loads(body_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {MOCK_RESPONSE_MAP_HEADER} header: {e}") from e

    # Get list of responses for this model
    responses = response_map.get(model) or response_map.get("*")
    if not responses:
        return None

    idx = call_tracker.get_and_increment(model) if call_tracker else 0
    # Get response at `idx` index for this model
    response_body = responses[min(idx, len(responses) - 1)]
    logger.debug(f"Using dynamic response for model '{model}' (call {idx})")

    return MockResponseConfig(
        body=response_body.get("response_body", {}), status_code=response_body.get("response_code", 200)
    )
