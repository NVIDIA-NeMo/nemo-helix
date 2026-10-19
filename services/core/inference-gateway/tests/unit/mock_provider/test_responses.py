# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mock_provider responses module."""

import json
from unittest.mock import Mock

import pytest
from fastapi import Request
from nhx.core.inference_gateway.api.mock_provider.responses import (
    DEFAULT_MODELS_RESPONSE,
    MOCK_RESPONSE_HEADER,
    MOCK_SERVED_MODELS_HEADER,
    MOCK_STATUS_HEADER,
    MockResponseConfig,
    get_mock_response_config,
)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock(spec=Request)
    request.method = "GET"
    request.headers = {}
    return request


def test_mock_response_config_defaults():
    """Test MockResponseConfig has correct default values."""
    config = MockResponseConfig(body={"test": True})
    assert config.body == {"test": True}
    assert config.status_code == 200


def test_mock_response_config_custom_status():
    """Test MockResponseConfig with custom status code."""
    config = MockResponseConfig(body={"error": "not found"}, status_code=404)
    assert config.body == {"error": "not found"}
    assert config.status_code == 404


def test_get_mock_response_from_request_header(mock_request):
    """Test extracting mock response from request header."""
    mock_response = {"id": "test", "result": "success"}
    mock_request.headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}

    config = get_mock_response_config(mock_request, "v1/chat/completions")

    assert config is not None
    assert config.body == mock_response
    assert config.status_code == 200


def test_get_mock_response_from_request_header_with_status(mock_request):
    """Test extracting mock response with custom status from request headers."""
    mock_response = {"error": "rate limited"}
    mock_request.headers = {
        MOCK_RESPONSE_HEADER: json.dumps(mock_response),
        MOCK_STATUS_HEADER: "429",
    }

    config = get_mock_response_config(mock_request, "v1/chat/completions")

    assert config is not None
    assert config.body == mock_response
    assert config.status_code == 429


def test_get_mock_response_from_provider_defaults(mock_request):
    """Test extracting mock response from provider default_extra_headers."""
    mock_response = {"provider": "default"}
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(mock_response)}

    config = get_mock_response_config(
        mock_request,
        "v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert config is not None
    assert config.body == mock_response
    assert config.status_code == 200


def test_request_header_takes_priority_over_provider(mock_request):
    """Test that request header takes priority over provider defaults."""
    request_response = {"source": "request"}
    provider_response = {"source": "provider"}

    mock_request.headers = {MOCK_RESPONSE_HEADER: json.dumps(request_response)}
    default_extra_headers = {MOCK_RESPONSE_HEADER: json.dumps(provider_response)}

    config = get_mock_response_config(
        mock_request,
        "v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert config is not None
    assert config.body == request_response


@pytest.mark.parametrize(
    "trailing_uri",
    [
        "v1/health/ready",
        "/v1/health/ready",
        "health/ready",
        "/health/ready",
    ],
)
def test_smart_defaults_health_ready(mock_request, trailing_uri):
    """Test smart defaults for health/ready endpoint."""
    mock_request.method = "GET"

    config = get_mock_response_config(mock_request, trailing_uri)

    assert config is not None
    assert config.body == {"status": "ready"}
    assert config.status_code == 200


@pytest.mark.parametrize(
    "trailing_uri",
    [
        "v1/health/live",
        "/v1/health/live",
        "health/live",
        "/health/live",
    ],
)
def test_smart_defaults_health_live(mock_request, trailing_uri):
    """Test smart defaults for health/live endpoint."""
    mock_request.method = "GET"

    config = get_mock_response_config(mock_request, trailing_uri)

    assert config is not None
    assert config.body == {"status": "live"}
    assert config.status_code == 200


@pytest.mark.parametrize(
    "trailing_uri",
    [
        "v1/models",
        "/v1/models",
        "models",
        "/models",
    ],
)
def test_smart_defaults_models(mock_request, trailing_uri):
    """Test smart defaults for models endpoint."""
    mock_request.method = "GET"

    config = get_mock_response_config(mock_request, trailing_uri)

    assert config is not None
    assert config.body == DEFAULT_MODELS_RESPONSE
    assert config.status_code == 200


@pytest.mark.parametrize(
    "trailing_uri",
    ["v1/models", "/v1/models", "models", "/models"],
)
def test_served_models_header_for_models_endpoint(mock_request, trailing_uri):
    """Test that MOCK_SERVED_MODELS_HEADER returns the configured model IDs for GET /v1/models."""
    mock_request.method = "GET"
    default_extra_headers = {MOCK_SERVED_MODELS_HEADER: '["my-model", "other-model"]'}

    config = get_mock_response_config(mock_request, trailing_uri, default_extra_headers=default_extra_headers)

    assert config is not None
    assert config.body["object"] == "list"
    model_ids = [m["id"] for m in config.body["data"]]
    assert model_ids == ["my-model", "other-model"]


def test_served_models_header_takes_priority_over_smart_default(mock_request):
    """MOCK_SERVED_MODELS_HEADER takes priority over the DEFAULT_MODELS_RESPONSE smart default."""
    mock_request.method = "GET"
    default_extra_headers = {
        MOCK_SERVED_MODELS_HEADER: '["e2e-chat-abc123"]',
        MOCK_RESPONSE_HEADER: '{"id": "chatcmpl-1", "object": "chat.completion", "choices": []}',
    }

    config = get_mock_response_config(mock_request, "v1/models", default_extra_headers=default_extra_headers)

    assert config is not None
    model_ids = [m["id"] for m in config.body["data"]]
    assert model_ids == ["e2e-chat-abc123"]
    assert model_ids != [m["id"] for m in DEFAULT_MODELS_RESPONSE["data"]]


def test_served_models_header_does_not_affect_other_endpoints(mock_request):
    """MOCK_SERVED_MODELS_HEADER should not affect non-/v1/models endpoints."""
    mock_request.method = "POST"
    chat_response = {"id": "chatcmpl-1", "object": "chat.completion"}
    default_extra_headers = {
        MOCK_SERVED_MODELS_HEADER: '["my-model"]',
        MOCK_RESPONSE_HEADER: json.dumps(chat_response),
    }

    config = get_mock_response_config(mock_request, "v1/chat/completions", default_extra_headers=default_extra_headers)

    assert config is not None
    assert config.body == chat_response


def test_invalid_json_in_header_raises_error(mock_request):
    """Test that invalid JSON in header raises ValueError."""
    mock_request.headers = {MOCK_RESPONSE_HEADER: "not valid json"}

    with pytest.raises(ValueError, match="Invalid JSON"):
        get_mock_response_config(mock_request, "v1/chat/completions")


def test_no_response_available_returns_none(mock_request):
    """Test that None is returned when no mock response is available."""
    mock_request.method = "POST"

    config = get_mock_response_config(mock_request, "v1/chat/completions")

    assert config is None


def test_invalid_status_code_raises_value_error(mock_request):
    """Test that invalid status code raises ValueError."""
    mock_response = {"test": True}
    mock_request.headers = {
        MOCK_RESPONSE_HEADER: json.dumps(mock_response),
        MOCK_STATUS_HEADER: "not_a_number",
    }

    with pytest.raises(ValueError) as exc_info:
        get_mock_response_config(mock_request, "v1/chat/completions")

    assert "Invalid status code" in str(exc_info.value)
    assert "not_a_number" in str(exc_info.value)


def test_smart_defaults_only_apply_to_get(mock_request):
    """Test that smart defaults only apply to GET requests."""
    mock_request.method = "POST"

    config = get_mock_response_config(mock_request, "v1/health/ready")

    assert config is None


def test_provider_defaults_with_status(mock_request):
    """Test provider defaults with custom status code."""
    mock_response = {"error": "service unavailable"}
    default_extra_headers = {
        MOCK_RESPONSE_HEADER: json.dumps(mock_response),
        MOCK_STATUS_HEADER: "503",
    }

    config = get_mock_response_config(
        mock_request,
        "v1/chat/completions",
        default_extra_headers=default_extra_headers,
    )

    assert config is not None
    assert config.body == mock_response
    assert config.status_code == 503
