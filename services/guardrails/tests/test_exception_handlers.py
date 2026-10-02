# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nhx.guardrails.app.exceptions.application_exceptions import (
    CustomHTTPException,
    LLMCallException,
)
from nhx.guardrails.app.exceptions.exception_handlers import (
    _format_field_path,
    _format_validation_message,
    _request_has_image_urls,
    authentication_error_handler,
    custom_404_handler,
    custom_exception_handler,
    llm_call_exception_handler,
    rate_limit_error_handler,
    validation_error_handler,
)
from openai import APIError, AuthenticationError, RateLimitError
from starlette.exceptions import HTTPException as StarletteHTTPException


@pytest.mark.asyncio
async def test_llm_call_exception_handler():
    request = Request(scope={"type": "http"})

    # HTTPException
    inner_exc = HTTPException(status_code=400, detail="Bad Request")
    exc = LLMCallException(inner_exception=inner_exc)
    response = await llm_call_exception_handler(request, exc)
    assert response.status_code == 400

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": "Bad Request"}

    # APIError
    inner_exc = APIError(
        request=httpx.Request("GET", "http://testserver"),
        message="API Error",
        body=None,
    )
    exc = LLMCallException(inner_exception=inner_exc)
    response = await llm_call_exception_handler(request, exc)
    assert response.status_code == 500

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": "API Error"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code,detail",
    [
        (401, "Authentication failed. Please verify your API key."),
        (404, "The /completions endpoint is not available."),
        (429, "Rate limit exceeded."),
        (500, "Server error."),
        (502, "Bad gateway."),
        (503, "Service unavailable."),
    ],
)
async def test_llm_call_exception_handler_preserves_status_codes(status_code, detail):
    """Test that llm_call_exception_handler preserves status codes from HTTPException."""
    request = Request(scope={"type": "http"})

    inner_exc = HTTPException(status_code=status_code, detail=detail)
    exc = LLMCallException(inner_exception=inner_exc)
    response = await llm_call_exception_handler(request, exc)

    assert response.status_code == status_code

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": detail}


@pytest.mark.asyncio
async def test_llm_call_exception_handler_with_openai_authentication_error():
    """Test that llm_call_exception_handler returns 401 for OpenAI AuthenticationError."""
    request = Request(scope={"type": "http"})

    mock_response = httpx.Response(status_code=401, request=httpx.Request("GET", "http://testserver"))
    inner_exc = AuthenticationError(message="Unauthorized", response=mock_response, body=None)
    exc = LLMCallException(inner_exception=inner_exc)
    response = await llm_call_exception_handler(request, exc)

    assert response.status_code == 401

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert "Authentication failed" in parsed_body["detail"]


@pytest.mark.asyncio
async def test_llm_call_exception_handler_with_openai_rate_limit_error():
    """Test that llm_call_exception_handler returns 429 for OpenAI RateLimitError."""
    request = Request(scope={"type": "http"})

    mock_response = httpx.Response(status_code=429, request=httpx.Request("GET", "http://testserver"))
    inner_exc = RateLimitError(message="Rate limit exceeded", response=mock_response, body=None)
    exc = LLMCallException(inner_exception=inner_exc)
    response = await llm_call_exception_handler(request, exc)

    assert response.status_code == 429

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert "Rate limit exceeded" in parsed_body["detail"]


@pytest.mark.asyncio
async def test_custom_exception_handler():
    request = Request(scope={"type": "http"})
    exc = CustomHTTPException(status_code=403, message="Forbidden")
    response = await custom_exception_handler(request, exc)
    assert response.status_code == 403

    body_str = bytes(response.body).decode("utf-8")

    # Parse the JSON
    parsed_body = json.loads(body_str)
    assert parsed_body == {"message": "Forbidden"}


@pytest.mark.asyncio
async def test_authentication_error_handler():
    request = Request(scope={"type": "http"})

    mock_response = httpx.Response(status_code=401, request=httpx.Request("GET", "http://testserver"))

    exc = AuthenticationError(message="Authentication Failed", response=mock_response, body=None)
    exc.code = "401"
    response = await authentication_error_handler(request, exc)
    assert response.status_code == 401

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": "Authentication Failed"}


@pytest.mark.asyncio
async def test_rate_limit_error_handler():
    request = Request(scope={"type": "http"})

    mock_response = httpx.Response(status_code=429, request=httpx.Request("GET", "http://testserver"))

    exc = RateLimitError(message="Rate Limit Exceeded", response=mock_response, body=None)
    response = await rate_limit_error_handler(request, exc)
    assert response.status_code == 429

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": "Rate Limit Exceeded"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail, expected_detail",
    [
        (None, "The requested resource was not found."),
        ("My custom detailed message from the API handler", "My custom detailed message from the API handler"),
    ],
)
async def test_custom_404_handler(detail, expected_detail):
    request = Request(scope={"type": "http", "path": "/test-path", "method": "GET", "headers": []})
    exc = StarletteHTTPException(status_code=404, detail=detail)
    response = await custom_404_handler(request, exc)
    assert response.status_code == 404

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {
        "detail": expected_detail,
        "path": "/test-path",
        "method": "GET",
    }


# Tests for validation error formatting helpers
def test_format_field_path():
    """Test field path formatting."""
    # Normal case
    assert _format_field_path(("body", "data", "prompts")) == "body.data.prompts"

    # Single element
    assert _format_field_path(("body",)) == "body"

    # Empty tuple
    assert _format_field_path(()) == ""

    # With numeric indices
    assert _format_field_path(("body", "messages", 0, "content")) == "body.messages.0.content"


@pytest.mark.parametrize(
    "error_message, error_loc, include_prefix, expected_result",
    [
        # Result message should contain the default prefix and error location
        ("Field required", ("body", "model"), True, "Validation error at body.model: Field required"),
        # Result message should contain the error location, but not a prefix
        ("Field required", ("body", "model"), False, "body.model: Field required"),
        # Result message shuold contain a prefix, but not an error location
        ("Invalid value", (), True, "Invalid value"),
        # Result message should not contain a prefix or an error location
        ("Invalid value", (), False, "Invalid value"),
        # Result message should contain the default prefix and deeply nested error location
        (
            "Input should be greater than 0",
            ("body", "data", "rails", "input"),
            True,
            "Validation error at body.data.rails.input: Input should be greater than 0",
        ),
    ],
)
def test_format_validation_message(error_message, error_loc, include_prefix, expected_result):
    """Test validation message formatting with various inputs."""
    result = _format_validation_message(error_message, error_loc, include_prefix)
    assert result == expected_result


@pytest.mark.asyncio
async def test_validation_error_handler_with_ctx_error():
    """Test handling of validation error with custom error in the Pydantic validation error's ctx."""
    request = Request(scope={"type": "http"})

    # Create a mock validation error with ctx.error
    custom_error = InvalidRailsConfigurationError(
        "Missing a `content_safety_check_input` prompt, which is required for the `content safety check input` rail."
    )
    validation_error = [
        {
            "type": "value_error",
            "loc": ("body", "data"),
            "msg": "Value error, Missing prompt",
            "input": {"some": "data"},
            "ctx": {"error": custom_error},
        }
    ]

    exc = RequestValidationError(errors=validation_error)
    response = await validation_error_handler(request, exc)

    assert response.status_code == 422

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {
        "detail": "Validation error at body.data: Missing a `content_safety_check_input` prompt, which is required for the `content safety check input` rail."
    }


@pytest.mark.asyncio
async def test_validation_error_handler_single_standard_error():
    """Test handling of single standard Pydantic validation error."""
    request = Request(scope={"type": "http"})

    # Mock Pydantic validation error (no ctx.error)
    validation_errors = [
        {
            "type": "missing",
            "loc": ("body", "model"),
            "msg": "Field required",
            "input": {"messages": []},
        }
    ]

    exc = RequestValidationError(errors=validation_errors)
    response = await validation_error_handler(request, exc)

    assert response.status_code == 422

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {"detail": "Validation error at body.model: Field required"}


@pytest.mark.asyncio
async def test_validation_error_handler_multiple_errors():
    """Test handling of multiple validation errors."""
    request = Request(scope={"type": "http"})

    # Mock multiple validation errors
    validation_errors = [
        {
            "type": "missing",
            "loc": ("body", "model"),
            "msg": "Field required",
            "input": {},
        },
        {
            "type": "missing",
            "loc": ("body", "messages"),
            "msg": "Field required",
            "input": {},
        },
        {
            "type": "float_parsing",
            "loc": ("body", "temperature"),
            "msg": "Input should be a valid number",
            "input": "invalid",
        },
    ]

    exc = RequestValidationError(errors=validation_errors)
    response = await validation_error_handler(request, exc)

    assert response.status_code == 422

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    # Multiple errors should be concatenated with semicolons
    assert parsed_body == {
        "detail": "body.model: Field required; body.messages: Field required; body.temperature: Input should be a valid number"
    }


@pytest.mark.asyncio
async def test_validation_error_handler_error_without_location():
    """Test handling of validation error without location field."""
    request = Request(scope={"type": "http"})

    # Mock validation error without location
    validation_errors = [
        {
            "type": "value_error",
            "loc": (),
            "msg": "Invalid request body",
            "input": None,
        }
    ]

    exc = RequestValidationError(errors=validation_errors)
    response = await validation_error_handler(request, exc)

    assert response.status_code == 422

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    # Should return just the message without location
    assert parsed_body == {"detail": "Invalid request body"}


@pytest.mark.asyncio
async def test_validation_error_handler_nested_field_path():
    """Test handling of validation error with deeply nested field path."""
    request = Request(scope={"type": "http"})

    # Mock validation error with nested path
    validation_errors = [
        {
            "type": "less_than_equal",
            "loc": ("body", "data", "rails", "input", "parallel"),
            "msg": "Input should be less than or equal to 1",
            "input": 5,
        }
    ]

    exc = RequestValidationError(errors=validation_errors)
    response = await validation_error_handler(request, exc)

    assert response.status_code == 422

    body_str = bytes(response.body).decode("utf-8")
    parsed_body = json.loads(body_str)

    assert parsed_body == {
        "detail": "Validation error at body.data.rails.input.parallel: Input should be less than or equal to 1"
    }


# ---------------------------------------------------------------------------
# Tests for _request_has_image_urls
# ---------------------------------------------------------------------------


class TestRequestHasImageUrls:
    def test_returns_false_for_non_list(self):
        assert _request_has_image_urls(None) is False
        assert _request_has_image_urls({}) is False
        assert _request_has_image_urls("not a list") is False

    def test_returns_false_for_empty_list(self):
        assert _request_has_image_urls([]) is False

    def test_returns_false_for_text_only_messages(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert _request_has_image_urls(messages) is False

    def test_returns_false_for_string_content(self):
        """String content (non-multimodal) has no image parts."""
        messages = [{"role": "user", "content": "plain text"}]
        assert _request_has_image_urls(messages) is False

    def test_returns_true_for_message_with_image_url(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        assert _request_has_image_urls(messages) is True

    def test_returns_true_for_data_uri(self):
        """data: URIs are also image_url parts and should be detected."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}},
                ],
            }
        ]
        assert _request_has_image_urls(messages) is True

    def test_returns_true_when_image_url_in_second_message(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            },
        ]
        assert _request_has_image_urls(messages) is True

    def test_skips_malformed_parts(self):
        """Parts that are not dicts are silently skipped."""
        messages = [{"role": "user", "content": ["not-a-dict", None, 42]}]
        assert _request_has_image_urls(messages) is False
