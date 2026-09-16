# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for LLM error handling in the completion NIM class."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from nhx.guardrails.app.llms.completion.nim import NIM

# Expected error messages - must match exactly with nim.py
EXPECTED_401_MESSAGE = (
    "Authentication failed. Please verify your API key is valid and configured to be used by this endpoint."
)
EXPECTED_404_MESSAGE = (
    "The endpoint was not found. This can occur when the /completions endpoint is not supported for this model. "
    "Please try the /chat/completions endpoint instead."
)


@pytest.fixture
def mock_main_model():
    """Patch get_main_model_from_context to allow NIM instantiation."""
    with patch("nhx.guardrails.app.llms.utils.get_main_model_from_context") as mock:
        mock.return_value = None
        yield mock


def create_mock_response(status_code: int, is_success: bool = False, json_data: dict = None):
    """Create a mock httpx Response object."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.is_success = is_success
    mock_response.json.return_value = json_data or {"error": "test error"}
    return mock_response


def test_handle_response_error_success_returns_none(mock_main_model):
    """Test that successful responses don't raise exceptions."""
    nim = NIM(model="test-model")
    mock_response = create_mock_response(status_code=200, is_success=True)

    result = nim._handle_response_error(mock_response)
    assert result is None


def test_handle_response_error_401_raises_http_exception(mock_main_model):
    """Test that 401 responses raise HTTPException with exact authentication message."""
    nim = NIM(model="test-model")
    mock_response = create_mock_response(status_code=401, is_success=False)

    with pytest.raises(HTTPException) as exc_info:
        nim._handle_response_error(mock_response)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == EXPECTED_401_MESSAGE


def test_handle_response_error_404_raises_http_exception(mock_main_model):
    """Test that 404 responses raise HTTPException with exact completions endpoint message."""
    nim = NIM(model="test-model")
    mock_response = create_mock_response(status_code=404, is_success=False)

    with pytest.raises(HTTPException) as exc_info:
        nim._handle_response_error(mock_response)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == EXPECTED_404_MESSAGE


@pytest.mark.parametrize("status_code", [400, 429, 500, 502, 503])
def test_handle_response_error_other_errors_passthrough_json(mock_main_model, status_code):
    """Test that other error responses raise HTTPException with JSON response as detail."""
    nim = NIM(model="test-model")
    error_json = {"error": f"Error {status_code}"}
    mock_response = create_mock_response(
        status_code=status_code,
        is_success=False,
        json_data=error_json,
    )

    with pytest.raises(HTTPException) as exc_info:
        nim._handle_response_error(mock_response)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == error_json
