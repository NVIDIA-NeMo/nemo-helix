# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.guardrails.app.exceptions.exception_transformers import (
    MAX_RETRIES_EXCEEDED_SUBSTRING,
    matches_authentication_error,
    matches_connection_error,
    matches_model_initialization_error,
    matches_model_not_found_error,
    transform_authentication_error,
    transform_connection_error,
    transform_model_initialization_error,
    transform_model_not_found_error,
)


class TestModelInitializationErrorTransformer:
    def test_matches_model_initialization_error(self):
        message = (
            "Failed to initialize model 'meta/llama-3.3-70b-instruct' "
            "with provider 'nimchat' in 'chat' mode: Invalid API key"
        )
        assert matches_model_initialization_error(message)

    def test_matches_model_initialization_error_case_insensitive(self):
        message = (
            "FAILED TO INITIALIZE MODEL 'meta/llama-3.3-70b-instruct' "
            "with provider 'nimchat' in 'chat' mode: Invalid API key"
        )
        assert matches_model_initialization_error(message)

    def test_transform_model_initialization_error_strips_prefix(self):
        message = (
            "Failed to initialize model 'meta/llama-3.3-70b-instruct' "
            "with provider 'nimchat' in 'chat' mode: Invalid API key"
        )
        assert transform_model_initialization_error(message) == "Invalid API key"

    def test_transform_model_initialization_error_no_colon_returns_original(self):
        message = "Failed to initialize model 'x' with provider 'y' in 'chat' mode"
        assert transform_model_initialization_error(message) == message


class TestModelNotFoundTransformer:
    def test_matches_model_not_found_error(self):
        assert matches_model_not_found_error("[404] Not Found")

    def test_matches_model_not_found_error_case_insensitive(self):
        assert matches_model_not_found_error("[404] NOT FOUND")

    def test_transform_model_not_found_error(self):
        assert (
            transform_model_not_found_error("anything")
            == "Model not found. Please check if the model exists at this endpoint."
        )


class TestConnectionErrorTransformer:
    def test_matches_connection_error_name_resolution(self):
        message = (
            "NameResolutionError(\"HTTPConnection(host='internal', port=8000): "
            "Failed to resolve 'internal' ([Errno 8] nodename nor servname provided, or not known)\")"
        )
        assert matches_connection_error(message)

    def test_matches_connection_error_name_resolution_case_insensitive(self):
        message = (
            "NAMEresolutionerror(\"HTTPConnection(host='internal', port=8000): "
            "Failed to resolve 'internal' ([Errno 8] nodename nor servname provided, or not known)\")"
        )
        assert matches_connection_error(message)

    def test_transform_connection_error_name_resolution(self):
        message = (
            "NameResolutionError(\"HTTPConnection(host='internal', port=8000): "
            "Failed to resolve 'internal' ([Errno 8] nodename nor servname provided, or not known)\")"
        )
        assert transform_connection_error(message) == (
            "Failed to connect to 'internal:8000'. Please check the URL and network connectivity."
        )

    def test_matches_connection_error_max_retries(self):
        message = (
            f"HTTPConnectionPool(host='localhost', port=8080): {MAX_RETRIES_EXCEEDED_SUBSTRING}: /v1/chat/completions"
        )
        assert matches_connection_error(message)

    def test_matches_connection_error_max_retries_case_insensitive(self):
        message = "HTTPConnectionPool(host='localhost', port=8080): Max Retries Exceeded With Url: /v1/chat/completions"
        assert matches_connection_error(message)

    def test_transform_connection_error_max_retries(self):
        message = (
            f"HTTPConnectionPool(host='localhost', port=8080): {MAX_RETRIES_EXCEEDED_SUBSTRING}: /v1/chat/completions"
        )
        assert transform_connection_error(message) == (
            "Failed to connect to the model endpoint. Please check the URL and network connectivity."
        )


class TestAuthenticationErrorTransformer:
    def test_matches_authentication_error_with_401(self):
        message = "Error code: 401 Unauthorized"
        assert matches_authentication_error(message)

    def test_matches_authentication_error_with_unauthorized(self):
        message = "Unauthorized access to the API"
        assert matches_authentication_error(message)

    def test_does_not_match_generic_authentication_message(self):
        message = "Authentication failed for this request"
        assert not matches_authentication_error(message)

    def test_matches_authentication_error_case_insensitive(self):
        message = "UNAUTHORIZED ACCESS"
        assert matches_authentication_error(message)

    def test_does_not_match_unrelated_error(self):
        message = "Connection timeout"
        assert not matches_authentication_error(message)

    def test_transform_authentication_error(self):
        message = "Error code: 401 Unauthorized"
        assert transform_authentication_error(message) == (
            "Authentication failed. Please check your API key or provider credentials."
        )
