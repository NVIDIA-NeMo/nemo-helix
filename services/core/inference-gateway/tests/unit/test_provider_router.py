# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for provider router endpoints."""

import json

from fastapi.testclient import TestClient
from multidict import CIMultiDict
from nhx.core.inference_gateway.api.model_cache import ModelCache


def test_provider_proxy_endpoint(client: TestClient, mock_proxy_client, mock_proxy_response):
    """Test the provider proxy endpoint."""
    content = b"{'hello': 'world'}"
    mock_proxy_response._body = [content]

    # workspace_id (default) is used as the workspace, provider_name is ollama
    response = client.get("/v2/workspaces/default/provider/ollama/-/v1/models")
    mock_proxy_client.request.assert_called_once()
    assert response.status_code == 200
    assert response.content == content


def test_provider_proxy_with_headers(client: TestClient, mock_proxy_client):
    """Test the provider proxy endpoint with custom headers."""
    headers = {
        "Authorization": "Bearer test-token",  # should be filtered out
        "Content-Type": "application/json",
    }

    response = client.get("/v2/workspaces/default/provider/ollama/-/v1/models", headers=headers)
    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()
    headers = CIMultiDict(mock_proxy_client.request.call_args.kwargs["headers"])
    assert "authorization" not in headers
    assert "content-type" in headers


def test_provider_proxy_with_auth(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that the cached raw secret is sent as a Bearer token by default."""
    model_info = model_cache.get_from_provider("default", "tot")
    assert model_info
    model_info.secret_value = "my-secret-token-123"

    headers = {
        "Authorization": "Bearer test-token",  # should be filtered out
        "Content-Type": "application/json",
    }

    response = client.post("/v2/workspaces/default/provider/tot/-/v1/models", headers=headers)
    assert response.status_code == 200

    call_args = mock_proxy_client.request.call_args
    headers = CIMultiDict(call_args.kwargs["headers"])
    assert headers["authorization"] == "Bearer my-secret-token-123"


def test_provider_proxy_with_unresolved_provider_secret(client: TestClient, model_cache: ModelCache):
    """Test that unresolved configured provider secrets return a dependency error."""
    model_info = model_cache.get_from_provider("default", "tot")
    assert model_info
    model_info.secret_value = None
    model_info.model_provider.api_key_secret_name = "my-secret-id"

    headers = {
        "Authorization": "Bearer test-token",  # should be filtered out
        "Content-Type": "application/json",
    }

    response = client.post("/v2/workspaces/default/provider/tot/-/v1/models", headers=headers)
    assert response.status_code == 424
    assert "secret not found or unreachable" in response.json()["detail"]


def test_provider_proxy_with_default_extra_body(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that provider's default_extra_body provides defaults for the upstream request."""

    # Set up provider with default_extra_body (can be overridden by request)
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.default_extra_body = {"stream": True, "temperature": 0.7}

    # Make request with a body
    request_body = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

    response = client.post(
        "/v2/workspaces/default/provider/ollama/-/v1/chat/completions",
        json=request_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that the proxied request body includes both original and default_extra_body
    call_args = mock_proxy_client.request.call_args
    proxied_body = json.loads(call_args.kwargs["data"])

    # Request body fields should be present
    assert proxied_body["model"] == "gpt-4"
    assert proxied_body["messages"] == [{"role": "user", "content": "test"}]

    # Default extra body fields should be merged
    assert proxied_body["stream"] is True
    assert proxied_body["temperature"] == 0.7


def test_provider_proxy_with_required_extra_body(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that provider's required_extra_body cannot be overridden by request."""

    # Set up provider with required_extra_body (cannot be overridden)
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.required_extra_body = {"stream": True, "enforce_distillable_text": True}

    # Make request with body that tries to override required fields
    request_body = {"model": "gpt-4", "stream": False, "enforce_distillable_text": False}

    response = client.post(
        "/v2/workspaces/default/provider/ollama/-/v1/chat/completions",
        json=request_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that required values cannot be overridden
    call_args = mock_proxy_client.request.call_args
    proxied_body = json.loads(call_args.kwargs["data"])

    # Required fields should take precedence over request
    assert proxied_body["stream"] is True  # Required value, not request's false
    assert proxied_body["enforce_distillable_text"] is True  # Required value, not request's false
    assert proxied_body["model"] == "gpt-4"  # Request value (not in required)


def test_provider_proxy_with_default_extra_headers(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that provider's default_extra_headers are passed to the upstream request."""
    from multidict import CIMultiDict

    # Set up provider with default_extra_headers (can be overridden)
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.default_extra_headers = {"X-Provider": "ollama", "X-Local": "true"}

    response = client.get(
        "/v2/workspaces/default/provider/ollama/-/v1/models",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that the proxied request headers include default_extra_headers
    call_args = mock_proxy_client.request.call_args
    headers = CIMultiDict(call_args.kwargs["headers"])

    assert headers["x-provider"] == "ollama"
    assert headers["x-local"] == "true"


def test_provider_proxy_with_required_extra_headers(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that provider's required_extra_headers cannot be overridden by request."""
    from multidict import CIMultiDict

    # Set up provider with required_extra_headers (cannot be overridden)
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.required_extra_headers = {"X-Required": "required-value", "X-Override": "provider-value"}

    # Make request with header that tries to override X-Override
    response = client.get(
        "/v2/workspaces/default/provider/ollama/-/v1/models",
        headers={"X-Override": "request-value"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that required header value takes precedence
    call_args = mock_proxy_client.request.call_args
    headers = CIMultiDict(call_args.kwargs["headers"])

    assert headers["x-required"] == "required-value"  # From required
    assert headers["x-override"] == "provider-value"  # Required value, not request's


def test_provider_proxy_request_overrides_default_extra_body(
    client: TestClient, mock_proxy_client, model_cache: ModelCache
):
    """Test that request body values take precedence over provider's default_extra_body."""

    # Set up provider with default_extra_body that has overlapping field
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.default_extra_body = {"temperature": 0.5, "stream": True}

    # Make request with body that overrides temperature
    request_body = {"model": "gpt-4", "temperature": 0.9}

    response = client.post(
        "/v2/workspaces/default/provider/ollama/-/v1/chat/completions",
        json=request_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that request value takes precedence over default
    call_args = mock_proxy_client.request.call_args
    proxied_body = json.loads(call_args.kwargs["data"])

    assert proxied_body["temperature"] == 0.9  # Request value, not default's 0.5
    assert proxied_body["stream"] is True  # Default's value (no conflict)
    assert proxied_body["model"] == "gpt-4"  # Request value


def test_provider_proxy_request_overrides_default_extra_headers(
    client: TestClient, mock_proxy_client, model_cache: ModelCache
):
    """Test that request headers take precedence over provider's default_extra_headers."""
    from multidict import CIMultiDict

    # Set up provider with default_extra_headers that has overlapping header
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.default_extra_headers = {"X-Provider": "provider-value", "X-Organization": "test-org"}

    # Make request with header that overrides X-Provider
    response = client.get(
        "/v2/workspaces/default/provider/ollama/-/v1/models",
        headers={"X-Provider": "request-value"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    # Check that request value takes precedence over default
    call_args = mock_proxy_client.request.call_args
    headers = CIMultiDict(call_args.kwargs["headers"])

    assert headers["x-provider"] == "request-value"  # Request value, not default's
    assert headers["x-organization"] == "test-org"  # Default's value (no conflict)


def test_provider_proxy_with_all_extra_fields(client: TestClient, mock_proxy_client, model_cache: ModelCache):
    """Test that default and required extra fields work together correctly."""
    from multidict import CIMultiDict

    # Set up provider with all extra fields
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info
    model_info.model_provider.default_extra_body = {"max_tokens": 100, "temperature": 0.5}
    model_info.model_provider.default_extra_headers = {"X-Default": "default"}
    model_info.model_provider.required_extra_body = {"stream": True}
    model_info.model_provider.required_extra_headers = {"X-Required": "required"}

    # Make request with body that overrides temperature
    request_body = {"model": "gpt-4", "temperature": 0.9}

    response = client.post(
        "/v2/workspaces/default/provider/ollama/-/v1/chat/completions",
        json=request_body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    mock_proxy_client.request.assert_called_once()

    call_args = mock_proxy_client.request.call_args
    proxied_body = json.loads(call_args.kwargs["data"])
    headers = CIMultiDict(call_args.kwargs["headers"])

    # Merge order: default < request < required
    assert proxied_body["model"] == "gpt-4"  # From request
    assert proxied_body["temperature"] == 0.9  # From request (overrides default)
    assert proxied_body["max_tokens"] == 100  # From default (not in request)
    assert proxied_body["stream"] is True  # From required (enforced)

    assert headers["x-default"] == "default"  # From default
    assert headers["x-required"] == "required"  # From required


def test_provider_proxy_invalid_workspace_returns_422(client: TestClient):
    """Test that invalid workspace (e.g. single char) returns 422 Unprocessable Entity."""
    response = client.get("/v2/workspaces/a/provider/ollama/-/v1/models")
    assert response.status_code == 422
    assert "workspace" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_provider_proxy_invalid_name_returns_422(client: TestClient):
    """Test that invalid provider name (e.g. single char) returns 422 Unprocessable Entity."""
    response = client.get("/v2/workspaces/default/provider/a/-/v1/models")
    assert response.status_code == 422
    assert "name" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_provider_ready_returns_200_when_status_ready(client: TestClient):
    """Provider in cache with status=READY returns 200 and ready true."""
    response = client.get("/v2/workspaces/default/provider/ollama/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["provider"] == "default/ollama"
    assert "host_url" in data


def test_provider_ready_returns_404_when_status_created(client: TestClient, model_cache: ModelCache):
    """Provider in cache with status=CREATED returns 404."""
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info is not None
    model_info.model_provider.status = "CREATED"
    response = client.get("/v2/workspaces/default/provider/ollama/ready")
    assert response.status_code == 404
    assert "not ready" in response.json()["detail"].lower()


def test_provider_ready_returns_404_when_status_none(client: TestClient, model_cache: ModelCache):
    """Provider in cache with status=None returns 404."""
    model_info = model_cache.get_from_provider("default", "ollama")
    assert model_info is not None
    model_info.model_provider.status = None
    response = client.get("/v2/workspaces/default/provider/ollama/ready")
    assert response.status_code == 404
    assert "not ready" in response.json()["detail"].lower()


def test_provider_ready_returns_404_when_not_in_cache(client: TestClient):
    """Provider not in cache returns 404."""
    response = client.get("/v2/workspaces/default/provider/nonexistent/ready")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
