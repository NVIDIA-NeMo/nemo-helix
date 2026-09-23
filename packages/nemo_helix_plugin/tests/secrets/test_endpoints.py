# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Secrets service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_origin

from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.secrets import endpoints
from nemo_helix_plugin.secrets.types import (
    HelixSecretAccessResponse,
    HelixSecretAdminRotationResponse,
    HelixSecretCreateRequest,
    HelixSecretResponse,
    HelixSecretUpdateRequest,
)


def test_create_secret() -> None:
    body = HelixSecretCreateRequest(name="hf-token", value="nvapi-xyz")
    prepared = endpoints.create_secret(workspace="default", body=body)

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/secrets/v2/workspaces/{workspace}/secrets"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content_type == "application/json"
    assert prepared.response_type is HelixSecretResponse


def test_create_secret_serializes_real_value() -> None:
    """Regression: SecretStr must serialize the plaintext, not the '****' mask."""
    body = HelixSecretCreateRequest(name="hf-token", value="nvapi-xyz")
    prepared = endpoints.create_secret(workspace="default", body=body)

    content = json.loads(prepared.content)
    assert content == {"name": "hf-token", "value": "nvapi-xyz"}


def test_create_secret_workspace_optional() -> None:
    body = HelixSecretCreateRequest(name="hf-token", value="v")
    prepared = endpoints.create_secret(body=body)

    assert prepared.path_params == {}


def test_list_secrets() -> None:
    prepared = endpoints.list_secrets(workspace="default")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/secrets/v2/workspaces/{workspace}/secrets"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content is None
    assert get_origin(prepared.response_type) is Paginated


def test_list_secrets_with_query_params() -> None:
    prepared = endpoints.list_secrets(workspace="default", query_params={"page": 2, "page_size": 5})

    assert prepared.query_params == {"page": 2, "page_size": 5}


def test_get_secret() -> None:
    prepared = endpoints.get_secret(workspace="default", name="hf-token")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "hf-token"}
    assert prepared.response_type is HelixSecretResponse


def test_update_secret() -> None:
    body = HelixSecretUpdateRequest(description="updated")
    prepared = endpoints.update_secret(workspace="default", name="hf-token", body=body)

    assert prepared.method == "PATCH"
    assert prepared.path_params == {"workspace": "default", "name": "hf-token"}
    assert prepared.response_type is HelixSecretResponse


def test_update_secret_excludes_unset_and_serializes_value() -> None:
    body = HelixSecretUpdateRequest(value="newvalue")
    prepared = endpoints.update_secret(workspace="default", name="hf-token", body=body)

    content = json.loads(prepared.content)
    assert content == {"value": "newvalue"}
    assert "description" not in content


def test_delete_secret_returns_none() -> None:
    prepared = endpoints.delete_secret(workspace="default", name="hf-token")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "name": "hf-token"}
    assert prepared.content is None
    assert prepared.response_type is None


def test_access_secret() -> None:
    prepared = endpoints.access_secret(workspace="default", name="hf-token")

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access"
    assert prepared.path_params == {"workspace": "default", "name": "hf-token"}
    assert prepared.response_type is HelixSecretAccessResponse


def test_rotate_encryption_keys() -> None:
    prepared = endpoints.rotate_encryption_keys()

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/secrets/v2/rotate-encryption-keys"
    assert prepared.path_params == {}
    assert prepared.content is None
    assert prepared.response_type is HelixSecretAdminRotationResponse
