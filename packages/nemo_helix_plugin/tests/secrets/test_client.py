# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SecretsClient / AsyncSecretsClient via mocked httpx transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.secrets.client import AsyncSecretsClient, SecretsClient
from nemo_helix_plugin.secrets.types import (
    PlatformSecretCreateRequest,
    PlatformSecretUpdateRequest,
)

BASE = "http://test:8000"


def test_create_secret_sends_plaintext_value() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        201,
        request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
        json={"name": "hf-token", "workspace": "default", "description": None},
    )

    client = SecretsClient(base_url=BASE, workspace="default", http_client=mock_http)
    resp = client.create_secret(body=PlatformSecretCreateRequest(name="hf-token", value="nvapi-xyz"))

    assert resp.data().name == "hf-token"
    _, kwargs = mock_http.request.call_args
    assert b'"value":"nvapi-xyz"' in kwargs["content"]


def test_access_secret_returns_value() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/hf-token/access"),
        json={"name": "hf-token", "workspace": "default", "value": "nvapi-xyz"},
    )

    client = SecretsClient(base_url=BASE, workspace="default", http_client=mock_http)
    resp = client.access_secret(name="hf-token")

    assert resp.data().value == "nvapi-xyz"


def test_get_secret_not_found_raises() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        404,
        request=httpx.Request("GET", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/missing"),
        json={"detail": "Secret default/missing not found"},
    )

    client = SecretsClient(base_url=BASE, workspace="default", http_client=mock_http)
    with pytest.raises(NotFoundError) as exc:
        client.get_secret(name="missing")
    assert exc.value.status_code == 404


def test_update_secret_sends_plaintext_value() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("PATCH", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/hf-token"),
        json={"name": "hf-token", "workspace": "default", "description": "d"},
    )

    client = SecretsClient(base_url=BASE, workspace="default", http_client=mock_http)
    client.update_secret(name="hf-token", body=PlatformSecretUpdateRequest(value="newvalue"))

    _, kwargs = mock_http.request.call_args
    assert b'"value":"newvalue"' in kwargs["content"]


def test_rotate_encryption_keys() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        202,
        request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/rotate-encryption-keys"),
        json={"rotated_secrets": 0, "success": True},
    )

    client = SecretsClient(base_url=BASE, http_client=mock_http)
    resp = client.rotate_encryption_keys()

    assert resp.data().success is True


@pytest.mark.asyncio
async def test_async_create_secret() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = httpx.Response(
        201,
        request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
        json={"name": "hf-token", "workspace": "default", "description": None},
    )

    client = AsyncSecretsClient(base_url=BASE, workspace="default", http_client=mock_http)
    resp = await client.create_secret(body=PlatformSecretCreateRequest(name="hf-token", value="nvapi-xyz"))

    assert resp.data().name == "hf-token"
