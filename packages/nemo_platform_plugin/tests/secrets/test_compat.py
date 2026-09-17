# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform import AsyncNeMoPlatform, ConflictError, NeMoPlatform
from nemo_platform.resources.secrets import SecretsResource
from nemo_platform.resources.secrets.admin import AdminResource
from nemo_platform.resources.secrets.secrets import SecretsResource as ModuleSecretsResource
from nemo_platform.types.secrets import PlatformSecretAccessResponse, PlatformSecretResponse, SecretCreateParams
from nemo_platform.types.secrets.platform_secret_response import PlatformSecretResponse as ModulePlatformSecretResponse
from nemo_platform.types.secrets.secret_create_params import SecretCreateParams as ModuleSecretCreateParams

BASE = "http://test:8000"


def test_legacy_access_response_exposes_data_alias() -> None:
    response = PlatformSecretAccessResponse(name="api-key", workspace="default", value="secret")

    assert response.data == "secret"


def test_legacy_sdk_alias_imports_resolve() -> None:
    assert SecretsResource.__name__ == "SecretsResource"
    assert AdminResource.__name__ == "SecretsAdminResource"
    assert ModuleSecretsResource is SecretsResource
    assert ModulePlatformSecretResponse is PlatformSecretResponse
    assert ModuleSecretCreateParams is SecretCreateParams


def test_legacy_secrets_resource_preserves_create_retrieve_list_and_delete() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.Response(
            201,
            request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
            json={"name": "hf-token", "workspace": "default", "description": None},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/hf-token"),
            json={"name": "hf-token", "workspace": "default", "description": None},
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
            json={
                "data": [{"name": "hf-token", "workspace": "default", "description": None}],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 1,
                    "total_pages": 1,
                    "total_results": 1,
                },
            },
        ),
        httpx.Response(
            204,
            request=httpx.Request("DELETE", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/hf-token"),
        ),
    ]

    sdk = NeMoPlatform(base_url=BASE, workspace="default", http_client=mock_http)

    created = sdk.secrets.create(name="hf-token", value="nvapi-real")
    retrieved = sdk.secrets.retrieve("hf-token")
    page = sdk.secrets.list()
    deleted = sdk.secrets.delete("hf-token")

    assert isinstance(created, PlatformSecretResponse)
    assert retrieved.name == "hf-token"
    assert page.data[0].name == "hf-token"
    assert [secret.name for secret in page.data] == ["hf-token"]
    assert page.pagination is not None
    assert page.pagination.total_results == 1
    assert deleted is None

    create_call = mock_http.request.call_args_list[0]
    assert b'"value":"nvapi-real"' in create_call.kwargs["content"]
    assert b"description" not in create_call.kwargs["content"]


def test_legacy_secrets_resource_raises_sdk_conflict_error() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        409,
        request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
        json={"detail": "Secret already exists"},
    )
    sdk = NeMoPlatform(base_url=BASE, workspace="default", http_client=mock_http)

    with pytest.raises(ConflictError) as exc_info:
        sdk.secrets.create(name="hf-token", value="nvapi-real")

    assert exc_info.value.status_code == 409
    assert exc_info.value.message == "Secret already exists"


def test_legacy_secrets_resource_accepts_data_alias_for_create() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        201,
        request=httpx.Request("POST", f"{BASE}/apis/secrets/v2/workspaces/default/secrets"),
        json={"name": "api-key", "workspace": "default", "description": None},
    )
    sdk = NeMoPlatform(base_url=BASE, workspace="default", http_client=mock_http)

    sdk.secrets.create(name="api-key", data="legacy-secret")

    create_call = mock_http.request.call_args
    assert b'"value":"legacy-secret"' in create_call.kwargs["content"]


def test_legacy_secrets_resource_accepts_data_alias_for_update() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("PATCH", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/api-key"),
        json={"name": "api-key", "workspace": "default", "description": None},
    )
    sdk = NeMoPlatform(base_url=BASE, workspace="default", http_client=mock_http)

    sdk.secrets.update("api-key", data="legacy-secret")

    update_call = mock_http.request.call_args
    assert b'"value":"legacy-secret"' in update_call.kwargs["content"]


@pytest.mark.asyncio
async def test_async_legacy_secrets_resource_preserves_retrieve() -> None:
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/secrets/v2/workspaces/default/secrets/hf-token"),
        json={"name": "hf-token", "workspace": "default", "description": None},
    )
    async_sdk = AsyncNeMoPlatform(base_url=BASE, workspace="default", http_client=mock_http)

    retrieved = await async_sdk.secrets.retrieve("hf-token")

    assert retrieved.name == "hf-token"
