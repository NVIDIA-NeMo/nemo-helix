# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve typed permission denials through the Models HTTP routers."""

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.core.models.api.dependencies import (
    get_adapter_entity_service,
    get_files_client,
    get_model_deployment_config_service,
    get_model_deployment_service,
    get_model_entity_service,
    get_model_provider_service,
    get_prompt_service,
    get_secrets_client,
)
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.service.model_deployment_service import ModelDeploymentService
from nhx.core.models.api.service.model_entity_service import ModelEntityService
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.api.service.prompt_service import PromptService
from nhx.core.models.api.v2 import adapters, deployment_configs, deployments, models, prompts, providers
from nhx.core.models.schemas import ModelEntity
from pydantic import JsonValue

WS = "deleted-workspace"
DETAIL = f"Not allowed to list entities in workspace '{WS}'"
CONFIG_BODY = {
    "engine": "nim",
    "model_spec": {"model_namespace": "nvidia", "model_name": "test-model"},
    "executor_config": {"gpu": 1},
}
ADAPTER_BODY = {"name": "test-adapter", "fileset": "test-fileset", "finetuning_type": "lora"}

# Select bound methods directly so method names and call counts stay reviewable.
MODEL_CASES = [
    pytest.param("GET", "models", None, lambda s: s.list_model_entities, id="list-models"),
    pytest.param("GET", "models/test-model", None, lambda s: s.get_model_entity, id="get-model"),
    pytest.param("POST", "models", {"name": "test-model"}, lambda s: s.create_model_entity, id="create-model"),
    pytest.param("PATCH", "models/test-model", {}, lambda s: s.update_model_entity, id="update-model"),
    pytest.param("DELETE", "models/test-model", None, lambda s: s.delete_model_entity, id="delete-model"),
]
MODEL_ADAPTER_CASES = [
    pytest.param(
        "POST", "models/test-model/adapters", ADAPTER_BODY, lambda s: s.create_adapter, id="create-model-adapter"
    ),
    pytest.param(
        "PATCH", "models/test-model/adapters/test-adapter", {}, lambda s: s.update_adapter, id="update-model-adapter"
    ),
    pytest.param(
        "DELETE", "models/test-model/adapters/test-adapter", None, lambda s: s.delete_adapter, id="delete-model-adapter"
    ),
]
ADAPTER_CASES = [
    pytest.param("GET", "adapters", None, lambda s: s.list_adapters, id="list-adapters"),
    pytest.param("GET", "adapters/test-adapter", None, lambda s: s.get_adapter, id="get-adapter"),
    pytest.param(
        "POST", "adapters", {**ADAPTER_BODY, "model": "test-model"}, lambda s: s.create_adapter, id="create-adapter"
    ),
    pytest.param("PATCH", "adapters/test-adapter", {}, lambda s: s.update_adapter_in_workspace, id="update-adapter"),
    pytest.param("DELETE", "adapters/test-adapter", None, lambda s: s.delete_adapter_in_workspace, id="delete-adapter"),
]
DEPLOYMENT_CASES = [
    pytest.param("GET", "deployments", None, lambda s: s.list_deployments, id="list-deployments"),
    pytest.param("GET", "deployments/test-deployment", None, lambda s: s.get_deployment, id="get-deployment"),
    pytest.param(
        "GET", "deployments/test-deployment/models", None, lambda s: s.get_deployment, id="get-deployment-models"
    ),
    pytest.param(
        "GET",
        "deployments/test-deployment/versions",
        None,
        lambda s: s.list_deployment_versions,
        id="list-deployment-versions",
    ),
    pytest.param(
        "GET", "deployments/test-deployment/versions/1", None, lambda s: s.get_deployment, id="get-deployment-version"
    ),
    pytest.param(
        "POST",
        "deployments",
        {"name": "test-deployment", "config": "test-config"},
        lambda s: s.create_deployment,
        id="create-deployment",
    ),
    pytest.param(
        "POST",
        "deployments/test-deployment",
        {"config": "test-config"},
        lambda s: s.update_deployment,
        id="update-deployment",
    ),
    pytest.param(
        "POST",
        "deployments/test-deployment/status",
        {"status": "READY"},
        lambda s: s.update_deployment_status,
        id="update-deployment-status",
    ),
    pytest.param("DELETE", "deployments/test-deployment", None, lambda s: s.delete_deployment, id="delete-deployment"),
    pytest.param(
        "DELETE",
        "deployments/test-deployment/versions/1",
        None,
        lambda s: s.delete_deployment,
        id="delete-deployment-version",
    ),
]
CONFIG_CASES = [
    pytest.param("GET", "deployment-configs", None, lambda s: s.list_deployment_configs, id="list-configs"),
    pytest.param("GET", "deployment-configs/test-config", None, lambda s: s.get_deployment_config, id="get-config"),
    pytest.param(
        "GET",
        "deployment-configs/test-config/versions",
        None,
        lambda s: s.list_deployment_config_versions,
        id="list-config-versions",
    ),
    pytest.param(
        "GET",
        "deployment-configs/test-config/versions/1",
        None,
        lambda s: s.get_deployment_config,
        id="get-config-version",
    ),
    pytest.param(
        "POST",
        "deployment-configs",
        {"name": "test-config", **CONFIG_BODY},
        lambda s: s.create_deployment_config,
        id="create-config",
    ),
    pytest.param(
        "POST", "deployment-configs/test-config", CONFIG_BODY, lambda s: s.update_deployment_config, id="update-config"
    ),
    pytest.param(
        "DELETE", "deployment-configs/test-config", None, lambda s: s.delete_deployment_config, id="delete-config"
    ),
    pytest.param(
        "DELETE",
        "deployment-configs/test-config/versions/1",
        None,
        lambda s: s.delete_deployment_config,
        id="delete-config-version",
    ),
]
PROVIDER_CASES = [
    pytest.param("GET", "providers", None, lambda s: s.list_model_providers, id="list-providers"),
    pytest.param("GET", "providers/test-provider", None, lambda s: s.get_model_provider, id="get-provider"),
    pytest.param(
        "POST",
        "providers",
        {"name": "test-provider", "host_url": "http://provider"},
        lambda s: s.create_model_provider,
        id="create-provider",
    ),
    pytest.param(
        "PUT",
        "providers/test-provider",
        {"host_url": "http://provider"},
        lambda s: s.upsert_model_provider,
        id="upsert-provider",
    ),
    pytest.param(
        "PUT",
        "providers/test-provider/status",
        {},
        lambda s: s.update_model_provider_status,
        id="update-provider-status",
    ),
    pytest.param("DELETE", "providers/test-provider", None, lambda s: s.delete_model_provider, id="delete-provider"),
]
PROMPT_CASES = [
    pytest.param("GET", "prompts", None, lambda s: s.list_prompts, id="list-prompts"),
    pytest.param("GET", "prompts/test-prompt", None, lambda s: s.get_prompt, id="get-prompt"),
    pytest.param("POST", "prompts", {"name": "test-prompt"}, lambda s: s.create_prompt, id="create-prompt"),
    pytest.param("PUT", "prompts/test-prompt", {}, lambda s: s.update_prompt, id="update-prompt"),
    pytest.param("DELETE", "prompts/test-prompt", None, lambda s: s.delete_prompt, id="delete-prompt"),
]


@pytest.fixture(params=["permission-denied", "unexpected"])
def error(request: pytest.FixtureRequest) -> Exception:
    if request.param == "permission-denied":
        return PermissionDeniedError(
            httpx.Response(403, json={"detail": DETAIL}, request=httpx.Request("GET", "http://entities"))
        )
    return RuntimeError("unexpected failure")


@pytest.fixture
def app() -> Iterator[FastAPI]:
    app = FastAPI()
    register_sdk_exception_handlers(app)
    for router in (
        models.router,
        adapters.router,
        deployments.router,
        deployment_configs.router,
        providers.router,
        prompts.router,
    ):
        app.include_router(router)

    auth = Mock(spec=AuthClient, auth_enabled=False, principal=Principal(id="former-member"))
    auth.has_permissions = AsyncMock(return_value=True)
    app.dependency_overrides[get_auth_client] = lambda: auth
    files = AsyncMock(spec=AsyncFilesClient)
    files.get_fileset.return_value = Mock(data=Mock(return_value=Mock(spec=FilesetOutput)))
    app.dependency_overrides[get_files_client] = lambda: files
    app.dependency_overrides[get_secrets_client] = lambda: AsyncMock(spec=AsyncSecretsClient)
    # Exercise service errors independently of the host's runtime configuration.
    with (
        patch.object(deployments, "deployments_enabled", return_value=True),
        patch.object(deployment_configs, "deployments_enabled", return_value=True),
    ):
        yield app


def assert_error_response(
    app: FastAPI,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    service_method: AsyncMock,
    error: Exception,
) -> None:
    service_method.side_effect = error
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.request(method, f"/v2/workspaces/{WS}/{path}", json=payload)
    service_method.assert_awaited_once()
    if isinstance(error, PermissionDeniedError):
        assert response.status_code == 403, response.text
        assert response.json() == {"detail": DETAIL}
    else:
        assert response.status_code == 500, response.text


@pytest.mark.parametrize("method,path,payload,select_method", MODEL_CASES)
def test_model_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=ModelEntityService)
    service.entity_client = Mock(get=AsyncMock())
    service.get_model_entity.return_value = ModelEntity(
        id="model-id",
        name="test-model",
        workspace=WS,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    app.dependency_overrides[get_model_entity_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)


@pytest.mark.parametrize("method,path,payload,select_method", MODEL_ADAPTER_CASES + ADAPTER_CASES)
def test_adapter_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=AdapterEntityService)
    app.dependency_overrides[get_adapter_entity_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)


@pytest.mark.parametrize("method,path,payload,select_method", DEPLOYMENT_CASES)
def test_deployment_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=ModelDeploymentService)
    app.dependency_overrides[get_model_deployment_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)


@pytest.mark.parametrize("method,path,payload,select_method", CONFIG_CASES)
def test_config_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=ModelDeploymentConfigService)
    app.dependency_overrides[get_model_deployment_config_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)


@pytest.mark.parametrize("method,path,payload,select_method", PROVIDER_CASES)
def test_provider_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=ModelProviderService)
    app.dependency_overrides[get_model_provider_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)


@pytest.mark.parametrize("method,path,payload,select_method", PROMPT_CASES)
def test_prompt_errors(
    app: FastAPI,
    error: Exception,
    method: str,
    path: str,
    payload: dict[str, JsonValue] | None,
    select_method: Callable[[AsyncMock], AsyncMock],
) -> None:
    service = AsyncMock(spec=PromptService)
    app.dependency_overrides[get_prompt_service] = lambda: service
    assert_error_response(app, method, path, payload, select_method(service), error)
