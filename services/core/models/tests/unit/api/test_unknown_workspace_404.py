# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real routers and SDK handlers when the entity store cannot find a workspace."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.api import response_policy
from nemo_helix_plugin.client.errors import BadRequestError, InternalServerError, NotFoundError
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.entities.client import EntityNotFoundError
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.core.models.api.dependencies import (
    get_model_deployment_config_service,
    get_model_deployment_service,
    get_model_entity_service,
    get_model_provider_service,
    get_prompt_service,
)
from nhx.core.models.api.v2 import deployment_configs, deployments, models, prompts, providers

WS = "no-such-workspace"
DETAIL = f"Workspace '{WS}' not found"

# Each case covers a changed catch-all, including the deployment-config POST path.
ENDPOINTS = [
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "list_deployment_configs",
        "GET",
        "deployment-configs",
        None,
        "Failed to list deployment configs",
        id="list-deployment-configs",
    ),
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "create_deployment_config",
        "POST",
        "deployment-configs",
        {
            "name": "test-config",
            "engine": "nim",
            "model_spec": {"model_namespace": "nvidia", "model_name": "test-model"},
            "executor_config": {"gpu": 1},
        },
        "Failed to create deployment config",
        id="create-deployment-config",
    ),
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "get_deployment_config",
        "GET",
        "deployment-configs/test-config",
        None,
        "Failed to get deployment config",
        id="get-deployment-config",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "list_deployments",
        "GET",
        "deployments",
        None,
        "Failed to list deployments",
        id="list-deployments",
    ),
    pytest.param(
        models,
        get_model_entity_service,
        "list_model_entities",
        "GET",
        "models",
        None,
        "Failed to list model entities",
        id="list-models",
    ),
    pytest.param(
        prompts,
        get_prompt_service,
        "list_prompts",
        "GET",
        "prompts",
        None,
        "Internal server error",
        id="list-prompts",
    ),
    pytest.param(
        providers,
        get_model_provider_service,
        "list_model_providers",
        "GET",
        "providers",
        None,
        None,  # the providers list echoes the error text as its 500 detail
        id="list-providers",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "get_deployment",
        "GET",
        "deployments/x",
        None,
        "Failed to get deployment",
        id="get-deployment",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "get_deployment",
        "GET",
        "deployments/x/models",
        None,
        "Failed to get deployment models",
        id="get-deployment-models",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "list_deployment_versions",
        "GET",
        "deployments/x/versions",
        None,
        "Failed to list deployment versions",
        id="list-deployment-versions",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "create_deployment",
        "POST",
        "deployments",
        {"name": "test", "config": "cfg"},
        "Failed to create deployment",
        id="create-deployment",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "update_deployment_status",
        "POST",
        "deployments/x/status",
        {"status": "READY"},
        "Failed to update deployment status",
        id="update-deployment-status",
    ),
    pytest.param(
        deployments,
        get_model_deployment_service,
        "delete_deployment",
        "DELETE",
        "deployments/x",
        None,
        "Failed to delete deployment",
        id="delete-deployment",
    ),
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "list_deployment_config_versions",
        "GET",
        "deployment-configs/x/versions",
        None,
        "Failed to list deployment config versions",
        id="list-deployment-config-versions",
    ),
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "delete_deployment_config",
        "DELETE",
        "deployment-configs/x",
        None,
        "Failed to delete deployment config",
        id="delete-deployment-config",
    ),
    pytest.param(
        deployment_configs,
        get_model_deployment_config_service,
        "delete_deployment_config",
        "DELETE",
        "deployment-configs/x/versions/1",
        None,
        "Failed to delete deployment config version",
        id="delete-deployment-config-version",
    ),
]


def _upstream(error_type, status: int, detail: str):
    request = httpx.Request("GET", f"http://entities/apis/entities/v2/workspaces/{WS}/entities")
    return error_type(httpx.Response(status, json={"detail": detail}, request=request))


# error kind -> (raised error, expected status, expected detail or None for the route's generic 500 detail)
ERROR_KINDS = {
    "sdk-not-found": (lambda: _upstream(NotFoundError, 404, DETAIL), 404, DETAIL),
    "entity-not-found": (lambda: EntityNotFoundError(DETAIL), 404, DETAIL),
    "sdk-bad-request": (
        lambda: _upstream(BadRequestError, 400, "Invalid sort field -name"),
        400,
        "Invalid sort field -name",
    ),
    "sdk-server-error": (lambda: _upstream(InternalServerError, 500, "database down"), 500, None),
    "unexpected": (lambda: RuntimeError("boom"), 500, None),
}


@pytest.mark.parametrize("module,dependency,method,verb,path,payload,generic_detail", ENDPOINTS)
@pytest.mark.parametrize("error_kind", list(ERROR_KINDS))
def test_workspace_error_response(
    module, dependency, method, verb, path, payload, generic_detail, error_kind, monkeypatch
):
    """Upstream client errors keep their status; server failures keep the route's generic 500."""
    make_error, expected_status, expected_detail = ERROR_KINDS[error_kind]
    error = make_error()

    service = Mock(spec=[method])
    service_method = AsyncMock(side_effect=error)
    setattr(service, method, service_method)
    auth_client = Mock(spec=AuthClient, auth_enabled=False, principal=Principal(id="test-user"))
    auth_client.has_permissions = AsyncMock(return_value=True)
    monkeypatch.setattr(deployments, "deployments_enabled", lambda: True)
    monkeypatch.setattr(deployment_configs, "deployments_enabled", lambda: True)
    monkeypatch.setattr(response_policy, "current_auth_client", lambda request: auth_client)

    # No default argument: FastAPI would treat it as a request parameter.
    def override_service():
        return service

    app = FastAPI()
    register_sdk_exception_handlers(app)
    app.include_router(module.router)
    app.dependency_overrides[dependency] = override_service
    app.dependency_overrides[get_auth_client] = lambda: auth_client

    with TestClient(app) as client:
        response = client.request(verb, f"/v2/workspaces/{WS}/{path}", json=payload)

    # Prove request validation/dependency resolution reached the intended service call.
    service_method.assert_awaited_once()
    assert service_method.await_args is not None
    args, kwargs = service_method.await_args
    assert WS in args or kwargs.get("workspace") == WS
    assert response.status_code == expected_status, response.text
    assert response.json() == {"detail": expected_detail or generic_detail or str(error)}
