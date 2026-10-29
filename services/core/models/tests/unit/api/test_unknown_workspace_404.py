# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real routers and SDK handlers when the entity store cannot find a workspace."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.api import response_policy
from nemo_helix_plugin.client.errors import NotFoundError
from nhx.common.auth import AuthClient, get_auth_client
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
        "boom",
        id="list-providers",
    ),
]


@pytest.mark.parametrize("module,dependency,method,verb,path,payload,generic_detail", ENDPOINTS)
@pytest.mark.parametrize("error_kind", ["sdk-not-found", "entity-not-found", "unexpected"])
def test_workspace_error_response(
    module, dependency, method, verb, path, payload, generic_detail, error_kind, monkeypatch
):
    if error_kind == "sdk-not-found":
        request = httpx.Request("GET", f"http://entities/apis/entities/v2/workspaces/{WS}/entities")
        error = NotFoundError(httpx.Response(404, json={"detail": DETAIL}, request=request))
    elif error_kind == "entity-not-found":
        error = EntityNotFoundError(DETAIL)
    else:
        error = RuntimeError("boom")

    service = Mock(spec=[method])
    service_method = AsyncMock(side_effect=error)
    setattr(service, method, service_method)
    auth_client = Mock(spec=AuthClient, auth_enabled=False)
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
    expected_status = 500 if error_kind == "unexpected" else 404
    expected_detail = generic_detail if error_kind == "unexpected" else DETAIL
    assert response.status_code == expected_status, response.text
    assert response.json() == {"detail": expected_detail}
