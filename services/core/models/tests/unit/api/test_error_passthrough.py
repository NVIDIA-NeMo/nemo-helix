# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every Models route must preserve errors handled by the shared API layer."""

import inspect
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import NemoHTTPError, raise_for_status
from nhx.common.auth import AuthClient, Principal, get_auth_client
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.core.models.api import dependencies
from nhx.core.models.api.service.adapter_entity_service import AdapterEntityService
from nhx.core.models.api.service.model_deployment_config_service import ModelDeploymentConfigService
from nhx.core.models.api.service.model_deployment_service import ModelDeploymentService
from nhx.core.models.api.service.model_entity_service import ModelEntityService
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.api.service.prompt_service import PromptService
from nhx.core.models.api.v2 import adapters, deployment_configs, deployments, models, prompts, providers

ROUTERS = (models, adapters, deployment_configs, deployments, providers, prompts)
ROUTES = [route for module in ROUTERS for route in module.router.routes if isinstance(route, APIRoute)]

# Valid bodies keep failures at the service call, not request validation.
BODIES = {
    "CreateModelEntityRequest": {"name": "test"},
    "UpdateModelEntityRequest": {"description": "updated"},
    "CreateModelAdapterRequest": {"name": "test", "fileset": "default/test", "finetuning_type": "lora"},
    "CreateAdapterRequest": {
        "name": "test",
        "fileset": "default/test",
        "finetuning_type": "lora",
        "model": "test",
    },
    "UpdateAdapterRequest": {"description": "updated"},
    "CreateModelProviderRequest": {"name": "test", "host_url": "https://example.com"},
    "UpsertModelProviderRequest": {"host_url": "https://example.com"},
    "UpdateModelProviderStatusRequest": {"status_message": "updated"},
    "CreatePromptRequest": {"name": "test"},
    "UpdatePromptRequest": {"description": "updated"},
    "CreateModelDeploymentRequest": {"name": "test", "config": "test"},
    "UpdateModelDeploymentRequest": {"config": "test"},
    "UpdateModelDeploymentStatusRequest": {"status": "PENDING"},
    "CreateModelDeploymentConfigRequest": {
        "name": "test",
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    },
    "UpdateModelDeploymentConfigRequest": {
        "engine": "nim",
        "model_spec": {},
        "executor_config": {"gpu": 1},
    },
}


def _service_override(service):
    def override():
        return service

    return override


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 422, 429, 503, 500, None])
def test_routes_preserve_sdk_errors(route, status_code, monkeypatch):
    """In particular, a deleted workspace's entity-client 403 must never become 500."""
    detail = "Upstream request denied or resource unavailable"
    if status_code is None:
        error = RuntimeError("Unexpected service failure")
    else:
        with pytest.raises(NemoHTTPError) as caught:
            raise_for_status(httpx.Response(status_code, json={"detail": detail}))
        error = caught.value

    app = FastAPI()
    register_sdk_exception_handlers(app)
    service_mocks = []
    for factory, service_type in (
        (dependencies.get_model_entity_service, ModelEntityService),
        (dependencies.get_adapter_entity_service, AdapterEntityService),
        (dependencies.get_model_deployment_service, ModelDeploymentService),
        (dependencies.get_model_deployment_config_service, ModelDeploymentConfigService),
        (dependencies.get_model_provider_service, ModelProviderService),
        (dependencies.get_prompt_service, PromptService),
    ):
        service = MagicMock(spec=service_type)
        for name, _ in inspect.getmembers(service_type, inspect.iscoroutinefunction):
            setattr(service, name, AsyncMock(side_effect=error))
        service.entity_client = MagicMock()
        service.entity_client.get = AsyncMock(side_effect=error)
        app.dependency_overrides[factory] = _service_override(service)
        service_mocks.append(service)

    auth_client = MagicMock(spec=AuthClient)
    auth_client.principal = Principal(id="test-user", groups=[])
    auth_client.has_permissions = AsyncMock(return_value=True)
    app.dependency_overrides[get_auth_client] = lambda: auth_client
    files = MagicMock()
    files.get_fileset = AsyncMock(return_value=MagicMock())
    app.dependency_overrides[dependencies.get_files_client] = lambda: files
    app.dependency_overrides[dependencies.get_secrets_client] = lambda: MagicMock()
    monkeypatch.setattr(deployments, "deployments_enabled", lambda: True)
    monkeypatch.setattr(deployment_configs, "deployments_enabled", lambda: True)
    for module in ROUTERS:
        app.include_router(module.router)

    path = route.path.format(
        **{name: "1" if name == "name" and "/versions/" in route.path else "test" for name in route.param_convertors}
    )
    body = BODIES[route.body_field.field_info.annotation.__name__] if route.body_field else None
    response = TestClient(app, raise_server_exceptions=False).request(next(iter(route.methods)), path, json=body)

    assert response.status_code == (status_code or 500), response.text
    if status_code is not None:
        expected_detail = "Internal server error" if status_code >= 500 else detail
        assert response.json() == {"detail": expected_detail}
    assert any(service.mock_calls for service in service_mocks), "Request must reach a service"
