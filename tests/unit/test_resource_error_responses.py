# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep error response documentation aligned with the affected resource routes."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nhx.common.api.common import ErrorResponse

MODEL_MODULE = "nhx.core.models.api.v2."
AUTH_MODULE = "nhx.core.auth.oidc_broker.routes"
RESOURCE_ROUTES = [
    ("nhx.core.entities.api.v2.projects.endpoints", "get_project"),
    ("nhx.core.files.api.v2.filesets.endpoints", "retrieve_fileset"),
    ("nhx.core.inference_gateway.api.v2.openai", "openai_get_model"),
    (MODEL_MODULE + "adapters", "get_adapter"),
    (MODEL_MODULE + "deployment_configs", "get_latest_deployment_config"),
    (MODEL_MODULE + "deployments", "get_latest_deployment"),
    (MODEL_MODULE + "models", "get_model"),
    (MODEL_MODULE + "prompts", "get_prompt"),
    (MODEL_MODULE + "providers", "get_provider"),
    ("nhx.core.secrets.api.v2.secrets.endpoints", "get_secret"),
]


def route_app(module_name: str, endpoint_name: str) -> tuple[FastAPI, APIRoute]:
    module = import_module(module_name)
    route = next(
        route
        for route in module.router.routes
        if isinstance(route, APIRoute) and route.endpoint is getattr(module, endpoint_name)
    )
    app = FastAPI()
    app.include_router(module.router)
    return app, route


@pytest.mark.parametrize(
    ("module_name", "endpoint_name", "status", "success"),
    [(module, endpoint, 404, "200") for module, endpoint in RESOURCE_ROUTES]
    + [
        (AUTH_MODULE, "web_session", 401, "200"),
        (AUTH_MODULE, "start_web_session_login", 503, "302"),
    ],
)
def test_documented_error_schema(module_name: str, endpoint_name: str, status: int, success: str):
    app, route = route_app(module_name, endpoint_name)
    schema = app.openapi()
    responses = schema["paths"][route.path_format]["get"]["responses"]
    assert success in responses
    if "{workspace}" in route.path or endpoint_name == "start_web_session_login":
        assert "422" in responses
    error = responses[str(status)]
    assert error["description"]
    # In particular, login errors are JSON even though success is a redirect.
    assert error["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/ErrorResponse"}
    detail = schema["components"]["schemas"]["ErrorResponse"]
    assert detail["type"] == "object"
    assert detail["required"] == ["detail"]
    assert detail["properties"]["detail"]["type"] == "string"
    if endpoint_name == "start_web_session_login":
        assert "Location" in responses["302"]["headers"]


@pytest.mark.parametrize(
    ("module_name", "endpoint_name", "dependency_name", "method_name"),
    [
        (MODEL_MODULE + "adapters", "get_adapter", "get_adapter_entity_service", "get_adapter"),
        (
            MODEL_MODULE + "deployment_configs",
            "get_latest_deployment_config",
            "get_model_deployment_config_service",
            "get_deployment_config",
        ),
        (MODEL_MODULE + "deployments", "get_latest_deployment", "get_model_deployment_service", "get_deployment"),
        (MODEL_MODULE + "models", "get_model", "get_model_entity_service", "get_model_entity"),
        (MODEL_MODULE + "prompts", "get_prompt", "get_prompt_service", "get_prompt"),
        (MODEL_MODULE + "providers", "get_provider", "get_model_provider_service", "get_model_provider"),
    ],
)
def test_missing_model_resource_matches_documented_response(
    module_name: str, endpoint_name: str, dependency_name: str, method_name: str
):
    module = import_module(module_name)
    app, route = route_app(module_name, endpoint_name)
    service = AsyncMock()
    getattr(service, method_name).return_value = None
    app.dependency_overrides[getattr(module, dependency_name)] = lambda: service
    response = TestClient(app).get(route.path_format.format(workspace="default", name="missing-resource"))
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    error = ErrorResponse.model_validate(response.json())
    assert "missing-resource" in error.detail
    getattr(service, method_name).assert_awaited_once()


def test_error_metadata_is_not_a_router_wide_default():
    for module_name, endpoint_name in RESOURCE_ROUTES:
        module = import_module(module_name)
        _, target = route_app(module_name, endpoint_name)
        # Collection reads must not claim that an absent named resource causes 404.
        collection_path = target.path.split("/{name", 1)[0]
        for route in module.router.routes:
            if isinstance(route, APIRoute) and route.path == collection_path and "GET" in route.methods:
                assert 404 not in route.responses


@pytest.mark.parametrize(
    ("endpoint_name", "status", "detail"),
    [
        ("web_session", 401, "Missing session"),
        ("start_web_session_login", 503, "OIDC server-side login is not available"),
    ],
)
def test_auth_error_matches_documented_response(monkeypatch, endpoint_name: str, status: int, detail: str):
    module = import_module(AUTH_MODULE)
    app, route = route_app(AUTH_MODULE, endpoint_name)
    # Replace storage dependencies, not the endpoint's error handling.
    if endpoint_name == "web_session":
        for dependency in route.dependant.dependencies:
            assert dependency.call is not None
            app.dependency_overrides[dependency.call] = lambda: AsyncMock()
    else:
        monkeypatch.setattr(
            module, "get_auth_config", lambda: SimpleNamespace(oidc=SimpleNamespace(server_sessions=None))
        )
    response = TestClient(app).get(route.path, follow_redirects=False)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert ErrorResponse.model_validate(response.json()).detail == detail
