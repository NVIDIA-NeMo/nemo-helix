# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate names at HTTP boundaries, before constructing internal models."""

from importlib import import_module
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.entities.base import EntityBase, EntityValidationError
from nemo_helix_plugin.iam.types import RoleBindingInput
from nhx.common.auth.models import Principal
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from pydantic import ValidationError

MODELS = "nhx.core.models.api.v2."
AGENTS = "nemo_agents_plugin.api.v2.environments"
# Module, endpoint, HTTP method, valid body. Agents mount their workspace prefix externally.
ROUTES = [
    ("nhx.core.secrets.api.v2.secrets.endpoints", "create_secret", "POST", {"name": "secret-one", "value": "x"}),
    ("nhx.core.files.api.v2.filesets.endpoints", "create_fileset", "POST", {"name": "fileset-one"}),
    (
        "nhx.core.jobs.api.v2.jobs.endpoints",
        "create_job",
        "POST",
        {
            "name": "job-one",
            "source": "test",
            "spec": {},
            "platform_spec": {
                "steps": [
                    {
                        "name": "step-one",
                        "executor": {
                            "provider": "cpu",
                            "container": {"image": "busybox", "command": ["echo", "hello"]},
                        },
                    }
                ]
            },
        },
    ),
    (AGENTS, "create_compute_spec", "POST", {"name": "compute-one"}),
    (AGENTS, "create_environment_spec", "POST", {"name": "environment-one"}),
    (AGENTS, "create_environment", "POST", {"name": "environment-one"}),
    (MODELS + "prompts", "get_prompt", "GET", None),
    (MODELS + "prompts", "delete_prompt", "DELETE", None),
    (MODELS + "providers", "get_provider", "GET", None),
]


def override(value):
    return lambda: value


def route_client(module_name, endpoint_name):
    module = import_module(module_name)
    route = next(
        route
        for route in module.router.routes
        if isinstance(route, APIRoute)
        and getattr(route.endpoint, "__wrapped__", route.endpoint) is getattr(module, endpoint_name)
    )
    app = FastAPI()
    register_sdk_exception_handlers(app)
    app.include_router(module.router, prefix="/v2/workspaces/{workspace}" if module_name == AGENTS else "")
    dependencies = {}
    for dependency in route.dependant.dependencies:
        # Only replace infrastructure, never the route or its request validators.
        mock = AsyncMock()
        mock.auth_enabled = False
        dependencies[dependency.call] = mock
        app.dependency_overrides[dependency.call] = override(mock)
    path = ("/v2/workspaces/{workspace}" if module_name == AGENTS else "") + route.path_format
    return TestClient(app), path, dependencies


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES)
@pytest.mark.parametrize("workspace", ["ab\x1a", "ab\x00", "ab\ncd", "ab\U0001f680", "ab space"])
def test_invalid_workspace_is_field_specific_422(module_name, endpoint_name, method, body, workspace):
    client, path, dependencies = route_client(module_name, endpoint_name)
    response = client.request(method, path.format(workspace=quote(workspace, safe=""), name="missing-name"), json=body)
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "workspace"]]
    assert response.json()["detail"][0]["type"] == "string_pattern_mismatch"
    for mock in dependencies.values():
        assert not mock.mock_calls


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES[-3:])
@pytest.mark.parametrize("name", ["bad\x1a", "bad\nname", "bad\U0001f680", "bad name", "bad@name"])
def test_invalid_model_name_is_field_specific_422(module_name, endpoint_name, method, body, name):
    client, path, _ = route_client(module_name, endpoint_name)
    response = client.request(method, path.format(workspace="default", name=quote(name, safe="")))
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "name"]]


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES[-3:])
@pytest.mark.parametrize("workspace", ["default", "missing-workspace", "Upper_case.123", "caf\u00e9"])
@pytest.mark.parametrize("name", ["missing-name", "Upper_case.123", "caf\u00e9"])
def test_valid_missing_model_resource_still_returns_404(module_name, endpoint_name, method, body, workspace, name):
    client, path, dependencies = route_client(module_name, endpoint_name)
    for service in dependencies.values():
        service.get_prompt.return_value = None
        service.delete_prompt.return_value = False
        service.get_model_provider.return_value = None
    response = client.request(method, path.format(workspace=quote(workspace), name=quote(name)))
    assert response.status_code == 404, response.text
    assert name in response.json()["detail"]


@pytest.mark.parametrize("workspace", ["bad\x1a", "bad\nname", "bad\U0001f680", "bad name", ""])
def test_role_binding_workspace_body_is_field_specific_422(workspace):
    client, path, dependencies = route_client("nhx.core.auth.api.v2.iam.endpoints", "create_role_binding")
    response = client.post(path, json={"workspace": workspace, "principal": "user:example", "role": "Viewer"})
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["body", "workspace"]]
    for mock in dependencies.values():
        assert not mock.mock_calls


@pytest.mark.parametrize(
    "workspace", [None, "default", "missing-workspace", "Upper_case.123", "name+tag@example:1", "caf\u00e9"]
)
def test_role_binding_valid_workspace_matches_entity_pattern(workspace):
    binding = RoleBindingInput(workspace=workspace, principal="user:example", role="Viewer")
    assert binding.workspace == workspace
    if workspace is not None:
        assert EntityBase(workspace=workspace).workspace == workspace


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES)
@pytest.mark.parametrize("workspace", ["missing-workspace", "Upper_case.123", "caf\u00e9"])
def test_valid_workspace_constraints_match_internal_models(module_name, endpoint_name, method, body, workspace):
    # Validate exactly the route's FastAPI parameter field without executing storage operations.
    module = import_module(module_name)
    route = APIRoute(
        "/v2/workspaces/{workspace}", endpoint=getattr(module, endpoint_name), methods=[method], response_model=None
    )
    field = next(field for field in route.dependant.path_params if field.name == "workspace")
    value, errors = field.validate(workspace, {}, loc=("path", "workspace"))
    assert not errors
    assert value == workspace
    assert EntityBase(workspace=workspace).workspace == workspace


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES[:6])
@pytest.mark.parametrize("workspace", ["missing-workspace", "Upper_case.123", "caf\u00e9"])
def test_valid_missing_workspace_keeps_downstream_error(
    monkeypatch, module_name, endpoint_name, method, body, workspace
):
    client, path, dependencies = route_client(module_name, endpoint_name)
    error = EntityValidationError("Workspace does not exist", code="workspace_not_found")
    for mock in dependencies.values():
        mock.get.side_effect = error
        mock.create.side_effect = error
        mock.create_job.side_effect = error
        mock.principal = Principal(id="user:example")
        mock.ngc_api_key_secret = "default/ngc-api-key"
    if endpoint_name == "create_job":
        module = import_module(module_name)
        # Resource validation is independent of the host's configured execution backends.
        monkeypatch.setattr(module, "validate_job_spec", lambda *args: None)
    response = client.request(method, path.format(workspace=quote(workspace)), json=body)
    assert response.status_code == 422, response.text
    if endpoint_name == "create_job":
        assert response.json()["detail"].startswith("Invalid job request.")
    else:
        assert response.json()["detail"] == "Workspace does not exist"
    assert any(mock.mock_calls for mock in dependencies.values())


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES[-3:])
@pytest.mark.parametrize("field", ["workspace", "name"])
@pytest.mark.parametrize(("length", "status"), [(255, 404), (256, 422)])
def test_model_path_length_matches_internal_request(module_name, endpoint_name, method, body, field, length, status):
    client, path, dependencies = route_client(module_name, endpoint_name)
    for service in dependencies.values():
        service.get_prompt.return_value = None
        service.delete_prompt.return_value = False
        service.get_model_provider.return_value = None
    params = {"workspace": "default", "name": "valid-name", field: "a" * length}
    response = client.request(method, path.format(**params))
    assert response.status_code == status, response.text
    if status == 422:
        assert [error["loc"] for error in response.json()["detail"]] == [["path", field]]
        assert response.json()["detail"][0]["type"] == "string_too_long"


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES[:6])
def test_entity_workspace_preserves_additional_identifier_characters(module_name, endpoint_name, method, body):
    module = import_module(module_name)
    route = APIRoute(
        "/v2/workspaces/{workspace}", endpoint=getattr(module, endpoint_name), methods=[method], response_model=None
    )
    field = next(field for field in route.dependant.path_params if field.name == "workspace")
    workspace = "name+tag@example:1"
    value, errors = field.validate(workspace, {}, loc=("path", "workspace"))
    assert not errors
    assert value == EntityBase(workspace=workspace).workspace


def test_internal_validation_errors_are_not_globally_reclassified():
    client, path, dependencies = route_client(MODELS + "prompts", "get_prompt")
    service = next(iter(dependencies.values()))
    try:
        EntityBase(workspace="invalid space")
    except ValidationError as exc:
        service.get_prompt.side_effect = exc
    response = client.get(path.format(workspace="default", name="valid-name"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
