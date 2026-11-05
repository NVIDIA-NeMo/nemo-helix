# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate names at HTTP boundaries, before constructing internal models."""

from datetime import UTC, datetime
from importlib import import_module
from unittest.mock import AsyncMock
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.entities.base import EntityBase, EntityNotFoundError, EntityValidationError
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
    (MODELS + "providers", "delete_provider", "DELETE", None),
]
CREATE_ROUTES = ROUTES[:6]
MODEL_ROUTES = ROUTES[6:]


def override(value):
    """Build a dependency override without exposing mock values as query parameters."""
    return lambda: value


def route_client(module_name, endpoint_name):
    """Mount production routes while replacing only their infrastructure dependencies."""
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
    """Reject malformed workspace paths before any resource operation runs."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    response = client.request(method, path.format(workspace=quote(workspace, safe=""), name="missing-name"), json=body)
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "workspace"]]
    assert response.json()["detail"][0]["type"] == "string_pattern_mismatch"
    for mock in dependencies.values():
        assert not mock.mock_calls


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize("name", ["bad\x1a", "bad\nname", "bad\U0001f680", "bad name", "bad:name"])
def test_invalid_model_name_is_field_specific_422(module_name, endpoint_name, method, body, name):
    """Reject malformed lookup names with the offending path field identified."""
    client, path, _ = route_client(module_name, endpoint_name)
    response = client.request(method, path.format(workspace="default", name=quote(name, safe="")))
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "name"]]


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize("workspace", ["default", "missing-workspace", "Upper_case.123", "caf\u00e9"])
@pytest.mark.parametrize("name", ["missing-name", "Upper_case.123", "caf\u00e9", "missing+name", "missing@name"])
def test_valid_missing_model_resource_still_returns_404(module_name, endpoint_name, method, body, workspace, name):
    """Preserve missing-resource semantics for every accepted lookup-name shape."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    for service in dependencies.values():
        service.get_prompt.return_value = None
        service.delete_prompt.return_value = False
        service.get_model_provider.return_value = None
        service.delete_model_provider.return_value = False
    response = client.request(method, path.format(workspace=quote(workspace), name=quote(name)))
    assert response.status_code == 404, response.text
    assert name in response.json()["detail"]


@pytest.mark.parametrize("workspace", ["bad\x1a", "bad\nname", "bad\U0001f680", "bad name", ""])
def test_role_binding_workspace_body_is_field_specific_422(workspace):
    """Validate the workspace in the IAM request body, not inside the handler."""
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
    """Keep platform-level bindings and accepted workspace identifiers valid."""
    binding = RoleBindingInput(workspace=workspace, principal="user:example", role="Viewer")
    assert binding.workspace == workspace
    if workspace is not None:
        assert EntityBase(workspace=workspace).workspace == workspace


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES)
@pytest.mark.parametrize("workspace", ["missing-workspace", "Upper_case.123", "caf\u00e9"])
def test_valid_workspace_constraints_match_internal_models(module_name, endpoint_name, method, body, workspace):
    """Keep request validation compatible with existing entity workspace identifiers."""
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


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), CREATE_ROUTES)
@pytest.mark.parametrize("workspace", ["missing-workspace", "Upper_case.123", "caf\u00e9"])
def test_valid_missing_workspace_keeps_downstream_error(
    monkeypatch, module_name, endpoint_name, method, body, workspace
):
    """Valid missing workspaces must retain downstream errors rather than format errors."""
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


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize("field", ["workspace", "name"])
@pytest.mark.parametrize(("length", "status"), [(255, 404), (256, 422)])
def test_model_path_length_matches_internal_request(module_name, endpoint_name, method, body, field, length, status):
    """Enforce the internal models' existing length limit at the HTTP boundary."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    for service in dependencies.values():
        service.get_prompt.return_value = None
        service.delete_prompt.return_value = False
        service.get_model_provider.return_value = None
        service.delete_model_provider.return_value = False
    params = {"workspace": "default", "name": "valid-name", field: "a" * length}
    response = client.request(method, path.format(**params))
    assert response.status_code == status, response.text
    if status == 422:
        assert [error["loc"] for error in response.json()["detail"]] == [["path", field]]
        assert response.json()["detail"][0]["type"] == "string_too_long"


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), CREATE_ROUTES)
def test_entity_workspace_preserves_additional_identifier_characters(module_name, endpoint_name, method, body):
    """Do not replace the entity workspace pattern with a narrower resource-name rule."""
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
    """An unrelated internal model failure must remain a server error."""
    client, path, dependencies = route_client(MODELS + "prompts", "get_prompt")
    service = next(iter(dependencies.values()))
    try:
        EntityBase(workspace="invalid space")
    except ValidationError as exc:
        service.get_prompt.side_effect = exc
    response = client.get(path.format(workspace="default", name="valid-name"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def memory_entity_client():
    """Back real resource services with an in-memory store instead of HTTP or a database."""
    entities = {}
    store = AsyncMock()

    async def get(entity_type, name, *, workspace):
        """Look up a stored entity with the normal missing-entity exception."""
        try:
            return entities[workspace, name]
        except KeyError as exc:
            raise EntityNotFoundError("Resource not found") from exc

    async def create(entity):
        """Populate store-owned metadata and persist the actual entity model."""
        entity._id = "test-id"
        entity._created_at = datetime.now(UTC)
        entity._updated_at = entity._created_at
        entities[entity.workspace, entity.name] = entity
        return entity

    async def delete(entity_type, name, *, workspace, **kwargs):
        """Remove a stored entity, preserving missing-resource behavior."""
        try:
            del entities[workspace, name]
        except KeyError as exc:
            raise EntityNotFoundError("Resource not found") from exc

    store.get.side_effect = get
    store.create.side_effect = create
    store.delete.side_effect = delete
    return store


@pytest.mark.parametrize("resource", ["prompts", "providers"])
@pytest.mark.parametrize("name", ["resource+v1", "resource@v1", "resource+v1@tag"])
@pytest.mark.parametrize("encoded", [False, True])
def test_create_get_delete_roundtrip_preserves_supported_punctuation(resource, name, encoded):
    """Names accepted by creation must survive lookup/deletion and URL decoding unchanged."""
    module = import_module(MODELS + resource)
    endpoint = "create_prompt" if resource == "prompts" else "create_provider"
    client, collection, dependencies = route_client(MODELS + resource, endpoint)
    for mock in dependencies.values():
        mock.principal = Principal(id="user:example")
    store = memory_entity_client()
    if resource == "prompts":
        service = module.PromptService(store)
        dependency = module.get_prompt_service
        body = {"name": name, "messages": [{"role": "user", "content": "Hello"}]}
    else:
        service = module.ModelProviderService(store)
        dependency = module.get_model_provider_service
        body = {"name": name, "host_url": "http://localhost:9000/v1"}
    client.app.dependency_overrides[dependency] = override(service)
    collection = collection.format(workspace="default")
    created = client.post(collection, json=body)
    assert created.status_code == 201, created.text
    assert created.json()["name"] == name
    path = collection + "/" + (quote(name, safe="") if encoded else name)
    retrieved = client.get(path)
    assert retrieved.status_code == 200, retrieved.text
    assert retrieved.json()["name"] == name
    assert client.delete(path).status_code == 204
    assert client.get(path).status_code == 404
    assert client.delete(path).status_code == 404


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize("name", ["resource+v1", "resource@v1"])
def test_model_permission_errors_are_not_masked(module_name, endpoint_name, method, body, name):
    """Accepted names must still propagate service permission failures as 403s."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    error = PermissionDeniedError(httpx.Response(403, json={"detail": "Access denied"}))
    for service in dependencies.values():
        service.get_prompt.side_effect = error
        service.delete_prompt.side_effect = error
        service.get_model_provider.side_effect = error
        service.delete_model_provider.side_effect = error
    response = client.request(method, path.format(workspace="default", name=quote(name, safe="")))
    assert response.status_code == 403, response.text
    assert response.json() == {"detail": "Access denied"}


@pytest.mark.parametrize(
    ("module_name", "request_name"),
    [
        (MODELS + "prompts", "GetPromptRequest"),
        (MODELS + "prompts", "DeletePromptRequest"),
        (MODELS + "providers", "GetModelProviderRequest"),
        (MODELS + "providers", "DeleteModelProviderRequest"),
    ],
)
def test_lookup_request_names_remain_nonempty(module_name, request_name):
    """Adding supported punctuation must not admit empty internal lookup names."""
    model = getattr(import_module(module_name), request_name)
    with pytest.raises(ValidationError) as error:
        model(workspace="default", name="")
    assert error.value.errors()[0]["loc"] == ("name",)
