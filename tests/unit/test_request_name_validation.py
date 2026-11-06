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
from nemo_helix_plugin.client.errors import PermissionDeniedError, UnprocessableEntityError
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
PROVIDER_WORKSPACE_ROUTES = [
    ("list_providers", "GET", None, "list_model_providers"),
    (
        "create_provider",
        "POST",
        {"name": "provider-one", "host_url": "http://localhost:9000/v1"},
        "create_model_provider",
    ),
    ("upsert_provider", "PUT", {"host_url": "http://localhost:9000/v1"}, "upsert_model_provider"),
    ("update_provider_status", "PUT", {"status": "READY"}, "update_model_provider_status"),
]


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
        # Keep the actual query parser for collection routes as well as request validators.
        if dependency.name == "parsed_filter":
            continue
        mock = AsyncMock()
        mock.auth_enabled = False
        dependencies[dependency.call] = mock
        app.dependency_overrides[dependency.call] = override(mock)
    path = ("/v2/workspaces/{workspace}" if module_name == AGENTS else "") + route.path_format
    return TestClient(app), path, dependencies


def lookup_operation(module_name, endpoint_name, dependencies):
    """Select the actual service operation so assertions cannot pass on incidental calls."""
    module = import_module(module_name)
    dependency = module.get_prompt_service if module_name == MODELS + "prompts" else module.get_model_provider_service
    method = {
        "get_prompt": "get_prompt",
        "delete_prompt": "delete_prompt",
        "get_provider": "get_model_provider",
        "delete_provider": "delete_model_provider",
    }[endpoint_name]
    return getattr(dependencies[dependency], method)


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), ROUTES)
@pytest.mark.parametrize("workspace", ["ab\x1a", "ab\ncd", "ab\U0001f680", "ab space"])
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
@pytest.mark.parametrize("name", ["bad\x1a", "bad\U0001f680", "bad name", "bad:name"])
def test_invalid_model_name_is_field_specific_422(module_name, endpoint_name, method, body, name):
    """Reject malformed lookup names with the offending path field identified."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    response = client.request(method, path.format(workspace="default", name=quote(name, safe="")))
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "name"]]
    lookup_operation(module_name, endpoint_name, dependencies).assert_not_awaited()


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize(
    ("workspace", "name"),
    [
        ("default", "missing+name"),
        ("default", "missing@name"),
        ("Upper_case.123", "caf\u00e9"),
        ("caf\u00e9", "Upper_case.123"),
    ],
)
def test_valid_missing_model_resource_still_returns_404(module_name, endpoint_name, method, body, workspace, name):
    """Accepted lookup syntax reaches missing-resource handling without name normalization."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    operation = lookup_operation(module_name, endpoint_name, dependencies)
    operation.return_value = None
    response = client.request(method, path.format(workspace=quote(workspace), name=quote(name)))
    assert response.status_code == 404, response.text
    assert name in response.json()["detail"]
    operation.assert_awaited_once()
    request = operation.await_args.args[0]
    assert (request.workspace, request.name) == (workspace, name)


@pytest.mark.parametrize("workspace", ["bad\x1a", "bad\U0001f680", "bad name", ""])
def test_role_binding_workspace_body_is_field_specific_422(workspace):
    """Validate the workspace in the IAM request body, not inside the handler."""
    client, path, dependencies = route_client("nhx.core.auth.api.v2.iam.endpoints", "create_role_binding")
    response = client.post(path, json={"workspace": workspace, "principal": "user:example", "role": "Viewer"})
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["body", "workspace"]]
    for mock in dependencies.values():
        assert not mock.mock_calls


@pytest.mark.parametrize("workspace", [None, "Upper_case.123", "name+tag@example:1", "caf\u00e9"])
def test_role_binding_valid_workspace_matches_entity_pattern(workspace):
    """Keep platform-level bindings and accepted workspace identifiers valid."""
    binding = RoleBindingInput(workspace=workspace, principal="user:example", role="Viewer")
    assert binding.workspace == workspace
    if workspace is not None:
        assert EntityBase(workspace=workspace).workspace == workspace


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), CREATE_ROUTES)
@pytest.mark.parametrize("workspace", ["missing-workspace", "Name+tag@example:1", "caf\u00e9"])
def test_valid_missing_workspace_keeps_downstream_error(
    monkeypatch, module_name, endpoint_name, method, body, workspace
):
    """Valid missing workspaces must retain downstream errors rather than format errors."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    module = import_module(module_name)
    if endpoint_name == "create_job":
        operation = dependencies[module.dep_dispatcher].create_job
        dependencies[module.get_auth_client].principal = Principal(id="user:example")
        monkeypatch.setattr(module, "validate_job_spec", lambda *args: None)
    else:
        store = dependencies[module.get_entity_client]
        operation = store.get if endpoint_name in {"create_secret", "create_fileset"} else store.create
        if endpoint_name == "create_secret":
            dependencies[module.get_platform_config].ngc_api_key_secret = "default/ngc-api-key"
    operation.side_effect = EntityValidationError("Workspace does not exist", code="workspace_not_found")
    response = client.request(method, path.format(workspace=quote(workspace)), json=body)
    assert response.status_code == 422, response.text
    if endpoint_name == "create_job":
        assert response.json()["detail"].startswith("Invalid job request.")
    else:
        assert response.json()["detail"] == "Workspace does not exist"
    operation.assert_awaited_once()
    if endpoint_name == "create_job":
        assert operation.await_args.args[1] == workspace
    elif endpoint_name in {"create_secret", "create_fileset"}:
        assert operation.await_args.kwargs["workspace"] == workspace
    else:
        assert operation.await_args.args[0].workspace == workspace


@pytest.mark.parametrize(("module_name", "endpoint_name", "method", "body"), MODEL_ROUTES)
@pytest.mark.parametrize("field", ["workspace", "name"])
@pytest.mark.parametrize(("length", "status"), [(255, 404), (256, 422)])
def test_model_path_length_matches_internal_request(module_name, endpoint_name, method, body, field, length, status):
    """Enforce the internal models' existing length limit at the HTTP boundary."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    operation = lookup_operation(module_name, endpoint_name, dependencies)
    operation.return_value = None
    params = {"workspace": "default", "name": "valid-name", field: "a" * length}
    response = client.request(method, path.format(**params))
    assert response.status_code == status, response.text
    if status == 422:
        assert [error["loc"] for error in response.json()["detail"]] == [["path", field]]
        assert response.json()["detail"][0]["type"] == "string_too_long"
        operation.assert_not_awaited()
    else:
        operation.assert_awaited_once()
        request = operation.await_args.args[0]
        assert (request.workspace, request.name) == (params["workspace"], params["name"])


def test_internal_validation_errors_are_not_globally_reclassified():
    """An unrelated internal model failure must remain a server error."""
    client, path, dependencies = route_client(MODELS + "prompts", "get_prompt")
    operation = lookup_operation(MODELS + "prompts", "get_prompt", dependencies)
    with pytest.raises(ValidationError) as error:
        EntityBase(workspace="invalid space")
    operation.side_effect = error.value
    response = client.get(path.format(workspace="default", name="valid-name"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    operation.assert_awaited_once()


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
@pytest.mark.parametrize(
    ("name", "encoded"), [("resource+v1", False), ("resource@v1", False), ("resource+v1@tag", True)]
)
def test_create_get_delete_roundtrip_preserves_supported_punctuation(resource, name, encoded):
    """Names accepted by creation must survive lookup/deletion and URL decoding unchanged."""
    module = import_module(MODELS + resource)
    endpoint = "create_prompt" if resource == "prompts" else "create_provider"
    client, collection, dependencies = route_client(MODELS + resource, endpoint)
    store = memory_entity_client()
    if resource == "prompts":
        service = module.PromptService(store)
        dependency = module.get_prompt_service
        body = {"name": name, "messages": [{"role": "user", "content": "Hello"}]}
    else:
        dependencies[module.get_auth_client].principal = Principal(id="user:example")
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
def test_model_permission_errors_are_not_masked(module_name, endpoint_name, method, body):
    """Accepted lookup names preserve permission failures rather than returning server errors."""
    client, path, dependencies = route_client(module_name, endpoint_name)
    operation = lookup_operation(module_name, endpoint_name, dependencies)
    operation.side_effect = PermissionDeniedError(httpx.Response(403, json={"detail": "Access denied"}))
    response = client.request(method, path.format(workspace="default", name=quote("resource+v1@tag", safe="")))
    assert response.status_code == 403, response.text
    assert response.json() == {"detail": "Access denied"}
    operation.assert_awaited_once()


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


@pytest.mark.parametrize(("endpoint", "method", "body", "operation_name"), PROVIDER_WORKSPACE_ROUTES)
@pytest.mark.parametrize("workspace", ["bad\x1a", "a" * 256])
def test_remaining_provider_routes_validate_workspace(endpoint, method, body, operation_name, workspace):
    """Every provider route rejects malformed/overlong workspaces before its service runs."""
    module = import_module(MODELS + "providers")
    client, path, dependencies = route_client(MODELS + "providers", endpoint)
    operation = getattr(dependencies[module.get_model_provider_service], operation_name)
    response = client.request(method, path.format(workspace=quote(workspace, safe=""), name="provider-one"), json=body)
    assert response.status_code == 422, response.text
    assert [error["loc"] for error in response.json()["detail"]] == [["path", "workspace"]]
    operation.assert_not_awaited()


@pytest.mark.parametrize(("endpoint", "method", "body", "operation_name"), PROVIDER_WORKSPACE_ROUTES)
@pytest.mark.parametrize(
    ("workspace", "status"), [("missing-workspace", 422), ("Upper_case.caf\u00e9" + "a" * 240, 403)]
)
def test_remaining_provider_routes_preserve_service_errors(endpoint, method, body, operation_name, workspace, status):
    """Accepted workspace syntax and the 255-character boundary retain business/permission errors."""
    module = import_module(MODELS + "providers")
    client, path, dependencies = route_client(MODELS + "providers", endpoint)
    operation = getattr(dependencies[module.get_model_provider_service], operation_name)
    if endpoint in {"create_provider", "upsert_provider"}:
        dependencies[module.get_auth_client].principal = Principal(id="user:example")
    detail = "Workspace does not exist" if status == 422 else "Access denied"
    if status == 403:
        error = PermissionDeniedError(httpx.Response(status, json={"detail": detail}))
    elif endpoint == "list_providers":
        error = UnprocessableEntityError(httpx.Response(status, json={"detail": detail}))
    else:
        error = EntityValidationError(detail, code="workspace_not_found")
    operation.side_effect = error
    response = client.request(method, path.format(workspace=quote(workspace, safe=""), name="provider-one"), json=body)
    assert response.status_code == status, response.text
    assert response.json() == {"detail": detail}
    operation.assert_awaited_once()
    if endpoint == "list_providers":
        assert operation.await_args.kwargs["workspace"] == workspace
    elif endpoint == "create_provider":
        assert operation.await_args.args[1] == workspace
    else:
        assert operation.await_args.args[0] == workspace
