# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test wrapper error mapping separately from local query validation."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.api import response_policy
from nemo_helix_plugin.client.errors import BadRequestError, NotFoundError, UnprocessableEntityError
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.entities.client import EntityNotFoundError
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers
from nhx.core.models.api.dependencies import (
    get_adapter_entity_service,
    get_model_deployment_config_service,
    get_model_deployment_service,
    get_model_entity_service,
    get_model_provider_service,
)
from nhx.core.models.api.v2 import adapters, deployment_configs, deployments, models, providers

ENDPOINTS = [
    (models, get_model_entity_service, "list_model_entities", "models", "Failed to list model entities"),
    (adapters, get_adapter_entity_service, "list_adapters", "adapters", "Failed to list adapter entities"),
    (providers, get_model_provider_service, "list_model_providers", "providers", "internal value failure"),
    (deployments, get_model_deployment_service, "list_deployments", "deployments", "Failed to list deployments"),
    (
        deployment_configs,
        get_model_deployment_config_service,
        "list_deployment_configs",
        "deployment-configs",
        "Failed to list deployment configs",
    ),
]


@contextmanager
def _list_client(endpoint, monkeypatch):
    """Use real routing/SDK handlers; mock only the service boundary under test."""
    module, dependency, method, _, _ = endpoint
    service_method = AsyncMock()
    service = Mock(spec=[method])
    setattr(service, method, service_method)
    auth_client = Mock(spec=AuthClient, auth_enabled=False)
    monkeypatch.setattr(response_policy, "current_auth_client", lambda request: auth_client)
    app = FastAPI()
    register_sdk_exception_handlers(app)
    app.include_router(module.router)
    app.dependency_overrides[dependency] = lambda: service
    app.dependency_overrides[get_auth_client] = lambda: auth_client
    with TestClient(app) as client:
        yield client, service_method


FORWARDED_CASES = [
    pytest.param(endpoint, params, error_type, status, detail, id=f"{endpoint[3]}-{label}")
    for endpoint in ENDPOINTS
    for label, params, error_type, status, detail in [
        ("page", {"page": "0"}, UnprocessableEntityError, 422, "Input should be greater than or equal to 1"),
        (
            "page-size",
            {"page_size": str(2**63)},
            UnprocessableEntityError,
            422,
            "Input should be less than or equal to 1000",
        ),
        (
            "timestamp",
            {"filter[created_at][$gt]": "notadate"},
            BadRequestError,
            400,
            "Invalid timestamp filter value 'notadate'",
        ),
        ("sort", {"sort": "--name"}, BadRequestError, 400, "Invalid sort field -name"),
    ]
    if not (label == "page-size" and endpoint[0] is adapters)
    and not (label == "sort" and endpoint[0] in (models, providers))
]


@pytest.mark.parametrize("endpoint,params,error_type,status,detail", FORWARDED_CASES)
def test_upstream_list_query_errors(endpoint, params, error_type, status, detail, monkeypatch):
    """Forward query values and preserve the service's known 400/422 response."""
    with _list_client(endpoint, monkeypatch) as (client, service_method):
        service_method.side_effect = error_type(
            httpx.Response(status, json={"detail": detail}, request=httpx.Request("GET", "http://entities/list"))
        )
        response = client.get(f"/v2/workspaces/default/{endpoint[3]}", params=params)
    service_method.assert_awaited_once()
    kwargs = service_method.await_args.kwargs
    if "page" in params:
        assert kwargs["page"] == int(params["page"])
    elif "page_size" in params:
        assert kwargs["page_size"] == int(params["page_size"])
    elif "sort" in params:
        assert kwargs["sort"] == params["sort"]
    else:
        operation = kwargs["parsed_filter"].operation if "parsed_filter" in kwargs else kwargs["filter_operation"]
        assert operation.to_dict() == {"created_at": {"$gt": "notadate"}}
    assert response.status_code == status, response.text
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=[e[3] for e in ENDPOINTS])
def test_internal_value_error_is_not_a_client_error(endpoint, monkeypatch):
    """Reject broad ValueError-to-400 mapping without weakening existing 500 details."""
    with _list_client(endpoint, monkeypatch) as (client, service_method):
        service_method.side_effect = ValueError("internal value failure")
        response = client.get(f"/v2/workspaces/default/{endpoint[3]}")
    service_method.assert_awaited_once()
    assert response.status_code == 500, response.text
    assert response.json() == {"detail": endpoint[4]}


@pytest.mark.parametrize(
    "endpoint,params,field",
    [
        (ENDPOINTS[1], {"page_size": str(2**63)}, "page_size"),
        (ENDPOINTS[0], {"sort": "--name"}, "sort"),
        (ENDPOINTS[2], {"sort": "--name"}, "sort"),
    ],
)
def test_local_list_query_validation(endpoint, params, field, monkeypatch):
    """Locally rejected requests must not reach a service or use injected errors."""
    with _list_client(endpoint, monkeypatch) as (client, service_method):
        response = client.get(f"/v2/workspaces/default/{endpoint[3]}", params=params)
    service_method.assert_not_awaited()
    assert response.status_code == 422, response.text
    assert field in str(response.json()["detail"])


@pytest.mark.parametrize(
    "error_type,status", [(NotFoundError, 404), (EntityNotFoundError, 404), (RuntimeError, 500), (None, 200)]
)
def test_adapter_list_controls(error_type, status, monkeypatch):
    """Cover the adapter route absent from the existing workspace-error suite."""
    endpoint = ENDPOINTS[1]
    detail = "Workspace not found"
    with _list_client(endpoint, monkeypatch) as (client, service_method):
        if error_type is NotFoundError:
            service_method.side_effect = NotFoundError(
                httpx.Response(404, json={"detail": detail}, request=httpx.Request("GET", "http://entities/list"))
            )
        elif error_type is not None:
            service_method.side_effect = error_type(detail)
        else:
            service_method.return_value = {
                "data": [],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "total_results": 0,
                    "total_pages": 0,
                    "current_page_size": 0,
                },
            }
        response = client.get("/v2/workspaces/default/adapters")
    service_method.assert_awaited_once()
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["data"] == []
    else:
        assert response.json() == {"detail": endpoint[4] if status == 500 else detail}
