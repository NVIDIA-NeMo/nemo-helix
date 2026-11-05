# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List wrappers preserve known upstream request errors, not programming errors."""

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
    (models, get_model_entity_service, "list_model_entities", "models"),
    (adapters, get_adapter_entity_service, "list_adapters", "adapters"),
    (providers, get_model_provider_service, "list_model_providers", "providers"),
    (deployments, get_model_deployment_service, "list_deployments", "deployments"),
    (deployment_configs, get_model_deployment_config_service, "list_deployment_configs", "deployment-configs"),
]


@pytest.mark.parametrize("module,dependency,method,path", ENDPOINTS, ids=[e[3] for e in ENDPOINTS])
@pytest.mark.parametrize(
    "params,error_type,status,detail",
    [
        ({"page": "0"}, UnprocessableEntityError, 422, "Input should be greater than or equal to 1"),
        ({"page_size": str(2**63)}, UnprocessableEntityError, 422, "Input should be less than or equal to 1000"),
        ({"filter[created_at][$gt]": "notadate"}, BadRequestError, 400, "Invalid timestamp filter value 'notadate'"),
        ({"filter[nosuchfield]": "1"}, BadRequestError, 400, "Invalid filter field 'nosuchfield'"),
        ({"sort": "--name"}, BadRequestError, 400, "Invalid sort field -name"),
        ({}, NotFoundError, 404, "Workspace not found"),
        ({}, EntityNotFoundError, 404, "Workspace not found"),
        ({}, ValueError, 500, "internal value failure"),
        ({}, RuntimeError, 500, "unexpected failure"),
        ({}, None, 200, ""),
    ],
)
def test_list_query_error_response(module, dependency, method, path, params, error_type, status, detail, monkeypatch):
    """Preserve upstream status/detail without reclassifying programming failures."""
    service_method = AsyncMock()
    if error_type in (BadRequestError, UnprocessableEntityError, NotFoundError):
        service_method.side_effect = error_type(
            httpx.Response(status, json={"detail": detail}, request=httpx.Request("GET", "http://entities/list"))
        )
    elif error_type is not None:
        service_method.side_effect = error_type(detail)
    else:
        service_method.return_value = {
            "data": [],
            "pagination": {"page": 1, "page_size": 10, "total_results": 0, "total_pages": 0, "current_page_size": 0},
        }
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
        response = client.get(f"/v2/workspaces/default/{path}", params=params)

    # Enum sorts are rejected locally; the other routers forward sort validation.
    if params.get("sort") == "--name" and module in (models, providers):
        assert response.status_code == 422, response.text
        service_method.assert_not_awaited()
        return
    if "filter[nosuchfield]" in params:
        assert response.status_code == 400, response.text
        assert "nosuchfield" in response.json()["detail"]
        service_method.assert_not_awaited()
        return
    if "page_size" in params and module is adapters:
        assert response.status_code == 422, response.text
        service_method.assert_not_awaited()
        return
    service_method.assert_awaited_once()
    assert response.status_code == status, response.text
    if status in (400, 404, 422):
        assert response.json() == {"detail": detail}
    elif status == 200:
        assert response.json()["data"] == []
