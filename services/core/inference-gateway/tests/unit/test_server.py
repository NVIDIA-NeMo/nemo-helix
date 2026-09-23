# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for server functionality."""

from typing import Annotated

import pytest
from aiohttp import ClientSession
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from nhx.core.inference_gateway.api.dependencies import global_http_client


@pytest.mark.asyncio
@pytest.mark.usefixtures("app")
async def test_lifespan_model_cache(model_cache):
    assert model_cache.get_from_provider("default", "ollama")


def test_lifespan_http_client(app: FastAPI, client: TestClient, mock_proxy_client):
    @app.get("/hello")
    def _(http_client: Annotated[ClientSession, Depends(global_http_client)]):
        assert http_client is mock_proxy_client
        return "world"

    client.get("/hello")


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}", "post"),
        ("/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}", "put"),
        ("/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}", "patch"),
        ("/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}", "post"),
        ("/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}", "put"),
        ("/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}", "patch"),
        ("/v2/workspaces/{workspace}/openai/-/{trailing_uri}", "post"),
        ("/v2/workspaces/{workspace}/openai/-/{trailing_uri}", "put"),
        ("/v2/workspaces/{workspace}/openai/-/{trailing_uri}", "patch"),
    ],
)
def test_proxy_endpoints_have_request_body_schema(app: FastAPI, path: str, method: str):
    """Test that proxy POST/PUT/PATCH endpoints have requestBody in OpenAPI spec.

    This ensures the SDK can generate a proper `body` parameter instead of
    requiring users to use `extra_body`.
    """
    openapi_schema = app.openapi()
    endpoint_spec = openapi_schema["paths"][path][method]

    assert "requestBody" in endpoint_spec, f"{method.upper()} {path} should have requestBody"

    request_body = endpoint_spec["requestBody"]
    assert "content" in request_body
    assert "application/json" in request_body["content"]

    schema = request_body["content"]["application/json"]["schema"]
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is True
