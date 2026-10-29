# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared typed-client and entity-client exception handlers."""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import UnprocessableEntityError
from nemo_helix_plugin.entities.types import WORKSPACE_NOT_FOUND_CODE
from nhx.common.entities.client import EntityValidationError
from nhx.common.errors.sdk_exception_handlers import register_sdk_exception_handlers


def _client(error: Exception) -> TestClient:
    app = FastAPI()
    register_sdk_exception_handlers(app)

    @app.get("/")
    def fail() -> None:
        raise error

    return TestClient(app, raise_server_exceptions=False)


def _unprocessable(body: dict[str, str]) -> UnprocessableEntityError:
    request = httpx.Request("POST", "http://test")
    return UnprocessableEntityError(httpx.Response(422, json=body, request=request))


def test_entity_store_error_keeps_its_code():
    response = _client(EntityValidationError("Workspace 'ws' does not exist.", code=WORKSPACE_NOT_FOUND_CODE)).get("/")

    assert response.status_code == 422
    assert response.json() == {"detail": "Workspace 'ws' does not exist.", "code": WORKSPACE_NOT_FOUND_CODE}


def test_entity_store_error_without_code_has_detail_only():
    response = _client(EntityValidationError("Project 'p' does not exist.")).get("/")

    assert response.status_code == 422
    assert response.json() == {"detail": "Project 'p' does not exist."}


def test_typed_client_error_relays_downstream_code():
    error = _unprocessable({"detail": "Workspace 'ws' does not exist.", "code": WORKSPACE_NOT_FOUND_CODE})

    response = _client(error).get("/")

    assert response.status_code == 422
    assert response.json() == {"detail": "Workspace 'ws' does not exist.", "code": WORKSPACE_NOT_FOUND_CODE}


def test_typed_client_error_without_code_has_detail_only():
    response = _client(_unprocessable({"detail": "Invalid."})).get("/")

    assert response.json() == {"detail": "Invalid."}
