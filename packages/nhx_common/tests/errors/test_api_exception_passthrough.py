# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the shared API exception passthrough contract."""

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nhx.common.entities.client import EntityConflictError, EntityNotFoundError, EntityValidationError
from nhx.common.errors.sdk_exception_handlers import HANDLED_API_EXCEPTIONS, register_sdk_exception_handlers
from starlette.exceptions import HTTPException as StarletteHTTPException


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (PermissionDeniedError(httpx.Response(403, json={"detail": "Access denied"})), 403, "Access denied"),
        (EntityNotFoundError("Workspace not found"), 404, "Workspace not found"),
        (EntityConflictError("Stale version"), 409, "Stale version"),
        (EntityValidationError("Invalid entity"), 422, "Invalid entity"),
        (HTTPException(400, "Invalid request"), 400, "Invalid request"),
        (StarletteHTTPException(405, "Method not allowed"), 405, "Method not allowed"),
        (RuntimeError("Private failure details"), 500, "Unexpected failure"),
    ],
)
def test_shared_passthrough_preserves_handled_errors(error, status_code, detail):
    app = FastAPI()
    register_sdk_exception_handlers(app)

    @app.get("/test")
    async def endpoint():
        try:
            raise error
        except HANDLED_API_EXCEPTIONS:
            raise
        except Exception:
            raise HTTPException(500, "Unexpected failure") from None

    response = TestClient(app).get("/test")

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
