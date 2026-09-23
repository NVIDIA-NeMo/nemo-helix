# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for register_query_param_schemas / clear_query_param_schemas.

These schemas are attached to FastAPI endpoints via ``openapi_extra`` and are
not reachable through Pydantic's response-model walk. The runtime
``custom_openapi`` hook has to call ``register_query_param_schemas`` explicitly
or the live ``/openapi.json`` will contain dangling ``$ref``s to the filter
classes.
"""

from typing import Optional

import nemo_helix_plugin.jobs.openapi_utils as job_openapi_utils
import pytest
from fastapi import FastAPI, Query, Request
from fastapi.testclient import TestClient
from nhx.common.api.utils import (
    clear_query_param_schemas,
    generate_openapi_extra_params,
    install_query_param_schema_openapi_hook,
    register_query_param_schemas,
)
from pydantic import BaseModel, create_model


class _DummyFilter(BaseModel):
    type: Optional[str] = None


class DummyDatetimeFilter(BaseModel):
    gte: Optional[str] = None
    lte: Optional[str] = None


class DummyStringFilter(BaseModel):
    eq: Optional[str] = None
    like: Optional[str] = None


class DummyJobsListFilter(BaseModel):
    created_at: Optional[DummyDatetimeFilter] = None
    name: Optional[DummyStringFilter | str] = None
    updated_at: Optional[DummyDatetimeFilter] = None


@pytest.fixture(autouse=True)
def _reset_registry():
    """The registry is module-level global state; reset around each test."""
    clear_query_param_schemas()
    yield
    clear_query_param_schemas()


def test_register_injects_filter_schema():
    """A filter referenced via ``generate_openapi_extra_params`` should land in
    ``components.schemas`` after ``register_query_param_schemas`` runs.
    """
    generate_openapi_extra_params(filter_schema=_DummyFilter)

    spec = {"components": {"schemas": {}}}
    spec = register_query_param_schemas(spec)

    assert "_DummyFilter" in spec["components"]["schemas"]
    assert spec["components"]["schemas"]["_DummyFilter"]["properties"]["type"]


def test_register_preserves_existing_schemas():
    generate_openapi_extra_params(filter_schema=_DummyFilter)

    spec = {"components": {"schemas": {"Existing": {"type": "object"}}}}
    spec = register_query_param_schemas(spec)

    assert "Existing" in spec["components"]["schemas"]
    assert "_DummyFilter" in spec["components"]["schemas"]


def test_register_promotes_nested_filter_defs():
    """Nested filter refs should resolve without depending on offline postprocessing."""
    generate_openapi_extra_params(filter_schema=DummyJobsListFilter)

    spec = {"components": {"schemas": {}}}
    spec = register_query_param_schemas(spec)
    schemas = spec["components"]["schemas"]

    assert "$defs" not in schemas["DummyJobsListFilter"]
    assert "DummyDatetimeFilter" in schemas
    assert "DummyStringFilter" in schemas
    assert schemas["DummyJobsListFilter"]["properties"]["created_at"]["anyOf"][0]["$ref"] == (
        "#/components/schemas/DummyDatetimeFilter"
    )
    assert schemas["DummyJobsListFilter"]["properties"]["name"]["anyOf"][0]["$ref"] == (
        "#/components/schemas/DummyStringFilter"
    )
    assert schemas["DummyJobsListFilter"]["properties"]["updated_at"]["anyOf"][0]["$ref"] == (
        "#/components/schemas/DummyDatetimeFilter"
    )


def test_clear_resets_registry_between_services():
    generate_openapi_extra_params(filter_schema=_DummyFilter)
    clear_query_param_schemas()

    spec = register_query_param_schemas({"components": {"schemas": {}}})
    assert "_DummyFilter" not in spec["components"]["schemas"]


def test_custom_openapi_hook_resolves_filter_ref():
    """End-to-end: a FastAPI app that wires ``register_query_param_schemas``
    into its ``custom_openapi`` hook emits a spec where the filter $ref
    resolves — which is exactly the regression the runtime was missing.
    """
    app = FastAPI()

    @app.get(
        "/items",
        openapi_extra=generate_openapi_extra_params(filter_schema=_DummyFilter),
    )
    async def list_items(request: Request, page: int = Query(default=1)):
        return {"data": []}

    install_query_param_schema_openapi_hook(app)

    spec = TestClient(app).get("/openapi.json").json()

    assert "_DummyFilter" in spec["components"]["schemas"]
    param = next(p for p in spec["paths"]["/items"]["get"]["parameters"] if p["name"] == "filter")
    assert param["schema"]["$ref"] == "#/components/schemas/_DummyFilter"


def test_custom_openapi_hook_retries_registration_after_component_conflict(monkeypatch):
    ExistingNested = create_model("ConflictNested", count=(int, ...), __module__="existing_mod")
    FilterNested = create_model("ConflictNested", value=(str | None, None), __module__="filter_mod")

    class ConflictFilter(BaseModel):
        created_at: FilterNested | None = None

    app = FastAPI()

    @app.get(
        "/items",
        response_model=ExistingNested,
        openapi_extra=generate_openapi_extra_params(filter_schema=ConflictFilter),
    )
    async def list_items() -> dict[str, int]:
        return {"count": 1}

    attempts = 0
    original_register = job_openapi_utils.register_query_param_schemas

    def counting_register(spec):
        nonlocal attempts
        attempts += 1
        return original_register(spec)

    monkeypatch.setattr(job_openapi_utils, "register_query_param_schemas", counting_register)
    install_query_param_schema_openapi_hook(app)

    with pytest.raises(ValueError, match="conflicts with existing component"):
        app.openapi()
    with pytest.raises(ValueError, match="conflicts with existing component"):
        app.openapi()

    assert attempts == 2
