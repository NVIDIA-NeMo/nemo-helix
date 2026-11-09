# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared page dependency, exercised through a real FastAPI route."""

import pytest
from fastapi import Depends, FastAPI, Query
from fastapi.testclient import TestClient
from nemo_helix_plugin.api.pagination import MAX_OFFSET, PageParams, make_page_dep

app = FastAPI()


@app.get("/items")
def list_items(
    page_params: PageParams = Depends(make_page_dep(default_size=20, max_size=100)),
    sort: str = Query(default="name"),
) -> dict[str, int | str]:
    return {
        "page": page_params.page,
        "page_size": page_params.page_size,
        "offset": page_params.offset,
        "sort": sort,
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_defaults_and_other_query_params(client: TestClient):
    assert client.get("/items").json() == {"page": 1, "page_size": 20, "offset": 0, "sort": "name"}
    assert client.get("/items", params={"page": 3, "page_size": 7, "sort": "-name"}).json() == {
        "page": 3,
        "page_size": 7,
        "offset": 14,
        "sort": "-name",
    }


@pytest.mark.parametrize(
    "params,field,error_type",
    [
        ({"page": 0}, "page", "greater_than_equal"),
        ({"page_size": 0}, "page_size", "greater_than_equal"),
        ({"page_size": 101}, "page_size", "less_than_equal"),
        ({"page": "1.5"}, "page", "int_parsing"),
    ],
)
def test_field_bounds(client: TestClient, params: dict, field: str, error_type: str):
    response = client.get("/items", params=params)
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert (error["loc"], error["type"]) == (["query", field], error_type)


def test_offset_bound_depends_on_page_size(client: TestClient):
    # The last page whose offset fits at size 7 (2**63 - 1 is divisible by 7).
    last_page = MAX_OFFSET // 7 + 1
    ok = client.get("/items", params={"page": last_page, "page_size": 7})
    assert ok.status_code == 200
    assert ok.json()["offset"] == MAX_OFFSET

    response = client.get("/items", params={"page": last_page + 1, "page_size": 7})
    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert (error["loc"], error["type"]) == (["query", "page"], "value_error")
    assert "offset" in error["msg"]


def test_openapi_parameters():
    parameters = {p["name"]: p["schema"] for p in app.openapi()["paths"]["/items"]["get"]["parameters"]}
    assert parameters["page"] == {
        "type": "integer",
        "minimum": 1,
        "default": 1,
        "title": "Page",
        "description": "Page number.",
    }
    assert parameters["page_size"] == {
        "type": "integer",
        "maximum": 100,
        "minimum": 1,
        "default": 20,
        "title": "Page Size",
        "description": "Page size.",
    }
