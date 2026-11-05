# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate list query failures against the real SQL-backed HTTP routes."""

import json

import pytest
from httpx import AsyncClient

PATHS = [
    "/apis/entities/v2/workspaces",
    "/apis/entities/v2/workspaces/default/projects",
    "/apis/entities/v2/workspaces/default/entities/model",
]


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "field", ["nosuchfield", "metadata", "registry", "__class__", "to_pydantic", "data.", "data..key"]
)
async def test_unknown_or_noncolumn_filter(client: AsyncClient, ctx, path: str, field: str):
    """Unknown fields and ORM attributes must fail before SQL execution."""
    response = await client.get(path, params={f"filter[{field}]": "1"})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == f"Invalid filter field '{field}'"


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "params,status",
    [
        ({"page": "0"}, 422),
        ({"page_size": str(2**63)}, 422),
        ({"sort": "--name"}, None),
        ({"filter[created_at][$gt]": "notadate"}, 400),
        ({"page": "1", "page_size": "10", "sort": "name", "filter[name]": "default"}, 200),
        ({"filter[created_at][$gt]": "2020-01-01T00:00:00Z"}, 200),
    ],
)
async def test_list_query_validation(client: AsyncClient, ctx, path: str, params: dict, status: int | None):
    """Keep malformed queries as client errors and supported queries successful."""
    response = await client.get(path, params=params)
    if status is None:
        status = 400 if path.endswith("/entities/model") else 422
    assert response.status_code == status, response.text
    if status == 400:
        assert "detail" in response.json()


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "operator,value",
    [
        ("$eq", "notadate"),
        ("$in", ["2020-01-01T00:00:00Z", "notadate"]),
        ("$nin", [None, "notadate"]),
    ],
)
async def test_invalid_timestamp_equality_and_membership(client: AsyncClient, ctx, path: str, operator, value):
    """Validate every timestamp operand, including mixed membership lists."""
    response = await client.get(path, params={"filter": json.dumps({"created_at": {operator: value}})})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Invalid timestamp filter value 'notadate'"


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("operator", ["$eq", "$in", "$nin"])
@pytest.mark.parametrize("operand_kind", ["timestamp", "timestamp-and-null", "other-timestamp", "null"])
async def test_valid_timestamp_equality_and_membership(client: AsyncClient, ctx, path: str, operator, operand_kind):
    """Coercion preserves timestamp matching and SQL's existing null semantics."""
    if path.endswith("/workspaces"):
        seed = await client.get(f"{path}/default")
    else:
        payload = {"name": "test-item"}
        if path.endswith("/entities/model"):
            payload["data"] = {}
        seed = await client.post(path, json=payload)
    assert seed.status_code in (200, 201), seed.text
    timestamp = seed.json()["created_at"]
    values = {
        "timestamp": [timestamp],
        "timestamp-and-null": [timestamp, None],
        "other-timestamp": ["1900-01-01T00:00:00Z"],
        "null": [None],
    }[operand_kind]
    value = values[0] if operator == "$eq" else values
    response = await client.get(path, params={"filter": json.dumps({"created_at": {operator: value}})})
    assert response.status_code == 200, response.text
    # All seeded created_at values are non-null. NOT IN with a matching timestamp
    # or a null operand matches nothing; equality/IN match the exact timestamp.
    matches = (
        operand_kind == "other-timestamp" if operator == "$nin" else operand_kind in ("timestamp", "timestamp-and-null")
    )
    expected = [seed.json()["name"]] if matches else []
    assert [item["name"] for item in response.json()["data"]] == expected


@pytest.mark.integration
@pytest.mark.parametrize("operator", ["$eq", "$in", "$nin"])
async def test_json_timestamp_text_is_not_coerced(client: AsyncClient, ctx, operator):
    """Timestamp-like JSON keys remain schema-agnostic string metadata."""
    path = "/apis/entities/v2/workspaces/default/entities/model"
    seed = await client.post(path, json={"name": "test-item", "data": {"created_at": "notadate"}})
    assert seed.status_code == 201, seed.text
    value = "notadate" if operator == "$eq" else ["notadate"]
    response = await client.get(path, params={"filter": json.dumps({"data.created_at": {operator: value}})})
    assert response.status_code == 200, response.text
    expected = [] if operator == "$nin" else ["test-item"]
    assert [item["name"] for item in response.json()["data"]] == expected
