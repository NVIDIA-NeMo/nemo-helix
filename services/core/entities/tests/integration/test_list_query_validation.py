# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List-query failures and meaningful controls against SQL-backed HTTP routes."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient

WORKSPACES = "/apis/entities/v2/workspaces"
PROJECTS = "/apis/entities/v2/workspaces/default/projects"
ENTITIES = "/apis/entities/v2/workspaces/default/entities/model"
PATHS = [WORKSPACES, PROJECTS, ENTITIES]


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("field", ["nosuchfield", "metadata"])
async def test_unknown_or_noncolumn_filter(client: AsyncClient, ctx, path: str, field: str):
    """Each route translates unknown fields and non-column attributes to 400."""
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
        ({"filter[created_at][$gt]": "notadate"}, 400),
    ],
)
async def test_list_query_validation(client: AsyncClient, ctx, path: str, params: dict, status: int):
    """Bad pagination and malformed ordered timestamps remain client errors."""
    response = await client.get(path, params=params)
    assert response.status_code == status, response.text
    if status == 400:
        assert response.json()["detail"] == "Invalid timestamp filter value 'notadate'"


@pytest.mark.integration
@pytest.mark.parametrize("path", [WORKSPACES, PROJECTS])
async def test_invalid_enum_sort(client: AsyncClient, ctx, path):
    """Workspace/project sorts reject malformed values before repository access."""
    response = await client.get(path, params={"sort": "--name"})
    assert response.status_code == 422, response.text


@pytest.mark.integration
@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize(
    "operator,value",
    [
        ("$eq", "notadate"),
        ("$in", ["2020-01-01T00:00:00Z", "notadate"]),
        ("$nin", [None, "notadate"]),
        ("$in", ["2020-01-01T00:00:00Z", 42]),
    ],
)
async def test_invalid_timestamp_equality_and_membership(client: AsyncClient, ctx, path: str, operator, value):
    """Validate each timestamp operand, including mixed valid/invalid lists."""
    response = await client.get(path, params={"filter": json.dumps({"created_at": {operator: value}})})
    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith("Invalid timestamp filter value")


async def _seed_list_item(client: AsyncClient, path: str) -> dict:
    """Return one real row exposed by the requested list endpoint."""
    if path == WORKSPACES:
        response = await client.get(f"{path}/default")
    else:
        payload = {"name": "test-item"}
        if path == ENTITIES:
            payload["data"] = {}
        response = await client.post(path, json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()


@pytest.mark.integration
@pytest.mark.parametrize(
    "path,operator,operand_kind,matches",
    [
        (ENTITIES, "$eq", "timestamp", True),
        (ENTITIES, "$eq", "other", False),
        (ENTITIES, "$eq", "null", False),
        (ENTITIES, "$in", "timestamp-and-null", True),
        (ENTITIES, "$in", "other", False),
        (ENTITIES, "$in", "null", False),
        (ENTITIES, "$nin", "timestamp", False),
        (ENTITIES, "$nin", "other", True),
        (ENTITIES, "$nin", "other-and-null", False),
        (ENTITIES, "$nin", "null", False),
        (WORKSPACES, "$eq", "timestamp", True),
        (WORKSPACES, "$nin", "other", True),
        (PROJECTS, "$in", "timestamp", True),
    ],
)
async def test_valid_timestamp_equality_and_membership(client: AsyncClient, ctx, path, operator, operand_kind, matches):
    """Check actual matches; a nonmatching+null NOT IN list must exclude the row."""
    seed = await _seed_list_item(client, path)
    values = {
        "timestamp": [seed["created_at"]],
        "timestamp-and-null": [seed["created_at"], None],
        "other": ["1900-01-01T00:00:00Z"],
        "other-and-null": ["1900-01-01T00:00:00Z", None],
        "null": [None],
    }[operand_kind]
    value = values[0] if operator == "$eq" else values
    response = await client.get(path, params={"filter": json.dumps({"created_at": {operator: value}})})
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["data"]] == ([seed["name"]] if matches else [])


@pytest.mark.integration
@pytest.mark.parametrize("operator,matches", [("$eq", True), ("$gte", True), ("$in", True), ("$nin", False)])
async def test_offset_timestamp_matches_same_utc_instant(client: AsyncClient, ctx, operator, matches):
    """Equivalent offset strings compare to UTC storage, not to their wall time."""
    seed = await _seed_list_item(client, ENTITIES)
    utc = datetime.fromisoformat(seed["created_at"]).replace(tzinfo=timezone.utc)
    offset = utc.astimezone(timezone(timedelta(hours=2))).isoformat()
    value = [offset] if operator in ("$in", "$nin") else offset
    response = await client.get(ENTITIES, params={"filter": json.dumps({"created_at": {operator: value}})})
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["data"]] == ([seed["name"]] if matches else [])


@pytest.mark.integration
@pytest.mark.parametrize("operator", ["$eq", "$in", "$nin"])
async def test_json_timestamp_text_is_not_coerced(client: AsyncClient, ctx, operator):
    """Timestamp-like JSON keys remain schema-agnostic string metadata."""
    seed = await client.post(ENTITIES, json={"name": "test-item", "data": {"created_at": "notadate"}})
    assert seed.status_code == 201, seed.text
    value = "notadate" if operator == "$eq" else ["notadate"]
    response = await client.get(ENTITIES, params={"filter": json.dumps({"data.created_at": {operator: value}})})
    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["data"]] == ([] if operator == "$nin" else ["test-item"])
