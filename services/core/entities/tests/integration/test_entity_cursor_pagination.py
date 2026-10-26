# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in timestamp/ID cursors through the in-process entities HTTP API."""

from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient
from nhx.core.entities.api.v2.utils import ROLE_BINDING_ENTITY_TYPE, clear_principal_bindings_cache
from nhx.core.entities.app.repository.sqlalchemy.models import DBEntity
from sqlalchemy import delete

pytestmark = pytest.mark.integration

URL = "/apis/entities/v2/workspaces/default/entities/widget"


@pytest.fixture
async def cursor_rows(ctx, session_maker):
    timestamp = datetime(2026, 1, 1)
    async with session_maker() as session:
        # Timestamp ties are deliberately inserted out of ID order.
        for index in [8, 2, 5, 1, 4, 7, 0, 3, 6]:
            session.add(
                DBEntity(
                    id=f"widget-{index}",
                    workspace="default",
                    entity_type="widget",
                    name=f"item-{index}",
                    data={"rank": index},
                    created_at=timestamp + timedelta(seconds=index // 3),
                    updated_at=timestamp,
                )
            )
        await session.commit()
    return timestamp


async def collect_remaining(client: AsyncClient, *, sort: str, token: str, page_size: int):
    ids = []
    seen_tokens = set()
    while token is not None:
        assert token not in seen_tokens
        seen_tokens.add(token)
        response = await client.get(URL, params={"sort": sort, "page_size": page_size, "page_token": token})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pagination"]["page"] == 1
        assert body["pagination"]["page_size"] == page_size
        assert body["pagination"]["current_page_size"] == len(body["data"])
        ids.extend(row["id"] for row in body["data"])
        token = body["next_page_token"]
    return ids


def expected_ids(sort: str):
    groups = [range(0, 3), range(3, 6), range(6, 9)]
    if sort == "-created_at":
        groups.reverse()
    return [f"widget-{index}" for group in groups for index in group]


@pytest.mark.parametrize("sort", ["created_at", "-created_at"])
@pytest.mark.parametrize("page_size", [1, 2, 3, 4, 9, 10])
async def test_cursor_pages_visit_timestamp_ties_once(client, cursor_rows, sort, page_size):
    actual = await collect_remaining(client, sort=sort, token="", page_size=page_size)
    assert actual == expected_ids(sort)


@pytest.mark.parametrize("sort", ["created_at", "-created_at"])
async def test_insert_and_deleted_boundary_do_not_shift_cursor_pages(client, cursor_rows, session_maker, sort):
    response = await client.get(URL, params={"sort": sort, "page_size": 2, "page_token": ""})
    assert response.status_code == 200
    first = response.json()
    first_ids = [row["id"] for row in first["data"]]
    assert first_ids == expected_ids(sort)[:2]
    async with session_maker() as session:
        # Delete the already emitted rows, including the boundary itself. Tokens
        # must not depend on looking that row up again.
        await session.execute(delete(DBEntity).where(DBEntity.id.in_(first_ids)))
        session.add(
            DBEntity(
                id="widget-new-before-boundary",
                workspace="default",
                entity_type="widget",
                name="new",
                data={},
                created_at=cursor_rows + timedelta(days=1 if sort.startswith("-") else -1),
                updated_at=cursor_rows,
            )
        )
        await session.commit()
    remaining = await collect_remaining(client, sort=sort, token=first["next_page_token"], page_size=2)
    assert remaining == expected_ids(sort)[2:]


async def test_offset_mode_and_default_sort_remain_available(client, cursor_rows):
    response = await client.get(URL, params={"page": 2, "page_size": 2, "sort": "name"})
    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body["data"]] == ["widget-2", "widget-3"]
    assert body["pagination"]["page"] == 2
    assert body["next_page_token"] is None
    response = await client.get(URL, params={"page_token": "", "page_size": 2})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["data"]] == expected_ids("-created_at")[:2]


@pytest.mark.parametrize("sort", ["", "id", "name", "updated_at", "data.rank", "--created_at"])
async def test_cursor_mode_rejects_unsupported_sorts(client, cursor_rows, sort):
    response = await client.get(URL, params={"page_token": "", "sort": sort})
    assert response.status_code == 400
    assert response.json()["detail"] == "page_token supports only created_at and -created_at sorts"


async def test_cursor_mode_rejects_numbered_page_combination(client, cursor_rows):
    response = await client.get(URL, params={"page_token": "", "page": 2})
    assert response.status_code == 400


@pytest.mark.parametrize("token", ["not-a-token", "0", "é", "1", "zzzz", "[]"])
async def test_invalid_cursor_returns_client_error(client, ctx, token):
    response = await client.get(URL, params={"page_token": token})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid page_token"


async def test_oversize_cursor_returns_validation_error(client, ctx):
    response = await client.get(URL, params={"page_token": "a" * 4097})
    assert response.status_code == 422


@pytest.mark.parametrize("change", ["sort", "filter", "entity_type", "workspace"])
async def test_cursor_cannot_be_reused_for_another_query(client, cursor_rows, change):
    first = await client.get(URL, params={"page_token": "", "page_size": 2})
    assert first.status_code == 200
    params = {"page_token": first.json()["next_page_token"]}
    url = URL
    if change == "sort":
        params["sort"] = "created_at"
    elif change == "filter":
        params["filter"] = '{"name":{"$eq":"item-0"}}'
    elif change == "entity_type":
        url = URL.replace("widget", "other")
    else:
        created = await client.post("/apis/entities/v2/workspaces", json={"name": "other"})
        assert created.status_code == 201
        url = URL.replace("default", "other")
    response = await client.get(url, params=params)
    assert response.status_code == 400
    assert response.json()["detail"] == "page_token does not match the current entity query"


async def test_cursor_respects_filters_and_allows_page_size_changes(client, cursor_rows):
    params = {"page_token": "", "page_size": 1, "filter": '{"data.rank":{"$gte":3}}'}
    ids = []
    while params["page_token"] is not None:
        response = await client.get(URL, params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pagination"]["total_results"] == 6
        ids.extend(row["id"] for row in body["data"])
        params["page_token"] = body["next_page_token"]
        params["page_size"] = 2
    assert ids == [f"widget-{index}" for index in [6, 7, 8, 3, 4, 5]]


async def test_empty_cursor_query_is_terminal(client, ctx):
    response = await client.get(URL, params={"page_token": ""})
    assert response.status_code == 200
    body = response.json()
    assert body["data"] == []
    assert body["next_page_token"] is None
    assert body["pagination"]["total_results"] == 0


async def test_cross_workspace_cursor_rechecks_current_permissions(client_with_auth, repos):
    await clear_principal_bindings_cache()
    grants = []
    try:
        for workspace in ["default", "other"]:
            await repos["workspace"].create_workspace(name=workspace)
            grants.append(
                await repos["entity"].create_entity(
                    workspace=workspace,
                    entity_type=ROLE_BINDING_ENTITY_TYPE,
                    name="membership",
                    data={
                        "principal": "test-user@example.com",
                        "workspace": workspace,
                        "role": "Editor",
                        "revoked_at": None,
                    },
                )
            )
            for index in range(3):
                await repos["entity"].create_entity(
                    workspace=workspace,
                    entity_type="widget",
                    name=f"item-{index}",
                    data={},
                )
        url = URL.replace("default", "-")
        response = await client_with_auth.get(url, params={"page_token": "", "page_size": 1})
        assert response.status_code == 200, response.text
        token = response.json()["next_page_token"]
        assert token is not None
        await repos["entity"].delete_entity(entity_id=grants[1].id)
        await clear_principal_bindings_cache()
        response = await client_with_auth.get(url, params={"page_token": token, "page_size": 1})
        assert response.status_code == 400
        assert response.json()["detail"] == "page_token does not match the current entity query"
        # A fresh traversal now sees only the currently accessible workspace.
        response = await client_with_auth.get(url, params={"page_token": ""})
        assert response.status_code == 200
        assert {row["workspace"] for row in response.json()["data"]} == {"default"}
    finally:
        await clear_principal_bindings_cache()
