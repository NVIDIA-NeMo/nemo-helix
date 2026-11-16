# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP regressions for the derived role-binding active filter."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.filter_ops import FilterOperation
from nemo_helix_plugin.in_memory_filter import InMemoryFilterRepository
from nhx.common.auth import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.entities import EntityClient, ListResponse, PaginationInfo
from nhx.core.auth.api.v2.iam import endpoints
from nhx.core.auth.entities import RoleBindingEntity

PATH = "/v2/iam/role-bindings"
ROWS = [
    {"name": "active-null", "data": {"principal": "alice", "revoked_at": None}},
    {"name": "active-missing", "data": {"principal": "alice"}},
    {"name": "revoked", "data": {"principal": "alice", "revoked_at": "2026-11-01T00:00:00Z"}},
    {"name": "other", "data": {"principal": "bob", "revoked_at": None}},
]


@pytest.fixture
def route_client() -> Iterator[tuple[TestClient, AsyncMock, Mock]]:
    auth = Mock(spec=AuthClient)
    auth.auth_enabled = False
    entities = Mock(spec=EntityClient)
    list_bindings = AsyncMock(
        return_value=ListResponse[RoleBindingEntity](
            data=[],
            pagination=PaginationInfo(
                page=1,
                page_size=10,
                current_page_size=0,
                total_pages=0,
                total_results=0,
            ),
        )
    )
    entities.list = list_bindings
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.get_auth_client] = lambda: auth
    app.dependency_overrides[endpoints.get_entity_client] = lambda: entities
    with TestClient(app) as client:
        yield client, list_bindings, auth


@pytest.mark.parametrize(
    ("params", "names"),
    [
        ({}, ["active-null", "active-missing", "revoked", "other"]),
        ({"filter[is_active]": "true"}, ["active-null", "active-missing", "other"]),
        ({"filter[is_active]": "false"}, ["revoked"]),
        ({"filter[is_active]": "FALSE"}, ["revoked"]),
        ({"filter": '{"is_active":true}'}, ["active-null", "active-missing", "other"]),
        ({"filter": '{"is_active":false}'}, ["revoked"]),
        ({"filter": 'is_active:"false"'}, ["revoked"]),
        ({"filter[is_active][$eq]": "false"}, ["revoked"]),
        ({"filter[is_active]": "true", "filter[principal]": "alice"}, ["active-null", "active-missing"]),
        ({"filter": '{"$or":[{"is_active":false},{"principal":"bob"}]}'}, ["revoked", "other"]),
        ({"filter": '{"$not":{"is_active":true}}'}, ["revoked"]),
        (
            {"filter": '{"$and":[{"principal":"alice"},{"$or":[{"is_active":true},{"is_active":false}]}]}'},
            ["active-null", "active-missing", "revoked"],
        ),
        ({"filter": '{"$and":[{"is_active":true},{"is_active":false}]}'}, []),
        ({"filter": '{"is_active":{"$not":{"$eq":true}}}'}, ["revoked"]),
    ],
)
def test_active_filter_selects_bindings(
    route_client: tuple[TestClient, AsyncMock, Mock],
    params: dict[str, str],
    names: list[str],
) -> None:
    client, list_bindings, _ = route_client
    response = client.get(PATH, params=params)
    assert response.status_code == 200, response.text
    list_bindings.assert_awaited_once()
    assert list_bindings.await_args is not None
    operation = list_bindings.await_args.kwargs["filter_operation"]
    assert operation is None or isinstance(operation, FilterOperation)
    selected = [row["name"] for row in ROWS if operation is None or operation.apply(InMemoryFilterRepository(row))]
    assert selected == names


@pytest.mark.parametrize(
    "params",
    [
        {"filter[is_active]": "null"},
        {"filter[is_active]": ""},
        {"filter[is_active]": "garbage"},
        {"filter[is_active]": "0"},
        {"filter[is_active]": "1"},
        {"filter": '{"is_active":null}'},
        {"filter": '{"is_active":0}'},
        {"filter": '{"is_active":1}'},
        {"filter": '{"is_active":[]}'},
        {"filter": '{"is_active":{"$in":[true,false]}}'},
        {"filter": '{"is_active":{"$exists":true}}'},
        {"filter": '{"$or":[{"principal":"alice"},{"is_active":null}]}'},
    ],
)
def test_invalid_active_filter_returns_422(
    route_client: tuple[TestClient, AsyncMock, Mock],
    params: dict[str, str],
) -> None:
    client, list_bindings, _ = route_client
    response = client.get(PATH, params=params)
    assert response.status_code == 422, response.text
    assert "is_active" in response.json()["detail"]
    list_bindings.assert_not_awaited()


@pytest.mark.parametrize(
    ("principal", "expected_status"),
    [
        ("alice@example.com", 403),
        ("service:integration-test", 200),
    ],
)
def test_active_filter_preserves_service_principal_guard(
    route_client: tuple[TestClient, AsyncMock, Mock],
    principal: str,
    expected_status: int,
) -> None:
    client, list_bindings, auth = route_client
    auth.auth_enabled = True
    auth.principal = Principal(id=principal)
    response = client.get(PATH, params={"filter[is_active]": "false"})
    assert response.status_code == expected_status
    if expected_status == 403:
        list_bindings.assert_not_awaited()
    else:
        list_bindings.assert_awaited_once()
