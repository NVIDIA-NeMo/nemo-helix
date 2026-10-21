# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest
from api_test_fixture import app, client
from scaled_evals.api.db import get_conn
from task_fixtures import seed_task_revision


def _use_fetchall(rows: list[dict[str, Any]]) -> MagicMock:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows

    def _gen():
        yield conn

    app.dependency_overrides[get_conn] = _gen
    return conn


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.pop(get_conn, None)


def test_cursor_walks_every_task_once() -> None:
    for index in range(1, 6):
        seed_task_revision(f"task_{index}")
    _use_fetchall([])

    pages: list[list[str]] = []
    cursor = None
    while True:
        params = {"limit": 2} if cursor is None else {"limit": 2, "cursor": cursor}
        response = client.get("/v1/tasks", params=params)
        assert response.status_code == 200
        payload = response.json()
        pages.append([row["id"] for row in payload["data"]])
        cursor = payload["next_cursor"]
        if cursor is None:
            break

    assert [len(page) for page in pages] == [2, 2, 1]
    assert sorted(sum(pages, [])) == [f"task_{index}" for index in range(1, 6)]


def test_invalid_cursor_returns_400() -> None:
    _use_fetchall([])

    response = client.get("/v1/tasks", params={"cursor": "not-a-valid-cursor"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_cursor"


@pytest.mark.parametrize(
    "path",
    [
        "/v1/evaluations",
        "/v1/tasks",
        "/v1/config-profiles",
        "/v1/users/me/evaluations",
        "/v1/users/me/tasks",
        "/v1/users/me/activity",
        "/v1/admin/users/user-1/evaluations",
    ],
)
def test_list_endpoints_reject_invalid_cursor(path: str) -> None:
    _use_fetchall([])

    response = client.get(path, params={"cursor": "bogus"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_cursor"
