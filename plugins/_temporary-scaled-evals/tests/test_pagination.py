# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

try:
    from nemo_scaled_evals_plugin.service import ScaledEvalsService
    from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
    from scaled_evals.api.db import get_conn
    from scaled_evals.api.schemas.common import decode_cursor, encode_cursor
    from scaled_evals.api.settings import settings
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)

app = NemoServiceAdapter(ScaledEvalsService()).create_app()
client = TestClient(app)

NOW = datetime(2026, 6, 6, tzinfo=UTC)


def _task_row(index: int) -> dict[str, Any]:
    return {
        "id": f"task_{index}",
        "name": f"task {index}",
        "slug": f"task-{index}",
        "description": None,
        "visibility": "private",
        "current_revision": None,
        "created_at": NOW - timedelta(minutes=index),
        "updated_at": NOW - timedelta(minutes=index),
    }


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


def test_first_page_populates_next_cursor() -> None:
    rows = [_task_row(1), _task_row(2), _task_row(3)]
    _use_fetchall(rows)

    response = client.get("/v1/tasks", params={"limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert [row["id"] for row in payload["data"]] == ["task_1", "task_2"]
    assert decode_cursor(payload["next_cursor"]).id == "task_2"  # type: ignore[union-attr]


def test_middle_page_uses_cursor_and_populates_next_cursor() -> None:
    first_cursor = encode_cursor(_task_row(2)["created_at"], "task_2")
    conn = _use_fetchall([_task_row(3), _task_row(4), _task_row(5)])

    response = client.get("/v1/tasks", params={"limit": 2, "cursor": first_cursor})

    assert response.status_code == 200
    payload = response.json()
    assert [row["id"] for row in payload["data"]] == ["task_3", "task_4"]
    assert decode_cursor(payload["next_cursor"]).id == "task_4"  # type: ignore[union-attr]
    params = conn.cursor.return_value.__enter__.return_value.execute.call_args.args[1]
    assert params == [_task_row(2)["created_at"], "task_2", 3]


def test_last_page_has_no_next_cursor() -> None:
    cursor = encode_cursor(_task_row(4)["created_at"], "task_4")
    _use_fetchall([_task_row(5)])

    response = client.get("/v1/tasks", params={"limit": 2, "cursor": cursor})

    assert response.status_code == 200
    assert response.json()["next_cursor"] is None


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
        "/v1/credentials",
        "/v1/config-profiles",
        "/v1/users/me/evaluations",
        "/v1/users/me/tasks",
        "/v1/users/me/activity",
        "/v1/admin/users",
        "/v1/admin/users/user-1/evaluations",
    ],
)
def test_list_endpoints_reject_invalid_cursor(path: str, monkeypatch) -> None:
    # Exercise pagination behind the admin gate as the local dev principal.
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    _use_fetchall([])

    response = client.get(path, params={"cursor": "bogus"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "invalid_cursor"
