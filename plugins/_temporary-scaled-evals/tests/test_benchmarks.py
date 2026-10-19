# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benchmarks router.

Uses an in-process FastAPI TestClient with a mocked psycopg connection (via
dependency_overrides on the mounted /v1 sub-app). End-to-end coverage against a
real Postgres lives in tests/integration/test_benchmarks.py.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import get_conn
from scaled_evals.api.settings import settings


def _conn_returning(fetchones: list, fetchall=None) -> MagicMock:
    """Fake connection whose cursor.fetchone yields each value in turn."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = list(fetchones)
    if fetchall is not None:
        cur.fetchall.return_value = fetchall
    return conn


def _use_conn(conn: MagicMock) -> None:
    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen


def _empty_db() -> Iterator[MagicMock]:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    yield conn


def _benchmark_row(**overrides) -> dict:
    row = {
        "id": "bm_x",
        "name": "Suite",
        "slug": "suite",
        "description": None,
        "visibility": "private",
        "qualification_status": "registered",
        "qualification_evidence": {},
        "qualified_at": None,
        "qualified_by": None,
        "current_revision": 1,
        "created_at": "2026-06-26T00:00:00Z",
        "updated_at": "2026-06-26T00:00:00Z",
    }
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def _override_db():
    v1.dependency_overrides[get_conn] = _empty_db
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(current_principal, None)


# ---------- list / get 404 ------------------------------------------------


def test_list_returns_envelope_shape() -> None:
    response = client.get("/v1/benchmarks")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_get_by_id_returns_404_when_missing() -> None:
    response = client.get("/v1/benchmarks/bm_missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_get_by_slug_returns_404_when_missing() -> None:
    response = client.get("/v1/benchmarks/by-slug/missing")
    assert response.status_code == 404


def test_list_tasks_returns_404_when_benchmark_missing() -> None:
    response = client.get("/v1/benchmarks/bm_missing/tasks")
    assert response.status_code == 404


def test_list_tasks_returns_paginated_envelope() -> None:
    conn = _conn_returning(
        [_benchmark_row(current_revision=1)],
        fetchall=[
            {
                "task_id": "task_a",
                "task_revision": None,
                "position": 0,
                "task_slug": "task-a",
                "task_name": "Task A",
            },
            {
                "task_id": "task_b",
                "task_revision": 2,
                "position": 1,
                "task_slug": "task-b",
                "task_name": "Task B",
            },
        ],
    )
    _use_conn(conn)

    response = client.get("/v1/benchmarks/bm_x/tasks", params={"limit": 1})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data"] == [
        {
            "task_id": "task_a",
            "task_revision": None,
            "position": 0,
            "task_slug": "task-a",
            "task_name": "Task A",
        }
    ]
    assert body["next_cursor"] == "1"


def test_list_tasks_rejects_invalid_cursor() -> None:
    _use_conn(_conn_returning([_benchmark_row(current_revision=1)]))

    response = client.get("/v1/benchmarks/bm_x/tasks", params={"cursor": "not-a-position"})

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_cursor"


# ---------- revise / patch / delete: 404 when missing ---------------------


def test_revise_returns_404_when_missing() -> None:
    response = client.post("/v1/benchmarks/bm_missing/revisions", json={"tasks": []})
    assert response.status_code == 404


def test_patch_returns_404_when_missing() -> None:
    response = client.patch("/v1/benchmarks/bm_missing", json={"name": "new"})
    assert response.status_code == 404


def test_delete_returns_404_when_missing() -> None:
    response = client.delete("/v1/benchmarks/bm_missing")
    assert response.status_code == 404


def test_delete_succeeds_when_present() -> None:
    _use_conn(_conn_returning([{"id": "bm_done"}]))
    response = client.delete("/v1/benchmarks/bm_done")
    assert response.status_code == 200
    assert response.json() == {"id": "bm_done", "deleted": True}


def test_non_admin_cannot_qualify_or_promote() -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="hosted-user", source="starfleet_jwt"
    )
    qualify = client.post(
        "/v1/benchmarks/bm_x/qualification",
        json={"status": "qualified", "evidence": {}},
    )
    promote = client.post("/v1/benchmarks/bm_x/promote")
    assert qualify.status_code == 404
    assert promote.status_code == 404


def test_admin_can_qualify_and_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    _use_conn(
        _conn_returning(
            [
                _benchmark_row(
                    visibility="public",
                    qualification_status="qualified",
                    qualification_evidence={"review": "ok"},
                    qualified_at="2026-07-10T00:00:00Z",
                    qualified_by="dev",
                ),
                _benchmark_row(
                    visibility="public",
                    qualification_status="qualified",
                    qualification_evidence={"review": "ok"},
                    qualified_at="2026-07-10T00:00:00Z",
                    qualified_by="dev",
                ),
                _benchmark_row(
                    visibility="public",
                    qualification_status="qualified",
                    qualification_evidence={"review": "ok"},
                    qualified_at="2026-07-10T00:00:00Z",
                    qualified_by="dev",
                ),
                _benchmark_row(
                    visibility="public",
                    qualification_status="qualified",
                    qualification_evidence={"review": "ok"},
                    qualified_at="2026-07-10T00:00:00Z",
                    qualified_by="dev",
                ),
            ]
        )
    )
    qualify = client.post(
        "/v1/benchmarks/bm_x/qualification",
        json={"status": "qualified", "evidence": {"review": "ok"}},
    )
    promote = client.post("/v1/benchmarks/bm_x/promote")
    assert qualify.status_code == 200, qualify.text
    assert promote.status_code == 200, promote.text
    assert promote.json()["visibility"] == "public"
    assert promote.json()["qualification_status"] == "qualified"


# ---------- create --------------------------------------------------------


def test_create_with_no_tasks_returns_201_revision_1() -> None:
    _use_conn(_conn_returning([_benchmark_row()]))
    response = client.post("/v1/benchmarks", json={"name": "Suite"})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == "bm_x"
    assert body["revision"] == 1
    assert body["links"]["self"].startswith("/benchmarks/")


def test_create_with_floating_task_member() -> None:
    # INSERT benchmark RETURNING row, then SELECT-1 task-exists for the member.
    _use_conn(_conn_returning([_benchmark_row(), {"?column?": 1}]))
    response = client.post(
        "/v1/benchmarks",
        json={"name": "Suite", "tasks": [{"task_id": "task_a"}]},
    )
    assert response.status_code == 201, response.text


def test_create_pinned_task_member_validates_revision() -> None:
    _use_conn(_conn_returning([_benchmark_row(), {"?column?": 1}, {"?column?": 1}]))
    response = client.post(
        "/v1/benchmarks",
        json={"name": "Suite", "tasks": [{"task_id": "task_a", "task_revision": 2}]},
    )
    assert response.status_code == 201, response.text


def test_create_422_when_member_task_missing() -> None:
    # benchmark row inserted, then task-exists SELECT returns None.
    _use_conn(_conn_returning([_benchmark_row(), None]))
    response = client.post(
        "/v1/benchmarks",
        json={"name": "Suite", "tasks": [{"task_id": "task_ghost"}]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_task_reference"


def test_create_422_when_pinned_revision_missing() -> None:
    _use_conn(_conn_returning([_benchmark_row(), {"?column?": 1}, None]))
    response = client.post(
        "/v1/benchmarks",
        json={"name": "Suite", "tasks": [{"task_id": "task_a", "task_revision": 9}]},
    )
    assert response.status_code == 422


def test_create_422_on_duplicate_member() -> None:
    # First member validates (row + task-exists), second is a duplicate id.
    _use_conn(_conn_returning([_benchmark_row(), {"?column?": 1}]))
    response = client.post(
        "/v1/benchmarks",
        json={"name": "Suite", "tasks": [{"task_id": "task_a"}, {"task_id": "task_a"}]},
    )
    assert response.status_code == 422


# ---------- get detail with members --------------------------------------


def test_get_detail_includes_member_tasks() -> None:
    conn = _conn_returning(
        [
            _benchmark_row(current_revision=1),
            {
                "derived_from_benchmark_id": None,
                "derived_from_revision": None,
                "operational_policy": {},
            },
        ],
        fetchall=[
            {
                "task_id": "task_a",
                "task_revision": None,
                "position": 0,
                "task_slug": "nemo-secrets-crud-cli-easy",
                "task_name": "NeMo secrets-crud (cli-easy)",
            }
        ],
    )
    _use_conn(conn)
    response = client.get("/v1/benchmarks/bm_x")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["revision"] == 1
    assert body["derived_from"] is None
    assert body["operational_policy"] == {}
    # Members carry the task's human-readable slug/name, not just the opaque id.
    assert body["tasks"] == [
        {
            "task_id": "task_a",
            "task_revision": None,
            "position": 0,
            "task_slug": "nemo-secrets-crud-cli-easy",
            "task_name": "NeMo secrets-crud (cli-easy)",
        }
    ]


def test_get_detail_can_skip_member_tasks() -> None:
    conn = _conn_returning([_benchmark_row(current_revision=1)])
    cur = conn.cursor.return_value.__enter__.return_value
    _use_conn(conn)

    response = client.get("/v1/benchmarks/bm_x", params={"include_tasks": "false"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["revision"] == 1
    assert body["tasks"] == []
    assert not cur.fetchall.called


# ---------- validation ----------------------------------------------------


def test_create_rejects_bad_slug() -> None:
    response = client.post("/v1/benchmarks", json={"name": "Suite", "slug": "Bad Slug"})
    assert response.status_code == 422
