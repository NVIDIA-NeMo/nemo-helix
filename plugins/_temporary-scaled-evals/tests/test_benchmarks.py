# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benchmarks router.

Uses an in-process FastAPI TestClient over the fake Entity Store.
"""

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from task_fixtures import seed_benchmark, seed_task_revision


@pytest.fixture(autouse=True)
def _override_db():
    yield
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
    seed_task_revision("task_a", slug="task-a", name="Task A")
    seed_task_revision("task_b", 2, slug="task-b", name="Task B")
    seed_benchmark(tasks=(("task_a", None), ("task_b", 2)))

    first = client.get("/v1/benchmarks/bm_x/tasks", params={"limit": 1})
    second = client.get("/v1/benchmarks/bm_x/tasks", params={"limit": 1, "cursor": "1"})
    bad = client.get("/v1/benchmarks/bm_x/tasks", params={"cursor": "not-a-position"})

    assert first.status_code == 200, first.text
    assert first.json() == {
        "data": [
            {"task_id": "task_a", "task_revision": None, "position": 0, "task_slug": "task-a", "task_name": "Task A"}
        ],
        "next_cursor": "1",
    }
    assert [row["task_id"] for row in second.json()["data"]] == ["task_b"]
    assert second.json()["next_cursor"] is None
    assert bad.status_code == 422
    assert bad.json()["detail"]["error"]["code"] == "invalid_cursor"


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
    seed_benchmark("bm_done")
    response = client.delete("/v1/benchmarks/bm_done")
    assert response.status_code == 200
    assert response.json() == {"id": "bm_done", "deleted": True}
    assert client.get("/v1/benchmarks/bm_done").status_code == 404


def test_admin_can_qualify_and_promote() -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    seed_benchmark()
    early = client.post("/v1/benchmarks/bm_x/promote")
    qualify = client.post(
        "/v1/benchmarks/bm_x/qualification",
        json={"status": "qualified", "evidence": {"review": "ok"}},
    )
    promote = client.post("/v1/benchmarks/bm_x/promote")
    assert early.status_code == 409
    assert early.json()["detail"]["error"]["code"] == "benchmark_not_qualified"
    assert qualify.status_code == 200, qualify.text
    assert qualify.json()["qualification_evidence"] == {"review": "ok"}
    assert promote.status_code == 200, promote.text
    assert promote.json()["visibility"] == "public"
    assert promote.json()["qualification_status"] == "qualified"
    assert promote.json()["qualified_by"] == "dev"


# ---------- create --------------------------------------------------------


def test_create_with_no_tasks_returns_201_revision_1() -> None:
    response = client.post("/v1/benchmarks", json={"name": "Suite"})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"].startswith("bm_")
    assert body["slug"] == "suite"
    assert body["revision"] == 1
    assert body["links"]["self"] == f"/benchmarks/{body['id']}"
    duplicate = client.post("/v1/benchmarks", json={"name": "Suite"})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["error"]["code"] == "slug_conflict"


def test_created_benchmarks_and_variants_belong_to_the_caller() -> None:
    from scaled_evals.api.benchmark_store import BenchmarkStore

    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    base = client.post("/v1/benchmarks", json={"name": "Suite"})
    assert base.status_code == 201, base.text
    variant = client.post(
        f"/v1/benchmarks/{base.json()['id']}/variants",
        json={"name": "Suite long", "operational_policy": {"agent_timeout_floor_sec": 600}},
    )
    assert variant.status_code == 201, variant.text
    for benchmark_id in (base.json()["id"], variant.json()["id"]):
        stored = BenchmarkStore().get(benchmark_id)
        assert stored is not None
        assert stored["owner_id"] == "dev"


def test_concurrent_create_on_a_taken_slug_withdraws_the_later_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    from scaled_evals.api.benchmark_store import BenchmarkStore
    from scaled_evals.api.repositories.base_repository import Conflict

    seed_benchmark("bm_first", slug="suite")
    store = BenchmarkStore()
    # Both creates passed the up-front check before either was written.
    monkeypatch.setattr(store, "_ensure_slug_free", lambda slug, **_: None)

    with pytest.raises(Conflict) as exc:
        store.create_with_initial_revision(
            "bm_second", name="Suite", slug="suite", description=None, visibility="private", tasks=[]
        )

    assert exc.value.code == "slug_conflict"
    holder = store.get_by_slug("suite")
    assert holder is not None and holder["id"] == "bm_first"
    assert client.get("/v1/benchmarks/bm_second").status_code == 404


def test_create_validates_task_members() -> None:
    seed_task_revision("task_a", 2)
    floating = client.post("/v1/benchmarks", json={"name": "Floating", "tasks": [{"task_id": "task_a"}]})
    pinned = client.post(
        "/v1/benchmarks", json={"name": "Pinned", "tasks": [{"task_id": "task_a", "task_revision": 2}]}
    )
    assert floating.status_code == 201, floating.text
    assert pinned.status_code == 201, pinned.text
    for tasks in (
        [{"task_id": "task_ghost"}],
        [{"task_id": "task_a", "task_revision": 9}],
        [{"task_id": "task_a"}, {"task_id": "task_a"}],
    ):
        response = client.post("/v1/benchmarks", json={"name": "Bad", "tasks": tasks})
        assert response.status_code == 422, tasks
        assert response.json()["detail"]["error"]["code"] == "invalid_task_reference"
    assert client.get("/v1/benchmarks/by-slug/bad").status_code == 404


# ---------- get detail with members --------------------------------------


def test_get_detail_includes_member_tasks() -> None:
    seed_task_revision("task_a", slug="nemo-secrets-crud-cli-easy", name="NeMo secrets-crud (cli-easy)")
    seed_benchmark(tasks=(("task_a", None),))
    response = client.get("/v1/benchmarks/bm_x")
    skipped = client.get("/v1/benchmarks/bm_x", params={"include_tasks": "false"})
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
    assert skipped.status_code == 200, skipped.text
    assert skipped.json()["revision"] == 1
    assert skipped.json()["tasks"] == []


# ---------- validation ----------------------------------------------------


def test_create_rejects_bad_slug() -> None:
    response = client.post("/v1/benchmarks", json={"name": "Suite", "slug": "Bad Slug"})
    assert response.status_code == 422
