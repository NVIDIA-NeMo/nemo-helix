# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route-level tests for the /tasksets CRUD endpoints.

Drives the real FastAPI router + TasksetService through a TestClient with an in-memory entity store.
Covers route wiring, the get_taskset_service dependency, and status-code mapping (201/204/404/409/422).
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evaluator.api.dependencies import get_taskset_service
from nemo_evaluator.api.schemas import TaskRef, TasksetInput
from nemo_evaluator.api.service.taskset_service import TasksetService
from nemo_evaluator.api.v2 import tasksets as tasksets_routes
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError


class _FakeTaskService:
    """Resolves the member tasks the route tests reference so create-time validation passes."""

    async def head_by_id(self, task_id):
        raise NotImplementedError("Use the real TaskService for ID tests")

    async def resolve_head_revision(self, head):
        raise NotImplementedError("Use the real TaskService for ID tests")

    async def get_task(self, workspace: str, name: str) -> object | None:
        return object() if name in {"task-a", "task-b"} else None

    async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
        """A stable per-task digest; raises for an unknown task, as the real service does."""
        if name not in {"task-a", "task-b"}:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return hashlib.sha256(f"{workspace}/{name}".encode()).hexdigest()


@pytest.fixture
def client(entity_store) -> TestClient:
    app = FastAPI()
    app.include_router(tasksets_routes.router, prefix="/v2/workspaces/{workspace}")
    service = TasksetService(entity_store, _FakeTaskService(), authorize_task_read=AsyncMock())
    app.dependency_overrides[get_taskset_service] = lambda: service
    return TestClient(app)


def _body(*, description: str = "A grouping.", members: list[str] | None = None, tags: list[str] | None = None) -> dict:
    return TasksetInput(
        description=description,
        tasks=[TaskRef(m) for m in (members or ["task-a", "default/task-b"])],
        tags=tags or [],
    ).model_dump(mode="json", exclude_unset=True)


_BASE = "/v2/workspaces/default/tasksets"


def test_create_then_get(client: TestClient) -> None:
    resp = client.post(f"{_BASE}/ts-1", json=_body())
    assert resp.status_code == 201
    assert resp.json()["name"] == "ts-1"

    got = client.get(f"{_BASE}/ts-1")
    assert got.status_code == 200
    body = got.json()
    assert body["description"] == "A grouping."
    # TaskRef serializes to a bare string, and stored membership is workspace-qualified and
    # digest-pinned — a bare "task-a" is resolved to an exact revision on write.
    assert [t.split("#")[0] for t in body["tasks"]] == ["default/task-a", "default/task-b"]


def test_create_rejects_unknown_body_key(client: TestClient) -> None:
    # TasksetInput is extra="forbid" — an unknown key is a 422.
    body = _body()
    body["intent"] = "nope"
    assert client.post(f"{_BASE}/ts-1", json=body).status_code == 422


def test_create_rejects_duplicate_task_refs(client: TestClient) -> None:
    # Members are a set expressed as a list; a repeated ref is a 422, not a silent collapse.
    body = _body()
    body["tasks"] = ["task-a", "task-a"]
    assert client.post(f"{_BASE}/ts-1", json=body).status_code == 422


def test_create_rejects_refs_resolving_to_same_task(client: TestClient) -> None:
    # Distinct ref strings that resolve to the same task ("task-a" vs "default/task-a") are a 422.
    body = _body()
    body["tasks"] = ["task-a", "default/task-a"]
    assert client.post(f"{_BASE}/ts-1", json=body).status_code == 422


def test_create_rejects_missing_task_ref(client: TestClient) -> None:
    # A referenced task that does not exist is a 422 (client error in the submitted body).
    body = _body()
    body["tasks"] = ["task-a", "does-not-exist"]
    assert client.post(f"{_BASE}/ts-1", json=body).status_code == 422


def test_create_rejects_duplicate_metadata_keys(client: TestClient) -> None:
    body = _body()
    body["metadata"] = [{"key": "suite", "value": "smoke"}, {"key": "suite", "value": "regression"}]
    assert client.post(f"{_BASE}/ts-1", json=body).status_code == 422


def test_create_duplicate_returns_409(client: TestClient) -> None:
    assert client.post(f"{_BASE}/ts-1", json=_body()).status_code == 201
    assert client.post(f"{_BASE}/ts-1", json=_body()).status_code == 409


def test_create_rejects_invalid_name(client: TestClient) -> None:
    assert client.post(f"{_BASE}/bad name", json=_body()).status_code == 422


def test_get_missing_returns_404(client: TestClient) -> None:
    assert client.get(f"{_BASE}/nope").status_code == 404


def test_list_returns_created_tasksets(client: TestClient) -> None:
    client.post(f"{_BASE}/a", json=_body())
    client.post(f"{_BASE}/b", json=_body())

    resp = client.get(_BASE)
    assert resp.status_code == 200
    body = resp.json()
    assert {t["name"] for t in body["data"]} == {"a", "b"}
    assert body["pagination"]["total_results"] == 2


def test_delete_then_get_404(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    assert client.delete(f"{_BASE}/ts-1").status_code == 204
    assert client.get(f"{_BASE}/ts-1").status_code == 404


def test_delete_missing_returns_404(client: TestClient) -> None:
    assert client.delete(f"{_BASE}/nope").status_code == 404


def test_delete_conflict_returns_409() -> None:
    class _Service:
        async def delete_taskset(self, workspace: str, name: str) -> bool:
            raise NemoEntityConflictError("changed")

    app = FastAPI()
    app.include_router(tasksets_routes.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_taskset_service] = lambda: _Service()
    client = TestClient(app)

    assert client.delete(f"{_BASE}/ts-1").status_code == 409


# --- Publishing revisions (POST creates, PUT replaces) ------------------------


def test_create_publishes_revision_one(client: TestClient) -> None:
    created = client.post(f"{_BASE}/ts-1", json=_body()).json()
    assert created["revision"] == 1
    assert created["tags"] == {"latest": 1}


def test_members_are_stored_digest_pinned(client: TestClient) -> None:
    """A published grouping names exact revisions. Storing '#latest' would let a member republish
    silently change what this taskset contains — which is the whole failure this design prevents."""
    body = client.post(f"{_BASE}/ts-1", json=_body()).json()
    assert all(len(ref.split("#")[1]) == 64 for ref in body["tasks"])


def test_tag_pinned_member_is_resolved_to_a_digest(client: TestClient) -> None:
    """Tags are resolution *inputs*: accepted on the way in, never persisted."""
    body = client.post(f"{_BASE}/ts-1", json=_body(members=["task-a#latest"])).json()
    assert "#latest" not in body["tasks"][0]
    assert len(body["tasks"][0].split("#")[1]) == 64


def test_put_on_missing_taskset_creates_it(client: TestClient) -> None:
    response = client.put(f"{_BASE}/ts-1", json=_body())
    assert response.status_code == 201
    assert response.json()["revision"] == 1


def test_put_with_changed_membership_publishes_a_new_revision(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    response = client.put(f"{_BASE}/ts-1", json=_body(members=["task-a"]))
    assert response.status_code == 201
    assert response.json()["revision"] == 2


def test_put_with_identical_membership_publishes_nothing(client: TestClient) -> None:
    """Idempotent when nothing underneath moved — the member digests resolve the same."""
    client.post(f"{_BASE}/ts-1", json=_body())
    response = client.put(f"{_BASE}/ts-1", json=_body())
    assert response.status_code == 200
    assert response.json()["revision"] == 1


def test_put_applies_tags_without_publishing(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    response = client.put(f"{_BASE}/ts-1", json=_body(tags=["blessed"]))
    assert response.status_code == 200
    assert response.json()["tags"] == {"latest": 1, "blessed": 1}


def test_post_on_existing_taskset_still_conflicts(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    assert client.post(f"{_BASE}/ts-1", json=_body(description="Other.")).status_code == 409


# --- Reading and tagging a specific revision ---------------------------------


def test_list_revisions_newest_first(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    client.put(f"{_BASE}/ts-1", json=_body(members=["task-a"]))

    revisions = client.get(f"{_BASE}/ts-1/revisions").json()["data"]
    assert [r["revision"] for r in revisions] == [2, 1]
    assert revisions[0]["tags"] == ["latest"]


def test_get_by_digest_returns_published_membership(client: TestClient) -> None:
    """The reason a dataset is reproducible: a pinned read returns the membership as published."""
    client.post(f"{_BASE}/ts-1", json=_body())
    digest = client.get(f"{_BASE}/ts-1/revisions").json()["data"][0]["content_hash"]
    client.put(f"{_BASE}/ts-1", json=_body(members=["task-a"]))

    pinned = client.get(f"{_BASE}/ts-1/revisions/{digest}").json()
    assert len(pinned["tasks"]) == 2
    assert len(client.get(f"{_BASE}/ts-1").json()["tasks"]) == 1


def test_tag_an_existing_revision(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    digest = client.get(f"{_BASE}/ts-1/revisions").json()["data"][0]["content_hash"]
    client.put(f"{_BASE}/ts-1", json=_body(members=["task-a"]))

    tagged = client.put(f"{_BASE}/ts-1/tags/blessed", params={"revision": digest})
    assert tagged.status_code == 200
    assert tagged.json()["tags"] == {"latest": 2, "blessed": 1}


def test_cannot_move_latest_by_hand(client: TestClient) -> None:
    client.post(f"{_BASE}/ts-1", json=_body())
    digest = client.get(f"{_BASE}/ts-1/revisions").json()["data"][0]["content_hash"]
    assert client.put(f"{_BASE}/ts-1/tags/latest", params={"revision": digest}).status_code == 422


def test_list_revisions_missing_taskset_returns_404(client: TestClient) -> None:
    assert client.get(f"{_BASE}/nope/revisions").status_code == 404
