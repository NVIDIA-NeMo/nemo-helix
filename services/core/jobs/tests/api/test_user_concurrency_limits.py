# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.entities.client import EntityConflictError, EntityNotFoundError
from nhx.core.jobs.api.dependencies import dep_system_entity_client
from nhx.core.jobs.api.v2.jobs.user_limits import router
from nhx.core.jobs.entities import JobsUserConcurrencyLimit
from pytest import fixture


class FakeEntityStore:
    """In-memory stand-in for the system ``EntityClient``, keyed by entity name."""

    def __init__(self) -> None:
        self.items: dict[str, JobsUserConcurrencyLimit] = {}

    async def create(self, entity):
        if entity.name in self.items:
            raise EntityConflictError("exists")
        self.items[entity.name] = entity
        return entity

    async def get(self, entity_type, name, *, workspace=None, parent=None):
        try:
            return self.items[name]
        except KeyError as exc:
            raise EntityNotFoundError(name) from exc

    async def list(self, entity_type, *, workspace, sort=None, page=1, page_size=100, **_):
        data = list(self.items.values())
        pagination = SimpleNamespace(
            model_dump=lambda: {
                "page": page,
                "page_size": page_size,
                "current_page_size": len(data),
                "total_pages": 1,
                "total_results": len(data),
            }
        )
        return SimpleNamespace(data=data, pagination=pagination)

    async def update(self, entity, *, original_name=None):
        self.items[entity.name] = entity
        return entity

    async def delete_by_id(self, entity_type, entity_id, **_):
        for name, entity in list(self.items.items()):
            if entity.id == entity_id:
                del self.items[name]


@fixture
def store() -> FakeEntityStore:
    return FakeEntityStore()


@fixture
def api(store: FakeEntityStore) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[dep_system_entity_client] = lambda: store
    return TestClient(app)


def test_create_then_get(api, store):
    created = api.post("/v2/user-concurrency-limits", json={"user_id": "Alice@Example.com", "max_concurrent_jobs": 7})

    assert created.status_code == 201
    assert created.json()["user_id"] == "Alice@Example.com"
    # The stored name is a valid entity name even though the user id has upper case and punctuation.
    assert list(store.items) == [JobsUserConcurrencyLimit.name_for_user("Alice@Example.com")]
    fetched = api.get("/v2/user-concurrency-limits/Alice@Example.com")
    assert fetched.status_code == 200
    assert fetched.json()["max_concurrent_jobs"] == 7


def test_create_duplicate_conflicts(api):
    body = {"user_id": "alice", "max_concurrent_jobs": 2}
    assert api.post("/v2/user-concurrency-limits", json=body).status_code == 201

    assert api.post("/v2/user-concurrency-limits", json=body).status_code == 409


def test_rejects_non_positive_limit(api):
    assert (
        api.post("/v2/user-concurrency-limits", json={"user_id": "alice", "max_concurrent_jobs": 0}).status_code == 422
    )
    assert api.post("/v2/user-concurrency-limits", json={"user_id": "", "max_concurrent_jobs": 1}).status_code == 422


def test_list(api):
    api.post("/v2/user-concurrency-limits", json={"user_id": "alice", "max_concurrent_jobs": 2})
    api.post("/v2/user-concurrency-limits", json={"user_id": "bob", "max_concurrent_jobs": 3})

    body = api.get("/v2/user-concurrency-limits").json()

    assert {item["user_id"] for item in body["data"]} == {"alice", "bob"}
    assert body["pagination"]["total_results"] == 2


def test_update(api):
    api.post("/v2/user-concurrency-limits", json={"user_id": "alice", "max_concurrent_jobs": 2})

    updated = api.put("/v2/user-concurrency-limits/alice", json={"max_concurrent_jobs": 9})

    assert updated.status_code == 200
    assert api.get("/v2/user-concurrency-limits/alice").json()["max_concurrent_jobs"] == 9


def test_delete(api):
    api.post("/v2/user-concurrency-limits", json={"user_id": "alice", "max_concurrent_jobs": 2})

    assert api.delete("/v2/user-concurrency-limits/alice").status_code == 204
    assert api.get("/v2/user-concurrency-limits/alice").status_code == 404


def test_missing_user_is_404(api):
    assert api.get("/v2/user-concurrency-limits/nobody").status_code == 404
    assert api.put("/v2/user-concurrency-limits/nobody", json={"max_concurrent_jobs": 2}).status_code == 404
    assert api.delete("/v2/user-concurrency-limits/nobody").status_code == 404


def test_name_for_user_is_stable_and_valid():
    name = JobsUserConcurrencyLimit.name_for_user("Alice@Example.com")

    assert name == JobsUserConcurrencyLimit.name_for_user("Alice@Example.com")
    assert name != JobsUserConcurrencyLimit.name_for_user("alice@example.com")
    assert name == name.lower() and "@" not in name


def test_rejects_user_id_that_cannot_be_addressed_by_path(api):
    for user in ("org/alice", " alice", "alice "):
        response = api.post("/v2/user-concurrency-limits", json={"user_id": user, "max_concurrent_jobs": 4})
        assert response.status_code == 422, user


def test_email_style_user_id_round_trips(api):
    user = "a+b@example.com"
    assert api.post("/v2/user-concurrency-limits", json={"user_id": user, "max_concurrent_jobs": 4}).status_code == 201

    assert api.get(f"/v2/user-concurrency-limits/{user}").status_code == 200
    assert api.delete(f"/v2/user-concurrency-limits/{user}").status_code == 204


def test_concurrent_delete_maps_to_404_on_update(api, store, monkeypatch):
    api.post("/v2/user-concurrency-limits", json={"user_id": "alice", "max_concurrent_jobs": 2})

    async def gone(entity, *, original_name=None):
        raise EntityNotFoundError("gone")

    monkeypatch.setattr(store, "update", gone)

    assert api.put("/v2/user-concurrency-limits/alice", json={"max_concurrent_jobs": 3}).status_code == 404
