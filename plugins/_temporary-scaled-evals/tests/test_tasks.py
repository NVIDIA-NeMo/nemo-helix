# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tasks router.

Uses an in-process FastAPI TestClient against the in-memory Entity Store,
with a mocked psycopg connection (via dependency_overrides on the mounted
/v1 sub-app) for the evaluation lookups that remain in Postgres.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")
from api_test_fixture import client, v1
from scaled_evals.api import build
from scaled_evals.api.db import get_conn
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskStore
from task_fixtures import stored_revision, stored_task


def _conn_returning(fetchones: list) -> MagicMock:
    """Fake connection whose cursor.fetchone yields each value in turn."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = list(fetchones)
    cur.rowcount = 1
    return conn


def _use_conn(conn: MagicMock) -> None:
    """Point get_conn at a specific fake connection for one test."""

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen


def _empty_db() -> Iterator[MagicMock]:
    """Fake connection whose queries return nothing — empty list, no row."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    yield conn


@pytest.fixture(autouse=True)
def _override_db():
    v1.dependency_overrides[get_conn] = _empty_db
    yield
    v1.dependency_overrides.pop(get_conn, None)


# ---------- list ----------------------------------------------------------


def _seed(task_id: str = "task_x", *, owner_id: str | None = "dev", revisions: int = 1) -> TaskStore:
    store = TaskStore()
    store.create_with_initial_revision(
        task_id,
        name=task_id,
        slug=task_id.replace("_", "-"),
        description=None,
        visibility="private",
        object_key="k",
        owner_id=owner_id,
    )
    for _ in range(revisions - 1):
        store.create_next_revision(task_id)
    return store


def _make_ready(store: TaskStore, task_id: str, revision: int) -> None:
    store.mark_latest_revision_building(
        task_id, build_backend="buildkit", expected_revision=revision, exact_revision=True
    )
    store.record_success(task_id, revision, image_ref=f"image:{revision}", image_digest="sha256:" + "a" * 64)


def test_list_returns_envelope_shape() -> None:
    response = client.get("/v1/tasks")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_reconcile_packs_repairs_missing_owner_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed("task_abc", revisions=2)
    _make_ready(store, "task_abc", 1)
    _make_ready(store, "task_abc", 2)
    object_key = stored_revision("task_abc", 2)["tarball_object_key"]
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.object_exists", lambda key: key != object_key)

    response = client.post("/v1/tasks/reconcile-packs?repair=true")

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["owner_id"], body["checked"], body["missing"], body["repaired"]) == ("dev", 2, 1, 1)
    assert body["items"][0] == {
        "task_id": "task_abc",
        "revision": 2,
        "status": "ready",
        "object_key": object_key,
        "missing": True,
        "repaired": True,
    }
    assert stored_revision("task_abc", 2)["status"] == "failed"
    assert stored_task("task_abc")["current_revision"] == 1


# ---------- get -----------------------------------------------------------


def test_get_by_id_returns_404_when_missing() -> None:
    response = client.get("/v1/tasks/task_does_not_exist")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_get_by_slug_returns_404_when_missing() -> None:
    response = client.get("/v1/tasks/by-slug/missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


# ---------- revise / patch / delete: 404 when missing ---------------------


def test_revise_returns_404_when_missing() -> None:
    response = client.post("/v1/tasks/task_missing/revisions")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_patch_returns_404_when_missing() -> None:
    response = client.patch("/v1/tasks/task_missing", json={"name": "new"})
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_delete_returns_404_when_missing() -> None:
    response = client.delete("/v1/tasks/task_missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_delete_returns_409_when_active_evaluation_references_task() -> None:
    store = _seed("task_in_use")
    _use_conn(_conn_returning([{"id": "ev_active"}]))

    response = client.delete("/v1/tasks/task_in_use")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "task_in_use"
    assert store.get("task_in_use") is not None


def test_delete_allows_task_when_no_active_evaluation_references_it() -> None:
    store = _seed("task_done")
    _use_conn(_conn_returning([None, None]))

    response = client.delete("/v1/tasks/task_done")

    assert response.status_code == 200
    assert response.json() == {"id": "task_done", "deleted": True}
    assert store.get("task_done") is None
    assert client.delete("/v1/tasks/task_done").status_code == 404


# ---------- finalize ------------------------------------------------------
#
# The build itself is exercised by the durable worker tests; here we assert the
# handler's state machine and the queue metadata persisted on the revision.


def _mock_task_pack_size(monkeypatch: pytest.MonkeyPatch, size_bytes: int | None) -> MagicMock:
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.object_size", lambda _key: size_bytes)
    deleted = MagicMock()
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.delete_object", deleted)
    return deleted


def _revision(revision: int = 1, task_id: str = "task_x") -> dict:
    return stored_revision(task_id, revision)


def test_finalize_returns_404_when_missing() -> None:
    response = client.post("/v1/tasks/task_missing/finalize")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_finalize_returns_409_when_not_uploading() -> None:
    _make_ready(_seed(), "task_x", 1)
    response = client.post("/v1/tasks/task_x/finalize")
    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "not_finalizable"


def test_finalize_prebuilt_image_marks_ready_without_buildkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "disabled")
    _mock_task_pack_size(monkeypatch, 12)
    scheduled = MagicMock()
    monkeypatch.setattr(build, "run_finalize_build", scheduled)
    _seed()

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"image_ref": "registry.example.com/bp:dev"},
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["revision"]) == ("ready", 1)
    assert _revision()["image_ref"] == "registry.example.com/bp:dev"
    scheduled.assert_not_called()


def test_finalize_prebuilt_image_queues_registry_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    _mock_task_pack_size(monkeypatch, 12)
    _seed()

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/team/task:signed",
            "image_digest": "sha256:" + "a" * 64,
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "building"
    revision = _revision()
    assert (revision["status"], revision["build_backend"], revision["build_credentials"]) == (
        "building",
        "prebuilt",
        {},
    )
    assert revision["build_payload"] == {
        "image_ref": "registry.example.com/team/task:signed",
        "expected_digest": "sha256:" + "a" * 64,
    }


def test_finalize_prebuilt_records_verifier_image_without_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "disabled")
    _mock_task_pack_size(monkeypatch, 12)
    _seed()

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/bp:dev",
            "verifier_image_ref": "registry.example.com/bp-verifier:dev",
            "verifier_image_digest": "sha256:" + "b" * 64,
        },
    )

    assert response.status_code == 200, response.text
    revision = _revision()
    assert revision["image_ref"] == "registry.example.com/bp:dev"
    assert revision["verifier_image_ref"] == "registry.example.com/bp-verifier:dev"
    assert revision["verifier_image_digest"] == "sha256:" + "b" * 64


def test_finalize_rejects_verifier_image_without_digest_when_validation_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_validation_mode", "disabled")

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/bp:dev",
            "verifier_image_ref": "registry.example.com/bp-verifier:dev",
        },
    )

    assert response.status_code == 422
    error = response.json()["detail"]["error"]
    assert error["code"] == "invalid_request"
    assert "verifier_image_digest" in error["message"]


def test_finalize_prebuilt_queues_verifier_image_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    _mock_task_pack_size(monkeypatch, 12)
    _seed()

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/team/task:signed",
            "image_digest": "sha256:" + "a" * 64,
            "verifier_image_ref": "registry.example.com/team/task-verifier:signed",
            "verifier_image_digest": "sha256:" + "b" * 64,
        },
    )

    assert response.status_code == 202, response.text
    assert _revision()["build_payload"] == {
        "image_ref": "registry.example.com/team/task:signed",
        "expected_digest": "sha256:" + "a" * 64,
        "verifier_image_ref": "registry.example.com/team/task-verifier:signed",
        "verifier_expected_digest": "sha256:" + "b" * 64,
    }


def test_finalize_rejects_verifier_image_from_disallowed_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/team/task:signed",
            "verifier_image_ref": "evil.example.com/team/task-verifier:signed",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_verifier_image"


@pytest.mark.parametrize(
    "body",
    [
        {"image_ref": "registry.example.com/bp:dev", "verifier_image_digest": "sha256:" + "b" * 64},
        {"verifier_image_ref": "registry.example.com/bp-verifier:dev"},
    ],
)
def test_finalize_rejects_incomplete_verifier_image_request(body: dict) -> None:
    response = client.post("/v1/tasks/task_x/finalize", json=body)

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_request"


def test_finalize_rejects_legacy_source_fields() -> None:
    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "source_project": "nemo/platform",
            "source_ref": "a" * 40,
            "context_hash": "8c366d6f324b86faee15f99af6a26483519f1624b",
        },
    )

    assert response.status_code == 422
    assert "source_project" in response.text


def test_finalize_rejects_digest_without_image_ref() -> None:
    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"image_digest": "sha256:" + "a" * 64},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_request"


def test_finalize_rejects_revision_changed_after_pack_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _seed()

    def size_then_revise(_key: str) -> int:
        store.create_next_revision("task_x")
        return 12

    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.object_size", size_then_revise)
    deleted = MagicMock()
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.delete_object", deleted)

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 409
    error = response.json()["detail"]["error"]
    assert error["code"] == "not_finalizable"
    assert error["details"] == {"expected_revision": 1, "actual_revision": 2}
    deleted.assert_not_called()
    assert _revision(1)["status"] == _revision(2)["status"] == "uploading"


def test_finalize_persists_supplied_tarball_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    _mock_task_pack_size(monkeypatch, 12)
    _seed()

    response = client.post("/v1/tasks/task_x/finalize", json={"tarball_sha256": digest})

    assert response.status_code == 202
    revision = _revision()
    assert (revision["status"], revision["tarball_sha256"], revision["tarball_size_bytes"]) == ("building", digest, 12)


def test_finalize_can_pin_exact_reserved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_task_pack_size(monkeypatch, 12)
    _seed(revisions=2)

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"revision": 1, "tarball_sha256": "a" * 64},
    )

    assert response.status_code == 202
    assert (_revision(1)["status"], _revision(2)["status"]) == ("building", "uploading")


def test_finalize_rejects_missing_task_pack_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    _seed()

    # On Files, a not-yet-uploaded pack simply has no size (object_size -> None); there is no
    # separate object-store error to raise.
    _mock_task_pack_size(monkeypatch, None)

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 409
    error = response.json()["detail"]["error"]
    assert error["code"] == "task_pack_missing"
    assert "upload the tarball" in error["message"]


def test_finalize_rejects_cloud_build_after_files_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cloud Build reads the pack from a GCS storageSource, but packs now live in the Files
    # service, so a Cloud Build finalize would build against an object that was never written.
    # With cloud build the only configured builder (no image-builder URL), finalize must refuse.
    monkeypatch.setattr(settings, "cloud_build_enabled", True)
    monkeypatch.setattr(settings, "image_builder_service_url", "")
    _seed()

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 503
    error = response.json()["detail"]["error"]
    assert error["code"] == "cloud_build_unsupported"
    assert "Files service" in error["message"]


def test_finalize_rejects_oversized_task_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    deleted = _mock_task_pack_size(monkeypatch, 101)
    _seed()

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 413
    error = response.json()["detail"]["error"]
    assert error["code"] == "task_pack_too_large"
    assert error["details"] == {"object_key": "k", "limit_bytes": 100, "uploaded_bytes": 101}
    deleted.assert_called_once_with("k")
    assert _revision()["status"] == "uploading"


def test_finalize_rejects_prebuilt_bypass_when_pack_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    _mock_task_pack_size(monkeypatch, 101)
    _seed()

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"image_ref": "registry.example.com/bp:dev"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["error"]["code"] == "task_pack_too_large"
    assert _revision()["status"] == "uploading"


def test_finalize_rejects_tenant_storage_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    monkeypatch.setattr(settings, "task_pack_tenant_storage_quota_bytes", 100)
    store = _seed("task_old")
    _mock_task_pack_size(monkeypatch, 90)
    assert client.post("/v1/tasks/task_old/finalize").status_code == 202
    store.create_with_initial_revision(
        "task_x", name="x", slug="x", description=None, visibility="private", object_key="quota-pack", owner_id="dev"
    )
    deleted = _mock_task_pack_size(monkeypatch, 11)

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 413
    error = response.json()["detail"]["error"]
    assert error["code"] == "tenant_storage_quota_exceeded"
    assert error["details"] == {
        "object_key": "quota-pack",
        "quota_bytes": 100,
        "used_bytes": 90,
        "uploaded_bytes": 11,
    }
    deleted.assert_called_once_with("quota-pack")
    assert _revision()["status"] == "uploading"


# ---------- patch: request-body validation (no DB hit) --------------------


def test_patch_rejects_uppercase_slug() -> None:
    response = client.patch("/v1/tasks/task_x", json={"slug": "Upper"})
    assert response.status_code == 422


def test_patch_rejects_empty_name() -> None:
    response = client.patch("/v1/tasks/task_x", json={"name": ""})
    assert response.status_code == 422


# ---------- create: request-body validation (no DB hit) -------------------
#
# Pydantic validation runs before dependency resolution, so these don't
# need a mock that emulates an INSERT — they're guarded at the boundary.


def test_create_rejects_empty_name() -> None:
    response = client.post("/v1/tasks", json={"name": ""})
    assert response.status_code == 422


def test_create_rejects_uppercase_slug() -> None:
    response = client.post("/v1/tasks", json={"name": "x", "slug": "Upper"})
    assert response.status_code == 422


def test_create_rejects_slug_with_special_chars() -> None:
    response = client.post("/v1/tasks", json={"name": "x", "slug": "bad slug!"})
    assert response.status_code == 422


def test_create_rejects_overlong_slug() -> None:
    response = client.post("/v1/tasks", json={"name": "x", "slug": "a" * 64})
    assert response.status_code == 422


def test_create_rejects_unknown_visibility() -> None:
    response = client.post("/v1/tasks", json={"name": "x", "visibility": "secret"})
    assert response.status_code == 422
