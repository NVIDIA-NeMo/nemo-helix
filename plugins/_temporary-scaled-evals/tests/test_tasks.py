# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tasks router.

Uses an in-process FastAPI TestClient with a mocked psycopg connection
(via dependency_overrides on the mounted /v1 sub-app). End-to-end
coverage against a real Postgres lives in tests/integration/.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")
from api_test_fixture import client, v1
from scaled_evals.api import build
from scaled_evals.api.db import get_conn
from scaled_evals.api.settings import settings


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


def _conn_with_fetchall(rows: list[dict]) -> MagicMock:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    cur.fetchone.return_value = None
    cur.rowcount = 1
    return conn


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


def test_list_returns_envelope_shape() -> None:
    response = client.get("/v1/tasks")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_reconcile_packs_repairs_missing_owner_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _conn_with_fetchall(
        [
            {
                "task_id": "task_abc",
                "revision": 2,
                "status": "ready",
                "tarball_object_key": "task_abc/rev/2/tarball.tar.gz",
            }
        ]
    )
    _use_conn(conn)
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.object_exists", lambda _key: False)

    response = client.post("/v1/tasks/reconcile-packs?repair=true")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["owner_id"] == "dev"
    assert body["checked"] == 1
    assert body["missing"] == 1
    assert body["repaired"] == 1
    assert body["items"][0] == {
        "task_id": "task_abc",
        "revision": 2,
        "status": "ready",
        "object_key": "task_abc/rev/2/tarball.tar.gz",
        "missing": True,
        "repaired": True,
    }
    executed = _executed_sql(conn)
    assert any("UPDATE task_revisions" in sql and "task_object_missing" not in sql for sql in executed)
    assert any("current_revision = latest.revision" in sql for sql in executed)


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
    _use_conn(_conn_returning([{"id": "ev_active"}]))

    response = client.delete("/v1/tasks/task_in_use")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "task_in_use"


def test_delete_allows_task_when_no_active_evaluation_references_it() -> None:
    _use_conn(_conn_returning([None, {"id": "task_done"}]))

    response = client.delete("/v1/tasks/task_done")

    assert response.status_code == 200
    assert response.json() == {"id": "task_done", "deleted": True}


# ---------- finalize ------------------------------------------------------
#
# The build itself is exercised by the durable worker tests; here we assert the
# handler's state machine and the queue metadata persisted on task_revisions.


def _build_queue_params(conn: MagicMock) -> tuple:
    cur = conn.cursor.return_value.__enter__.return_value
    for call in cur.execute.call_args_list:
        sql, params = call.args
        if "build_backend = %s" in sql:
            return params
    raise AssertionError("build queue update was not executed")


def _latest_revision_row(*, revision: int = 1, status: str = "uploading", object_key: str = "k") -> dict:
    return {
        "revision": revision,
        "status": status,
        "tarball_object_key": object_key,
    }


def _mock_task_pack_size(monkeypatch: pytest.MonkeyPatch, size_bytes: int | None) -> MagicMock:
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.object_size", lambda _key: size_bytes)
    deleted = MagicMock()
    monkeypatch.setattr("scaled_evals.api.routers.tasks.artifacts.delete_object", deleted)
    return deleted


def _recorded_upload_size(conn: MagicMock) -> int:
    cur = conn.cursor.return_value.__enter__.return_value
    for call in cur.execute.call_args_list:
        sql, params = call.args
        if "tarball_size_bytes = %s" in sql:
            return params[0]
    raise AssertionError("upload size was not recorded")


def _executed_sql(conn: MagicMock) -> list[str]:
    cur = conn.cursor.return_value.__enter__.return_value
    return [call.args[0] for call in cur.execute.call_args_list]


def test_finalize_returns_404_when_missing() -> None:
    _use_conn(_conn_returning([None]))  # task row absent
    response = client.post("/v1/tasks/task_missing/finalize")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_finalize_returns_409_when_not_uploading() -> None:
    _use_conn(
        _conn_returning(
            [
                _latest_revision_row(status="ready"),
            ]
        )
    )
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
    _use_conn(
        _conn_returning(
            [
                _latest_revision_row(),
                {"id": "task_x"},
                {"revision": 1, "status": "uploading", "tarball_object_key": "k"},
                {"previous_storage_bytes": 0},
            ]
        )
    )

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"image_ref": "registry.example.com/bp:dev"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["revision"] == 1
    scheduled.assert_not_called()


def test_finalize_prebuilt_image_queues_registry_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "buildkit_enabled", False)
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    _mock_task_pack_size(monkeypatch, 12)
    conn = _conn_returning(
        [
            _latest_revision_row(),
            {"id": "task_x"},
            {"revision": 1, "status": "uploading", "tarball_object_key": "k"},
            {"previous_storage_bytes": 0},
        ]
    )
    _use_conn(conn)

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={
            "image_ref": "registry.example.com/team/task:signed",
            "image_digest": "sha256:" + "a" * 64,
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "building"
    backend, payload, credentials, *_ = _build_queue_params(conn)
    assert backend == "prebuilt"
    assert payload.obj == {
        "image_ref": "registry.example.com/team/task:signed",
        "expected_digest": "sha256:" + "a" * 64,
    }
    assert credentials.obj == {}


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
    deleted = _mock_task_pack_size(monkeypatch, 12)
    conn = _conn_returning(
        [
            _latest_revision_row(object_key="rev-1-pack"),
            {"id": "task_x"},
            {"revision": 2, "status": "uploading", "tarball_object_key": "rev-2-pack"},
        ]
    )
    _use_conn(conn)

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 409
    error = response.json()["detail"]["error"]
    assert error["code"] == "not_finalizable"
    assert error["details"] == {"expected_revision": 1, "actual_revision": 2}
    deleted.assert_not_called()
    with pytest.raises(AssertionError, match="build queue update"):
        _build_queue_params(conn)


def test_finalize_persists_supplied_tarball_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    _mock_task_pack_size(monkeypatch, 12)
    conn = _conn_returning(
        [
            _latest_revision_row(),
            {"id": "task_x"},
            {"revision": 1, "status": "uploading", "tarball_object_key": "key"},
            {"previous_storage_bytes": 0},
        ]
    )
    _use_conn(conn)

    response = client.post("/v1/tasks/task_x/finalize", json={"tarball_sha256": digest})

    assert response.status_code == 202
    _backend, _payload, _credentials, stored_hash, *_ = _build_queue_params(conn)
    assert stored_hash == digest


def test_finalize_can_pin_exact_reserved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_task_pack_size(monkeypatch, 12)
    conn = _conn_returning(
        [
            _latest_revision_row(revision=1, object_key="rev-1-pack"),
            {"id": "task_x"},
            _latest_revision_row(revision=1, object_key="rev-1-pack"),
            {"previous_storage_bytes": 0},
        ]
    )
    _use_conn(conn)

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"revision": 1, "tarball_sha256": "a" * 64},
    )

    assert response.status_code == 202
    revision_queries = [sql for sql in _executed_sql(conn) if "FROM task_revisions" in sql]
    assert revision_queries
    assert all("AND revision = %s" in sql for sql in revision_queries)


def test_finalize_rejects_missing_task_pack_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_conn(_conn_returning([_latest_revision_row(object_key="missing")]))

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
    _use_conn(_conn_returning([_latest_revision_row(object_key="k")]))

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 503
    error = response.json()["detail"]["error"]
    assert error["code"] == "cloud_build_unsupported"
    assert "Files service" in error["message"]


def test_finalize_rejects_oversized_task_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    deleted = _mock_task_pack_size(monkeypatch, 101)
    conn = _conn_returning([_latest_revision_row(object_key="too-big")])
    _use_conn(conn)

    response = client.post("/v1/tasks/task_x/finalize")

    assert response.status_code == 413
    error = response.json()["detail"]["error"]
    assert error["code"] == "task_pack_too_large"
    assert error["details"] == {
        "object_key": "too-big",
        "limit_bytes": 100,
        "uploaded_bytes": 101,
    }
    deleted.assert_called_once_with("too-big")
    with pytest.raises(AssertionError, match="build queue update"):
        _build_queue_params(conn)


def test_finalize_rejects_prebuilt_bypass_when_pack_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_allowed_registries", "registry.example.com")
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    _mock_task_pack_size(monkeypatch, 101)
    conn = _conn_returning([_latest_revision_row(object_key="too-big")])
    _use_conn(conn)

    response = client.post(
        "/v1/tasks/task_x/finalize",
        json={"image_ref": "registry.example.com/bp:dev"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["error"]["code"] == "task_pack_too_large"
    with pytest.raises(AssertionError, match="build queue update"):
        _build_queue_params(conn)


def test_finalize_rejects_tenant_storage_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "task_pack_max_size_bytes", 100)
    monkeypatch.setattr(settings, "task_pack_tenant_storage_quota_bytes", 100)
    deleted = _mock_task_pack_size(monkeypatch, 11)
    conn = _conn_returning(
        [
            _latest_revision_row(object_key="quota-pack"),
            {"id": "task_x"},
            {"revision": 1, "status": "uploading", "tarball_object_key": "quota-pack"},
            {"previous_storage_bytes": 90},
        ]
    )
    _use_conn(conn)

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
    sql = _executed_sql(conn)
    lock_index = next(i for i, statement in enumerate(sql) if "pg_advisory_xact_lock" in statement)
    usage_index = next(i for i, statement in enumerate(sql) if "SUM(r2.tarball_size_bytes)" in statement)
    assert lock_index < usage_index


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
