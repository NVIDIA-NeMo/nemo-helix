# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request isolation with in-memory entities and controlled authorization decisions."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx2
import pytest

pytest.importorskip("scaled_evals")

from fake_entity_store import FakeEntityStore
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from nemo_helix_plugin.auth import AuthContext
from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmarkRun, ScaledEvalsEvaluation
from scaled_evals.api import auth
from scaled_evals.api import entity_store as entity_storage
from scaled_evals.api.db import RequestDatabase, get_request_db, get_request_stream_database_factory
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.resource_access import ResourceAccess
from scaled_evals.api.routers import benchmark_runs, evaluations
from scaled_evals.api.runnability import caller_secret_readable
from scaled_evals.api.task_store import TaskStore
from scaled_evals.api.tenancy import caller_is_admin
from task_fixtures import seed_benchmark_run, seed_evaluation, seed_task_revision

OWNER = "synthetic-owner-a"
OTHER = "synthetic-owner-b"
BASE = "/apis/scaled-evals/v1"
Method = Literal["GET", "POST", "DELETE"]


@dataclass
class Harness:
    client: TestClient
    entities: FakeEntityStore
    submissions: list[str]
    secrets: list[str]
    teardowns: list[str]
    artifacts: list[str]

    def request(self, method: Method, path: str, *, user: str = OWNER, admin: bool = False) -> httpx2.Response:
        return self.client.request(
            method,
            f"{BASE}/{path}",
            headers={"x-user": user, "x-admin": str(admin).lower()},
            json={} if method == "POST" else None,
        )

    def snapshot(self) -> dict[tuple[str, str, str], str]:
        return {key: entity.model_dump_json() for key, entity in self.entities.stored.items()}


@pytest.fixture
def harness(entity_store: FakeEntityStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    app = FastAPI()
    app.include_router(evaluations.router, prefix=BASE)
    app.include_router(benchmark_runs.router, prefix=BASE)
    submissions: list[str] = []
    secrets: list[str] = []
    teardowns: list[str] = []
    artifacts: list[str] = []

    def principal(request: Request) -> auth.CurrentPrincipal:
        return auth.CurrentPrincipal(owner_type="USER", owner_id=request.headers["x-user"], source="platform")

    def admin(request: Request) -> bool:
        return request.headers.get("x-admin") == "true"

    def secret_readable(reference: str) -> bool:
        secrets.append(reference)
        return True

    def teardown(_database: evaluations.Database, row: dict) -> dict:
        teardowns.append(row["id"])
        return row

    def objects(prefix: str) -> list[dict[str, str]]:
        artifacts.append(prefix)
        return []

    app.dependency_overrides[auth.current_principal] = principal
    app.dependency_overrides[caller_is_admin] = admin
    app.dependency_overrides[caller_secret_readable] = lambda: secret_readable
    monkeypatch.setattr(evaluations, "_submit_now", submissions.append)
    monkeypatch.setattr(evaluations, "teardown_cancelled_evaluation", teardown)
    monkeypatch.setattr(benchmark_runs, "teardown_cancelled_evaluation", teardown)
    monkeypatch.setattr(evaluations.artifacts, "list_objects", objects)
    seed_task_revision("task_x", 1)
    for evaluation_id, owner, visibility in (
        ("ev_a", OWNER, "private"),
        ("ev_b", OTHER, "private"),
        ("ev_public", OWNER, "public"),
        ("ev_team", OWNER, "team"),
        ("ev_org", OWNER, "org"),
        ("ev_null", None, "private"),
    ):
        seed_evaluation({"id": evaluation_id, "owner_id": owner, "visibility": visibility})
    with TestClient(app) as client:
        yield Harness(client, entity_store, submissions, secrets, teardowns, artifacts)


def _evaluation(evaluation_id: str) -> ScaledEvalsEvaluation:
    return entity_storage.entity_client().get(
        ScaledEvalsEvaluation, evaluation_id, workspace=entity_storage.private_workspace()
    )


def _run(run_id: str) -> ScaledEvalsBenchmarkRun:
    return entity_storage.entity_client().get(
        ScaledEvalsBenchmarkRun, run_id, workspace=entity_storage.private_workspace()
    )


def _seed_run(run_id: str, owner: str, visibility: str = "private", *, member_owner: str = OWNER) -> None:
    seed_benchmark_run(
        {
            "id": run_id,
            "name": run_id,
            "owner_id": owner,
            "framework": "harbor",
            "benchmark_id": "bm_x",
            "benchmark_revision": 1,
            "runtime": "sandbox_k8s",
            "parallelism": 1,
            "visibility": visibility,
        }
    )
    seed_evaluation(
        {"id": f"{run_id}_member", "owner_id": member_owner, "visibility": visibility, "benchmark_run_id": run_id}
    )


@pytest.mark.parametrize(("user", "evaluation_id"), [(OWNER, "ev_b"), (OTHER, "ev_a")])
@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("GET", ""),
        ("GET", "/telemetry"),
        ("GET", "/reproduce"),
        ("GET", "/logs"),
        ("GET", "/logs/stream"),
        ("GET", "/events"),
        ("GET", "/events/stream"),
        ("GET", "/artifacts"),
        ("GET", "/artifacts/private.txt"),
        ("GET", "/archive"),
        ("GET", "/archive/download"),
        ("GET", "/harbor-viewer/archive"),
        ("POST", "/retry"),
        ("POST", "/cancel"),
        ("POST", "/archive"),
        ("DELETE", ""),
    ],
)
def test_private_evaluation_denial_precedes_effects(
    harness: Harness, user: str, evaluation_id: str, method: Method, suffix: str
) -> None:
    before = harness.snapshot()
    response = harness.request(method, f"evaluations/{evaluation_id}{suffix}", user=user)
    assert response.status_code == 404, response.text
    assert harness.snapshot() == before
    assert not harness.submissions
    assert not harness.secrets
    assert not harness.teardowns
    assert not harness.artifacts


@pytest.mark.parametrize("evaluation_id", ["ev_public", "ev_team", "ev_org"])
@pytest.mark.parametrize(
    ("method", "suffix"), [("POST", "/cancel"), ("POST", "/retry"), ("POST", "/archive"), ("DELETE", "")]
)
def test_shared_read_does_not_authorize_mutation(
    harness: Harness, evaluation_id: str, method: Method, suffix: str
) -> None:
    assert harness.request("GET", f"evaluations/{evaluation_id}", user=OTHER).status_code == 200
    before = harness.snapshot()
    assert harness.request(method, f"evaluations/{evaluation_id}{suffix}", user=OTHER).status_code == 404
    assert harness.snapshot() == before
    assert not harness.secrets and not harness.submissions and not harness.teardowns


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize(("method", "suffix"), [("GET", ""), ("POST", "/cancel"), ("DELETE", "")])
def test_owner_and_admin_positive_controls(harness: Harness, admin: bool, method: Method, suffix: str) -> None:
    response = harness.request(method, f"evaluations/ev_a{suffix}", user=OTHER if admin else OWNER, admin=admin)
    assert response.status_code == 200, response.text
    entity = _evaluation("ev_a")
    if method == "DELETE":
        assert entity.deleted_at is not None
    elif method == "POST":
        assert entity.status == "cancelled"
        assert harness.teardowns == ["ev_a"]


def test_lists_apply_access_before_search_and_pagination(harness: Harness) -> None:
    response = harness.request("GET", "evaluations?limit=100")
    assert {row["id"] for row in response.json()["data"]} == {"ev_a", "ev_public", "ev_team", "ev_org"}
    assert harness.request("GET", "evaluations?q=ev_b").json()["data"] == []
    assert {row["id"] for row in harness.request("GET", "evaluations?mine=true").json()["data"]} == {
        "ev_a",
        "ev_public",
        "ev_team",
        "ev_org",
    }
    page = harness.request("GET", "evaluations?limit=1").json()
    assert len(page["data"]) == 1
    cursor = page["next_cursor"]
    following = harness.request("GET", f"evaluations?limit=100&cursor={cursor}").json()
    assert not {"ev_b", "ev_null"}.intersection(row["id"] for row in following["data"])
    assert {row["id"] for row in harness.request("GET", "evaluations?shared=true", user=OTHER).json()["data"]} == {
        "ev_public",
        "ev_team",
        "ev_org",
    }


def test_null_owner_is_admin_only_and_disabled_auth_remains_privileged(harness: Harness) -> None:
    assert harness.request("GET", "evaluations/ev_null").status_code == 404
    assert harness.request("GET", "evaluations/ev_null", admin=True).status_code == 200
    disabled = auth.CurrentPrincipal(owner_type="DEV", owner_id="dev", source="disabled")
    database = get_request_db(disabled, False)
    assert database.evaluations.get("ev_null") is not None


def test_denial_happens_before_task_enrichment(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(_store: TaskStore, _task_id: str, _revision: int) -> None:
        pytest.fail("unauthorized evaluation reached task enrichment")

    monkeypatch.setattr(TaskStore, "revision", unavailable)
    assert harness.request("GET", "evaluations/ev_b").status_code == 404


def test_retry_and_archive_check_write_access_before_secret_or_state_details(harness: Harness) -> None:
    entity = _evaluation("ev_public")
    entity.status = "failed"
    entity.credentials = {"openai": "private/credential"}
    entity.archive_status = "ready"
    entity.archive_object_key = "private/artifact"
    entity_storage.entity_client().update(entity)
    before = harness.snapshot()
    assert harness.request("POST", "evaluations/ev_public/retry", user=OTHER).status_code == 404
    assert harness.request("POST", "evaluations/ev_public/archive", user=OTHER).status_code == 404
    assert not harness.secrets and not harness.submissions
    assert harness.snapshot() == before


@pytest.mark.parametrize("visibility", ["private", "public"])
def test_cas_retry_rechecks_owner_before_mutating(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, visibility: str
) -> None:
    entity = _evaluation("ev_a")
    entity.visibility = visibility
    entity_storage.entity_client().update(entity)
    update = harness.entities.update
    raced = False

    def transfer_owner(entity: ScaledEvalsEvaluation) -> ScaledEvalsEvaluation:
        nonlocal raced
        if not raced and entity.name == "ev_a":
            raced = True
            current = _evaluation("ev_a")
            current.owner_id = OTHER
            update(current)
        return update(entity)

    monkeypatch.setattr(harness.entities, "update", transfer_owner)
    assert harness.request("POST", "evaluations/ev_a/cancel").status_code == 404
    stored = _evaluation("ev_a")
    assert stored.owner_id == OTHER and stored.status == "queued" and stored.events == []
    assert not harness.teardowns


@pytest.mark.parametrize("admin", [False, True])
def test_deleted_rows_are_not_request_visible_even_on_history_fallbacks(harness: Harness, admin: bool) -> None:
    entity = _evaluation("ev_a")
    entity.deleted_at = datetime.now(tz=UTC)
    entity_storage.entity_client().update(entity)
    database = RequestDatabase(ResourceAccess(OWNER, admin))
    assert database.evaluations.load_status_runtime("ev_a") is None
    assert database.evaluations.list_events("ev_a", limit=10, cursor=None, offset=0) == []
    assert harness.request("GET", "evaluations/ev_a/events", admin=admin).status_code == 404


def test_stream_factory_rechecks_live_scope_on_every_poll(harness: Harness) -> None:
    principal = auth.CurrentPrincipal(owner_type="USER", owner_id=OWNER, source="platform")
    factory = get_request_stream_database_factory(principal, False)
    with factory() as database:
        assert database.evaluations.load_stream_status("ev_a") == "queued"
    entity = _evaluation("ev_a")
    entity.owner_id = OTHER
    entity_storage.entity_client().update(entity)
    with factory() as database:
        assert database.evaluations.load_stream_status("ev_a") is None
        assert database.evaluations.load_event_batch("ev_a", limit=10) == []


@pytest.mark.parametrize(
    ("method", "suffix"),
    [
        ("GET", ""),
        ("GET", "/evaluations"),
        ("GET", "/reproduce"),
        ("GET", "/archive"),
        ("GET", "/archive/download"),
        ("POST", "/cancel"),
        ("POST", "/archive"),
        ("DELETE", ""),
    ],
)
def test_private_run_denial_precedes_member_or_archive_effects(harness: Harness, method: Method, suffix: str) -> None:
    _seed_run("run_other", OTHER, member_owner=OTHER)
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_other{suffix}").status_code == 404
    assert harness.snapshot() == before
    assert harness.teardowns == []


@pytest.mark.parametrize(("method", "suffix"), [("POST", "/cancel"), ("POST", "/archive"), ("DELETE", "")])
def test_public_run_is_readable_but_not_mutable_by_non_owner(harness: Harness, method: Method, suffix: str) -> None:
    _seed_run("run_public", OWNER, "public")
    assert harness.request("GET", "benchmark-runs/run_public", user=OTHER).status_code == 200
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_public{suffix}", user=OTHER).status_code == 404
    assert harness.snapshot() == before


def test_hidden_members_make_run_unreadable_and_unlisted(harness: Harness) -> None:
    _seed_run("run_mixed", OWNER, member_owner=OTHER)
    assert harness.request("GET", "benchmark-runs/run_mixed").status_code == 404
    assert harness.request("GET", "benchmark-runs").json()["data"] == []
    assert harness.request("GET", "benchmark-runs/run_mixed", admin=True).status_code == 200


@pytest.mark.parametrize(("method", "suffix"), [("POST", "/cancel"), ("POST", "/archive"), ("DELETE", "")])
def test_bulk_mutation_requires_write_access_to_every_member(harness: Harness, method: Method, suffix: str) -> None:
    _seed_run("run_mixed", OWNER, "public", member_owner=OTHER)
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_mixed{suffix}").status_code == 404
    assert harness.snapshot() == before
    assert harness.teardowns == []


def test_port_preserves_parent_only_deletion_without_adding_cascade(harness: Harness) -> None:
    _seed_run("run_owner", OWNER)
    assert harness.request("DELETE", "benchmark-runs/run_owner").status_code == 200
    assert _run("run_owner").deleted_at is not None
    assert _evaluation("run_owner_member").deleted_at is None
    assert harness.request("GET", "evaluations/run_owner_member").status_code == 200


def test_internal_workers_keep_explicit_unrestricted_access(harness: Harness) -> None:
    assert EvaluationStore().cancel("ev_b")[0] is not None
    assert _evaluation("ev_b").status == "cancelled"


@pytest.mark.parametrize("admin", [False, True])
def test_owner_and_admin_can_retry_without_changing_evaluation_identity(harness: Harness, admin: bool) -> None:
    entity = _evaluation("ev_a")
    entity.status = "failed"
    entity_storage.entity_client().update(entity)
    response = harness.request("POST", "evaluations/ev_a/retry", user=OTHER if admin else OWNER, admin=admin)
    assert response.status_code == 202, response.text
    entity = _evaluation("ev_a")
    assert entity.name == "ev_a" and entity.status == "queued" and entity.current_execution == 2
    assert harness.submissions == ["ev_a"]


@pytest.mark.parametrize("admin", [False, True])
def test_owner_and_admin_can_request_archive(harness: Harness, admin: bool) -> None:
    entity = _evaluation("ev_a")
    entity.status = "succeeded"
    entity_storage.entity_client().update(entity)
    response = harness.request("POST", "evaluations/ev_a/archive", user=OTHER if admin else OWNER, admin=admin)
    assert response.status_code == 202, response.text
    assert _evaluation("ev_a").archive_status == "building"


def test_terminal_cancel_fallback_does_not_grant_shared_mutation(harness: Harness) -> None:
    entity = _evaluation("ev_public")
    entity.status = "cancelled"
    entity_storage.entity_client().update(entity)
    before = harness.snapshot()
    assert harness.request("POST", "evaluations/ev_public/cancel", user=OTHER).status_code == 404
    assert harness.snapshot() == before
    assert not harness.teardowns


def test_admin_mine_and_shared_filters_only_narrow_access(harness: Harness) -> None:
    rows = harness.request("GET", "evaluations?mine=true", admin=True).json()["data"]
    assert {row["id"] for row in rows} == {"ev_a", "ev_public", "ev_team", "ev_org"}
    assert harness.request("GET", "evaluations?mine=true&shared=true", user=OTHER).json()["data"] == []


def test_delegated_request_scope_uses_the_effective_owner(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    context = AuthContext(principal_id="service:scaled-evals", principal_on_behalf_of=OWNER)
    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_auth_context", lambda: context)
    database = get_request_db(auth.current_principal(), False)
    assert database.evaluations.get("ev_a") is not None
    assert database.evaluations.get("ev_b") is None


@pytest.mark.parametrize("method", ["POST", "DELETE"])
@pytest.mark.parametrize("visibility", ["private", "public"])
def test_run_cas_retry_rechecks_owner(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, method: Method, visibility: str
) -> None:
    _seed_run("run_owner", OWNER, visibility)
    update = harness.entities.update
    raced = False

    def transfer_owner(entity: ScaledEvalsBenchmarkRun) -> ScaledEvalsBenchmarkRun:
        nonlocal raced
        if not raced and entity.name == "run_owner":
            raced = True
            current = _run("run_owner")
            current.owner_id = OTHER
            update(current)
        return update(entity)

    monkeypatch.setattr(harness.entities, "update", transfer_owner)
    suffix = "/cancel" if method == "POST" else ""
    assert harness.request(method, f"benchmark-runs/run_owner{suffix}").status_code == 404
    stored = _run("run_owner")
    assert stored.owner_id == OTHER and stored.cancelled_at is None and stored.deleted_at is None
    assert _evaluation("run_owner_member").status == "queued"
    assert not harness.teardowns


def test_admin_can_cancel_another_owners_run_and_members(harness: Harness) -> None:
    _seed_run("run_other", OTHER, member_owner=OTHER)
    response = harness.request("POST", "benchmark-runs/run_other/cancel", admin=True)
    assert response.status_code == 200, response.text
    assert _run("run_other").cancelled_at is not None
    assert _evaluation("run_other_member").status == "cancelled"
    assert harness.teardowns == ["run_other_member"]


def test_log_stream_stops_cleanly_when_access_is_revoked(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def log_lines(row: Mapping[str, object], *, tail_lines: int | None) -> list[str]:
        assert tail_lines is None
        calls.append(str(row["id"]))
        entity = _evaluation("ev_a")
        entity.owner_id = OTHER
        entity_storage.entity_client().update(entity)
        return ["authorized log batch"]

    monkeypatch.setattr(evaluations, "collect_log_lines", log_lines)
    monkeypatch.setattr(evaluations.settings, "api_sse_poll_interval_seconds", 0)
    monkeypatch.setattr(evaluations.settings, "api_sse_max_duration_seconds", 1)
    response = harness.request("GET", "evaluations/ev_a/logs/stream")
    assert response.status_code == 200
    assert response.text.count("authorized log batch") == 1
    assert calls == ["ev_a"]
    assert evaluations._sse_active_connections == 0
