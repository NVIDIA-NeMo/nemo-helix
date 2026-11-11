# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request isolation with in-memory entities and controlled authorization decisions."""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import signature
from typing import Literal
from unittest.mock import patch

import httpx
import httpx2
import pytest

pytest.importorskip("scaled_evals")

from fake_entity_store import FakeEntityStore
from fastapi import FastAPI, HTTPException, Request
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.dependencies import get_request_authorizer
from nemo_helix_plugin.entities.base import EntityT, SyncEntityClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmarkRun, ScaledEvalsEvaluation, ScaledEvalsEvaluationEvent
from scaled_evals.api import auth
from scaled_evals.api import benchmark_run_store as run_storage
from scaled_evals.api import entity_store as entity_storage
from scaled_evals.api.benchmark_archive_store import BenchmarkArchiveRequestRepository
from scaled_evals.api.benchmark_run_store import (
    BenchmarkRunGlobalQueries,
    BenchmarkRunRequestRepository,
    BenchmarkRunWorkerRepository,
    enrich_run_reads,
)
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.db import (
    RequestDatabase,
    get_db,
    get_evaluation_global_queries,
    get_request_db,
    get_request_stream_database_factory,
)
from scaled_evals.api.evaluation_store import (
    EvaluationGlobalQueries,
    EvaluationRequestRepository,
    EvaluationWorkerRepository,
)
from scaled_evals.api.resource_access import ResourceAccess
from scaled_evals.api.routers import benchmark_runs, config_profiles, evaluations, tasks, users
from scaled_evals.api.runnability import caller_secret_readable
from scaled_evals.api.task_store import TaskStore
from scaled_evals.api.tenancy import caller_is_admin
from task_fixtures import seed_benchmark_run, seed_evaluation, seed_task_revision

OWNER = "synthetic-owner-a"
OTHER = "synthetic-owner-b"
BASE = "/apis/scaled-evals/v1"
Method = Literal["GET", "POST", "DELETE"]
RunChange = Literal["owner_transfer", "deleted_parent", "hidden_member", "visibility", "metadata"]


@dataclass
class Harness:
    client: TestClient
    entities: FakeEntityStore
    submissions: list[str]
    secrets: list[str]
    teardowns: list[str]
    artifacts: list[str]
    app: FastAPI

    def request(self, method: Method, path: str, *, user: str = OWNER, admin: bool = False) -> httpx2.Response:
        """Call a real route with controlled principal and admin decisions."""
        return self.client.request(
            method,
            f"{BASE}/{path}",
            headers={"x-user": user, "x-admin": str(admin).lower()},
            json={} if method == "POST" else None,
        )

    def snapshot(self) -> dict[tuple[str, str, str], str]:
        """Capture persisted entity state so denied requests can prove no writes occurred."""
        return {key: entity.model_dump_json() for key, entity in self.entities.stored.items()}


@pytest.fixture
def harness(entity_store: FakeEntityStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """Exercise request-scoped stores over isolated entities with external effects recorded."""
    app = FastAPI()
    app.include_router(evaluations.router, prefix=BASE)
    app.include_router(benchmark_runs.router, prefix=BASE)
    submissions: list[str] = []
    secrets: list[str] = []
    teardowns: list[str] = []
    artifacts: list[str] = []

    def principal(request: Request) -> auth.CurrentPrincipal:
        """Resolve the synthetic effective owner without exercising platform authentication."""
        return auth.CurrentPrincipal(owner_type="USER", owner_id=request.headers["x-user"], source="platform")

    def admin(request: Request) -> bool:
        """Supply a controlled admin decision rather than claiming real role verification."""
        return request.headers.get("x-admin") == "true"

    def secret_readable(reference: str) -> bool:
        """Record secret checks so denial tests can detect unauthorized lookups."""
        secrets.append(reference)
        return True

    def teardown(_database: evaluations.RequestDatabase, row: dict) -> dict:
        """Record runtime teardown attempts without reaching an external backend."""
        teardowns.append(row["id"])
        return row

    def objects(prefix: str) -> list[dict[str, str]]:
        """Record object-storage reads without reaching an external service."""
        artifacts.append(prefix)
        return []

    app.dependency_overrides[auth.current_principal] = principal
    app.dependency_overrides[caller_is_admin] = admin
    app.dependency_overrides[caller_secret_readable] = lambda: secret_readable
    monkeypatch.setattr(evaluations, "_submit_now", submissions.append)
    monkeypatch.setattr(evaluations, "teardown_cancelled_evaluation", teardown)

    def teardown_runtime(row: dict) -> str | None:
        """Record cleanup while the accepted cancellation persists its own result."""
        teardowns.append(row["id"])
        return None

    monkeypatch.setattr(benchmark_runs, "teardown_cancelled_runtime", teardown_runtime)
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
        yield Harness(client, entity_store, submissions, secrets, teardowns, artifacts, app)


def _evaluation(evaluation_id: str) -> ScaledEvalsEvaluation:
    """Read persisted evaluation state through the privileged fixture client."""
    return entity_storage.entity_client().get(
        ScaledEvalsEvaluation, evaluation_id, workspace=entity_storage.private_workspace()
    )


def _run(run_id: str) -> ScaledEvalsBenchmarkRun:
    """Read persisted run state through the privileged fixture client."""
    return entity_storage.entity_client().get(
        ScaledEvalsBenchmarkRun, run_id, workspace=entity_storage.private_workspace()
    )


def _members_for_runs(store: BenchmarkRunRequestRepository, run_ids: list[str]) -> list[dict]:
    """Flatten member snapshots without losing eligibility in the production API."""
    return [member for read in enrich_run_reads(list(store.read_for_runs(run_ids).values())) for member in read.members]


def _seed_run(run_id: str, owner: str, visibility: str = "private", *, member_owner: str = OWNER) -> None:
    """Store one run and member with independently selectable ownership."""
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
    """Deny private access in both user directions without writes or external effects."""
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
    """Shared visibility allows reads while preserving owner-only mutations."""
    assert harness.request("GET", f"evaluations/{evaluation_id}", user=OTHER).status_code == 200
    before = harness.snapshot()
    assert harness.request(method, f"evaluations/{evaluation_id}{suffix}", user=OTHER).status_code == 404
    assert harness.snapshot() == before
    assert not harness.secrets and not harness.submissions and not harness.teardowns


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize(("method", "suffix"), [("GET", ""), ("POST", "/cancel"), ("DELETE", "")])
def test_owner_and_admin_positive_controls(harness: Harness, admin: bool, method: Method, suffix: str) -> None:
    """Owners and authorized admins retain read, cancel and delete access."""
    response = harness.request(method, f"evaluations/ev_a{suffix}", user=OTHER if admin else OWNER, admin=admin)
    assert response.status_code == 200, response.text
    entity = _evaluation("ev_a")
    if method == "DELETE":
        assert entity.deleted_at is not None
    elif method == "POST":
        assert entity.status == "cancelled"
        assert harness.teardowns == ["ev_a"]


def test_lists_apply_access_before_search_and_pagination(harness: Harness) -> None:
    """Default lists, search and every page exclude inaccessible private rows."""
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
    """Private legacy rows require admin access unless authentication is disabled."""
    assert harness.request("GET", "evaluations/ev_null").status_code == 404
    assert harness.request("GET", "evaluations/ev_null", admin=True).status_code == 200
    disabled = auth.CurrentPrincipal(owner_type="DEV", owner_id="dev", source="disabled")
    database = get_request_db(disabled, False)
    assert database.evaluations.get("ev_null") is not None


def test_denial_happens_before_task_enrichment(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """A private denial must not reach the independently authorized task store."""

    def unavailable(_store: TaskStore, _task_id: str, _revision: int) -> None:
        pytest.fail("unauthorized evaluation reached task enrichment")

    monkeypatch.setattr(TaskStore, "revision", unavailable)
    assert harness.request("GET", "evaluations/ev_b").status_code == 404


def test_retry_and_archive_check_write_access_before_secret_or_state_details(harness: Harness) -> None:
    """Reject shared mutations before credential lookup or ready-archive shortcuts."""
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
    """A conflicting ownership transfer must revoke the retried evaluation write."""
    entity = _evaluation("ev_a")
    entity.visibility = visibility
    entity_storage.entity_client().update(entity)
    update = harness.entities.update
    raced = False

    def transfer_owner(entity: ScaledEvalsEvaluation) -> ScaledEvalsEvaluation:
        """Change stored ownership before the first write to force an authorization retry."""
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
    """Request history reads cannot recover deleted rows, including for admins."""
    entity = _evaluation("ev_a")
    entity.deleted_at = datetime.now(tz=UTC)
    entity_storage.entity_client().update(entity)
    database = RequestDatabase(ResourceAccess(OWNER, admin))
    assert database.evaluations.load_status_runtime("ev_a") is None
    assert database.evaluations.list_events("ev_a", limit=10, cursor=None, offset=0) == []
    assert harness.request("GET", "evaluations/ev_a/events", admin=admin).status_code == 404


def test_stream_factory_rechecks_live_scope_on_every_poll(harness: Harness) -> None:
    """Each stream poll applies the retained scope to fresh resource state."""
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
    """Deny private run routes before exposing members or accessing archives."""
    _seed_run("run_other", OTHER, member_owner=OTHER)
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_other{suffix}").status_code == 404
    assert harness.snapshot() == before
    assert harness.teardowns == []


@pytest.mark.parametrize(("method", "suffix"), [("POST", "/cancel"), ("POST", "/archive"), ("DELETE", "")])
def test_public_run_is_readable_but_not_mutable_by_non_owner(harness: Harness, method: Method, suffix: str) -> None:
    """Sharing a run does not grant other owners mutation access."""
    _seed_run("run_public", OWNER, "public")
    assert harness.request("GET", "benchmark-runs/run_public", user=OTHER).status_code == 200
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_public{suffix}", user=OTHER).status_code == 404
    assert harness.snapshot() == before


def test_hidden_members_make_run_unreadable_and_unlisted(harness: Harness) -> None:
    """One inaccessible live member hides its parent run from ordinary readers."""
    _seed_run("run_mixed", OWNER, member_owner=OTHER)
    assert harness.request("GET", "benchmark-runs/run_mixed").status_code == 404
    assert harness.request("GET", "benchmark-runs").json()["data"] == []
    assert harness.request("GET", "benchmark-runs/run_mixed", admin=True).status_code == 200


@pytest.mark.parametrize(("method", "suffix"), [("POST", "/cancel"), ("POST", "/archive"), ("DELETE", "")])
def test_bulk_mutation_requires_write_access_to_every_member(harness: Harness, method: Method, suffix: str) -> None:
    """Reject mixed-owner run mutations without changing any parent or member."""
    _seed_run("run_mixed", OWNER, "public", member_owner=OTHER)
    before = harness.snapshot()
    assert harness.request(method, f"benchmark-runs/run_mixed{suffix}").status_code == 404
    assert harness.snapshot() == before
    assert harness.teardowns == []


def test_port_preserves_parent_only_deletion_without_adding_cascade(harness: Harness) -> None:
    """Run deletion retains the existing independently accessible member entity."""
    _seed_run("run_owner", OWNER)
    assert harness.request("DELETE", "benchmark-runs/run_owner").status_code == 200
    assert _run("run_owner").deleted_at is not None
    assert _evaluation("run_owner_member").deleted_at is None
    assert harness.request("GET", "evaluations/run_owner_member").status_code == 200


def test_internal_workers_keep_explicit_unrestricted_access(harness: Harness) -> None:
    """Internal storage remains privileged for background lifecycle work."""
    assert EvaluationWorkerRepository().cancel("ev_b")[0] is not None
    assert _evaluation("ev_b").status == "cancelled"


@pytest.mark.parametrize("admin", [False, True])
def test_owner_and_admin_can_retry_without_changing_evaluation_identity(harness: Harness, admin: bool) -> None:
    """Authorized retries retain identity and queue the next execution."""
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
    """Authorized owners and admins can request archive generation."""
    entity = _evaluation("ev_a")
    entity.status = "succeeded"
    entity_storage.entity_client().update(entity)
    response = harness.request("POST", "evaluations/ev_a/archive", user=OTHER if admin else OWNER, admin=admin)
    assert response.status_code == 202, response.text
    assert _evaluation("ev_a").archive_status == "building"


def test_terminal_cancel_fallback_does_not_grant_shared_mutation(harness: Harness) -> None:
    """Idempotent cancellation still requires write access to a shared evaluation."""
    entity = _evaluation("ev_public")
    entity.status = "cancelled"
    entity_storage.entity_client().update(entity)
    before = harness.snapshot()
    assert harness.request("POST", "evaluations/ev_public/cancel", user=OTHER).status_code == 404
    assert harness.snapshot() == before
    assert not harness.teardowns


def test_admin_mine_and_shared_filters_only_narrow_access(harness: Harness) -> None:
    """Optional list filters narrow access without overriding authorization."""
    rows = harness.request("GET", "evaluations?mine=true", admin=True).json()["data"]
    assert {row["id"] for row in rows} == {"ev_a", "ev_public", "ev_team", "ev_org"}
    assert harness.request("GET", "evaluations?mine=true&shared=true", user=OTHER).json()["data"] == []


def test_delegated_request_scope_uses_the_effective_owner(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """A service acting on behalf of a user receives that user's ownership scope."""
    context = AuthContext(principal_id="service:scaled-evals", principal_on_behalf_of=OWNER)
    monkeypatch.setattr(auth, "platform_auth_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_auth_context", lambda: context)
    database = get_request_db(auth.current_principal(), False)
    assert database.evaluations.get("ev_a") is not None
    assert database.evaluations.get("ev_b") is None


@pytest.mark.parametrize("method", ["POST", "DELETE"])
@pytest.mark.parametrize("visibility", ["private", "public"])
def test_run_cas_retry_preserves_cancellation_grant_and_rechecks_deletion(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, method: Method, visibility: str
) -> None:
    """Accepted cancellation survives ownership changes; deletion still reauthorizes on retry."""
    _seed_run("run_owner", OWNER, visibility)
    update = harness.entities.update
    raced = False

    def transfer_owner(entity: ScaledEvalsBenchmarkRun) -> ScaledEvalsBenchmarkRun:
        """Change stored ownership before the first write to force an authorization retry."""
        nonlocal raced
        if not raced and entity.name == "run_owner":
            raced = True
            current = _run("run_owner")
            current.owner_id = OTHER
            update(current)
        return update(entity)

    monkeypatch.setattr(harness.entities, "update", transfer_owner)
    suffix = "/cancel" if method == "POST" else ""
    response = harness.request(method, f"benchmark-runs/run_owner{suffix}")
    assert response.status_code == (200 if method == "POST" else 404), response.text
    stored = _run("run_owner")
    assert stored.owner_id == OTHER and stored.deleted_at is None
    if method == "POST":
        assert stored.cancelled_at is not None
        assert _evaluation("run_owner_member").status == "cancelled"
        assert harness.teardowns == ["run_owner_member"]
    else:
        assert stored.cancelled_at is None
        assert _evaluation("run_owner_member").status == "queued"
        assert not harness.teardowns


def test_admin_can_cancel_another_owners_run_and_members(harness: Harness) -> None:
    """Authorized admins can cancel both a foreign run and its active members."""
    _seed_run("run_other", OTHER, member_owner=OTHER)
    response = harness.request("POST", "benchmark-runs/run_other/cancel", admin=True)
    assert response.status_code == 200, response.text
    assert _run("run_other").cancelled_at is not None
    assert _evaluation("run_other_member").status == "cancelled"
    assert harness.teardowns == ["run_other_member"]


def test_log_stream_stops_cleanly_when_access_is_revoked(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """A log stream closes after revocation without leaking another batch or its slot."""
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


@pytest.mark.parametrize("admin", [False, True])
def test_run_search_batches_authorization_before_pagination(harness: Harness, admin: bool) -> None:
    """Searching more than one store page must not fetch every run individually."""
    for index in range(103):
        _seed_run(f"run_{index}", OWNER)
    _seed_run("run_hidden_member", OWNER, member_owner=OTHER)
    _seed_run("run_hidden_parent", OTHER)
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, admin))
    with (
        patch.object(harness.entities, "get", wraps=harness.entities.get) as get,
        patch.object(run_storage, "derive_run_view", wraps=run_storage.derive_run_view) as rollup,
    ):
        rows = store.list(limit=1, cursor=None, order="asc", q="running")
    assert len(rows) == 2
    assert {row["id"] for row in rows} == {"run_0", "run_1"}
    assert get.call_count == 1  # One shared task, regardless of how many runs match.
    assert rollup.call_count == 2  # Full diagnostics only for the page and cursor lookahead.
    with patch.object(harness.entities, "get", wraps=harness.entities.get) as get:
        full = store.list(limit=200, cursor=None, order="asc", q="running")
    ids = {row["id"] for row in full}
    assert len(ids) == (105 if admin else 103)
    for run_id in ("run_hidden_member", "run_hidden_parent"):
        assert (run_id in ids) is admin
    assert get.call_count == 1


def test_batched_members_recheck_live_parents_and_hidden_members(harness: Harness) -> None:
    """A fresh batch must exclude revoked/deleted parents and all members of a hidden run."""
    for run_id in ("run_allowed", "run_revoked", "run_deleted", "run_hidden_member"):
        _seed_run(run_id, OWNER)
    run_ids = ["run_allowed", "run_revoked", "run_deleted", "run_hidden_member", "run_missing"]
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    assert len(_members_for_runs(store, run_ids)) == 4
    revoked = _run("run_revoked")
    revoked.owner_id = OTHER
    harness.entities.update(revoked)
    deleted = _run("run_deleted")
    deleted.deleted_at = datetime.now(tz=UTC)
    harness.entities.update(deleted)
    hidden = _evaluation("run_hidden_member_member")
    hidden.owner_id = OTHER
    harness.entities.update(hidden)
    # A second readable member must not leak from a run with one private foreign member.
    seed_evaluation({"id": "ev_visible_sibling", "owner_id": OWNER, "benchmark_run_id": "run_hidden_member"})
    assert [row["id"] for row in _members_for_runs(store, run_ids)] == ["run_allowed_member"]
    assert {row["id"] for row in store.list(limit=100, cursor=None, order="asc", q="running")} == {"run_allowed"}


def test_batched_members_allow_shared_parents_and_ignore_deleted_members(harness: Harness) -> None:
    """Batch reads preserve shared access and ignore inaccessible members already deleted."""
    _seed_run("run_shared", OTHER, "public", member_owner=OTHER)
    _seed_run("run_live", OWNER)
    seed_evaluation({"id": "ev_deleted_foreign", "owner_id": OTHER, "benchmark_run_id": "run_live"})
    deleted = _evaluation("ev_deleted_foreign")
    deleted.deleted_at = datetime.now(tz=UTC)
    deleted.deleted = True
    harness.entities.update(deleted)
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    assert {row["id"] for row in _members_for_runs(store, ["run_shared", "run_live"])} == {
        "run_shared_member",
        "run_live_member",
    }


def test_member_batch_queries_only_requested_parents(harness: Harness) -> None:
    """Reading one run must not scan unrelated parents or issue per-run GETs."""
    for index in range(103):
        _seed_run(f"run_unrelated_{index}", OWNER)
    _seed_run("run_requested", OWNER)
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    with (
        patch.object(harness.entities, "get", wraps=harness.entities.get) as get,
        patch.object(harness.entities, "list", wraps=harness.entities.list) as scan,
    ):
        members = _members_for_runs(store, ["run_requested"])
    assert [row["id"] for row in members] == ["run_requested_member"]
    assert get.call_count == 1  # Only task enrichment needs an individual read.
    assert scan.call_count == 2  # One filtered parent page and one member page.


@pytest.mark.parametrize("query", [None, "running"])
def test_large_run_filters_are_bounded_before_pagination(harness: Harness, query: str | None) -> None:
    """Large lists and searches keep every filter URL bounded and retain every member page."""
    run_ids = [f"bmr_{index:032x}" for index in range(2000)]
    for run_id in run_ids:
        seed_benchmark_run(
            {
                "id": run_id,
                "name": run_id,
                "owner_id": OWNER,
                "framework": "harbor",
                "benchmark_id": "bm_x",
                "benchmark_revision": 1,
                "runtime": "sandbox_k8s",
                "parallelism": 1,
                "visibility": "private",
            }
        )
    # The first batch spans several member pages; the last batch must also survive.
    for index in range(205):
        seed_evaluation({"id": f"ev_batch_{index}", "owner_id": OWNER, "benchmark_run_id": run_ids[0]})
    seed_evaluation({"id": "ev_last_batch", "owner_id": OWNER, "benchmark_run_id": run_ids[-1]})
    seed_evaluation({"id": "ev_hidden_batch", "owner_id": OTHER, "benchmark_run_id": run_ids[1]})
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    with patch.object(entity_storage, "_list_matching", wraps=entity_storage._list_matching) as scans:
        rows = store.list(limit=1, cursor=None, order="asc", q=query)
        full = store.list(limit=len(run_ids), cursor=None, order="asc", q=query)
        members = _members_for_runs(store, [run_ids[0], run_ids[-1], run_ids[0]])
    assert [row["id"] for row in rows] == [run_ids[0], run_ids[2]]
    assert [row["id"] for row in full] == [run_id for run_id in run_ids if run_id != run_ids[1]]
    assert {row["id"] for row in members} == {"ev_last_batch", *(f"ev_batch_{i}" for i in range(205))}
    assert len(members) == 206
    filters = [request.args[2] for request in scans.call_args_list if request.args[2] is not None]
    pending = list(filters)
    bounded_fields: set[str] = set()
    while pending:
        operation = pending.pop()
        if isinstance(operation, LogicalOperation):
            pending.extend(operation.operations)
        elif isinstance(operation, ComparisonOperation) and operation.operator == FilterOperator.IN:
            assert isinstance(operation.value, list)
            assert len(operation.value) <= entity_storage.FILTER_BATCH_SIZE
            bounded_fields.add(operation.field)
    assert {"name", "data.benchmark_run_id"} <= bounded_fields
    for match in filters:
        # Also enforce the actual HTTPX query limit, independently of batch tuning.
        httpx.URL("http://entity-store.test/entities", params={"filter": json.dumps(match.to_dict())})


def test_empty_run_batches_do_not_issue_unfiltered_reads(harness: Harness) -> None:
    """Empty parent/member filters return no rows without reading unrelated entities."""
    with patch.object(entity_storage, "_list_matching", wraps=entity_storage._list_matching) as scans:
        assert entity_storage.list_by_names(entity_storage.entity_client(), ScaledEvalsBenchmarkRun, []) == []
        assert EvaluationWorkerRepository().list_for_runs([]) == []
        assert _members_for_runs(BenchmarkRunWorkerRepository(), []) == []
    scans.assert_not_called()


def _revoke_before_parent_batch(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, run_id: str, change: RunChange
) -> list[str]:
    """Change stored access or metadata once, immediately before the fresh parent query."""
    original = entity_storage.list_by_names
    changed: list[str] = []

    def change_before_batch(
        client: SyncEntityClient, entity_type: type[EntityT], names: Sequence[str]
    ) -> list[EntityT]:
        """Interleave the write without replacing the production authorization checks."""
        if entity_type is ScaledEvalsBenchmarkRun and run_id in names and not changed:
            changed.append(change)
            if change == "hidden_member":
                member = _evaluation(f"{run_id}_member")
                member.owner_id = OTHER
                harness.entities.update(member)
            else:
                parent = _run(run_id)
                if change == "owner_transfer":
                    parent.owner_id = OTHER
                elif change == "deleted_parent":
                    parent.deleted_at = datetime.now(tz=UTC)
                elif change == "metadata":
                    parent.display_name = "updated run"
                    parent.cancelled_at = datetime.now(tz=UTC)
                else:
                    parent.visibility = "private"
                harness.entities.update(parent)
        return original(client, entity_type, names)

    monkeypatch.setattr(entity_storage, "list_by_names", change_before_batch)
    return changed


@pytest.mark.parametrize("change", ["owner_transfer", "deleted_parent", "hidden_member", "visibility"])
@pytest.mark.parametrize("through_router", [False, True])
@pytest.mark.parametrize("query", [None, "running"])
def test_run_revocation_between_scan_and_batch_excludes_parent_before_pagination(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, change: RunChange, through_router: bool, query: str | None
) -> None:
    """A fresh denial removes the parent before search, page membership and response enrichment."""
    if change == "visibility":
        _seed_run("run_a", OTHER, "public", member_owner=OTHER)
    else:
        _seed_run("run_a", OWNER)
    _seed_run("run_b", OWNER)
    _seed_run("run_c", OWNER)
    changed = _revoke_before_parent_batch(harness, monkeypatch, "run_a", change)
    if through_router:
        path = "benchmark-runs?limit=1&order=asc" + (f"&q={query}" if query else "")
        response = harness.request("GET", path)
        assert response.status_code == 200, response.text
        page = response.json()
        assert [row["id"] for row in page["data"]] == ["run_b"]
        assert page["next_cursor"] is not None
        following = harness.request("GET", f"{path}&cursor={page['next_cursor']}")
        assert [row["id"] for row in following.json()["data"]] == ["run_c"]
    else:
        store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
        rows = store.list(limit=1, cursor=None, order="asc", q=query)
        assert [row["id"] for row in rows] == ["run_b", "run_c"]
    assert changed


def test_readable_empty_run_is_not_confused_with_denial(harness: Harness) -> None:
    """An authorized parent with no live members remains readable and searchable."""
    _seed_run("run_empty", OWNER)
    harness.entities.delete(ScaledEvalsEvaluation, "run_empty_member", workspace=entity_storage.private_workspace())
    for path in ("benchmark-runs", "benchmark-runs?q=running", "benchmark-runs/run_empty"):
        response = harness.request("GET", path)
        assert response.status_code == 200, response.text
        rows = response.json()["data"] if "data" in response.json() else [response.json()]
        assert [row["id"] for row in rows] == ["run_empty"]
        assert rows[0]["status"] == "running"


@pytest.mark.parametrize("change", ["owner_transfer", "deleted_parent", "hidden_member", "visibility"])
@pytest.mark.parametrize("suffix", ["", "/reproduce"])
def test_run_response_rechecks_eligibility(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, change: RunChange, suffix: str
) -> None:
    """Direct responses and reproduction deny a parent revoked before the final batch read."""
    if change == "visibility":
        _seed_run("run_a", OTHER, "public", member_owner=OTHER)
    else:
        _seed_run("run_a", OWNER)
    changed = _revoke_before_parent_batch(harness, monkeypatch, "run_a", change)
    response = harness.request("GET", f"benchmark-runs/run_a{suffix}")
    assert response.status_code == 404, response.text
    assert changed


@pytest.mark.parametrize(("query", "visible"), [("running", False), ("cancelled", True), ("updated", True)])
def test_search_uses_fresh_parent_metadata(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, query: str, visible: bool
) -> None:
    """Search and derived status use the eligible parent reread, rather than the discovery row."""
    _seed_run("run_a", OWNER)
    changed = _revoke_before_parent_batch(harness, monkeypatch, "run_a", "metadata")
    response = harness.request("GET", f"benchmark-runs?q={query}")
    assert response.status_code == 200, response.text
    rows = response.json()["data"]
    assert [row["id"] for row in rows] == (["run_a"] if visible else [])
    if visible:
        assert rows[0]["name"] == "updated run"
        assert rows[0]["status"] == "cancelled"
    assert changed


def test_shared_filter_uses_fresh_parent_visibility(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """An owner retaining access still sees a newly private run excluded from a shared-only list."""
    _seed_run("run_a", OWNER, "public")
    changed = _revoke_before_parent_batch(harness, monkeypatch, "run_a", "visibility")
    response = harness.request("GET", "benchmark-runs?shared=true")
    assert response.status_code == 200, response.text
    assert response.json()["data"] == []
    assert changed


def test_default_run_list_enriches_only_the_selected_page(harness: Harness) -> None:
    """Authorization checks all candidates while task hydration stays bounded to the page."""
    for index in range(103):
        task_id = f"task_page_{index}"
        seed_task_revision(task_id)
        _seed_run(f"run_page_{index}", OWNER)
        member = _evaluation(f"run_page_{index}_member")
        member.task_id = task_id
        harness.entities.update(member)
    with patch.object(harness.entities, "get", wraps=harness.entities.get) as get:
        response = harness.request("GET", "benchmark-runs?limit=1&order=asc")
    assert response.status_code == 200, response.text
    assert [row["id"] for row in response.json()["data"]] == ["run_page_0"]
    assert get.call_count == 2  # The returned row and one cursor lookahead row.


@pytest.mark.parametrize("through_router", [False, True])
def test_run_cancel_reuses_member_authorization_reads(harness: Harness, through_router: bool) -> None:
    """Cancellation reads membership once, then refreshes only the response records after cleanup."""
    _seed_run("run_cancel", OWNER)
    with patch.object(harness.entities, "list", wraps=harness.entities.list) as scans:
        if through_router:
            response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "cancelled"
        else:
            cancellation = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False)).cancel("run_cancel")
            assert cancellation is not None and cancellation.cancelled_now
            assert list(cancellation.cancelled_members) == ["run_cancel_member"]
    member_scans = [call for call in scans.call_args_list if call.args[0] is ScaledEvalsEvaluation]
    assert len(member_scans) == (2 if through_router else 1)
    assert _evaluation("run_cancel_member").status == "cancelled"


@pytest.mark.parametrize("change", ["owner_transfer", "deleted_parent", "hidden_member"])
def test_run_cancel_keeps_initial_authorization_after_marking_parent(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, change: RunChange
) -> None:
    """An accepted cancellation finishes for its authorized IDs even when parent or member access changes."""
    _seed_run("run_cancel", OWNER)
    update = harness.entities.update
    raced = False

    def revoke_after_mark(entity: EntityT) -> EntityT:
        """Commit the marker, then interleave revocation before the fresh member read."""
        nonlocal raced
        saved = update(entity)
        if isinstance(entity, ScaledEvalsBenchmarkRun) and not raced:
            raced = True
            if change == "hidden_member":
                member = _evaluation("run_cancel_member")
                member.owner_id = OTHER
                member.visibility = "public"  # Readable does not grant cancellation.
                update(member)
            else:
                parent = _run("run_cancel")
                if change == "owner_transfer":
                    parent.owner_id = OTHER
                    parent.visibility = "public"
                else:
                    parent.deleted_at = datetime.now(tz=UTC)
                update(parent)
        return saved

    monkeypatch.setattr(harness.entities, "update", revoke_after_mark)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert raced
    assert _evaluation("run_cancel_member").status == "cancelled"
    assert harness.teardowns == ["run_cancel_member"]


def test_run_cancel_retries_fresh_member_versions_under_initial_grant(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Member retries preserve current ownership while completing the initially authorized cancellation."""
    _seed_run("run_cancel", OWNER)
    update = harness.entities.update
    raced = False

    def transfer_member(entity: EntityT) -> EntityT:
        """Force a member compare-and-swap retry with a newly inaccessible owner."""
        nonlocal raced
        if isinstance(entity, ScaledEvalsEvaluation) and entity.name == "run_cancel_member" and not raced:
            raced = True
            member = _evaluation(entity.name)
            member.owner_id = OTHER
            update(member)
        return update(entity)

    monkeypatch.setattr(harness.entities, "update", transfer_member)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert raced
    assert _evaluation("run_cancel_member").status == "cancelled"
    assert harness.teardowns == ["run_cancel_member"]
    assert _evaluation("run_cancel_member").owner_id == OTHER


def test_run_cancel_completes_cleanup_after_parent_ownership_changes(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup keeps the accepted grant, while subsequent requests use the new ownership."""
    _seed_run("run_cancel", OWNER)
    update = harness.entities.update
    raced = False

    def revoke_after_member(entity: EntityT) -> EntityT:
        """Transfer the private parent after the member write succeeds."""
        nonlocal raced
        saved = update(entity)
        if isinstance(entity, ScaledEvalsEvaluation) and entity.name == "run_cancel_member" and not raced:
            raced = True
            parent = _run("run_cancel")
            parent.owner_id = OTHER
            update(parent)
        return saved

    monkeypatch.setattr(harness.entities, "update", revoke_after_member)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert raced
    assert _evaluation("run_cancel_member").status == "cancelled"
    assert _evaluation("run_cancel_member").cancel_teardown_status == "succeeded"
    assert harness.teardowns == ["run_cancel_member"]
    assert harness.request("GET", "benchmark-runs/run_cancel").status_code == 404


def test_run_get_authorizes_once_before_enrichment(harness: Harness) -> None:
    """A detail GET uses one fresh parent/member authorization read and one task lookup."""
    _seed_run("run_detail", OWNER)
    with (
        patch.object(harness.entities, "get", wraps=harness.entities.get) as get,
        patch.object(harness.entities, "list", wraps=harness.entities.list) as scans,
    ):
        response = harness.request("GET", "benchmark-runs/run_detail")
    assert response.status_code == 200, response.text
    assert response.json()["id"] == "run_detail"
    assert len(response.json()["result"]["per_task"]) == 1
    assert get.call_count == 1  # Task metadata only; no preliminary parent GET.
    assert len([call for call in scans.call_args_list if call.args[0] is ScaledEvalsEvaluation]) == 1


@pytest.mark.parametrize(
    ("status", "reward", "cancelled", "expected"),
    [
        ("queued", None, False, "running"),
        ("running", None, False, "running"),
        ("succeeded", 0.0, False, "succeeded"),
        ("succeeded", None, False, "failed"),
        ("failed", None, False, "failed"),
        ("cancelled", None, False, "failed"),
        ("queued", None, True, "cancelled"),
    ],
)
def test_run_status_search_preserves_detail_status(
    harness: Harness,
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"],
    reward: float | None,
    cancelled: bool,
    expected: str,
) -> None:
    """Status search preserves terminal, missing-score, zero-score and explicit-cancel behavior."""
    _seed_run("run_status", OWNER)
    member = _evaluation("run_status_member")
    member.status, member.reward = status, reward
    harness.entities.update(member)
    if cancelled:
        parent = _run("run_status")
        parent.cancelled_at = datetime.now(tz=UTC)
        harness.entities.update(parent)
    detail = harness.request("GET", "benchmark-runs/run_status")
    assert detail.status_code == 200, detail.text
    assert detail.json()["status"] == expected
    for query in ("running", "succeeded", "failed", "cancelled"):
        response = harness.request("GET", f"benchmark-runs?q={query}")
        assert response.status_code == 200, response.text
        rows = response.json()["data"]
        assert [row["id"] for row in rows] == (["run_status"] if query == expected else [])
        if rows:
            assert rows[0]["status"] == expected


def test_admin_retry_denies_unreadable_owner_secret(harness: Harness) -> None:
    """Admin mutation rights do not bypass the caller's credential-secret ACL."""
    entity = _evaluation("ev_a")
    entity.status = "failed"
    entity.credentials = {"openai": "private/credential"}
    harness.entities.update(entity)
    before = harness.snapshot()

    def denied_secret(reference: str) -> bool:
        """Record the caller's denied secret check without exposing the value."""
        harness.secrets.append(reference)
        return False

    app = harness.client.app
    assert isinstance(app, FastAPI)
    app.dependency_overrides[caller_secret_readable] = lambda: denied_secret
    response = harness.request("POST", "evaluations/ev_a/retry", user=OTHER, admin=True)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"]["code"] == "invalid_reference"
    assert harness.snapshot() == before
    assert harness.secrets == ["private/credential"]
    assert not harness.submissions


def test_event_stream_stops_cleanly_when_access_is_revoked(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Event polling emits the authorized batch once and hides later events after revocation."""
    entity = _evaluation("ev_a")
    entity.events = [
        ScaledEvalsEvaluationEvent(
            id=1, status="running", detail="authorized event batch", created_at=datetime.now(UTC)
        )
    ]
    harness.entities.update(entity)
    original = EvaluationRequestRepository.load_event_batch
    batches: list[list[int]] = []

    def revoke_after_batch(
        store: EvaluationRequestRepository,
        evaluation_id: str,
        *,
        after_created_at: datetime | None = None,
        after_id: int | None = None,
        limit: int,
    ) -> list[dict]:
        """Interleave a real stored ownership change after the first authorized poll."""
        rows = original(store, evaluation_id, after_created_at=after_created_at, after_id=after_id, limit=limit)
        batches.append([row["id"] for row in rows])
        if len(batches) == 1:
            revoked = _evaluation(evaluation_id)
            revoked.owner_id = OTHER
            revoked.events.append(
                ScaledEvalsEvaluationEvent(
                    id=2, status="running", detail="revoked event batch", created_at=datetime.now(UTC)
                )
            )
            harness.entities.update(revoked)
        return rows

    monkeypatch.setattr(EvaluationRequestRepository, "load_event_batch", revoke_after_batch)
    monkeypatch.setattr(evaluations.settings, "api_sse_poll_interval_seconds", 0)
    monkeypatch.setattr(evaluations.settings, "api_sse_max_duration_seconds", 1)
    response = harness.request("GET", "evaluations/ev_a/events/stream")
    assert response.status_code == 200, response.text
    assert response.text.count("authorized event batch") == 1
    assert "revoked event batch" not in response.text
    assert batches == [[1], []]
    assert evaluations._sse_active_connections == 0


@pytest.mark.parametrize("query", [None, "running"])
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_run_list_stops_member_reads_after_filling_page(harness: Harness, query: str | None, order: str) -> None:
    """A small page reads only the first candidate batch and leaves later members untouched."""
    batch_size = entity_storage.FILTER_BATCH_SIZE
    run_ids = [f"run_incremental_{index:04d}" for index in range(2 * batch_size + 5)]
    for run_id in run_ids:
        _seed_run(run_id, OWNER)
    ordered = run_ids if order == "asc" else list(reversed(run_ids))
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    with patch.object(
        EvaluationGlobalQueries,
        "members_for_runs",
        autospec=True,
        side_effect=EvaluationGlobalQueries.members_for_runs,
    ) as scans:
        rows = store.list(limit=1, cursor=None, order=order, q=query)
    assert [row["id"] for row in rows] == ordered[:2]
    assert scans.call_count == 1
    requested = scans.call_args.args[1]
    assert len(requested) <= entity_storage.FILTER_BATCH_SIZE
    assert set(requested) == set(ordered[: entity_storage.FILTER_BATCH_SIZE])


@pytest.mark.parametrize("case", ["denied", "search"])
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_run_list_fills_page_across_batches_and_skips_prior_cursor_rows(
    harness: Harness, case: str, order: str
) -> None:
    """Denied or nonmatching batches cannot underfill pages or make cursors repeat earlier work."""
    batch_size = entity_storage.FILTER_BATCH_SIZE
    run_ids = [f"run_incremental_{index:04d}" for index in range(2 * batch_size + 5)]
    ordered = run_ids if order == "asc" else list(reversed(run_ids))
    skipped = set(ordered[:batch_size])
    for run_id in run_ids:
        _seed_run(run_id, OWNER, member_owner=OTHER if case == "denied" and run_id in skipped else OWNER)
        if case == "search" and run_id not in skipped:
            member = _evaluation(f"{run_id}_member")
            member.status, member.reward = "succeeded", 1.0
            harness.entities.update(member)
    path = f"benchmark-runs?limit=1&order={order}" + ("&q=succeeded" if case == "search" else "")
    with patch.object(
        EvaluationGlobalQueries,
        "members_for_runs",
        autospec=True,
        side_effect=EvaluationGlobalQueries.members_for_runs,
    ) as scans:
        response = harness.request("GET", path)
    assert response.status_code == 200, response.text
    page = response.json()
    assert [row["id"] for row in page["data"]] == [ordered[batch_size]]
    assert page["next_cursor"] is not None
    assert scans.call_count == 2
    requested = {run_id for call in scans.call_args_list for run_id in call.args[1]}
    assert requested == set(ordered[: 2 * batch_size])
    with patch.object(
        EvaluationGlobalQueries,
        "members_for_runs",
        autospec=True,
        side_effect=EvaluationGlobalQueries.members_for_runs,
    ) as scans:
        following = harness.request("GET", f"{path}&cursor={page['next_cursor']}")
    assert following.status_code == 200, following.text
    assert [row["id"] for row in following.json()["data"]] == [ordered[batch_size + 1]]
    assert scans.call_count == 1
    assert set(scans.call_args.args[1]) == set(ordered[batch_size + 1 : 2 * batch_size + 1])


def test_run_search_exhausts_batches_when_no_rows_match(harness: Harness) -> None:
    """An empty search result means every candidate was checked, including the final partial batch."""
    batch_size = entity_storage.FILTER_BATCH_SIZE
    run_ids = [f"run_incremental_{index:04d}" for index in range(2 * batch_size + 5)]
    for run_id in run_ids:
        _seed_run(run_id, OWNER)
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    with patch.object(
        EvaluationGlobalQueries,
        "members_for_runs",
        autospec=True,
        side_effect=EvaluationGlobalQueries.members_for_runs,
    ) as scans:
        rows = store.list(limit=1, cursor=None, order="asc", q="no-matching-run")
    assert rows == []
    assert scans.call_count == 3
    assert {run_id for call in scans.call_args_list for run_id in call.args[1]} == set(run_ids)


@pytest.mark.parametrize("query", [None, "running"])
def test_run_list_returns_final_partial_page_after_denials(harness: Harness, query: str | None) -> None:
    """Exhausting denied candidates returns the remaining authorized row without a next cursor."""
    batch_size = entity_storage.FILTER_BATCH_SIZE
    run_ids = [f"run_partial_{index:04d}" for index in range(batch_size + 1)]
    for run_id in run_ids:
        _seed_run(run_id, OWNER, member_owner=OWNER if run_id == run_ids[-1] else OTHER)
    path = "benchmark-runs?limit=2&order=asc" + (f"&q={query}" if query else "")
    response = harness.request("GET", path)
    assert response.status_code == 200, response.text
    assert [row["id"] for row in response.json()["data"]] == [run_ids[-1]]
    assert response.json()["next_cursor"] is None


def test_run_search_caches_missing_task_lookups_across_batches(harness: Harness) -> None:
    """A missing task is looked up once per search rather than once for every candidate batch."""
    for index in range(2 * entity_storage.FILTER_BATCH_SIZE + 5):
        run_id = f"run_missing_task_{index:04d}"
        _seed_run(run_id, OWNER)
        member = _evaluation(f"{run_id}_member")
        member.task_id = "task_missing"
        harness.entities.update(member)
    store = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False))
    with patch.object(harness.entities, "get", wraps=harness.entities.get) as get:
        rows = store.list(limit=1, cursor=None, order="asc", q="no-matching-run")
    assert rows == []
    assert get.call_count == 1


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("has_member", [False, True])
def test_cancel_initializing_run_preserves_state(harness: Harness, admin: bool, has_member: bool) -> None:
    """An authorized cancellation cannot interrupt creation or change the partial member set."""
    _seed_run("run_initializing", OWNER)
    parent = _run("run_initializing")
    parent.initialization_status = "created"
    harness.entities.update(parent)
    if not has_member:
        harness.entities.delete(
            ScaledEvalsEvaluation, "run_initializing_member", workspace=entity_storage.private_workspace()
        )
    before = harness.snapshot()
    response = harness.request(
        "POST", "benchmark-runs/run_initializing/cancel", user=OTHER if admin else OWNER, admin=admin
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error"]["code"] == "benchmark_run_initializing"
    assert harness.snapshot() == before
    assert not harness.teardowns
    detail = harness.request("GET", "benchmark-runs/run_initializing")
    assert detail.status_code == 200, detail.text
    assert detail.json()["initialization_status"] == "created"
    assert detail.json()["status"] == "initializing"
    reproduce = harness.request("GET", "benchmark-runs/run_initializing/reproduce")
    if has_member:
        assert reproduce.status_code == 200, reproduce.text
        assert reproduce.json()["source_status"] == "initializing"
    else:
        assert reproduce.status_code == 409, reproduce.text
    search = harness.request("GET", "benchmark-runs?q=initializing")
    assert [row["id"] for row in search.json()["data"]] == ["run_initializing"]


def test_initializing_run_does_not_reveal_state_to_another_owner(harness: Harness) -> None:
    """A foreign caller receives 404 rather than learning that a private run is initializing."""
    _seed_run("run_initializing", OWNER)
    parent = _run("run_initializing")
    parent.initialization_status = "created"
    harness.entities.update(parent)
    before = harness.snapshot()
    assert harness.request("POST", "benchmark-runs/run_initializing/cancel", user=OTHER).status_code == 404
    assert harness.snapshot() == before
    assert not harness.teardowns


def test_legacy_run_defaults_to_completed_initialization(harness: Harness) -> None:
    """Records without the new field retain their pre-existing cancellation behavior."""
    _seed_run("run_legacy", OWNER)
    legacy = ScaledEvalsBenchmarkRun.model_validate(_run("run_legacy").model_dump(exclude={"initialization_status"}))
    assert legacy.initialization_status == "complete"
    response = harness.request("POST", "benchmark-runs/run_legacy/cancel")
    assert response.status_code == 200, response.text
    assert response.json()["initialization_status"] == "complete"
    assert response.json()["status"] == "cancelled"


@pytest.mark.parametrize("state", ["created", "deleted", "missing", "cancelled"])
def test_worker_does_not_admit_members_of_unavailable_runs(harness: Harness, state: str) -> None:
    """Queued members cannot escape the initialization gate through cancellation or parent deletion."""
    _seed_run("run_unavailable", OWNER)
    parent = _run("run_unavailable")
    if state == "created":
        parent.initialization_status = "created"
    elif state == "deleted":
        parent.deleted_at = datetime.now(UTC)
    elif state == "cancelled":
        parent.cancelled_at = datetime.now(UTC)
    if state == "missing":
        harness.entities.delete(ScaledEvalsBenchmarkRun, parent.name, workspace=parent.workspace)
    else:
        harness.entities.update(parent)
    before = harness.snapshot()
    assert (
        EvaluationWorkerRepository().claim_next(
            claim_timeout=60, worker_id="initialization-test", evaluation_id="run_unavailable_member"
        )
        is None
    )
    assert harness.snapshot() == before


def test_run_cancel_does_not_expand_the_authorized_member_set(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record attached after authorization is neither cancelled, cleaned up nor returned."""
    _seed_run("run_cancel", OWNER)
    update = harness.entities.update
    added = False

    def attach_after_parent_write(entity: EntityT) -> EntityT:
        """Model an out-of-band insert after the normal membership boundary has closed."""
        nonlocal added
        saved = update(entity)
        if isinstance(entity, ScaledEvalsBenchmarkRun) and not added:
            added = True
            seed_evaluation({"id": "ev_late_foreign", "owner_id": OTHER, "benchmark_run_id": "run_cancel"})
        return saved

    monkeypatch.setattr(harness.entities, "update", attach_after_parent_write)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert added
    assert [member["evaluation_id"] for member in response.json()["result"]["per_task"]] == ["run_cancel_member"]
    assert _evaluation("ev_late_foreign").status == "queued"
    assert harness.teardowns == ["run_cancel_member"]


def test_run_cancel_uses_current_member_status_after_authorization(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member requeued after the initial read is cancelled using its current version."""
    _seed_run("run_cancel", OWNER)
    member = _evaluation("run_cancel_member")
    member.status = "failed"
    harness.entities.update(member)
    update = harness.entities.update
    requeued = False

    def requeue_during_parent_write(entity: EntityT) -> EntityT:
        """Interleave a fresh execution state before the member cancellation begins."""
        nonlocal requeued
        if isinstance(entity, ScaledEvalsBenchmarkRun) and not requeued:
            requeued = True
            current = _evaluation("run_cancel_member")
            current.status = "queued"
            current.current_execution += 1
            update(current)
        return update(entity)

    monkeypatch.setattr(harness.entities, "update", requeue_during_parent_write)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert requeued
    current = _evaluation("run_cancel_member")
    assert current.status == "cancelled" and current.current_execution == 2
    assert response.json()["result"]["per_task"][0]["attempt"] == 2


def test_run_cancel_response_omits_deleted_members(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """A member deleted after authorization is omitted rather than rendered from the old row."""
    _seed_run("run_cancel", OWNER)
    update = harness.entities.update
    removed = False

    def delete_after_parent_write(entity: EntityT) -> EntityT:
        """Delete the member after the operation accepts its initial work set."""
        nonlocal removed
        saved = update(entity)
        if isinstance(entity, ScaledEvalsBenchmarkRun) and not removed:
            removed = True
            current = _evaluation("run_cancel_member")
            current.deleted_at, current.deleted = datetime.now(UTC), True
            update(current)
        return saved

    monkeypatch.setattr(harness.entities, "update", delete_after_parent_write)
    response = harness.request("POST", "benchmark-runs/run_cancel/cancel")
    assert response.status_code == 200, response.text
    assert removed
    assert response.json()["result"]["per_task"] == []
    assert not harness.teardowns


def test_run_cancellation_cleanup_cannot_target_unapproved_ids(harness: Harness) -> None:
    """The accepted operation's cleanup recorder rejects IDs outside its cancelled set."""
    _seed_run("run_cancel", OWNER)
    cancellation = BenchmarkRunRequestRepository(access=ResourceAccess(OWNER, False)).cancel("run_cancel")
    assert cancellation is not None
    before = harness.snapshot()
    with pytest.raises(KeyError):
        cancellation.record_teardown("ev_b", "unexpected cleanup")
    assert harness.snapshot() == before


def test_run_cancel_batches_terminal_state_checks(harness: Harness) -> None:
    """Finished members are checked in a batch instead of fetched and mutated individually."""
    _seed_run("run_terminal", OWNER)
    first = _evaluation("run_terminal_member")
    first.status, first.reward = "succeeded", 1.0
    first = harness.entities.update(first)
    for index, status in enumerate(("failed", "cancelled")):
        seed_evaluation(
            {"id": f"ev_terminal_{index}", "owner_id": OWNER, "benchmark_run_id": "run_terminal", "status": status}
        )
    before = {
        member.name: member.model_dump_json()
        for member in (first, _evaluation("ev_terminal_0"), _evaluation("ev_terminal_1"))
    }
    with (
        patch.object(harness.entities, "get", wraps=harness.entities.get) as get,
        patch.object(harness.entities, "list", wraps=harness.entities.list) as scans,
    ):
        response = harness.request("POST", "benchmark-runs/run_terminal/cancel")
    assert response.status_code == 200, response.text
    assert not [call for call in get.call_args_list if call.args[0] is ScaledEvalsEvaluation]
    assert len([call for call in scans.call_args_list if call.args[0] is ScaledEvalsEvaluation]) == 3
    assert {_id: _evaluation(_id).model_dump_json() for _id in before} == before
    assert not harness.teardowns


@pytest.fixture(params=["http_503", "transport", "pdp_503"])
def failed_admin_probe(harness: Harness, request: pytest.FixtureRequest) -> list[tuple[str, str]]:
    """Exercise the real admin dependency with an unavailable optional privilege check."""
    probes: list[tuple[str, str]] = []

    async def authorize(method: str, path: str) -> None:
        probes.append((method, path))
        if request.param == "http_503":
            raise HTTPException(503)
        if request.param == "transport":
            raise httpx.ConnectError("synthetic admin probe failure")
        pdp_request = httpx.Request("POST", "https://synthetic.invalid/allow")
        raise httpx.HTTPStatusError(
            "synthetic PDP failure", request=pdp_request, response=httpx.Response(503, request=pdp_request)
        )

    harness.app.dependency_overrides.pop(caller_is_admin)
    harness.app.dependency_overrides[get_request_authorizer] = lambda: authorize
    return probes


@pytest.mark.parametrize("resource", ["evaluations", "benchmark-runs"])
def test_admin_probe_failure_preserves_owner_access_and_private_isolation(
    harness: Harness, failed_admin_probe: list[tuple[str, str]], resource: Literal["evaluations", "benchmark-runs"]
) -> None:
    """Probe outages permit normal reads and owner cancellation without granting access to private peers."""
    if resource == "benchmark-runs":
        _seed_run("run_a", OWNER)
        _seed_run("run_b", OTHER, member_owner=OTHER)
        own_id, other_id = "run_a", "run_b"
    else:
        own_id, other_id = "ev_a", "ev_b"
    assert harness.request("GET", f"{resource}/{own_id}").status_code == 200
    assert failed_admin_probe == [("GET", "/apis/scaled-evals/v1/admin/capacity")]
    visible_ids = {row["id"] for row in harness.request("GET", resource).json()["data"]}
    assert own_id in visible_ids and other_id not in visible_ids
    assert harness.request("GET", f"{resource}/{other_id}").status_code == 404
    before = harness.snapshot()
    assert harness.request("POST", f"{resource}/{other_id}/cancel").status_code == 404
    assert harness.snapshot() == before
    assert harness.request("GET", "evaluations/ev_public", user=OTHER).status_code == 200
    assert harness.request("POST", f"{resource}/{own_id}/cancel").status_code == 200


@pytest.mark.parametrize("stream", ["logs", "events"])
def test_admin_probe_failure_preserves_stream_isolation(
    harness: Harness, failed_admin_probe: list[tuple[str, str]], stream: Literal["logs", "events"]
) -> None:
    """Stream admission tolerates a failed admin probe while retaining private-record isolation."""
    member = _evaluation("ev_a")
    member.status = "succeeded"
    harness.entities.update(member)
    assert harness.request("GET", f"evaluations/ev_a/{stream}/stream").status_code == 200
    assert failed_admin_probe == [("GET", "/apis/scaled-evals/v1/admin/capacity")]
    assert harness.request("GET", f"evaluations/ev_b/{stream}/stream").status_code == 404


@pytest.mark.parametrize(
    "repository", [EvaluationRequestRepository, BenchmarkRunRequestRepository, BenchmarkArchiveRequestRepository]
)
def test_request_repository_requires_an_explicit_scope(
    repository: type[EvaluationRequestRepository]
    | type[BenchmarkRunRequestRepository]
    | type[BenchmarkArchiveRequestRepository],
) -> None:
    """An omitted request scope must fail instead of granting service access."""
    with pytest.raises(TypeError, match="access"):
        signature(repository).bind()


@pytest.mark.parametrize(
    "repository", [EvaluationRequestRepository, BenchmarkRunRequestRepository, BenchmarkArchiveRequestRepository]
)
def test_request_interfaces_do_not_expose_worker_or_global_operations(
    repository: type[EvaluationRequestRepository]
    | type[BenchmarkRunRequestRepository]
    | type[BenchmarkArchiveRequestRepository],
) -> None:
    forbidden = {
        "claim_next",
        "claim",
        "set_status",
        "persist_result",
        "heartbeat",
        "finish",
        "for_cleanup",
        "count",
        "fleet_totals",
        "task_in_active_use",
        "load_experiment_identity",
    }
    assert forbidden.isdisjoint(dir(repository))


@pytest.mark.parametrize("queries", [EvaluationGlobalQueries, BenchmarkRunGlobalQueries])
def test_global_query_interfaces_have_no_public_mutations(
    queries: type[EvaluationGlobalQueries] | type[BenchmarkRunGlobalQueries],
) -> None:
    forbidden = {
        "create",
        "create_run",
        "cancel",
        "soft_delete",
        "retry_failed",
        "claim_next",
        "set_status",
        "persist_result",
        "request_archive_build",
        "record_cancel_teardown_failure",
        "_mutate",
        "_claim",
    }
    assert forbidden.isdisjoint(dir(queries))


def test_catalog_database_cannot_supply_privileged_repositories() -> None:
    assert {"evaluations", "benchmark_runs", "benchmark_archives"}.isdisjoint(dir(get_db()))


def test_resource_routes_only_depend_on_scoped_repositories() -> None:
    def dependencies(node: Dependant) -> Iterator[object]:
        yield node.call
        for child in node.dependencies:
            yield from dependencies(child)

    for router in (evaluations.router, benchmark_runs.router):
        for route in router.routes:
            assert isinstance(route, APIRoute)
            calls = list(dependencies(route.dependant))
            assert get_db not in calls and get_evaluation_global_queries not in calls
            assert get_request_db in calls or get_request_stream_database_factory in calls


def test_global_reference_checks_protect_resources_used_by_another_owner(harness: Harness) -> None:
    """A private evaluation owned by someone else still blocks deleting its task or shared profile."""
    harness.app.include_router(tasks.router, prefix=BASE)
    harness.app.include_router(config_profiles.router, prefix=BASE)
    ConfigProfileStore().create("shared_profile", name="shared", type="gym", config={}, owner_id=OWNER)
    for stored in list(harness.entities.stored.values()):
        if isinstance(stored, ScaledEvalsEvaluation):
            member = stored.model_copy(deep=True)
            member.status = "running" if member.name == "ev_b" else "succeeded"
            if member.name == "ev_b":
                member.framework_profile_id = "shared_profile"
            harness.entities.update(member)
    queries = EvaluationGlobalQueries()
    assert queries.task_in_active_use("task_x") and queries.profile_in_active_use("shared_profile")
    assert EvaluationRequestRepository(access=ResourceAccess(OWNER, False)).get("ev_b") is None
    before = harness.snapshot()
    assert harness.request("DELETE", "tasks/task_x").status_code == 409
    assert harness.request("DELETE", "config-profiles/shared_profile").status_code == 409
    assert harness.snapshot() == before


def test_users_views_keep_the_callers_scope(harness: Harness) -> None:
    harness.app.include_router(users.router, prefix=BASE)
    response = harness.request("GET", "users/me/evaluations")
    assert response.status_code == 200, response.text
    assert {row["id"] for row in response.json()["data"]} == {"ev_a", "ev_public", "ev_team", "ev_org"}
