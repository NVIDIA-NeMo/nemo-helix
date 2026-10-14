# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benchmark-runs router + derive-on-read rollup.

A benchmark run is identity + config in `benchmark_runs`; its members are
ordinary `evaluations` rows carrying benchmark_run_id. Status/reward/per-task are
derived on read from the members (derive_run_view) — nothing is materialized and
the worker is not involved. Cluster-free: TestClient over the fake Entity Store.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client
from scaled_evals.api.benchmark_run_store import BenchmarkRunStore, derive_run_view
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.evaluation_store import EvaluationStore
from scaled_evals.api.settings import settings
from task_fixtures import seed_benchmark, seed_benchmark_run, seed_evaluation, seed_task_revision

NOW = datetime(2026, 6, 29, tzinfo=UTC)


def _seed_suite(*, members: list[dict], **benchmark_fields: Any) -> None:
    """Store `members` (load_members shape) as revision 1 of benchmark `bm_suite`."""
    refs = tuple((member["task_id"], member["task_revision"]) for member in members)
    seed_benchmark("bm_suite", refs, **benchmark_fields)


@pytest.fixture(autouse=True)
def _stub_execution_snapshot(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        EvaluationStore,
        "_capture_execution_snapshot",
        MagicMock(
            return_value={
                "schema_version": "scaled-evals-execution-inputs-v1",
                "captured_at": NOW.isoformat(),
                "evaluation": {},
                "task": {},
                "profiles": {},
                "credentials": {},
                "submission_identity": {},
            }
        ),
    )


def _run_row(**overrides) -> dict:
    """A benchmark_runs row as the repo returns it (no materialized status/reward)."""
    row = {
        "id": "bmr_1",
        "owner_id": None,
        "name": "suite run",
        "framework": "harbor",
        "benchmark_id": "bm_suite",
        "benchmark_revision": 1,
        "framework_profile_id": None,
        "harbor_profile_id": None,
        "intake_profile_id": None,
        "credentials": {},
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "parallelism": 1,
        "visibility": "private",
        "cancelled_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    row.update(overrides)
    return row


def _member(position: int, **overrides) -> dict:
    """A resolved benchmark member (load_members shape), with its task stored."""
    m: dict[str, Any] = {
        "task_id": f"task_{position}",
        "position": position,
        "task_revision": 1,
        "task_slug": f"nemo-task-{position}",
        "revision_status": "ready",
    }
    m.update(overrides)
    seed_task_revision(
        m["task_id"],
        m["task_revision"],
        slug=m["task_slug"],
        status=m["revision_status"],
        **{field: m[field] for field in ("image_ref", "image_digest", "tarball_object_key") if field in m},
    )
    return m


def _member_eval(position: int, **overrides) -> dict:
    """A member evaluation row (members_for_runs shape)."""
    row = {
        "id": f"ev_m{position}",
        "benchmark_run_id": "bmr_1",
        "task_id": f"task_{position}",
        "task_slug": f"nemo-task-{position}",
        "task_name": f"NeMo task {position}",
        "task_revision": 1,
        "status": "succeeded",
        "reward": 1.0,
        "n_trials": 1,
        "n_completed": 1,
        "n_errored": 0,
        "n_failed_solve": 0,
        "exception_counts": {},
        "status_detail": None,
        "current_execution": 1,
        "max_executions": 3,
        "next_retry_at": None,
        "last_failure_code": None,
        "last_failure_category": None,
        "cancel_teardown_status": "not_requested",
        "cancel_teardown_error": None,
        "finished_at": None,
    }
    row.update(overrides)
    seed_task_revision(row["task_id"], row["task_revision"], slug=row["task_slug"], name=row["task_name"])
    return row


def _seed_run(*members: dict, **overrides) -> None:
    """Store run `bmr_1` and the given member evaluations."""
    seed_benchmark_run(_run_row(**overrides))
    for member in members:
        seed_evaluation(member)


def _created(response) -> tuple[dict, list[dict]]:  # noqa: ANN001
    """Stored run and member evaluations (by task) of a create response."""
    run_id = response.json()["id"]
    run = BenchmarkRunStore().get(run_id)
    assert run is not None
    members = BenchmarkRunStore().members_for_runs([run_id])
    stored = [EvaluationStore().load_for_dispatch(member["id"]) for member in members]
    assert None not in stored
    return run, [row for row in stored if row is not None]


# ---- derive_run_view (pure read-time rollup) ------------------------------


def test_derive_succeeded_means_reward() -> None:
    view = derive_run_view(_run_row(), [_member_eval(0, reward=1.0), _member_eval(1, reward=0.0)])
    assert view["status"] == "succeeded"
    assert view["reward"] == 0.5
    assert view["n_trials"] == 2
    assert view["n_completed"] == 2
    assert view["n_errored"] == 0
    assert view["n_failed_solve"] == 0
    assert view["exception_counts"] == {}
    assert view["result"]["kind"] == "benchmark"
    assert [t["evaluation_id"] for t in view["result"]["per_task"]] == ["ev_m0", "ev_m1"]
    # Breakdown carries human-readable task slug/name, not just the id.
    assert [t["task_slug"] for t in view["result"]["per_task"]] == ["nemo-task-0", "nemo-task-1"]


def test_derive_running_while_a_member_unfinished() -> None:
    view = derive_run_view(_run_row(), [_member_eval(0), _member_eval(1, status="running", reward=None)])
    assert view["status"] == "running"


def test_derive_failed_when_a_member_failed() -> None:
    view = derive_run_view(
        _run_row(),
        [_member_eval(0, reward=1.0), _member_eval(1, status="failed", reward=None, n_errored=1)],
    )
    assert view["status"] == "failed"
    assert view["reward"] == 1.0  # mean over the member that scored
    assert view["n_errored"] == 1


def test_derive_reports_failed_solves_and_infrastructure_errors_separately() -> None:
    view = derive_run_view(
        _run_row(),
        [
            _member_eval(0, reward=0.0, n_failed_solve=1),
            _member_eval(
                1,
                status="failed",
                reward=None,
                n_completed=0,
                n_errored=1,
                exception_counts={"SandboxExecutionError": 1},
            ),
        ],
    )

    assert view["n_completed"] == 1
    assert view["n_failed_solve"] == 1
    assert view["n_errored"] == 1
    assert view["exception_counts"] == {"SandboxExecutionError": 1}
    assert view["result"]["aggregate"] == {
        "reward": 0.0,
        "n_tasks": 2,
        "n_tasks_scored": 1,
        "n_trials": 2,
        "n_completed": 1,
        "n_errored": 1,
        "n_failed_solve": 1,
        "exception_counts": {"SandboxExecutionError": 1},
        "n_teardown_pending": 0,
        "n_teardown_failed": 0,
        "n_retryable_failures": 0,
        "n_recovered": 0,
        "failure_counts": {"infrastructure": 1},
        "recovered_counts": {},
    }
    assert view["result"]["per_task"][0]["n_failed_solve"] == 1
    assert view["result"]["per_task"][1]["exception_counts"] == {"SandboxExecutionError": 1}


def test_derive_exposes_failure_diagnostics_and_recovered_members() -> None:
    failed_member = _member_eval(
        0,
        status="failed",
        reward=None,
        n_errored=1,
        status_detail="pod timed out waiting for runner",
        current_execution=2,
        max_executions=3,
        last_failure_code="runner_disappeared",
        last_failure_category="infrastructure",
    )
    recovered_member = _member_eval(
        1,
        reward=1.0,
        current_execution=2,
        max_executions=3,
        last_failure_code="RateLimitError",
        last_failure_category="provider",
    )

    view = derive_run_view(_run_row(), [failed_member, recovered_member])

    assert view["failure_counts"] == {"infrastructure": 1}
    assert view["recovered_counts"] == {"provider": 1}
    assert view["n_retryable_failures"] == 1
    assert view["n_recovered"] == 1
    assert view["result"]["aggregate"]["failure_counts"] == {"infrastructure": 1}
    assert view["result"]["per_task"][0]["retryable"] is True
    assert view["result"]["per_task"][0]["failure_evidence"] == {
        "code": "runner_disappeared",
        "category": "infrastructure",
        "detail": "pod timed out waiting for runner",
        "exception_counts": {},
        "attempt": 2,
        "max_attempts": 5,
        "next_retry_at": None,
    }
    assert view["result"]["per_task"][1]["recovered"] is True
    assert [task["task_id"] for task in view["result"]["original_failures"]] == [
        "task_0",
        "task_1",
    ]
    assert [task["task_id"] for task in view["result"]["recovered_tasks"]] == ["task_1"]


def test_derive_marks_nonzero_agent_exit_as_automatically_retryable() -> None:
    failed_member = _member_eval(
        0,
        status="failed",
        reward=None,
        n_errored=1,
        exception_counts={"NonZeroAgentExitCodeError": 1},
        current_execution=1,
        max_executions=3,
    )

    view = derive_run_view(_run_row(), [failed_member])

    assert view["failure_counts"] == {"task": 1}
    assert view["n_retryable_failures"] == 1
    assert view["result"]["per_task"][0]["failure_code"] == "NonZeroAgentExitCodeError"
    assert view["result"]["per_task"][0]["retryable"] is True


def test_retrying_same_member_moves_existing_benchmark_back_to_running() -> None:
    failed_member = _member_eval(1, status="failed", reward=None, n_errored=1)
    failed = derive_run_view(_run_row(), [_member_eval(0), failed_member])
    assert failed["status"] == "failed"

    retried_member = {
        **failed_member,
        "status": "queued",
        "n_trials": None,
        "n_errored": None,
        "finished_at": None,
    }
    running = derive_run_view(_run_row(), [_member_eval(0), retried_member])
    assert running["status"] == "running"
    assert [task["evaluation_id"] for task in running["result"]["per_task"]] == [
        "ev_m0",
        "ev_m1",
    ]

    succeeded = derive_run_view(
        _run_row(),
        [_member_eval(0), {**retried_member, "status": "succeeded", "reward": 1.0}],
    )
    assert succeeded["status"] == "succeeded"
    assert succeeded["reward"] == 1.0


def test_derive_cancelled_overrides_members() -> None:
    view = derive_run_view(_run_row(cancelled_at=NOW), [_member_eval(0), _member_eval(1)])
    assert view["status"] == "cancelled"


def test_derive_cancelled_reports_member_teardown_progress() -> None:
    view = derive_run_view(
        _run_row(cancelled_at=NOW),
        [
            _member_eval(
                0,
                status="cancelled",
                reward=None,
                cancel_teardown_status="pending",
            ),
            _member_eval(
                1,
                status="cancelled",
                reward=None,
                cancel_teardown_status="failed",
                cancel_teardown_error="cleanup failed",
            ),
        ],
    )

    assert view["n_teardown_pending"] == 1
    assert view["n_teardown_failed"] == 1
    assert view["result"]["aggregate"]["n_teardown_pending"] == 1
    assert view["result"]["per_task"][1]["cancel_teardown_error"] == "cleanup failed"
    assert "teardown pending: 1" in view["status_detail"]
    assert "teardown failed: 1" in view["status_detail"]


def test_benchmark_request_supports_full_member_execution_contract() -> None:
    from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest

    request = CreateBenchmarkRunRequest(
        name="full contract",
        benchmark_id="bm_test",
        n_attempts=3,
        extra_skill_object_keys=["skills/review/SKILL.md"],
        instruction_prefix="inspect first",
        instruction_postfix="summarize last",
        initial_user_turns=["Initialize", "/review"],
    )

    assert request.n_attempts == 3
    assert request.extra_skill_object_keys == ["skills/review/SKILL.md"]
    assert request.initial_user_turns == ["Initialize", "/review"]


# ---- POST /benchmark-runs: spawn run + member evaluations -----------------


def test_create_run_spawns_run_and_members() -> None:
    _seed_suite(members=[_member(0), _member(1)])

    resp = client.post("/v1/benchmark-runs", json={"name": "suite run", "benchmark_id": "bm_suite"})

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["id"].startswith("bmr_")
    assert body["benchmark_id"] == "bm_suite"
    assert body["status"] == "running"  # members queued → derived running
    run, members = _created(resp)
    assert run["name"] == "suite run"
    assert [(member["task_id"], member["status"]) for member in members] == [
        ("task_0", "queued"),
        ("task_1", "queued"),
    ]


def test_create_run_applies_member_framework_profile_overrides() -> None:
    overrides = {"task_0": "cfg_member"}
    for profile_id in ("cfg_default", "cfg_member"):
        ConfigProfileStore().create(profile_id, name=profile_id, type="harbor", config={})
    _seed_suite(members=[_member(0), _member(1)])

    response = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "suite run",
            "benchmark_id": "bm_suite",
            "framework_profile_id": "cfg_default",
            "member_framework_profile_ids": overrides,
        },
    )

    assert response.status_code == 202, response.text
    run, members = _created(response)
    assert run["runner_metadata"]["member_framework_profile_ids"] == overrides
    assert [(member["framework_profile_id"], member["harbor_profile_id"]) for member in members] == [
        ("cfg_member", "cfg_member"),
        ("cfg_default", "cfg_default"),
    ]


def test_preflight_run_reports_members_without_creating_run() -> None:
    _seed_suite(
        members=[_member(0), _member(1)],
        qualification_status="qualified",
        qualification_evidence={"receipt": "qe_1"},
        qualified_at=NOW,
    )

    response = client.post(
        "/v1/benchmark-runs/preflight",
        json={"name": "suite check", "benchmark_id": "bm_suite"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runnable"] is True
    assert body["member_summary"] == {
        "total": 2,
        "ready": 2,
        "blocked": 0,
        "failures": [],
        "failures_truncated": False,
    }
    qualification = next(check for check in body["checks"] if check["prerequisite"] == "benchmark_qualification")
    assert qualification["state"] == "ready"
    assert qualification["details"]["evidence_present"] is True
    assert client.get("/v1/benchmark-runs").json()["data"] == []


def test_preflight_run_reports_bounded_member_failure() -> None:
    _seed_suite(
        members=[_member(0, revision_status="building"), _member(1)],
    )

    response = client.post(
        "/v1/benchmark-runs/preflight",
        json={"name": "suite check", "benchmark_id": "bm_suite"},
    )

    assert response.status_code == 200, response.text
    summary = response.json()["member_summary"]
    assert summary["blocked"] == 1
    assert summary["ready"] == 1
    assert summary["failures"][0]["code"] == "task_not_ready"


def test_preflight_run_truncates_large_member_failure_details() -> None:
    members = [_member(i, revision_status="building") for i in range(52)]
    _seed_suite(members=members)

    response = client.post(
        "/v1/benchmark-runs/preflight",
        json={"name": "suite check", "benchmark_id": "bm_suite"},
    )

    assert response.status_code == 200, response.text
    summary = response.json()["member_summary"]
    assert summary["blocked"] == 52
    assert summary["ready"] == 0
    assert len(summary["failures"]) == 50
    assert summary["failures_truncated"] is True


def test_create_run_rejects_member_with_missing_task_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _member(0, tarball_object_key="task_0/rev/1/tarball.tar.gz")
    _seed_suite(
        members=[member],
    )
    monkeypatch.setattr("scaled_evals.api.routers.benchmark_runs.artifacts.object_exists", lambda _key: False)

    resp = client.post("/v1/benchmark-runs", json={"name": "suite run", "benchmark_id": "bm_suite"})

    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "task_object_missing"
    assert client.get("/v1/benchmark-runs").json()["data"] == []


def test_create_run_fans_out_full_member_execution_contract() -> None:
    _seed_suite(members=[_member(0)])

    response = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "suite run",
            "benchmark_id": "bm_suite",
            "n_attempts": 3,
            "extra_skill_object_keys": ["skills/review/SKILL.md"],
            "instruction_prefix": "inspect first",
            "instruction_postfix": "summarize last",
            "initial_user_turns": ["Initialize", "/review"],
        },
    )

    assert response.status_code == 202, response.text
    [member] = _created(response)[1]
    assert member["extra_skill_object_keys"] == ["skills/review/SKILL.md"]
    assert (member["instruction_prefix"], member["instruction_postfix"]) == ("inspect first", "summarize last")
    assert member["initial_user_turns"] == ["Initialize", "/review"]
    assert member["n_attempts"] == 3


def test_create_run_persists_member_cap() -> None:
    _seed_suite(members=[_member(0)])

    resp = client.post(
        "/v1/benchmark-runs",
        json={"name": "suite run", "benchmark_id": "bm_suite", "max_concurrent_members": 50},
    )

    assert resp.status_code == 202, resp.text
    assert _created(resp)[0]["max_concurrent_members"] == 50


def test_create_run_fans_out_network_policy() -> None:
    _seed_suite(members=[_member(0), _member(1)])

    resp = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "offline suite",
            "benchmark_id": "bm_suite",
            "network_policy": "default_deny",
        },
    )

    assert resp.status_code == 202, resp.text
    assert resp.json()["network_policy"] == "default_deny"
    run, members = _created(resp)
    assert run["network_policy"] == "default_deny"
    assert {member["network_policy"] for member in members} == {"default_deny"}


def test_create_run_resolves_version_once_for_run_and_members() -> None:
    _seed_suite(members=[_member(0)])

    response = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "versioned suite",
            "benchmark_id": "bm_suite",
            "framework_version": "stable",
        },
    )

    assert response.status_code == 202, response.text
    run, [member] = _created(response)
    version_fields = (
        "requested_framework_version",
        "framework_version",
        "runner_image_ref",
        "runner_image_digest",
        "framework_adapter_version",
        "sandbox_k8s_version",
    )
    assert tuple(run[field] for field in version_fields) == (
        "stable",
        "0.13.2",
        "scaled-evals-api:dev",
        None,
        "nemo-helix-plugin-overlay-v1",
        "0.1.13",
    )
    assert tuple(member[field] for field in version_fields) == tuple(run[field] for field in version_fields)
    assert member["runner_metadata"] == run["runner_metadata"]
    assert run["runner_metadata"]["qualification"]["release"]["version"] == "0.13.2"


def test_create_run_404_when_benchmark_unknown() -> None:
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_missing"})
    assert resp.status_code == 404, resp.text


def test_create_run_409_when_member_not_ready() -> None:
    _seed_suite(
        members=[_member(0, revision_status="building")],
    )
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_suite"})
    assert resp.status_code == 409, resp.text


def test_create_run_rejects_member_image_from_unapproved_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "us-central1-docker.pkg.dev")
    _seed_suite(
        members=[
            _member(
                0,
                image_ref="artifactory.nvidia.com/team/task:rev1",
                image_digest="sha256:" + "a" * 64,
            )
        ],
    )

    response = client.post(
        "/v1/benchmark-runs",
        json={"name": "x", "benchmark_id": "bm_suite"},
    )

    assert response.status_code == 422
    error = response.json()["detail"]["error"]
    assert error["code"] == "invalid_task_image"
    assert error["message"] == (
        "member task task_0 rev 1: task image registry 'artifactory.nvidia.com' "
        "is not approved; allowed registries: us-central1-docker.pkg.dev"
    )
    assert client.get("/v1/benchmark-runs").json()["data"] == []


def test_create_run_422_when_empty_benchmark() -> None:
    _seed_suite(members=[])
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_suite"})
    assert resp.status_code == 422, resp.text


def test_create_run_422_on_bad_profile_id() -> None:
    resp = client.post(
        "/v1/benchmark-runs",
        json={"name": "x", "benchmark_id": "bm_suite", "framework_profile_id": "nope"},
    )
    assert resp.status_code == 422, resp.text

    resp = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "x",
            "benchmark_id": "bm_suite",
            "member_framework_profile_ids": {"task_0": "nope"},
        },
    )
    assert resp.status_code == 422, resp.text


# ---- GET derives status/reward from members -------------------------------


def test_get_run_derives_succeeded() -> None:
    _seed_run(_member_eval(0), _member_eval(1))
    resp = client.get("/v1/benchmark-runs/bmr_1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["reward"] == 1.0
    assert body["n_trials"] == 2
    assert body["result"]["kind"] == "benchmark"
    assert len(body["result"]["per_task"]) == 2


def test_get_run_derives_running() -> None:
    _seed_run(_member_eval(0), _member_eval(1, status="running", reward=None))
    body = client.get("/v1/benchmark-runs/bmr_1").json()
    assert body["status"] == "running"


def test_reproduce_run_returns_complete_frozen_request_and_command() -> None:
    run = dict(
        framework_version="0.6.3",
        runner_metadata={
            "agent_bundle": {"bundle_id": "ab_codex"},
            "member_framework_profile_ids": {"task_0": "cfg_member"},
        },
        framework_profile_id="cfg_h",
        harbor_profile_id="cfg_h",
        credentials={"openai": "team-a/openai-key"},
        parallelism=2,
        max_concurrent_members=4,
    )
    member = _member_eval(0, status="failed", reward=None)
    member = {
        **member,
        "name": "suite run · nemo-task-0",
        "framework": "harbor",
        "framework_version": "0.6.3",
        "runner_metadata": run["runner_metadata"],
        "framework_profile_id": None,
        "intake_profile_id": None,
        "credentials": run["credentials"],
        "extra_skill_object_keys": ["skills/review/SKILL.md"],
        "instruction_prefix": "inspect first",
        "instruction_postfix": "summarize last",
        "initial_user_turns": ["Initialize", "/review"],
        "runtime": "sandbox_k8s",
        "network_policy": "unrestricted",
        "network_policy_config": {},
        "n_attempts": 3,
        "parallelism": 2,
        "visibility": "private",
    }
    _seed_run(member, **run)

    response = client.get("/v1/benchmark-runs/bmr_1/reproduce")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["benchmark_run_id"] == "bmr_1"
    assert body["source_status"] == "failed"
    assert body["request"]["benchmark_revision"] == 1
    assert body["request"]["n_attempts"] == 3
    assert body["request"]["extra_skill_object_keys"] == ["skills/review/SKILL.md"]
    assert body["request"]["initial_user_turns"] == ["Initialize", "/review"]
    assert body["request"]["agent_bundle_id"] == "ab_codex"
    assert body["request"]["member_framework_profile_ids"] == {"task_0": "cfg_member"}
    assert body["request"]["max_concurrent_members"] == 4
    assert body["cli_command"][:3] == ["scaled-evals", "benchmark-run", "create"]
    assert "--n-attempts" in body["cli_command"]
    assert "--max-concurrent-members" in body["cli_command"]
    assert "task_0=cfg_member" in body["cli_command"]


def test_get_run_404() -> None:
    assert client.get("/v1/benchmark-runs/bmr_missing").status_code == 404


# ---- cancel stamps cancelled_at, derived status is cancelled --------------


def test_cancel_run_derives_cancelled() -> None:
    _seed_run(_member_eval(0, status="queued", reward=None))
    resp = client.post("/v1/benchmark-runs/bmr_1/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"
    assert (EvaluationStore().load_for_dispatch("ev_m0") or {}).get("status") == "cancelled"


def test_cancel_run_tears_down_launched_members(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MagicMock()
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    _seed_run(
        _member_eval(
            0,
            status="running",
            reward=None,
            runtime="sandbox_k8s",
            backend_handle=json.dumps({"backend": "sandbox_k8s", "external_id": "job-123"}),
        )
    )

    response = client.post("/v1/benchmark-runs/bmr_1/cancel")

    assert response.status_code == 200, response.text
    assert backend.teardown.call_args.args[0].external_id == "job-123"
