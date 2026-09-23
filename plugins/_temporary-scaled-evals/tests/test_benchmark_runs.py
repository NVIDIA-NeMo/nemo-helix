# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benchmark-runs router + derive-on-read rollup.

A benchmark run is identity + config in `benchmark_runs`; its members are
ordinary `evaluations` rows carrying benchmark_run_id. Status/reward/per-task are
derived on read from the members (derive_run_view) — nothing is materialized and
the worker is not involved. Cluster-free: TestClient + mocked psycopg connection.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.db import get_conn
from scaled_evals.api.repositories.benchmark_run_repository import derive_run_view
from scaled_evals.api.repositories.evaluation_repository import EvaluationRepository
from scaled_evals.api.settings import settings

NOW = datetime(2026, 6, 29, tzinfo=UTC)


def _conn(*, fetchone=None, fetchall=None) -> MagicMock:
    """Mock connection. fetchone/fetchall accept a list used as side_effect."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    if fetchone is not None:
        cur.fetchone.side_effect = list(fetchone)
    if fetchall is not None:
        cur.fetchall.side_effect = list(fetchall)
    return conn


def _use_conn(conn: MagicMock) -> None:
    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    v1.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _stub_execution_snapshot(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        EvaluationRepository,
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
        "switchyard_profile_id": None,
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
    """A resolved benchmark member (load_members shape)."""
    m = {
        "task_id": f"task_{position}",
        "position": position,
        "task_revision": 1,
        "task_slug": f"task-{position}",
        "revision_status": "ready",
    }
    m.update(overrides)
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
    return row


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


def test_max_concurrent_members_does_not_require_switchyard() -> None:
    from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest

    request = CreateBenchmarkRunRequest(
        name="capped",
        benchmark_id="bm_test",
        parallelism=4,
        max_concurrent_members=50,
    )

    assert request.max_concurrent_members == 50
    assert request.switchyard_profile_id is None


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


def test_shared_switchyard_campaign_preserves_trial_parallelism_semantics() -> None:
    from pydantic import ValidationError
    from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest

    request = CreateBenchmarkRunRequest(
        name="shared",
        benchmark_id="bm_test",
        switchyard_profile_id="cfg_switchyard",
        parallelism=1,
        max_concurrent_members=1024,
    )
    assert request.max_concurrent_members == 1024

    with pytest.raises(ValidationError, match="parallelism=1"):
        CreateBenchmarkRunRequest(
            name="not-exact",
            benchmark_id="bm_test",
            switchyard_profile_id="cfg_switchyard",
            parallelism=2,
        )


# ---- POST /benchmark-runs: spawn run + member evaluations -----------------


def test_create_run_spawns_run_and_members() -> None:
    conn = _conn(
        # fetchone: current_revision, run RETURNING, member0 RETURNING, member1 RETURNING.
        fetchone=[{"current_revision": 1}, _run_row(), _member_eval(0), _member_eval(1)],
        # fetchall: load_members (benchmark tasks), then members_for_runs (derive on response).
        fetchall=[[_member(0), _member(1)], [_member_eval(0, status="queued", reward=None)]],
    )
    _use_conn(conn)

    resp = client.post("/v1/benchmark-runs", json={"name": "suite run", "benchmark_id": "bm_suite"})

    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["id"] == "bmr_1"
    assert body["benchmark_id"] == "bm_suite"
    assert body["status"] == "running"  # members queued → derived running
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    run_inserts = [c for c in execs if "INSERT INTO benchmark_runs" in c.args[0]]
    member_inserts = [c for c in execs if "INSERT INTO evaluations" in c.args[0]]
    assert len(run_inserts) == 1
    assert len(member_inserts) == 2  # one member evaluation per member task
    assert conn.commit.called


def test_create_run_applies_member_framework_profile_overrides() -> None:
    overrides = {"task_0": "cfg_member"}
    conn = _conn(
        fetchone=[
            {"config": {}},
            {"config": {}},
            {"current_revision": 1},
            _run_row(
                framework_profile_id="cfg_default",
                harbor_profile_id="cfg_default",
                runner_metadata={"member_framework_profile_ids": overrides},
            ),
            _member_eval(0),
            _member_eval(1),
        ],
        fetchall=[
            [
                {"id": "cfg_default", "type": "harbor"},
                {"id": "cfg_member", "type": "harbor"},
            ],
            [_member(0), _member(1)],
            [_member_eval(0, status="queued", reward=None)],
        ],
    )
    _use_conn(conn)

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
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    run_insert = next(c for c in execs if "INSERT INTO benchmark_runs" in c.args[0])
    assert run_insert.args[1][10].obj["member_framework_profile_ids"] == overrides
    member_inserts = [c for c in execs if "INSERT INTO evaluations" in c.args[0]]
    assert [(c.args[1][14], c.args[1][15]) for c in member_inserts] == [
        ("cfg_member", "cfg_member"),
        ("cfg_default", "cfg_default"),
    ]


def test_preflight_run_reports_members_without_creating_run() -> None:
    conn = _conn(
        fetchone=[
            {
                "current_revision": 1,
                "qualification_status": "qualified",
                "qualification_evidence": {"receipt": "qe_1"},
                "qualified_at": NOW,
            }
        ],
        fetchall=[[_member(0), _member(1)]],
    )
    _use_conn(conn)

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
    assert not any(
        "INSERT INTO benchmark_runs" in call.args[0]
        for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    )
    assert not conn.commit.called


def test_preflight_run_reports_bounded_member_failure() -> None:
    conn = _conn(
        fetchone=[{"current_revision": 1}],
        fetchall=[[_member(0, revision_status="building"), _member(1)]],
    )
    _use_conn(conn)

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
    conn = _conn(fetchone=[{"current_revision": 1}], fetchall=[members])
    _use_conn(conn)

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
    conn = _conn(
        fetchone=[{"current_revision": 1}],
        fetchall=[[member]],
    )
    _use_conn(conn)
    monkeypatch.setattr("scaled_evals.api.routers.benchmark_runs.artifacts.object_exists", lambda _key: False)

    resp = client.post("/v1/benchmark-runs", json={"name": "suite run", "benchmark_id": "bm_suite"})

    assert resp.status_code == 409
    assert resp.json()["detail"]["error"]["code"] == "task_object_missing"
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    assert not any("INSERT INTO benchmark_runs" in call.args[0] for call in execs)


def test_create_run_fans_out_full_member_execution_contract() -> None:
    conn = _conn(
        fetchone=[{"current_revision": 1}, _run_row(), _member_eval(0)],
        fetchall=[[_member(0)], [_member_eval(0, status="queued", reward=None)]],
    )
    _use_conn(conn)

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
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    member_insert = next(c for c in execs if "INSERT INTO evaluations" in c.args[0])
    params = member_insert.args[1]
    assert params[19:23] == (
        ["skills/review/SKILL.md"],
        "inspect first",
        "summarize last",
        ["Initialize", "/review"],
    )
    assert params[26] == 3


def test_create_run_persists_non_switchyard_member_cap() -> None:
    conn = _conn(
        fetchone=[{"current_revision": 1}, _run_row(max_concurrent_members=50), _member_eval(0)],
        fetchall=[[_member(0)], [_member_eval(0, status="queued", reward=None)]],
    )
    _use_conn(conn)

    resp = client.post(
        "/v1/benchmark-runs",
        json={"name": "suite run", "benchmark_id": "bm_suite", "max_concurrent_members": 50},
    )

    assert resp.status_code == 202, resp.text
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    run_insert = next(c for c in execs if "INSERT INTO benchmark_runs" in c.args[0])
    assert run_insert.args[1][22] == 50


def test_create_run_fans_out_network_policy() -> None:
    conn = _conn(
        fetchone=[
            {"current_revision": 1},
            _run_row(network_policy="default_deny"),
            _member_eval(0),
            _member_eval(1),
        ],
        fetchall=[
            [_member(0), _member(1)],
            [_member_eval(0, status="queued", reward=None)],
        ],
    )
    _use_conn(conn)

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
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    run_insert = next(c for c in execs if "INSERT INTO benchmark_runs" in c.args[0])
    member_insert = next(c for c in execs if "INSERT INTO evaluations" in c.args[0])
    assert run_insert.args[1][19] == "default_deny"
    assert member_insert.args[1][24] == "default_deny"


def test_create_run_resolves_version_once_for_run_and_members() -> None:
    conn = _conn(
        fetchone=[
            {"current_revision": 1},
            _run_row(requested_framework_version="stable", framework_version="0.13.2"),
            _member_eval(0),
        ],
        fetchall=[[_member(0)], [_member_eval(0, status="queued", reward=None)]],
    )
    _use_conn(conn)

    response = client.post(
        "/v1/benchmark-runs",
        json={
            "name": "versioned suite",
            "benchmark_id": "bm_suite",
            "framework_version": "stable",
        },
    )

    assert response.status_code == 202, response.text
    execs = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    run_insert = next(c for c in execs if "INSERT INTO benchmark_runs" in c.args[0])
    member_insert = next(c for c in execs if "INSERT INTO evaluations" in c.args[0])
    assert run_insert.args[1][4:10] == member_insert.args[1][4:10]
    assert run_insert.args[1][10].obj == member_insert.args[1][10].obj
    assert run_insert.args[1][4:10] == (
        "stable",
        "0.13.2",
        "scaled-evals-api:dev",
        None,
        "nemo-helix-plugin-overlay-v1",
        "0.1.13",
    )
    assert run_insert.args[1][10].obj["qualification"]["release"]["version"] == "0.13.2"


def test_create_run_404_when_benchmark_unknown() -> None:
    _use_conn(_conn(fetchone=[None]))
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_missing"})
    assert resp.status_code == 404, resp.text


def test_create_run_409_when_member_not_ready() -> None:
    conn = _conn(
        fetchone=[{"current_revision": 1}],
        fetchall=[[_member(0, revision_status="building")]],
    )
    _use_conn(conn)
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_suite"})
    assert resp.status_code == 409, resp.text


def test_create_run_rejects_member_image_from_unapproved_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "task_image_validation_mode", "resolve")
    monkeypatch.setattr(settings, "task_image_allowed_registries", "us-central1-docker.pkg.dev")
    conn = _conn(
        fetchone=[{"current_revision": 1}],
        fetchall=[
            [
                _member(
                    0,
                    image_ref="artifactory.nvidia.com/team/task:rev1",
                    image_digest="sha256:" + "a" * 64,
                )
            ]
        ],
    )
    _use_conn(conn)

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
    assert not any(
        "INSERT INTO benchmark_runs" in call.args[0]
        for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
    )


def test_create_run_422_when_empty_benchmark() -> None:
    conn = _conn(fetchone=[{"current_revision": 1}], fetchall=[[]])
    _use_conn(conn)
    resp = client.post("/v1/benchmark-runs", json={"name": "x", "benchmark_id": "bm_suite"})
    assert resp.status_code == 422, resp.text


def test_create_run_422_on_bad_profile_id() -> None:
    _use_conn(MagicMock())
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
    conn = _conn(
        fetchone=[_run_row()],  # get()
        fetchall=[[_member_eval(0), _member_eval(1)]],  # members_for_runs()
    )
    _use_conn(conn)
    resp = client.get("/v1/benchmark-runs/bmr_1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["reward"] == 1.0
    assert body["n_trials"] == 2
    assert body["result"]["kind"] == "benchmark"
    assert len(body["result"]["per_task"]) == 2


def test_get_run_derives_running() -> None:
    conn = _conn(
        fetchone=[_run_row()],
        fetchall=[[_member_eval(0), _member_eval(1, status="running", reward=None)]],
    )
    _use_conn(conn)
    body = client.get("/v1/benchmark-runs/bmr_1").json()
    assert body["status"] == "running"


def test_reproduce_run_returns_complete_frozen_request_and_command() -> None:
    run = _run_row(
        framework_version="0.6.3",
        runner_metadata={
            "agent_bundle": {"bundle_id": "ab_codex"},
            "member_framework_profile_ids": {"task_0": "cfg_member"},
        },
        framework_profile_id="cfg_h",
        harbor_profile_id="cfg_h",
        credentials={"openai": "cred_openai"},
        parallelism=2,
        max_concurrent_members=4,
    )
    member = _member_eval(0, status="failed", reward=None)
    source = {
        **member,
        "name": "suite run · nemo-task-0",
        "framework": "harbor",
        "framework_version": "0.6.3",
        "runner_metadata": run["runner_metadata"],
        "framework_profile_id": None,
        "switchyard_profile_id": None,
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
    conn = _conn(fetchone=[run, source], fetchall=[[member]])
    _use_conn(conn)

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
    _use_conn(_conn(fetchone=[None]))
    assert client.get("/v1/benchmark-runs/bmr_missing").status_code == 404


# ---- cancel stamps cancelled_at, derived status is cancelled --------------


def test_cancel_run_derives_cancelled() -> None:
    conn = _conn(
        # cancel() UPDATE … RETURNING → run row with cancelled_at; then members_for_runs.
        fetchone=[
            _run_row(cancelled_at=NOW),
            _member_eval(
                0,
                status="cancelled",
                reward=None,
                cancel_teardown_status="succeeded",
            ),
        ],
        fetchall=[
            [_member_eval(0, status="cancelled", reward=None)],
            [_member_eval(0, status="cancelled", reward=None)],
        ],
    )
    _use_conn(conn)
    resp = client.post("/v1/benchmark-runs/bmr_1/cancel")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "cancelled"


def test_cancel_run_tears_down_launched_members(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MagicMock()
    monkeypatch.setattr(
        "scaled_evals.api.routers.evaluations._resolve_backend",
        lambda runtime: backend,
    )
    cancelled = _member_eval(
        0,
        status="cancelled",
        reward=None,
        runtime="sandbox_k8s",
        backend_handle={"backend": "sandbox_k8s", "external_id": "job-123"},
        dispatch_job_name=None,
        switchyard_profile_id=None,
        cancel_teardown_status="pending",
    )
    conn = _conn(
        fetchone=[
            _run_row(cancelled_at=NOW),
            {**cancelled, "cancel_teardown_status": "succeeded"},
        ],
        fetchall=[[cancelled], [cancelled]],
    )
    _use_conn(conn)

    response = client.post("/v1/benchmark-runs/bmr_1/cancel")

    assert response.status_code == 200, response.text
    assert backend.teardown.call_args.args[0].external_id == "job-123"
