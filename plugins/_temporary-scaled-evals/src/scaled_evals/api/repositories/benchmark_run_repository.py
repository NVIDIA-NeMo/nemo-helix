# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# The repository below defines a ``list`` method, which shadows the builtin for
# annotations in the same class body, so those spell the type ``builtins.list``.
import builtins
from typing import Any

import psycopg
from psycopg.types.json import Json

from scaled_evals.api.failure_diagnostics import failure_category_for_code, is_retryable_failure
from scaled_evals.api.repositories.base_repository import (
    created_at_cursor_clause,
    join_where,
    normalize_order,
    order_by_clause,
    substring_search_pattern,
)
from scaled_evals.api.repositories.evaluation_repository import (
    EVALUATION_COLUMNS,
    EvaluationRepository,
)

# benchmark_runs stores only identity + run config + the explicit-cancel marker.
# status / reward / per-task breakdown are derived on read from member
# evaluations (see derive_run_view), never materialized here.
BENCHMARK_RUN_COLUMNS = (
    "id, owner_id, name, framework, requested_framework_version, framework_version, "
    "runner_image_ref, runner_image_digest, framework_adapter_version, sandbox_k8s_version, "
    "runner_metadata, "
    "benchmark_id, benchmark_revision, "
    "framework_profile_id, harbor_profile_id, intake_profile_id, "
    "credentials, runtime, network_policy, network_policy_config, parallelism, "
    "visibility, "
    "cancelled_at, created_at, updated_at"
)
_TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")
_CANCELLABLE_SQL = "('blocked', 'queued', 'provisioning', 'running')"
_DERIVED_STATUS_SQL = """
    CASE
        WHEN br.cancelled_at IS NOT NULL THEN 'cancelled'
        WHEN NOT EXISTS (
            SELECT 1
            FROM evaluations member
            WHERE member.benchmark_run_id = br.id AND member.deleted_at IS NULL
        ) OR EXISTS (
            SELECT 1
            FROM evaluations member
            WHERE member.benchmark_run_id = br.id
              AND member.deleted_at IS NULL
              AND member.status NOT IN ('succeeded', 'failed', 'cancelled')
        ) THEN 'running'
        WHEN NOT EXISTS (
            SELECT 1
            FROM evaluations member
            WHERE member.benchmark_run_id = br.id
              AND member.deleted_at IS NULL
              AND (member.status <> 'succeeded' OR member.reward IS NULL)
        ) THEN 'succeeded'
        ELSE 'failed'
    END
"""

# Resolved member tasks of a benchmark revision, in benchmark order. A member's
# NULL task_revision floats to the task's current revision (COALESCE); the
# LEFT JOIN yields status for the resolved revision (NULL when the task has no
# revision yet — the router validates `ready`).
_LOAD_MEMBERS_SQL = """
    SELECT
        m.task_id,
        m.position,
        COALESCE(m.task_revision, t.current_revision) AS task_revision,
        t.slug AS task_slug,
        r.status AS revision_status,
        r.image_ref,
        r.image_digest,
        r.tarball_object_key
    FROM benchmark_revision_tasks m
    JOIN tasks t ON t.id = m.task_id
    LEFT JOIN task_revisions r
        ON r.task_id = m.task_id
       AND r.revision = COALESCE(m.task_revision, t.current_revision)
    WHERE m.benchmark_id = %s AND m.revision = %s
    ORDER BY m.position, m.task_id
"""

# Member executions of one or more runs (for derive_run_view). Ordered by
# task_id for a stable per-task breakdown.
_MEMBERS_FOR_RUNS_SQL = """
    SELECT
        e.id, e.benchmark_run_id, e.status, e.reward, e.n_trials, e.n_completed,
        e.n_errored, e.n_failed_solve, e.exception_counts, e.status_detail,
        e.current_execution, e.max_executions, e.infrastructure_retries,
        e.max_infrastructure_retries, e.next_retry_at,
        e.last_failure_code, e.last_failure_category,
        e.cancel_teardown_status, e.cancel_teardown_error,
        e.finished_at, e.task_id, e.task_revision,
        t.slug AS task_slug, t.name AS task_name
    FROM evaluations e
    JOIN tasks t ON t.id = e.task_id
    WHERE e.benchmark_run_id = ANY(%s) AND e.deleted_at IS NULL
    ORDER BY e.task_id, e.id
"""


def derive_run_view(run: dict, members: list[dict]) -> dict:
    """Derive a benchmark run's status + reward + per-task breakdown from members.

    Pure read-time rollup — nothing is written. ``run`` is a benchmark_runs row;
    ``members`` are its member evaluation rows. Status: ``cancelled`` if the run
    was cancelled; else ``running`` while any member is non-terminal; else
    ``succeeded`` only if every member succeeded, otherwise ``failed``. Reward is
    the mean of the per-member rewards (over members that scored). Returns the run
    dict augmented with status/status_detail/reward/trial outcome counts/finished_at
    and the ``result`` envelope (kind="benchmark") the UI/CLI render.
    """
    per_task: list[dict[str, Any]] = []
    rewards: list[float] = []
    n_trials = 0
    n_completed = 0
    n_errored = 0
    n_failed_solve = 0
    exception_counts: dict[str, int] = {}
    finished_ats: list[str] = []
    failed: list[str] = []
    n_teardown_pending = 0
    n_teardown_failed = 0
    n_retryable_failures = 0
    n_recovered = 0
    failure_counts: dict[str, int] = {}
    recovered_counts: dict[str, int] = {}
    original_failures: list[dict[str, Any]] = []
    recovered_tasks: list[dict[str, Any]] = []
    all_terminal = True

    for m in members:
        status = m.get("status")
        reward = m.get("reward")
        terminal = status in _TERMINAL_STATUSES
        teardown_status = m.get("cancel_teardown_status") or "not_requested"
        if teardown_status == "pending":
            n_teardown_pending += 1
        elif teardown_status == "failed":
            n_teardown_failed += 1
        if not terminal:
            all_terminal = False
        succeeded = status == "succeeded" and reward is not None
        attempts = int(m.get("current_execution") or 1)
        max_executions = int(m.get("max_executions") or attempts)
        infrastructure_retries = int(m.get("infrastructure_retries") or 0)
        raw_max_infrastructure_retries = m.get("max_infrastructure_retries")
        max_infrastructure_retries = int(
            2 if raw_max_infrastructure_retries is None else raw_max_infrastructure_retries
        )
        max_attempts = max_executions + max_infrastructure_retries
        member_exception_counts = m.get("exception_counts") or {}
        last_failure_code = m.get("last_failure_code")
        if last_failure_code is None and member_exception_counts:
            last_failure_code = next(iter(member_exception_counts.keys()))
        raw_failure_category = m.get("last_failure_category")
        failure_category = raw_failure_category or failure_category_for_code(
            last_failure_code,
            m.get("status_detail"),
            default="task" if status == "failed" else "unknown",
        )
        if failure_category == "retryable_task":
            failure_category = "infrastructure"
        elif failure_category == "non_retryable":
            failure_category = "task"
        retry_budget_available = (
            infrastructure_retries < max_infrastructure_retries
            if failure_category == "infrastructure"
            else attempts - 1 - infrastructure_retries < max_executions - 1
        )
        retryable_failure = (
            status == "failed"
            and retry_budget_available
            and is_retryable_failure(last_failure_code, m.get("status_detail"))
        )
        recovered = succeeded and attempts > 1 and last_failure_code is not None
        if status == "failed":
            failure_counts[failure_category] = failure_counts.get(failure_category, 0) + 1
            if retryable_failure:
                n_retryable_failures += 1
        if recovered:
            n_recovered += 1
            recovered_counts[failure_category] = recovered_counts.get(failure_category, 0) + 1

        n_trials += m.get("n_trials") or 1
        n_completed += m.get("n_completed") or 0
        n_failed_solve += m.get("n_failed_solve") or 0
        for name, count in member_exception_counts.items():
            exception_counts[str(name)] = exception_counts.get(str(name), 0) + int(count)
        member_errored = m.get("n_errored")
        if member_errored is None:
            member_errored = 1 if (terminal and status != "succeeded") else 0
        elif terminal and status != "succeeded" and member_errored == 0:
            member_errored = 1
        n_errored += member_errored

        if succeeded:
            rewards.append(reward)
        elif terminal:
            failed.append(m.get("task_id") or m["id"])
        finished = _isoformat(m.get("finished_at"))
        if finished:
            finished_ats.append(finished)

        failure_evidence = {
            "code": last_failure_code,
            "category": failure_category if last_failure_code or status == "failed" else None,
            "detail": m.get("status_detail"),
            "exception_counts": member_exception_counts,
            "attempt": attempts,
            "max_attempts": max_attempts,
            "next_retry_at": _isoformat(m.get("next_retry_at")),
        }
        member_view = {
            "evaluation_id": m["id"],
            "task_id": m.get("task_id"),
            "task_slug": m.get("task_slug"),
            "task_name": m.get("task_name"),
            "task_revision": m.get("task_revision"),
            "status": status,
            "reward": reward,
            "attempt": attempts,
            "max_attempts": max_attempts,
            "retryable": retryable_failure,
            "recovered": recovered,
            "failure_category": failure_evidence["category"],
            "failure_code": last_failure_code,
            "failure_evidence": failure_evidence,
            "n_trials": m.get("n_trials"),
            "n_completed": m.get("n_completed"),
            "n_errored": m.get("n_errored"),
            "n_failed_solve": m.get("n_failed_solve"),
            "exception_counts": member_exception_counts,
            "finished_at": finished,
            "detail": m.get("status_detail"),
            "cancel_teardown_status": teardown_status,
            "cancel_teardown_error": m.get("cancel_teardown_error"),
        }
        per_task.append(member_view)
        if status == "failed" or last_failure_code:
            original_failures.append(member_view)
        if recovered:
            recovered_tasks.append(member_view)

    if run.get("cancelled_at") is not None:
        status = "cancelled"
    elif not members or not all_terminal:
        status = "running"
    elif not failed and len(rewards) == len(members):
        status = "succeeded"
    else:
        status = "failed"

    reward = sum(rewards) / len(rewards) if rewards else None
    detail = f"{len(rewards)}/{len(members)} member task(s) succeeded"
    if failed:
        detail = f"{detail}; failed: {', '.join(str(f) for f in failed)}"
    if n_teardown_pending:
        detail = f"{detail}; teardown pending: {n_teardown_pending}"
    if n_teardown_failed:
        detail = f"{detail}; teardown failed: {n_teardown_failed}"

    view = dict(run)
    view.update(
        status=status,
        status_detail=detail,
        reward=reward,
        n_trials=n_trials,
        n_completed=n_completed,
        n_errored=n_errored,
        n_failed_solve=n_failed_solve,
        exception_counts=exception_counts,
        n_teardown_pending=n_teardown_pending,
        n_teardown_failed=n_teardown_failed,
        n_retryable_failures=n_retryable_failures,
        n_recovered=n_recovered,
        failure_counts=failure_counts,
        recovered_counts=recovered_counts,
        finished_at=max(finished_ats) if (status in _TERMINAL_STATUSES and finished_ats) else None,
        result={
            "kind": "benchmark",
            "benchmark_id": run.get("benchmark_id"),
            "benchmark_revision": run.get("benchmark_revision"),
            "aggregate": {
                "reward": reward,
                "n_tasks": len(members),
                "n_tasks_scored": len(rewards),
                "n_trials": n_trials,
                "n_completed": n_completed,
                "n_errored": n_errored,
                "n_failed_solve": n_failed_solve,
                "exception_counts": exception_counts,
                "n_teardown_pending": n_teardown_pending,
                "n_teardown_failed": n_teardown_failed,
                "n_retryable_failures": n_retryable_failures,
                "n_recovered": n_recovered,
                "failure_counts": failure_counts,
                "recovered_counts": recovered_counts,
            },
            "per_task": per_task,
            "original_failures": original_failures,
            "recovered_tasks": recovered_tasks,
        },
    )
    return view


def _isoformat(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


class BenchmarkRunRepository:
    """A benchmark run: identity + config; its members are ordinary evaluations.

    Running a benchmark spawns one ``evaluations`` row per member task
    (``benchmark_run_id`` set); the worker pool runs them independently. The
    run's status/reward are derived on read from those members (derive_run_view),
    not stored — so there is no worker-side fan-in.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def benchmark_revision_for_run(self, benchmark_id: str, revision: int | None = None) -> dict[str, Any] | None:
        """Resolve a live benchmark revision and its advisory qualification state."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT current_revision, qualification_status,
                       qualification_evidence, qualified_at
                FROM benchmarks
                WHERE id = %s AND deleted_at IS NULL
                """,
                (benchmark_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            resolved_revision = revision if revision is not None else row["current_revision"]
            if revision is not None:
                cur.execute(
                    "SELECT 1 FROM benchmark_revisions WHERE benchmark_id = %s AND revision = %s",
                    (benchmark_id, revision),
                )
                if cur.fetchone() is None:
                    return None
            return {
                "revision": resolved_revision,
                "qualification_status": row.get("qualification_status"),
                "qualification_evidence": row.get("qualification_evidence") or {},
                "qualified_at": row.get("qualified_at"),
            }

    def load_experiment_identity(self, benchmark_run_id: str) -> dict | None:
        """Resolve a benchmark run to its intake Experiment identity.

        Returns ``{benchmark_slug, run_name}`` — the benchmark slug groups the
        Experiment (one group per benchmark) and names its dataset. ``None`` when
        the run (or its benchmark) is missing.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT b.slug AS benchmark_slug, br.name AS run_name
                FROM benchmark_runs br
                JOIN benchmarks b ON b.id = br.benchmark_id
                WHERE br.id = %s
                """,
                (benchmark_run_id,),
            )
            return cur.fetchone()

    def load_members(self, benchmark_id: str, revision: int) -> builtins.list[dict]:
        """Member tasks of a benchmark revision, each resolved to a task revision."""
        with self.conn.cursor() as cur:
            cur.execute(_LOAD_MEMBERS_SQL, (benchmark_id, revision))
            return cur.fetchall()

    def create_run(
        self,
        run_id: str,
        *,
        name: str,
        framework: str,
        requested_framework_version: str | None,
        framework_version: str | None,
        runner_image_ref: str | None,
        runner_image_digest: str | None,
        framework_adapter_version: str | None,
        sandbox_k8s_version: str | None,
        runner_metadata: dict[str, Any],
        benchmark_id: str,
        benchmark_revision: int,
        members: builtins.list[dict],
        framework_profile_id: str | None,
        harbor_profile_id: str | None,
        intake_profile_id: str | None,
        credentials: dict[str, str],
        runtime: str,
        network_policy: str,
        network_policy_config: dict[str, Any],
        parallelism: int,
        visibility: str,
        owner_id: str,
        n_attempts: int = 1,
        extra_skill_object_keys: builtins.list[str] | None = None,
        instruction_prefix: str | None = None,
        instruction_postfix: str | None = None,
        initial_user_turns: builtins.list[str] | None = None,
    ) -> dict:
        """Insert the run row plus one member evaluation per member task.

        Each entry in ``members`` is ``{id, task_id, task_revision, task_slug}``.
        Members are ordinary queued task runs (``benchmark_run_id`` set)
        inheriting the run's profiles/credentials/runtime/parallelism. Returns the
        run row.
        """
        evaluations = EvaluationRepository(self.conn)
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id) VALUES (%s)
                ON CONFLICT (id) DO UPDATE SET last_seen_at = NOW(), updated_at = NOW()
                """,
                (owner_id,),
            )
            cur.execute(
                f"""
                INSERT INTO benchmark_runs (
                    id, owner_id, name, framework,
                    requested_framework_version, framework_version,
                    runner_image_ref, runner_image_digest,
                    framework_adapter_version, sandbox_k8s_version, runner_metadata,
                    benchmark_id, benchmark_revision,
                    framework_profile_id, harbor_profile_id,
                    intake_profile_id, credentials, runtime, network_policy,
                    network_policy_config, parallelism, visibility
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING {BENCHMARK_RUN_COLUMNS}
                """,
                (
                    run_id,
                    owner_id,
                    name,
                    framework,
                    requested_framework_version,
                    framework_version,
                    runner_image_ref,
                    runner_image_digest,
                    framework_adapter_version,
                    sandbox_k8s_version,
                    Json(runner_metadata),
                    benchmark_id,
                    benchmark_revision,
                    framework_profile_id,
                    harbor_profile_id,
                    intake_profile_id,
                    Json(credentials),
                    runtime,
                    network_policy,
                    Json(network_policy_config),
                    parallelism,
                    visibility,
                ),
            )
            run = cur.fetchone()
            for member in members:
                member_framework_profile_id = member.get("framework_profile_id") or framework_profile_id
                evaluations.create(
                    member["id"],
                    name=f"{name} · {member.get('task_slug') or member['task_id']}",
                    framework=framework,
                    requested_framework_version=requested_framework_version,
                    framework_version=framework_version,
                    runner_image_ref=runner_image_ref,
                    runner_image_digest=runner_image_digest,
                    framework_adapter_version=framework_adapter_version,
                    sandbox_k8s_version=sandbox_k8s_version,
                    runner_metadata=runner_metadata,
                    task_id=member["task_id"],
                    task_revision=member["task_revision"],
                    framework_profile_id=member_framework_profile_id,
                    harbor_profile_id=(member_framework_profile_id if framework == "harbor" else harbor_profile_id),
                    intake_profile_id=intake_profile_id,
                    credentials=credentials,
                    extra_skill_object_keys=extra_skill_object_keys or [],
                    instruction_prefix=instruction_prefix,
                    instruction_postfix=instruction_postfix,
                    initial_user_turns=initial_user_turns,
                    runtime=runtime,
                    network_policy=network_policy,
                    network_policy_config=network_policy_config,
                    n_attempts=n_attempts,
                    parallelism=parallelism,
                    visibility=visibility,
                    benchmark_run_id=run_id,
                    owner_id=owner_id,
                )
        return run

    def get(self, run_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT {BENCHMARK_RUN_COLUMNS} FROM benchmark_runs WHERE id = %s AND deleted_at IS NULL",
                (run_id,),
            )
            return cur.fetchone()

    def exists(self, run_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM benchmark_runs WHERE id = %s AND deleted_at IS NULL",
                (run_id,),
            )
            return cur.fetchone() is not None

    def list(
        self,
        *,
        limit: int,
        cursor: str | None,
        order: str,
        benchmark_id: str | None = None,
        shared: bool = False,
        q: str | None = None,
    ) -> builtins.list[dict]:
        direction = normalize_order(order)
        ordering = order_by_clause(("created_at", "id"), direction)
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        if benchmark_id is not None:
            clauses.append("benchmark_id = %s")
            params.append(benchmark_id)
        if shared:
            clauses.append("visibility <> 'private'")
        if search := substring_search_pattern(q):
            clauses.append(
                "(id ILIKE %s ESCAPE '\\' OR name ILIKE %s ESCAPE '\\' "
                "OR benchmark_id ILIKE %s ESCAPE '\\' "
                "OR framework ILIKE %s ESCAPE '\\' OR runtime ILIKE %s ESCAPE '\\' "
                f"OR ({_DERIVED_STATUS_SQL}) ILIKE %s ESCAPE '\\')"
            )
            params.extend([search] * 6)
        cursor_filter, cursor_params = created_at_cursor_clause(cursor, direction)
        if cursor_filter:
            clauses.append(cursor_filter)
            params.extend(cursor_params)
        params.append(limit + 1)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {BENCHMARK_RUN_COLUMNS}
                FROM benchmark_runs AS br
                WHERE {join_where(clauses)}
                ORDER BY {ordering}
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def members_for_runs(self, run_ids: builtins.list[str]) -> builtins.list[dict]:
        """Member evaluation rows for the given runs (for derive_run_view)."""
        if not run_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(_MEMBERS_FOR_RUNS_SQL, (run_ids,))
            return cur.fetchall()

    def cancel(self, run_id: str) -> tuple[dict | None, bool, builtins.list[dict]]:
        """Cancel a run (stamp cancelled_at) and its active members (soft cancel).

        Returns ``(run_row, cancelled_now, cancelled_members)``;
        ``cancelled_now`` is False if the run was already cancelled / not found.
        Newly cancelled member rows are returned so the API can apply the same
        runtime teardown contract as single-evaluation cancellation.
        """
        cancelled_members: list[dict] = []
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE benchmark_runs
                SET cancelled_at = NOW(), updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL AND cancelled_at IS NULL
                RETURNING {BENCHMARK_RUN_COLUMNS}
                """,
                (run_id,),
            )
            run = cur.fetchone()
            cancelled_now = run is not None
            if not cancelled_now:
                cur.execute(
                    f"SELECT {BENCHMARK_RUN_COLUMNS} FROM benchmark_runs WHERE id = %s AND deleted_at IS NULL",
                    (run_id,),
                )
                run = cur.fetchone()
            else:
                cur.execute(
                    f"""
                    UPDATE evaluations
                    SET status = 'cancelled',
                        status_detail = 'cancelled (benchmark run cancelled)',
                        cancel_teardown_status = 'pending',
                        cancel_teardown_error = NULL,
                        cancel_teardown_updated_at = NOW(),
                        finished_at = COALESCE(finished_at, NOW()),
                        dispatch_claimed_at = NULL,
                        dispatch_claimed_by = NULL,
                        evidence_status = 'building',
                        evidence_requested_at = NOW(),
                        evidence_error = NULL,
                        evidence_claimed_at = NULL,
                        evidence_claimed_by = NULL,
                        evidence_build_attempts = 0,
                        archive_status = 'building',
                        archive_requested_at = NOW(),
                        updated_at = NOW()
                    WHERE benchmark_run_id = %s AND deleted_at IS NULL
                      AND status IN {_CANCELLABLE_SQL}
                    RETURNING {EVALUATION_COLUMNS}
                    """,
                    (run_id,),
                )
                cancelled_members = cur.fetchall()
                for member in cancelled_members:
                    EvaluationRepository.insert_status_event(
                        cur,
                        member["id"],
                        "cancelled",
                        member.get("status_detail"),
                    )
        return run, cancelled_now, cancelled_members

    def soft_delete(self, run_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE benchmark_runs SET deleted_at = NOW(), updated_at = NOW() "
                "WHERE id = %s AND deleted_at IS NULL RETURNING id",
                (run_id,),
            )
            return cur.fetchone() is not None
