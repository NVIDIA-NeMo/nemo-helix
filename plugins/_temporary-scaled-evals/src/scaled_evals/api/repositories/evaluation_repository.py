# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# The repository below defines a ``list`` method, which shadows the builtin for
# annotations in the same class body, so those spell the type ``builtins.list``.
import builtins
from datetime import UTC, datetime
from typing import Any, Literal, cast

import psycopg
from psycopg.types.json import Json

from scaled_evals.api.failure_diagnostics import failure_category_for_code
from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.api.repositories.base_repository import (
    InvalidReference,
    created_at_cursor_clause,
    join_where,
    normalize_order,
    order_by_clause,
    substring_search_pattern,
)
from scaled_evals.api.schemas.common import decode_cursor
from scaled_evals.models.evaluations import EvaluationResultWrite
from scaled_evals.models.execution_snapshot import (
    build_execution_snapshot,
    current_process_identity,
)

EVALUATION_COLUMNS = (
    "id, owner_id, name, framework, requested_framework_version, framework_version, "
    "runner_image_ref, runner_image_digest, framework_adapter_version, sandbox_k8s_version, "
    "runner_metadata, "
    "task_id, task_revision, "
    "benchmark_run_id, "
    "framework_profile_id, "
    "harbor_profile_id, switchyard_profile_id, intake_profile_id, "
    "credentials, runtime, network_policy, network_policy_config, n_attempts, parallelism, "
    "visibility, status, status_detail, "
    "cancel_teardown_status, cancel_teardown_error, cancel_teardown_updated_at, "
    "backend_handle, dispatch_job_name, dispatch_job_uid, "
    "current_execution, max_executions, infrastructure_retries, "
    "max_infrastructure_retries, next_retry_at, "
    "last_failure_code, last_failure_category, "
    "reward, reward_value, n_trials, n_completed, n_errored, n_failed_solve, "
    "exception_counts, finished_at, created_at, updated_at"
)
_TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")
EVALUATION_DETAIL_COLUMNS = (
    "e.id, e.owner_id, e.name, e.framework, e.requested_framework_version, "
    "e.framework_version, e.runner_image_ref, e.runner_image_digest, "
    "e.framework_adapter_version, e.sandbox_k8s_version, e.runner_metadata, "
    "e.task_id, e.task_revision, "
    "e.benchmark_run_id, "
    "e.framework_profile_id, "
    "e.harbor_profile_id, e.switchyard_profile_id, e.intake_profile_id, "
    "e.credentials, e.extra_skill_object_keys, e.instruction_prefix, "
    "e.instruction_postfix, e.initial_user_turns, "
    "e.runtime, e.network_policy, e.network_policy_config, e.n_attempts, "
    "e.parallelism, e.visibility, "
    "e.status, e.status_detail, "
    "e.cancel_teardown_status, e.cancel_teardown_error, e.cancel_teardown_updated_at, "
    "e.backend_handle, e.dispatch_job_name, e.dispatch_job_uid, "
    "e.current_execution, e.max_executions, e.infrastructure_retries, "
    "e.max_infrastructure_retries, e.next_retry_at, "
    "e.last_failure_code, e.last_failure_category, "
    "e.reward, e.reward_value, e.n_trials, e.n_completed, e.n_errored, "
    "e.n_failed_solve, e.exception_counts, e.finished_at, e.created_at, "
    "e.updated_at, e.result, e.evidence_status, e.evidence_error, "
    "e.archive_status, e.archive_error, "
    "r.image_ref AS image_ref, r.image_digest AS image_digest"
)
_CANCELLABLE_SQL = "('blocked', 'queued', 'provisioning', 'running')"
RetryBlockReason = Literal["terminal_artifacts_finalizing", "benchmark_unavailable"]

_DISPATCH_CLAIM_LOCK_ID = 1936024438
_CLAIM_LOCK_SQL = "SELECT pg_advisory_xact_lock(%s)"

_CLAIM_SQL = """
    WITH active_usage AS MATERIALIZED (
        SELECT COALESCE(SUM(parallelism), 0)::BIGINT AS cluster_slots
        FROM evaluations
        WHERE deleted_at IS NULL
          AND status IN ('provisioning', 'running')
    ),
    ranked_candidate AS MATERIALIZED (
        SELECT e.id, e.status
        FROM evaluations e
        CROSS JOIN active_usage usage
        LEFT JOIN benchmark_runs br
          ON br.id = e.benchmark_run_id
        LEFT JOIN LATERAL (
            SELECT COALESCE(SUM(owned.parallelism), 0)::BIGINT AS owner_slots
            FROM evaluations owned
            WHERE owned.deleted_at IS NULL
              AND owned.status IN ('provisioning', 'running')
              AND owned.owner_id IS NOT DISTINCT FROM e.owner_id
        ) owner_usage ON TRUE
        LEFT JOIN LATERAL (
            SELECT COUNT(*)::BIGINT AS active_members
            FROM evaluations member
            WHERE member.deleted_at IS NULL
              AND member.status IN ('provisioning', 'running')
              AND member.benchmark_run_id = e.benchmark_run_id
        ) benchmark_usage ON e.benchmark_run_id IS NOT NULL
        WHERE e.deleted_at IS NULL
          AND e.status IN ('queued', 'provisioning', 'running')
          AND e.dispatch_job_name IS NULL
          AND (e.next_retry_at IS NULL OR e.next_retry_at <= statement_timestamp())
          AND (
              e.dispatch_claimed_at IS NULL
              OR e.dispatch_claimed_at < statement_timestamp() - (%s * INTERVAL '1 second')
          )
          AND (
              e.status <> 'queued'
              OR (
                  e.parallelism <= %s
                  AND usage.cluster_slots + e.parallelism <= %s
                  AND owner_usage.owner_slots + e.parallelism <= %s
                  AND (
                      e.benchmark_run_id IS NULL
                      OR br.max_concurrent_members IS NULL
                      OR benchmark_usage.active_members < br.max_concurrent_members
                  )
              )
          )
        ORDER BY
            CASE e.status
                WHEN 'running' THEN 0
                WHEN 'provisioning' THEN 1
                WHEN 'queued' THEN 2
                ELSE 3
            END,
            -- Once recovery work is exhausted, admit the owner currently
            -- consuming the fewest sandbox slots. This prevents one older
            -- benchmark from monopolizing a smaller dispatch-worker pool.
            CASE WHEN e.status = 'queued' THEN owner_usage.owner_slots ELSE 0 END,
            e.created_at
        LIMIT 1
    ),
    candidate AS (
        -- Keep row locking in a separate, non-aggregate SELECT. PostgreSQL
        -- rejects FOR UPDATE anywhere in the ranked query because it contains
        -- aggregate capacity subqueries, even when the lock is qualified.
        SELECT e.id, e.status
        FROM evaluations e
        JOIN ranked_candidate ranked
          ON ranked.id = e.id
         AND ranked.status = e.status
        FOR UPDATE OF e SKIP LOCKED
    )
    UPDATE evaluations e
    SET status = CASE WHEN e.status = 'queued' THEN 'provisioning' ELSE e.status END,
        status_detail = CASE
            WHEN e.status = 'queued' THEN 'claimed by dispatch worker'
            ELSE e.status_detail
        END,
        next_retry_at = NULL,
        dispatch_claimed_at = statement_timestamp(),
        dispatch_claimed_by = %s,
        dispatch_attempts = dispatch_attempts + 1,
        updated_at = statement_timestamp()
    FROM candidate
    WHERE e.id = candidate.id
      AND (
          e.dispatch_claimed_at IS NULL
          OR e.dispatch_claimed_at < statement_timestamp() - (%s * INTERVAL '1 second')
      )
    RETURNING e.id, candidate.status AS previous_status, e.status
"""

# Every evaluation targets a task revision; a member evaluation of a benchmark
# run additionally carries benchmark_run_id (so the worker can finalize the run
# once its members finish).
_LOAD_FOR_DISPATCH_SQL = """
    SELECT
        e.id,
        e.name,
        e.framework,
        e.requested_framework_version,
        e.framework_version,
        e.runner_image_ref,
        e.runner_image_digest,
        e.framework_adapter_version,
        e.sandbox_k8s_version,
        e.runner_metadata,
        e.task_id,
        e.task_revision,
        e.benchmark_run_id,
        br.max_concurrent_members,
        e.runtime,
        e.network_policy,
        e.network_policy_config,
        e.n_attempts,
        e.parallelism,
        e.status,
        e.framework_profile_id,
        e.harbor_profile_id,
        e.switchyard_profile_id,
        e.intake_profile_id,
        e.credentials,
        e.execution_snapshot,
        e.backend_handle,
        e.dispatch_job_name,
        e.dispatch_job_uid,
        e.current_execution,
        e.max_executions,
        e.infrastructure_retries,
        e.max_infrastructure_retries,
        e.next_retry_at,
        e.last_failure_code,
        e.last_failure_category,
        b.slug AS task_slug,
        r.image_ref,
        r.image_digest,
        r.tarball_sha256,
        r.tarball_object_key,
        r.tarball_size_bytes,
        r.build_backend,
        r.build_payload,
        r.build_completed_at,
        e.extra_skill_object_keys,
        e.instruction_prefix,
        e.instruction_postfix,
        e.initial_user_turns,
        e.status_detail,
        e.result,
        e.reward_value,
        e.reward,
        e.n_trials,
        e.n_completed,
        e.n_errored,
        e.n_failed_solve,
        e.exception_counts,
        e.created_at,
        e.updated_at,
        e.finished_at
    FROM evaluations e
    JOIN task_revisions r
        ON r.task_id = e.task_id
       AND r.revision = e.task_revision
    JOIN tasks b ON b.id = e.task_id
    LEFT JOIN benchmark_runs br ON br.id = e.benchmark_run_id
    WHERE e.id = %s AND e.deleted_at IS NULL
"""

_HEARTBEAT_SQL = """
    UPDATE evaluations
    SET dispatch_claimed_at = NOW(),
        updated_at = NOW()
    WHERE id = %s
      AND dispatch_claimed_by = %s
      AND deleted_at IS NULL
      AND status IN ('queued', 'provisioning', 'running')
      AND (%s::integer IS NULL OR current_execution = %s::integer)
    RETURNING 1
"""

_ARCHIVE_CLAIM_SQL = """
    WITH candidate AS (
        SELECT id
        FROM evaluations
        WHERE deleted_at IS NULL
          AND status IN ('succeeded', 'failed', 'cancelled')
          AND archive_status = 'building'
          AND (evidence_status IN ('ready', 'missing') OR execution_snapshot IS NULL)
          AND (
              archive_claimed_at IS NULL
              OR archive_claimed_at < NOW() - (%s * INTERVAL '1 second')
          )
        ORDER BY COALESCE(archive_requested_at, updated_at), id
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE evaluations e
    SET archive_claimed_at = NOW(),
        archive_claimed_by = %s,
        archive_build_attempts = archive_build_attempts + 1,
        updated_at = NOW()
    FROM candidate
    WHERE e.id = candidate.id
    RETURNING e.id
"""

_EVIDENCE_CLAIM_SQL = """
    WITH candidate AS (
        SELECT id
        FROM evaluations
        WHERE deleted_at IS NULL
          AND status IN ('succeeded', 'failed', 'cancelled')
          AND evidence_status = 'building'
          AND evidence_build_attempts < 5
          AND (
              benchmark_run_id IS NULL
              OR NOT EXISTS (
                  SELECT 1 FROM benchmark_switchyard_campaigns c
                  WHERE c.benchmark_run_id = evaluations.benchmark_run_id
                    AND c.evidence_status NOT IN ('ready', 'unavailable')
              )
          )
          AND (
              evidence_claimed_at IS NULL
              OR evidence_claimed_at < NOW() - (%s * INTERVAL '1 second')
          )
        ORDER BY COALESCE(evidence_requested_at, updated_at), id
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE evaluations e
    SET evidence_claimed_at = NOW(),
        evidence_claimed_by = %s,
        evidence_build_attempts = evidence_build_attempts + 1,
        updated_at = NOW()
    FROM candidate
    WHERE e.id = candidate.id
    RETURNING e.id
"""


class EvaluationRepository:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    def task_revision_for_evaluation(self, task_id: str, revision: int) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, image_ref, image_digest, tarball_object_key
                FROM task_revisions
                WHERE task_id = %s AND revision = %s
                """,
                (task_id, revision),
            )
            return cur.fetchone()

    def validate_profile_references(self, profile_slots: builtins.list[tuple[str, str]]) -> None:
        if not profile_slots:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, type FROM config_profiles WHERE id = ANY(%s) AND deleted_at IS NULL",
                ([profile_id for profile_id, _ in profile_slots],),
            )
            found = {row["id"]: row["type"] for row in cur.fetchall()}
        for profile_id, expected_type in profile_slots:
            if profile_id not in found:
                raise InvalidReference(f"config profile not found: {profile_id}")
            actual_type = found[profile_id]
            if actual_type != expected_type:
                raise InvalidReference(f"{profile_id} is a '{actual_type}' profile, expected '{expected_type}'")

    def validate_credential_references(
        self,
        credential_ids: builtins.list[str],
        *,
        owner_id: str,
        include_unowned: bool = False,
    ) -> None:
        if not credential_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM credentials
                WHERE id = ANY(%s) AND deleted_at IS NULL
                  AND (owner_id = %s OR (%s AND owner_id IS NULL))
                """,
                (credential_ids, owner_id, include_unowned),
            )
            found = {row["id"] for row in cur.fetchall()}
        missing = [credential_id for credential_id in credential_ids if credential_id not in found]
        if missing:
            raise InvalidReference(f"credential not found: {missing[0]}")

    @staticmethod
    def _benchmark_variant_snapshot(cur: psycopg.Cursor, benchmark_run_id: str | None) -> dict[str, Any] | None:
        """Freeze the run's benchmark-variant lineage and operational policy.

        Dispatch and provenance read the frozen copy, so editing a variant later
        cannot change how an already-submitted evaluation runs or reports.
        """
        if not benchmark_run_id:
            return None
        cur.execute(
            """
            SELECT brv.derived_from_benchmark_id, brv.derived_from_revision, brv.operational_policy
            FROM benchmark_runs br
            JOIN benchmark_revisions brv
              ON brv.benchmark_id = br.benchmark_id
             AND brv.revision = br.benchmark_revision
            WHERE br.id = %s AND br.deleted_at IS NULL
            FOR KEY SHARE OF brv
            """,
            (benchmark_run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        policy = row["operational_policy"] if isinstance(row["operational_policy"], dict) else {}
        if not policy and row["derived_from_benchmark_id"] is None:
            return None
        return {
            "derived_from": (
                None
                if row["derived_from_benchmark_id"] is None
                else {
                    "benchmark_id": row["derived_from_benchmark_id"],
                    "revision": row["derived_from_revision"],
                }
            ),
            "operational_policy": dict(policy),
        }

    @staticmethod
    def _capture_execution_snapshot(
        cur: psycopg.Cursor,
        *,
        evaluation: dict[str, Any],
        task_id: str,
        task_revision: int,
        profile_ids: dict[str, str | None],
        credentials: dict[str, str],
    ) -> dict[str, Any]:
        """Capture mutable execution inputs under row locks in the insert transaction."""
        cur.execute(
            """
            SELECT
                t.id, t.name, t.slug, r.revision, r.status,
                r.image_ref, r.image_digest, r.tarball_object_key,
                r.tarball_size_bytes, r.tarball_sha256, r.build_backend,
                r.build_payload, r.build_completed_at, r.created_at
            FROM tasks t
            JOIN task_revisions r ON r.task_id = t.id
            WHERE t.id = %s AND r.revision = %s AND t.deleted_at IS NULL
            FOR KEY SHARE OF t, r
            """,
            (task_id, task_revision),
        )
        task = cur.fetchone()
        if task is None:
            raise InvalidReference(f"task revision not found: {task_id} rev {task_revision}")

        wanted_profiles = {role: value for role, value in profile_ids.items() if value}
        profiles_by_id: dict[str, dict[str, Any]] = {}
        if wanted_profiles:
            cur.execute(
                """
                SELECT id, name, type, config, updated_at
                FROM config_profiles
                WHERE id = ANY(%s) AND deleted_at IS NULL
                FOR KEY SHARE
                """,
                (sorted(set(wanted_profiles.values())),),
            )
            profiles_by_id = {str(row["id"]): row for row in cur.fetchall()}
        profiles: dict[str, dict[str, Any]] = {}
        for role, profile_id in wanted_profiles.items():
            if profile_id not in profiles_by_id:
                raise InvalidReference(f"config profile not found: {profile_id}")
            profiles[role] = dict(profiles_by_id[profile_id])

        credential_ids = sorted(set(credentials.values()))
        credentials_by_id: dict[str, dict[str, Any]] = {}
        if credential_ids:
            cur.execute(
                """
                SELECT id, provider, payload_kind, fingerprint, updated_at
                FROM credentials
                WHERE id = ANY(%s) AND deleted_at IS NULL
                FOR KEY SHARE
                """,
                (credential_ids,),
            )
            credentials_by_id = {str(row["id"]): row for row in cur.fetchall()}
        credential_snapshot: dict[str, dict[str, Any]] = {}
        for role, credential_id in credentials.items():
            if credential_id not in credentials_by_id:
                raise InvalidReference(f"credential not found: {credential_id}")
            credential_snapshot[role] = dict(credentials_by_id[credential_id])

        variant = EvaluationRepository._benchmark_variant_snapshot(cur, evaluation.get("benchmark_run_id"))
        if variant is not None:
            evaluation = {**evaluation, "benchmark_variant": variant}

        return build_execution_snapshot(
            captured_at=datetime.now(tz=UTC).isoformat(),
            evaluation=evaluation,
            task=task,
            profiles=profiles,
            credentials=credential_snapshot,
            submission_identity=current_process_identity(),
        )

    def _insert(
        self,
        cur: psycopg.Cursor,
        evaluation_id: str,
        *,
        status: str,
        name: str,
        framework: str,
        requested_framework_version: str | None,
        framework_version: str | None,
        runner_image_ref: str | None,
        runner_image_digest: str | None,
        framework_adapter_version: str | None,
        sandbox_k8s_version: str | None,
        runner_metadata: dict[str, Any],
        task_id: str,
        task_revision: int,
        benchmark_run_id: str | None,
        framework_profile_id: str | None,
        harbor_profile_id: str | None,
        switchyard_profile_id: str | None,
        intake_profile_id: str | None,
        credentials: dict[str, str],
        extra_skill_object_keys: builtins.list[str],
        instruction_prefix: str | None = None,
        instruction_postfix: str | None = None,
        initial_user_turns: builtins.list[str] | None = None,
        runtime: str,
        network_policy: str,
        network_policy_config: dict[str, Any],
        n_attempts: int = 1,
        parallelism: int,
        visibility: str,
        owner_id: str | None,
    ) -> dict:
        if owner_id is not None:
            cur.execute(
                "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                (owner_id,),
            )
        execution_snapshot = self._capture_execution_snapshot(
            cur,
            evaluation={
                "id": evaluation_id,
                "name": name,
                "framework": framework,
                "requested_framework_version": requested_framework_version,
                "framework_version": framework_version,
                "runner_image_ref": runner_image_ref,
                "runner_image_digest": runner_image_digest,
                "framework_adapter_version": framework_adapter_version,
                "sandbox_k8s_version": sandbox_k8s_version,
                "runner_metadata": runner_metadata,
                "benchmark_run_id": benchmark_run_id,
                "runtime": runtime,
                "network_policy": network_policy,
                "network_policy_config": network_policy_config,
                "n_attempts": n_attempts,
                "parallelism": parallelism,
                "visibility": visibility,
                "extra_skill_object_keys": extra_skill_object_keys,
                "instruction_prefix": instruction_prefix,
                "instruction_postfix": instruction_postfix,
                "initial_user_turns": initial_user_turns or [],
            },
            task_id=task_id,
            task_revision=task_revision,
            profile_ids={
                "framework": framework_profile_id,
                "harbor": harbor_profile_id,
                "switchyard": switchyard_profile_id,
                "intake": intake_profile_id,
            },
            credentials=credentials,
        )
        cur.execute(
            f"""
            INSERT INTO evaluations (
                id, owner_id, name, framework,
                requested_framework_version, framework_version,
                runner_image_ref, runner_image_digest,
                framework_adapter_version, sandbox_k8s_version, runner_metadata,
                task_id, task_revision,
                benchmark_run_id,
                framework_profile_id, harbor_profile_id, switchyard_profile_id,
                intake_profile_id, credentials, extra_skill_object_keys,
                instruction_prefix, instruction_postfix, initial_user_turns,
                runtime, network_policy, network_policy_config, n_attempts,
                parallelism, visibility, status, execution_snapshot
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING {EVALUATION_COLUMNS}
            """,
            (
                evaluation_id,
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
                task_id,
                task_revision,
                benchmark_run_id,
                framework_profile_id,
                harbor_profile_id,
                switchyard_profile_id,
                intake_profile_id,
                Json(credentials),
                extra_skill_object_keys,
                instruction_prefix,
                instruction_postfix,
                initial_user_turns or [],
                runtime,
                network_policy,
                Json(network_policy_config),
                n_attempts,
                parallelism,
                visibility,
                status,
                Json(execution_snapshot),
            ),
        )
        row = cur.fetchone()
        self.insert_status_event(cur, evaluation_id, status)
        return row

    def create(
        self,
        evaluation_id: str,
        *,
        name: str,
        framework: str,
        requested_framework_version: str | None = None,
        framework_version: str | None = None,
        runner_image_ref: str | None = None,
        runner_image_digest: str | None = None,
        framework_adapter_version: str | None = None,
        sandbox_k8s_version: str | None = None,
        runner_metadata: dict[str, Any] | None = None,
        task_id: str,
        task_revision: int,
        framework_profile_id: str | None,
        harbor_profile_id: str | None,
        switchyard_profile_id: str | None,
        intake_profile_id: str | None,
        credentials: dict[str, str],
        extra_skill_object_keys: builtins.list[str],
        instruction_prefix: str | None = None,
        instruction_postfix: str | None = None,
        initial_user_turns: builtins.list[str] | None = None,
        runtime: str,
        network_policy: str = "unrestricted",
        network_policy_config: dict[str, Any] | None = None,
        n_attempts: int = 1,
        parallelism: int,
        visibility: str,
        benchmark_run_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict:
        """Insert a queued task run (standalone, or a member of a benchmark run).

        A benchmark run sets ``benchmark_run_id`` so the row is one member
        execution of that run; otherwise it is NULL and the row is a standalone
        single-task run.
        """
        with self.conn.transaction(), self.conn.cursor() as cur:
            return self._insert(
                cur,
                evaluation_id,
                status="queued",
                name=name,
                framework=framework,
                requested_framework_version=requested_framework_version,
                framework_version=framework_version,
                runner_image_ref=runner_image_ref,
                runner_image_digest=runner_image_digest,
                framework_adapter_version=framework_adapter_version,
                sandbox_k8s_version=sandbox_k8s_version,
                runner_metadata=runner_metadata or {},
                task_id=task_id,
                task_revision=task_revision,
                benchmark_run_id=benchmark_run_id,
                framework_profile_id=framework_profile_id,
                harbor_profile_id=harbor_profile_id,
                switchyard_profile_id=switchyard_profile_id,
                intake_profile_id=intake_profile_id,
                credentials=credentials,
                extra_skill_object_keys=extra_skill_object_keys,
                instruction_prefix=instruction_prefix,
                instruction_postfix=instruction_postfix,
                initial_user_turns=initial_user_turns,
                runtime=runtime,
                network_policy=network_policy,
                network_policy_config=network_policy_config or {},
                n_attempts=n_attempts,
                parallelism=parallelism,
                visibility=visibility,
                owner_id=owner_id,
            )

    def list(
        self,
        *,
        limit: int,
        cursor: str | None,
        order: str,
        status: str | None,
        task_id: str | None,
        shared: bool,
        benchmark_run_id: str | None = None,
        owner_id: str | None = None,
        q: str | None = None,
    ) -> builtins.list[dict]:
        direction = normalize_order(order)
        ordering = order_by_clause(("created_at", "id"), direction)
        clauses = ["deleted_at IS NULL"]
        params: list[Any] = []
        if owner_id is not None:
            clauses.append("owner_id = %s")
            params.append(owner_id)
        if benchmark_run_id is not None:
            # List one benchmark run's member task evaluations.
            clauses.append("benchmark_run_id = %s")
            params.append(benchmark_run_id)
        else:
            # Top-level listing hides benchmark-run members (the run is listed via
            # /benchmark-runs); pass benchmark_run_id to drill into a run.
            clauses.append("benchmark_run_id IS NULL")
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if task_id is not None:
            clauses.append("task_id = %s")
            params.append(task_id)
        if shared:
            clauses.append("visibility <> 'private'")
        if search := substring_search_pattern(q):
            clauses.append(
                "(id ILIKE %s ESCAPE '\\' OR name ILIKE %s ESCAPE '\\' "
                "OR task_id ILIKE %s ESCAPE '\\' OR status::text ILIKE %s ESCAPE '\\' "
                "OR framework ILIKE %s ESCAPE '\\' OR runtime ILIKE %s ESCAPE '\\')"
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
                SELECT {EVALUATION_COLUMNS}
                FROM evaluations
                WHERE {join_where(clauses)}
                ORDER BY {ordering}
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def get(self, evaluation_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {EVALUATION_DETAIL_COLUMNS}
                FROM evaluations e
                LEFT JOIN task_revisions r
                  ON r.task_id = e.task_id
                 AND r.revision = e.task_revision
                WHERE e.id = %s AND e.deleted_at IS NULL
                """,
                (evaluation_id,),
            )
            return cur.fetchone()

    def exists(self, evaluation_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM evaluations WHERE id = %s AND deleted_at IS NULL",
                (evaluation_id,),
            )
            return cur.fetchone() is not None

    def load_observability_row(self, evaluation_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, status, status_detail, backend_handle, result, runtime,
                    created_at, updated_at, finished_at
                FROM evaluations
                WHERE id = %s AND deleted_at IS NULL
                """,
                (evaluation_id,),
            )
            return cur.fetchone()

    def load_status_runtime(self, evaluation_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, runtime, current_execution
                FROM evaluations
                WHERE id = %s
                """,
                (evaluation_id,),
            )
            return cur.fetchone()

    def list_cancelled_platform_jobs(self, *, limit: int = 100) -> builtins.list[dict[str, Any]]:
        """List cancelled evaluations whose outer Platform Job may still run."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, current_execution, dispatch_job_name, backend_handle
                FROM evaluations
                WHERE deleted_at IS NULL
                  AND status = 'cancelled'
                  AND dispatch_job_name LIKE 'scaled-evals-evaluation-%%'
                  AND cancel_teardown_status = 'pending'
                ORDER BY cancel_teardown_updated_at, id
                LIMIT %s
                """,
                (limit,),
            )
            return cast(builtins.list[dict[str, Any]], cur.fetchall())

    def load_archive_row(self, evaluation_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    archive_status,
                    archive_object_key,
                    archive_size_bytes,
                    archive_built_at,
                    archive_error
                FROM evaluations
                WHERE id = %s AND deleted_at IS NULL
                """,
                (evaluation_id,),
            )
            return cur.fetchone()

    def retry_failed(self, evaluation_id: str) -> dict | None:
        """Queue the next execution of a failed logical evaluation."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE evaluations
                SET status = 'queued',
                    status_detail = 'manual retry scheduled; execution '
                        || (current_execution + 1),
                    backend_handle = NULL,
                    dispatch_claimed_at = NULL,
                    dispatch_claimed_by = NULL,
                    dispatch_job_name = NULL,
                    dispatch_job_uid = NULL,
                    dispatch_reconcile_claimed_at = NULL,
                    dispatch_reconcile_claimed_by = NULL,
                    current_execution = current_execution + 1,
                    max_executions = GREATEST(
                        max_executions,
                        current_execution - infrastructure_retries + 1
                    ),
                    next_retry_at = NULL,
                    result = NULL,
                    reward_value = NULL,
                    reward = NULL,
                    n_trials = NULL,
                    n_completed = NULL,
                    n_errored = NULL,
                    n_failed_solve = NULL,
                    exception_counts = '{{}}'::jsonb,
                    finished_at = NULL,
                    evidence_status = 'missing',
                    evidence_requested_at = NULL,
                    evidence_built_at = NULL,
                    evidence_error = NULL,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    evidence_build_attempts = 0,
                    archive_status = 'missing',
                    archive_object_key = NULL,
                    archive_size_bytes = NULL,
                    archive_requested_at = NULL,
                    archive_built_at = NULL,
                    archive_error = NULL,
                    archive_claimed_at = NULL,
                    archive_claimed_by = NULL,
                    archive_build_attempts = 0,
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status = 'failed'
                  AND evidence_claimed_by IS NULL
                  AND archive_claimed_by IS NULL
                  AND (
                      benchmark_run_id IS NULL
                      OR EXISTS (
                          SELECT 1
                          FROM benchmark_runs
                          WHERE benchmark_runs.id = evaluations.benchmark_run_id
                            AND benchmark_runs.deleted_at IS NULL
                            AND benchmark_runs.cancelled_at IS NULL
                      )
                  )
                RETURNING {EVALUATION_COLUMNS}
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
            if row is not None:
                self.insert_event(
                    cur,
                    evaluation_id,
                    "queued",
                    f"manual retry {row['current_execution']}/{row['max_executions']} scheduled",
                    type="retry",
                )
            return row

    def retry_block_reason(self, evaluation_id: str) -> RetryBlockReason | None:
        """Explain why a failed evaluation did not transition to a retry."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT CASE
                    WHEN evidence_claimed_by IS NOT NULL
                      OR archive_claimed_by IS NOT NULL
                        THEN 'terminal_artifacts_finalizing'
                    WHEN benchmark_run_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM benchmark_runs
                          WHERE benchmark_runs.id = evaluations.benchmark_run_id
                            AND benchmark_runs.deleted_at IS NULL
                            AND benchmark_runs.cancelled_at IS NULL
                      )
                        THEN 'benchmark_unavailable'
                    ELSE NULL
                END AS reason
                FROM evaluations
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status = 'failed'
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row["reason"]

    def cancel(self, evaluation_id: str) -> tuple[dict | None, bool]:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE evaluations
                SET status = 'cancelled',
                    status_detail = 'cancelled',
                    cancel_teardown_status = 'pending',
                    cancel_teardown_error = NULL,
                    cancel_teardown_updated_at = NOW(),
                    dispatch_claimed_at = NULL,
                    dispatch_claimed_by = NULL,
                    finished_at = COALESCE(finished_at, NOW()),
                    evidence_status = 'building',
                    evidence_requested_at = NOW(),
                    evidence_error = NULL,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    evidence_build_attempts = 0,
                    archive_status = 'building',
                    archive_requested_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL AND status IN {_CANCELLABLE_SQL}
                RETURNING {EVALUATION_COLUMNS}
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
            cancelled_now = row is not None
            if cancelled_now:
                self.insert_status_event(cur, evaluation_id, "cancelled", row.get("status_detail"))
            if row is None:
                cur.execute(
                    f"SELECT {EVALUATION_COLUMNS} FROM evaluations WHERE id = %s AND deleted_at IS NULL",
                    (evaluation_id,),
                )
                row = cur.fetchone()
        return row, cancelled_now

    def record_cancel_teardown_failure(self, evaluation_id: str, detail: str) -> dict | None:
        detail = redact_secret_text(detail)
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE evaluations
                SET status_detail = %s,
                    cancel_teardown_status = 'failed',
                    cancel_teardown_error = %s,
                    cancel_teardown_updated_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL AND status = 'cancelled'
                RETURNING {EVALUATION_COLUMNS}
                """,
                (detail, detail, evaluation_id),
            )
            row = cur.fetchone()
            if row is not None:
                self.insert_status_event(cur, evaluation_id, "cancelled", detail)
        return row

    def record_cancel_teardown_succeeded(self, evaluation_id: str) -> dict | None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE evaluations
                SET cancel_teardown_status = 'succeeded',
                    cancel_teardown_error = NULL,
                    cancel_teardown_updated_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL AND status = 'cancelled'
                  AND cancel_teardown_status = 'pending'
                RETURNING {EVALUATION_COLUMNS}
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
            if row is not None:
                self.insert_status_event(
                    cur,
                    evaluation_id,
                    "cancelled",
                    "cancellation teardown succeeded",
                )
        return row

    def soft_delete(self, evaluation_id: str) -> bool:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET deleted_at = NOW(), updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING id
                """,
                (evaluation_id,),
            )
            return cur.fetchone() is not None

    def request_archive_build(self, evaluation_id: str, *, force: bool) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET archive_status = 'building',
                    archive_requested_at = NOW(),
                    archive_claimed_at = NULL,
                    archive_claimed_by = NULL,
                    archive_error = NULL,
                    archive_object_key = CASE WHEN %s THEN NULL ELSE archive_object_key END,
                    archive_size_bytes = CASE WHEN %s THEN NULL ELSE archive_size_bytes END,
                    archive_built_at = CASE WHEN %s THEN NULL ELSE archive_built_at END,
                    updated_at = NOW()
                WHERE id = %s AND deleted_at IS NULL
                RETURNING
                    id,
                    archive_status,
                    archive_object_key,
                    archive_size_bytes,
                    archive_built_at,
                    archive_error
                """,
                (force, force, force, evaluation_id),
            )
            return cur.fetchone()

    def list_events(
        self,
        evaluation_id: str,
        *,
        limit: int,
        cursor: str | None,
        offset: int,
    ) -> builtins.list[dict]:
        filters = ["evaluation_id = %s"]
        params: list[Any] = [evaluation_id]
        cursor_filter, cursor_params = self.event_cursor_clause(cursor)
        if cursor_filter:
            filters.append(cursor_filter)
            params.extend(cursor_params)
        params.extend([limit + 1, offset])
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, type, status, detail, created_at
                FROM evaluation_events
                WHERE {join_where(filters)}
                ORDER BY created_at ASC, id ASC
                LIMIT %s OFFSET %s
                """,
                params,
            )
            return cur.fetchall()

    def load_event_batch(
        self,
        evaluation_id: str,
        *,
        after_created_at: datetime | None = None,
        after_id: int | None = None,
        limit: int,
    ) -> builtins.list[dict]:
        filters = ["evaluation_id = %s"]
        params: list[Any] = [evaluation_id]
        if after_created_at is not None and after_id is not None:
            filters.append("(created_at, id) > (%s, %s)")
            params.extend([after_created_at, after_id])
        params.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, type, status, detail, created_at
                FROM evaluation_events
                WHERE {join_where(filters)}
                ORDER BY created_at ASC, id ASC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def load_stream_status(self, evaluation_id: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT status
                FROM evaluations
                WHERE id = %s AND deleted_at IS NULL
                """,
                (evaluation_id,),
            )
            row = cur.fetchone()
        return None if row is None else str(row["status"])

    def load_runtime_status(
        self,
        evaluation_id: str,
        *,
        expected_execution_number: int | None = None,
    ) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT status
                FROM evaluations
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )
            try:
                row = cur.fetchone()
            except StopIteration:
                row = None
        return None if row is None or "status" not in row else str(row["status"])

    def lock_current_execution(
        self,
        evaluation_id: str,
        *,
        expected_execution_number: int,
    ) -> bool:
        """Lock an evaluation while its current execution publishes stable objects."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM evaluations
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND current_execution = %s
                FOR UPDATE
                """,
                (evaluation_id, expected_execution_number),
            )
            return cur.rowcount != 0

    def claim_next(
        self,
        *,
        claim_timeout: float,
        worker_id: str,
        cluster_slot_limit: int = 500,
        per_user_slot_limit: int = 50,
    ) -> dict | None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            # Take the admission lock in its own statement. PostgreSQL establishes
            # a READ COMMITTED snapshot at statement start, so putting the lock in
            # the claim query lets waiters retain a snapshot from before the prior
            # worker committed. Those waiters can then all select and launch the
            # same evaluation. Acquiring first makes the claim query start with a
            # fresh snapshot after the previous claimant commits.
            cur.execute(_CLAIM_LOCK_SQL, (_DISPATCH_CLAIM_LOCK_ID,))
            cur.execute(
                _CLAIM_SQL,
                (
                    claim_timeout,
                    per_user_slot_limit,
                    cluster_slot_limit,
                    per_user_slot_limit,
                    worker_id,
                    claim_timeout,
                ),
            )
            row = cur.fetchone()
            if row is not None and row.get("previous_status") == "queued" and row.get("status") == "provisioning":
                self.insert_status_event(
                    cur,
                    row["id"],
                    "provisioning",
                    "claimed by dispatch worker",
                )
                cur.execute(
                    """
                    INSERT INTO evaluation_execution_telemetry (
                        evaluation_id, execution_number, provisioning_started_at
                    )
                    SELECT id, current_execution, statement_timestamp()
                    FROM evaluations
                    WHERE id = %s
                    ON CONFLICT (evaluation_id, execution_number) DO UPDATE SET
                        provisioning_started_at = COALESCE(
                            evaluation_execution_telemetry.provisioning_started_at,
                            EXCLUDED.provisioning_started_at
                        ),
                        updated_at = statement_timestamp()
                    """,
                    (row["id"],),
                )
            return row

    def claim_next_archive(self, *, claim_timeout: float, worker_id: str) -> dict | None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(_ARCHIVE_CLAIM_SQL, (claim_timeout, worker_id))
            return cur.fetchone()

    def claim_next_evidence(self, *, claim_timeout: float, worker_id: str) -> dict | None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(_EVIDENCE_CLAIM_SQL, (claim_timeout, worker_id))
            return cur.fetchone()

    def mark_evidence_ready(
        self,
        evaluation_id: str,
        *,
        expected_execution_number: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET evidence_status = 'ready',
                    evidence_built_at = NOW(),
                    evidence_error = NULL,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    archive_status = 'building',
                    archive_requested_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )

    def mark_evidence_failed(
        self,
        evaluation_id: str,
        detail: str,
        *,
        expected_execution_number: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET evidence_status = CASE
                        WHEN evidence_build_attempts >= 5 THEN 'missing'
                        ELSE evidence_status
                    END,
                    evidence_error = %s,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND evidence_status = 'building'
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    redact_secret_text(detail),
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )

    def heartbeat_claim(
        self,
        evaluation_id: str,
        *,
        worker_id: str,
        expected_execution_number: int | None = None,
    ) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                _HEARTBEAT_SQL,
                (
                    evaluation_id,
                    worker_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )
            return cur.fetchone() is not None

    def record_dispatch_job(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        name: str,
        uid: str,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET dispatch_job_name = %s,
                    dispatch_job_uid = NULLIF(%s, ''),
                    dispatch_claimed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                  AND status IN ('provisioning', 'running')
                  AND current_execution = %s
                  AND (dispatch_job_name IS NULL OR dispatch_job_name = %s)
                """,
                (name, uid, evaluation_id, execution_number, name),
            )

    def claim_stale_dispatch_job(
        self,
        *,
        stale_seconds: float,
        claim_timeout: float,
        worker_id: str,
    ) -> dict | None:
        """Lease one stale outer Job for bounded Kubernetes reconciliation."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate AS (
                    SELECT e.id
                    FROM evaluations e
                    WHERE e.deleted_at IS NULL
                      AND e.status IN ('provisioning', 'running')
                      AND e.dispatch_job_name IS NOT NULL
                      AND (
                          e.dispatch_claimed_at IS NULL
                          OR e.dispatch_claimed_at < NOW() - (%s * INTERVAL '1 second')
                      )
                      AND (
                          e.dispatch_reconcile_claimed_at IS NULL
                          OR e.dispatch_reconcile_claimed_at
                              < NOW() - (%s * INTERVAL '1 second')
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM evaluation_execution_cleanups cleanup
                          WHERE cleanup.evaluation_id = e.id
                            AND cleanup.execution_number = e.current_execution
                            AND cleanup.status <> 'deleted'
                      )
                    ORDER BY e.dispatch_claimed_at NULLS FIRST, e.created_at, e.id
                    LIMIT 1
                    FOR UPDATE OF e SKIP LOCKED
                )
                UPDATE evaluations e
                SET dispatch_reconcile_claimed_at = NOW(),
                    dispatch_reconcile_claimed_by = %s,
                    updated_at = NOW()
                FROM candidate
                WHERE e.id = candidate.id
                RETURNING e.id, e.status, e.current_execution, e.dispatch_job_name,
                          e.dispatch_job_uid
                """,
                (stale_seconds, claim_timeout, worker_id),
            )
            return cur.fetchone()

    def release_dispatch_reconcile_claim(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        dispatch_job_name: str,
        worker_id: str,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET dispatch_reconcile_claimed_at = NULL,
                    dispatch_reconcile_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND current_execution = %s
                  AND dispatch_job_name = %s
                  AND dispatch_reconcile_claimed_by = %s
                """,
                (evaluation_id, execution_number, dispatch_job_name, worker_id),
            )

    def record_dispatch_job_infrastructure_failure(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        dispatch_job_name: str,
        reconcile_worker_id: str,
        failure_code: str,
        detail: str,
        retry_delay_seconds: float,
    ) -> dict | None:
        """Own cleanup, then retry or terminalize one failed outer Job atomically."""
        detail = redact_secret_text(detail)
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, runtime, backend_handle, benchmark_run_id,
                       infrastructure_retries, max_infrastructure_retries
                FROM evaluations
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status IN ('provisioning', 'running')
                  AND current_execution = %s
                  AND dispatch_job_name = %s
                  AND dispatch_reconcile_claimed_by = %s
                FOR UPDATE
                """,
                (
                    evaluation_id,
                    execution_number,
                    dispatch_job_name,
                    reconcile_worker_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                """
                SELECT %s::text IS NULL OR EXISTS (
                    SELECT 1
                    FROM benchmark_runs
                    WHERE id = %s
                      AND deleted_at IS NULL
                      AND cancelled_at IS NULL
                ) AS benchmark_available
                """,
                (row.get("benchmark_run_id"), row.get("benchmark_run_id")),
            )
            benchmark_available = bool((cur.fetchone() or {}).get("benchmark_available"))
            retry_allowed = benchmark_available and int(row.get("infrastructure_retries") or 0) < int(
                row.get("max_infrastructure_retries") or 0
            )
            backend_handle = row.get("backend_handle")
            self._mark_failed_execution_resources(cur, evaluation_id)
            if backend_handle:
                cur.execute(
                    """
                    INSERT INTO evaluation_execution_cleanups (
                        evaluation_id, execution_number, runtime, backend_handle,
                        dispatch_job_name, failure_code, failure_detail,
                        retry_after_cleanup
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evaluation_id, execution_number) DO UPDATE
                    SET runtime = EXCLUDED.runtime,
                        backend_handle = EXCLUDED.backend_handle,
                        dispatch_job_name = EXCLUDED.dispatch_job_name,
                        failure_code = EXCLUDED.failure_code,
                        failure_detail = EXCLUDED.failure_detail,
                        retry_after_cleanup = EXCLUDED.retry_after_cleanup,
                        status = 'pending',
                        teardown_claimed_at = NULL,
                        teardown_claimed_by = NULL,
                        teardown_attempts = 0,
                        delete_error = NULL,
                        next_attempt_at = NOW(),
                        deleted_at = NULL,
                        updated_at = NOW()
                    WHERE evaluation_execution_cleanups.status = 'deleted'
                    RETURNING id
                    """,
                    (
                        evaluation_id,
                        execution_number,
                        row["runtime"],
                        backend_handle,
                        dispatch_job_name,
                        failure_code,
                        detail,
                        retry_allowed,
                    ),
                )
                cleanup = cur.fetchone()
                if cleanup is None:
                    return None
                cur.execute(
                    """
                    UPDATE evaluations
                    SET status_detail = %s,
                        last_failure_code = %s,
                        last_failure_category = 'infrastructure',
                        infrastructure_retries = infrastructure_retries + %s,
                        dispatch_reconcile_claimed_at = NULL,
                        dispatch_reconcile_claimed_by = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        f"{detail}; evaluation-runtime cleanup pending",
                        failure_code,
                        1 if retry_allowed else 0,
                        evaluation_id,
                    ),
                )
                retries_used = int(row.get("infrastructure_retries") or 0) + (1 if retry_allowed else 0)
                self.insert_event(
                    cur,
                    evaluation_id,
                    row["status"],
                    (
                        f"{failure_code}; runtime cleanup pending; infrastructure retries "
                        f"{retries_used}/"
                        f"{int(row.get('max_infrastructure_retries') or 0)}"
                    ),
                    type="infrastructure",
                )
                return {"action": "cleanup", "retry": retry_allowed}

            if retry_allowed:
                self._queue_infrastructure_retry(
                    cur,
                    evaluation_id=evaluation_id,
                    execution_number=execution_number,
                    failure_code=failure_code,
                    detail=detail,
                    retry_delay_seconds=retry_delay_seconds,
                    retry_already_reserved=False,
                )
                return {"action": "retry", "retry": True}
            self._terminalize_infrastructure_failure(
                cur,
                evaluation_id=evaluation_id,
                failure_code=failure_code,
                detail=detail,
            )
            return {"action": "failed", "retry": False}

    def complete_execution_cleanup(
        self,
        cleanup_id: int,
        *,
        worker_id: str,
        retry_delay_seconds: float,
    ) -> dict | None:
        """Finish cleanup and perform its fenced retry/terminal transition."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT cleanup.*,
                       cleanup.dispatch_job_name AS cleanup_dispatch_job_name,
                       evaluation.status AS evaluation_status,
                       evaluation.current_execution,
                       evaluation.dispatch_job_name AS evaluation_dispatch_job_name,
                       evaluation.benchmark_run_id,
                       (
                           evaluation.benchmark_run_id IS NULL OR EXISTS (
                               SELECT 1 FROM benchmark_runs
                               WHERE id = evaluation.benchmark_run_id
                                 AND deleted_at IS NULL
                                 AND cancelled_at IS NULL
                           )
                       ) AS benchmark_available
                FROM evaluation_execution_cleanups cleanup
                JOIN evaluations evaluation ON evaluation.id = cleanup.evaluation_id
                WHERE cleanup.id = %s
                  AND cleanup.status = 'deleting'
                  AND cleanup.teardown_claimed_by = %s
                FOR UPDATE OF cleanup, evaluation
                """,
                (cleanup_id, worker_id),
            )
            cleanup = cur.fetchone()
            if cleanup is None:
                return None
            cur.execute(
                """
                UPDATE evaluation_execution_cleanups
                SET status = 'deleted', deleted_at = NOW(), delete_error = NULL,
                    teardown_claimed_at = NULL, teardown_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (cleanup_id,),
            )
            if cleanup["evaluation_status"] not in {"provisioning", "running"}:
                return {"action": "cleaned", "retry": False}
            if (
                int(cleanup["current_execution"]) != int(cleanup["execution_number"])
                or cleanup["evaluation_dispatch_job_name"] != cleanup["cleanup_dispatch_job_name"]
            ):
                return {"action": "stale", "retry": False}
            if cleanup["retry_after_cleanup"] and cleanup["benchmark_available"]:
                self._queue_infrastructure_retry(
                    cur,
                    evaluation_id=cleanup["evaluation_id"],
                    execution_number=int(cleanup["execution_number"]),
                    failure_code=cleanup["failure_code"],
                    detail=cleanup["failure_detail"],
                    retry_delay_seconds=retry_delay_seconds,
                    retry_already_reserved=True,
                )
                return {"action": "retry", "retry": True}
            self._terminalize_infrastructure_failure(
                cur,
                evaluation_id=cleanup["evaluation_id"],
                failure_code=cleanup["failure_code"],
                detail=cleanup["failure_detail"],
            )
            return {"action": "failed", "retry": False}

    @staticmethod
    def _mark_failed_execution_resources(cur: psycopg.Cursor, evaluation_id: str) -> None:
        cur.execute(
            """
            UPDATE evaluation_runtime_resources AS resource
            SET status = 'draining', drain_until = NOW(),
                teardown_claimed_at = NULL, teardown_claimed_by = NULL,
                updated_at = NOW()
            FROM evaluations AS evaluation
            WHERE resource.evaluation_id = %s
              AND evaluation.id = resource.evaluation_id
              AND resource.execution_number = evaluation.current_execution
              AND resource.kind = 'switchyard'
              AND resource.status IN ('provisioned', 'draining', 'delete_failed')
            """,
            (evaluation_id,),
        )
        cur.execute(
            """
            UPDATE benchmark_switchyard_launches
            SET status = 'cleanup_pending', permit_expires_at = NOW(), updated_at = NOW()
            WHERE evaluation_id = %s AND status IN ('launching', 'running')
            """,
            (evaluation_id,),
        )

    def _queue_infrastructure_retry(
        self,
        cur: psycopg.Cursor,
        *,
        evaluation_id: str,
        execution_number: int,
        failure_code: str,
        detail: str,
        retry_delay_seconds: float,
        retry_already_reserved: bool,
    ) -> None:
        cur.execute(
            """
            UPDATE evaluations
            SET status = 'queued', status_detail = %s, backend_handle = NULL,
                dispatch_claimed_at = NULL, dispatch_claimed_by = NULL,
                dispatch_job_name = NULL, dispatch_job_uid = NULL,
                dispatch_reconcile_claimed_at = NULL,
                dispatch_reconcile_claimed_by = NULL,
                current_execution = current_execution + 1,
                infrastructure_retries = infrastructure_retries + %s,
                next_retry_at = NOW() + (%s * INTERVAL '1 second'),
                last_failure_code = %s, last_failure_category = 'infrastructure',
                result = NULL, reward_value = NULL, reward = NULL,
                n_trials = NULL, n_completed = NULL, n_errored = NULL,
                n_failed_solve = NULL, exception_counts = '{}'::jsonb,
                finished_at = NULL, updated_at = NOW()
            WHERE id = %s
              AND status IN ('provisioning', 'running')
              AND current_execution = %s
            RETURNING current_execution, infrastructure_retries,
                      max_infrastructure_retries
            """,
            (
                f"automatic infrastructure retry scheduled after {failure_code}; "
                f"execution {execution_number + 1}: {detail}",
                0 if retry_already_reserved else 1,
                retry_delay_seconds,
                failure_code,
                evaluation_id,
                execution_number,
            ),
        )
        retried = cur.fetchone()
        if retried is None:
            return
        self.insert_event(
            cur,
            evaluation_id,
            "queued",
            (
                f"automatic infrastructure retry execution {retried['current_execution']} "
                f"scheduled after {failure_code}; infrastructure retries "
                f"{retried['infrastructure_retries']}/"
                f"{retried['max_infrastructure_retries']}"
            ),
            type="retry",
        )

    def _terminalize_infrastructure_failure(
        self,
        cur: psycopg.Cursor,
        *,
        evaluation_id: str,
        failure_code: str,
        detail: str,
    ) -> None:
        cur.execute(
            """
            UPDATE evaluations
            SET status = 'failed', status_detail = %s, next_retry_at = NULL,
                last_failure_code = %s, last_failure_category = 'infrastructure',
                backend_handle = NULL, dispatch_claimed_at = NULL,
                dispatch_claimed_by = NULL, dispatch_reconcile_claimed_at = NULL,
                dispatch_reconcile_claimed_by = NULL,
                finished_at = COALESCE(finished_at, NOW()),
                evidence_status = 'building', evidence_requested_at = NOW(),
                evidence_error = NULL, evidence_claimed_at = NULL,
                evidence_claimed_by = NULL, evidence_build_attempts = 0,
                archive_status = 'building', archive_requested_at = NOW(),
                updated_at = NOW()
            WHERE id = %s AND status IN ('provisioning', 'running')
            """,
            (detail, failure_code, evaluation_id),
        )
        self.insert_status_event(cur, evaluation_id, "failed", detail)

    def list_stale_dispatch_jobs(self, *, stale_seconds: float, limit: int = 20) -> builtins.list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, current_execution, dispatch_job_name, dispatch_job_uid
                FROM evaluations
                WHERE deleted_at IS NULL
                  AND status IN ('provisioning', 'running')
                  AND dispatch_job_name IS NOT NULL
                  AND (
                      dispatch_claimed_at IS NULL
                      OR dispatch_claimed_at < NOW() - (%s * INTERVAL '1 second')
                  )
                ORDER BY dispatch_claimed_at NULLS FIRST, created_at, id
                LIMIT %s
                """,
                (stale_seconds, limit),
            )
            return list(cur.fetchall())

    def fail_stale_dispatch_job(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        dispatch_job_name: str,
        detail: str,
    ) -> bool:
        detail = redact_secret_text(detail)
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET status = 'failed',
                    status_detail = %s,
                    next_retry_at = NULL,
                    last_failure_code = 'dispatch_job_failed',
                    last_failure_category = 'task',
                    dispatch_claimed_at = NULL,
                    dispatch_claimed_by = NULL,
                    finished_at = COALESCE(finished_at, NOW()),
                    evidence_status = 'building',
                    evidence_requested_at = NOW(),
                    evidence_error = NULL,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    evidence_build_attempts = 0,
                    archive_status = 'building',
                    archive_requested_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status IN ('provisioning', 'running')
                  AND current_execution = %s
                  AND dispatch_job_name = %s
                """,
                (detail, evaluation_id, execution_number, dispatch_job_name),
            )
            updated = cur.rowcount != 0
            if updated:
                self.insert_status_event(cur, evaluation_id, "failed", detail)
                cur.execute(
                    """
                    UPDATE evaluation_runtime_resources AS resource
                    SET status = 'draining',
                        drain_until = NOW(),
                        teardown_claimed_at = NULL,
                        teardown_claimed_by = NULL,
                        updated_at = NOW()
                    FROM evaluations AS evaluation
                    WHERE resource.evaluation_id = %s
                      AND evaluation.id = resource.evaluation_id
                      AND resource.execution_number = evaluation.current_execution
                      AND resource.kind = 'switchyard'
                      AND resource.status IN ('provisioned', 'draining', 'delete_failed')
                    """,
                    (evaluation_id,),
                )
                cur.execute(
                    """
                    UPDATE benchmark_switchyard_launches
                    SET status = 'cleanup_pending',
                        permit_expires_at = NOW(),
                        updated_at = NOW()
                    WHERE evaluation_id = %s
                      AND status IN ('launching', 'running')
                    """,
                    (evaluation_id,),
                )
            return updated

    def load_for_dispatch(self, evaluation_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(_LOAD_FOR_DISPATCH_SQL, (evaluation_id,))
            return cur.fetchone()

    def schedule_retry(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        failure_code: str,
        failure_category: str,
        delay_seconds: float,
        expected_dispatch_owner: str | None = None,
    ) -> dict | None:
        infrastructure = failure_category == "infrastructure"
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET status = 'queued',
                    status_detail = %s,
                    backend_handle = NULL,
                    dispatch_claimed_at = NULL,
                    dispatch_claimed_by = NULL,
                    dispatch_job_name = NULL,
                    dispatch_job_uid = NULL,
                    dispatch_reconcile_claimed_at = NULL,
                    dispatch_reconcile_claimed_by = NULL,
                    current_execution = current_execution + 1,
                    infrastructure_retries = infrastructure_retries + %s,
                    next_retry_at = NOW() + (%s * INTERVAL '1 second'),
                    last_failure_code = %s,
                    last_failure_category = %s,
                    result = NULL,
                    reward_value = NULL,
                    reward = NULL,
                    n_trials = NULL,
                    n_completed = NULL,
                    n_errored = NULL,
                    n_failed_solve = NULL,
                    exception_counts = '{}'::jsonb,
                    finished_at = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND status IN ('provisioning', 'running')
                  AND current_execution = %s
                  AND (
                      (%s AND infrastructure_retries < max_infrastructure_retries)
                      OR (
                          NOT %s
                          AND current_execution - infrastructure_retries < max_executions
                      )
                  )
                  AND (%s::text IS NULL OR dispatch_claimed_by = %s::text)
                RETURNING current_execution, max_executions,
                          infrastructure_retries, max_infrastructure_retries,
                          next_retry_at
                """,
                (
                    (f"automatic retry scheduled after {failure_code}; execution {execution_number + 1}"),
                    1 if infrastructure else 0,
                    delay_seconds,
                    failure_code,
                    failure_category,
                    evaluation_id,
                    execution_number,
                    infrastructure,
                    infrastructure,
                    expected_dispatch_owner,
                    expected_dispatch_owner,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            current_execution = int(row.get("current_execution") or execution_number + 1)
            infrastructure_retries = int(row.get("infrastructure_retries") or 0)
            budget = (
                f"infrastructure retries {infrastructure_retries}/{row.get('max_infrastructure_retries') or 2}"
                if infrastructure
                else (
                    "non-infrastructure executions "
                    f"{current_execution - infrastructure_retries}/"
                    f"{row.get('max_executions') or 3}"
                )
            )
            self.insert_event(
                cur,
                evaluation_id,
                "queued",
                (f"automatic retry execution {current_execution} scheduled after {failure_code}; {budget}"),
                type="retry",
            )
            return row

    def load_intake_profile(self, profile_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM config_profiles
                WHERE id = %s AND type = 'intake' AND deleted_at IS NULL
                """,
                (profile_id,),
            )
            return cur.fetchone()

    def load_harbor_profile(self, profile_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM config_profiles
                WHERE id = %s AND type = 'harbor' AND deleted_at IS NULL
                """,
                (profile_id,),
            )
            return cur.fetchone()

    def load_framework_profile(self, profile_id: str) -> dict | None:
        """Load either supported framework profile for legacy unsnapshotted rows."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM config_profiles
                WHERE id = %s AND type IN ('harbor', 'gym') AND deleted_at IS NULL
                """,
                (profile_id,),
            )
            return cur.fetchone()

    def load_switchyard_profile(self, profile_id: str) -> dict | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT config
                FROM config_profiles
                WHERE id = %s AND type = 'switchyard' AND deleted_at IS NULL
                """,
                (profile_id,),
            )
            return cur.fetchone()

    def set_status(
        self,
        evaluation_id: str,
        status: str,
        *,
        detail: str | None = None,
        handle: str | None = None,
        failure_code: str | None = None,
        expected_dispatch_owner: str | None = None,
        expected_execution_number: int | None = None,
    ) -> bool:
        detail = redact_secret_text(detail) if detail is not None else None
        failure_category = failure_category_for_code(failure_code, detail) if status == "failed" else None
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET status = %s,
                    status_detail = %s,
                    backend_handle = COALESCE(%s, backend_handle),
                    last_failure_code = CASE
                        WHEN %s = 'failed' THEN COALESCE(%s, 'unknown')
                        ELSE last_failure_code
                    END,
                    last_failure_category = CASE
                        WHEN %s = 'failed' THEN %s
                        ELSE last_failure_category
                    END,
                    next_retry_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE next_retry_at
                    END,
                    dispatch_claimed_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE dispatch_claimed_at
                    END,
                    dispatch_claimed_by = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE dispatch_claimed_by
                    END,
                    finished_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled')
                            THEN COALESCE(finished_at, NOW())
                        ELSE finished_at
                    END,
                    evidence_status = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN 'building'
                        ELSE evidence_status
                    END,
                    evidence_requested_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NOW()
                        ELSE evidence_requested_at
                    END,
                    evidence_error = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE evidence_error
                    END,
                    evidence_claimed_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE evidence_claimed_at
                    END,
                    evidence_claimed_by = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NULL
                        ELSE evidence_claimed_by
                    END,
                    evidence_build_attempts = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN 0
                        ELSE evidence_build_attempts
                    END,
                    archive_status = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN 'building'
                        ELSE archive_status
                    END,
                    archive_requested_at = CASE
                        WHEN %s IN ('succeeded', 'failed', 'cancelled') THEN NOW()
                        ELSE archive_requested_at
                    END,
                    updated_at = NOW()
                WHERE id = %s
                  AND (%s::text IS NULL OR dispatch_claimed_by = %s::text)
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    status,
                    detail,
                    handle,
                    status,
                    failure_code,
                    status,
                    failure_category,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    status,
                    evaluation_id,
                    expected_dispatch_owner,
                    expected_dispatch_owner,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )
            updated = cur.rowcount != 0
            if updated:
                self.insert_status_event(cur, evaluation_id, status, detail)
            return updated

    def persist_result(
        self,
        evaluation_id: str,
        result: EvaluationResultWrite,
        *,
        terminal_status: str = "succeeded",
        status_detail: str | None = None,
        failure_code: str | None = None,
        expected_dispatch_owner: str | None = None,
        expected_execution_number: int | None = None,
    ) -> bool:
        """Write a terminal result envelope and its backend-neutral projections."""
        if terminal_status not in {"succeeded", "failed"}:
            raise ValueError(f"unsupported result terminal status: {terminal_status}")
        detail = status_detail if status_detail is not None else result.status_detail()
        detail = redact_secret_text(detail) if detail is not None else None
        failure_category = (
            failure_category_for_code(failure_code, detail, default="task") if terminal_status == "failed" else None
        )
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET status = %s,
                    status_detail = %s,
                    result = %s,
                    reward_value = %s,
                    reward = %s,
                    n_trials = %s,
                    n_errored = %s,
                    finished_at = %s,
                    n_completed = %s,
                    n_failed_solve = %s,
                    exception_counts = %s,
                    last_failure_code = CASE
                        WHEN %s = 'failed' THEN COALESCE(%s, 'unknown')
                        ELSE last_failure_code
                    END,
                    last_failure_category = CASE
                        WHEN %s = 'failed' THEN %s
                        ELSE last_failure_category
                    END,
                    evidence_status = 'building',
                    evidence_requested_at = NOW(),
                    evidence_error = NULL,
                    evidence_claimed_at = NULL,
                    evidence_claimed_by = NULL,
                    evidence_build_attempts = 0,
                    archive_status = 'building',
                    archive_requested_at = NOW(),
                    next_retry_at = NULL,
                    dispatch_claimed_at = NULL,
                    dispatch_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND (%s::text IS NULL OR dispatch_claimed_by = %s::text)
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    terminal_status,
                    detail,
                    Json(result.result),
                    Json(result.summary.reward) if result.summary.reward is not None else None,
                    result.summary.legacy_numeric_reward(),
                    result.summary.n_trials,
                    result.summary.n_errored,
                    result.summary.finished_at,
                    result.summary.n_completed,
                    result.summary.n_failed_solve,
                    Json(result.summary.exception_counts),
                    terminal_status,
                    failure_code,
                    terminal_status,
                    failure_category,
                    evaluation_id,
                    expected_dispatch_owner,
                    expected_dispatch_owner,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )
            updated = cur.rowcount != 0
            if updated:
                self.insert_status_event(cur, evaluation_id, terminal_status, detail)
            return updated

    def mark_archive_building(
        self,
        evaluation_id: str,
        *,
        worker_id: str,
        expected_execution_number: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET archive_status = 'building',
                    archive_requested_at = COALESCE(archive_requested_at, NOW()),
                    archive_claimed_at = NOW(),
                    archive_claimed_by = %s,
                    archive_build_attempts = archive_build_attempts + 1,
                    archive_error = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    worker_id,
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )

    def mark_archive_ready(
        self,
        evaluation_id: str,
        *,
        object_key: str,
        size_bytes: int,
        expected_execution_number: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET archive_status = 'ready',
                    archive_object_key = %s,
                    archive_size_bytes = %s,
                    archive_built_at = NOW(),
                    archive_error = NULL,
                    archive_claimed_at = NULL,
                    archive_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    object_key,
                    size_bytes,
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )

    def mark_archive_missing(
        self,
        evaluation_id: str,
        *,
        detail: str,
        expected_execution_number: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE evaluations
                SET archive_status = 'missing',
                    archive_error = %s,
                    archive_claimed_at = NULL,
                    archive_claimed_by = NULL,
                    updated_at = NOW()
                WHERE id = %s
                  AND deleted_at IS NULL
                  AND (%s::integer IS NULL OR current_execution = %s::integer)
                """,
                (
                    detail,
                    evaluation_id,
                    expected_execution_number,
                    expected_execution_number,
                ),
            )

    @staticmethod
    def insert_status_event(
        cur: psycopg.Cursor,
        evaluation_id: str,
        status: str,
        detail: str | None = None,
    ) -> None:
        EvaluationRepository.insert_event(
            cur,
            evaluation_id,
            status,
            detail,
            type="status",
        )

    @staticmethod
    def insert_event(
        cur: psycopg.Cursor,
        evaluation_id: str,
        status: str,
        detail: str | None = None,
        *,
        type: str = "status",
    ) -> None:
        detail = redact_secret_text(detail) if detail is not None else None
        cur.execute(
            """
            INSERT INTO evaluation_events (evaluation_id, type, status, detail)
            VALUES (%s, %s, %s, %s)
            """,
            (evaluation_id, type, status, detail),
        )

    def append_event(
        self,
        evaluation_id: str,
        *,
        status: str,
        detail: str | None,
        type: str = "status",
    ) -> None:
        with self.conn.cursor() as cur:
            self.insert_event(cur, evaluation_id, status, detail, type=type)

    @staticmethod
    def event_cursor_clause(cursor: str | None) -> tuple[str, builtins.list[Any]]:
        position = decode_cursor(cursor)
        if position is None:
            return "", []
        try:
            event_id = int(position.id)
        except ValueError:
            raise ValueError("invalid cursor") from None
        return "(created_at, id) > (%s, %s)", [position.created_at, event_id]
