# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluations and their per-execution state, stored in Entity Store as one entity per evaluation.

Every state change is a compare-and-swap on that one entity: read, check the
predicates the old SQL `WHERE` clause held, write, and rerun on a version
conflict. The dispatch, archive, evidence and cleanup queues are claimed the
same way, so of two claimants racing for one evaluation exactly one wins.
"""

from __future__ import annotations

# `EvaluationStore.list` shadows the builtin inside the class body.
import builtins
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_scaled_evals_plugin.entities import (
    ScaledEvalsEvaluation,
    ScaledEvalsEvaluationEvent,
    ScaledEvalsExecutionCleanup,
    ScaledEvalsExecutionTelemetry,
    ScaledEvalsResourceUsage,
)

from scaled_evals.api import config_profile_store, entity_store
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.failure_diagnostics import failure_category_for_code
from scaled_evals.api.redaction import redact_secret_text
from scaled_evals.api.repositories.base_repository import Conflict, InvalidReference
from scaled_evals.api.resource_access import INTERNAL_ACCESS, ResourceAccess
from scaled_evals.api.schemas.common import decode_cursor
from scaled_evals.api.task_store import TaskStore
from scaled_evals.models.evaluations import EvaluationResultWrite
from scaled_evals.models.execution_snapshot import build_execution_snapshot, current_process_identity
from scaled_evals.models.resource_usage import ResourceUsageSample

T = TypeVar("T")

_MAX_WRITE_ATTEMPTS = 5
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_ACTIVE = ("provisioning", "running")
_CLAIMABLE = ("queued", *_ACTIVE)
_CANCELLABLE = ("blocked", "queued", *_ACTIVE)
_CHILD_FIELDS = frozenset({"events", "telemetry", "resource_usage", "cleanups", "cleanup_open"})
_ROW_FIELDS = tuple(
    field
    for field in ScaledEvalsEvaluation.model_fields
    if field
    not in _CHILD_FIELDS | set(NemoEntity.model_fields) | {"display_name", "changed_at", "standalone", "deleted"}
)
RetryBlockReason = Literal["terminal_artifacts_finalizing", "benchmark_unavailable"]

# The `load_for_dispatch` fields a run changes. Everything else is fixed once
# the evaluation is created.
DISPATCH_STATE_COLUMNS = (
    "status",
    "backend_handle",
    "dispatch_job_name",
    "dispatch_job_uid",
    "current_execution",
    "infrastructure_retries",
    "next_retry_at",
    "last_failure_code",
    "last_failure_category",
    "status_detail",
    "result",
    "reward_value",
    "reward",
    "n_trials",
    "n_completed",
    "n_errored",
    "n_failed_solve",
    "exception_counts",
    "updated_at",
    "finished_at",
)
_DISPATCH_REVISION_FIELDS = (
    "image_ref",
    "image_digest",
    "verifier_image_ref",
    "verifier_image_digest",
    "tarball_sha256",
    "tarball_object_key",
    "tarball_size_bytes",
    "build_backend",
    "build_payload",
    "build_completed_at",
)
_TELEMETRY_PHASES = {"provisioning": "provisioning_started_at", "running": "running_started_at"}
# (sample-count prefix, measured field) for each resource a sample may report.
_MEASURES = (
    ("cpu", "cpu_usage_cores"),
    ("memory", "memory_usage_bytes"),
    ("gpu", "gpu_usage_percent"),
    ("gpu_memory", "gpu_memory_usage_bytes"),
)


def _now() -> datetime:
    return datetime.now(tz=UTC)


def row(entity: ScaledEvalsEvaluation) -> dict[str, Any]:
    """Return the evaluation as the column-named row its callers read."""
    return {
        "id": entity.name,
        "name": entity.display_name,
        **{field: getattr(entity, field) for field in _ROW_FIELDS},
        "created_at": entity.created_at,
        "updated_at": entity.changed_at,
    }


def _add_event(entity: ScaledEvalsEvaluation, status: str, detail: str | None = None, *, type: str = "status") -> None:
    entity.events.append(
        ScaledEvalsEvaluationEvent(
            id=len(entity.events) + 1,
            type=type,
            status=status,
            detail=redact_secret_text(detail) if detail is not None else None,
            created_at=_now(),
        )
    )


def _start_terminal_artifacts(entity: ScaledEvalsEvaluation, now: datetime) -> None:
    entity.finished_at = entity.finished_at or now
    entity.dispatch_claimed_at = None
    entity.dispatch_claimed_by = None
    entity.evidence_status = "building"
    entity.evidence_requested_at = now
    entity.evidence_error = None
    entity.evidence_claimed_at = None
    entity.evidence_claimed_by = None
    entity.evidence_build_attempts = 0
    entity.archive_status = "building"
    entity.archive_requested_at = now


def _requeue(entity: ScaledEvalsEvaluation, detail: str) -> None:
    """Return the evaluation to the queue as its next execution, dropping the last result."""
    entity.status = "queued"
    entity.status_detail = detail
    entity.backend_handle = None
    entity.dispatch_claimed_at = None
    entity.dispatch_claimed_by = None
    entity.dispatch_job_name = None
    entity.dispatch_job_uid = None
    entity.dispatch_reconcile_claimed_at = None
    entity.dispatch_reconcile_claimed_by = None
    entity.current_execution += 1
    entity.result = None
    entity.reward_value = None
    entity.reward = None
    entity.n_trials = None
    entity.n_completed = None
    entity.n_errored = None
    entity.n_failed_solve = None
    entity.exception_counts = {}
    entity.finished_at = None


def _queue_infrastructure_retry(
    entity: ScaledEvalsEvaluation,
    *,
    execution_number: int,
    failure_code: str,
    detail: str,
    retry_delay_seconds: float,
    retry_already_reserved: bool,
    now: datetime,
) -> None:
    _requeue(
        entity,
        f"automatic infrastructure retry scheduled after {failure_code}; execution {execution_number + 1}: {detail}",
    )
    entity.infrastructure_retries += 0 if retry_already_reserved else 1
    entity.next_retry_at = now + timedelta(seconds=retry_delay_seconds)
    entity.last_failure_code = failure_code
    entity.last_failure_category = "infrastructure"
    _add_event(
        entity,
        "queued",
        f"automatic infrastructure retry execution {entity.current_execution} scheduled after {failure_code}; "
        f"infrastructure retries {entity.infrastructure_retries}/{entity.max_infrastructure_retries}",
        type="retry",
    )


def _terminalize_infrastructure_failure(
    entity: ScaledEvalsEvaluation, *, failure_code: str, detail: str, now: datetime
) -> None:
    entity.status = "failed"
    entity.status_detail = detail
    entity.next_retry_at = None
    entity.last_failure_code = failure_code
    entity.last_failure_category = "infrastructure"
    entity.backend_handle = None
    entity.dispatch_reconcile_claimed_at = None
    entity.dispatch_reconcile_claimed_by = None
    _start_terminal_artifacts(entity, now)
    _add_event(entity, "failed", detail)


def _telemetry(entity: ScaledEvalsEvaluation, execution_number: int, now: datetime) -> ScaledEvalsExecutionTelemetry:
    """Return the execution's telemetry record, adding an empty one if it has none."""
    for record in entity.telemetry:
        if record.execution_number == execution_number:
            record.updated_at = now
            return record
    record = ScaledEvalsExecutionTelemetry(execution_number=execution_number, created_at=now, updated_at=now)
    entity.telemetry.append(record)
    return record


def _cleanup(entity: ScaledEvalsEvaluation, execution_number: int) -> ScaledEvalsExecutionCleanup | None:
    return next((item for item in entity.cleanups if item.execution_number == execution_number), None)


def cleanup_id(evaluation_id: str, execution_number: int) -> str:
    return f"{evaluation_id}/{execution_number}"


def _cleanup_row(entity: ScaledEvalsEvaluation, cleanup: ScaledEvalsExecutionCleanup) -> dict[str, Any]:
    return {
        "id": cleanup_id(entity.name, cleanup.execution_number),
        "evaluation_id": entity.name,
        **cleanup.model_dump(),
    }


def _stale(claimed_at: datetime | None, now: datetime, timeout: float) -> bool:
    return claimed_at is None or claimed_at < now - timedelta(seconds=timeout)


def _event_rows(entity: ScaledEvalsEvaluation) -> builtins.list[dict[str, Any]]:
    return [event.model_dump() for event in entity.events]


class EvaluationStore:
    def __init__(self, client: SyncEntityClient | None = None, *, access: ResourceAccess = INTERNAL_ACCESS) -> None:
        self._client = client or entity_store.entity_client()
        self._access = access

    def _get(self, evaluation_id: str, *, include_deleted: bool = False) -> ScaledEvalsEvaluation | None:
        try:
            entity = self._client.get(ScaledEvalsEvaluation, evaluation_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        if entity.deleted_at is not None and not (include_deleted and self._access.internal):
            return None
        return entity if self._access.can_read(entity.owner_id, entity.visibility) else None

    def _scan(
        self, *, include_deleted: bool = False, **where: Iterable[str | bool]
    ) -> builtins.list[ScaledEvalsEvaluation]:
        # ponytail: every scan lists all matching evaluations through the
        # store's pagination; the queue scans filter to rows with work, but the
        # list endpoint pages in memory and analytics read every live
        # evaluation. Upgrade path: push the keyset and limit into the store.
        filters = {field: list(values) for field, values in where.items()}
        if not include_deleted or not self._access.internal:
            filters["deleted"] = [False]
        return [
            entity
            for entity in entity_store.list_all(self._client, ScaledEvalsEvaluation, filters)
            if self._access.can_read(entity.owner_id, entity.visibility)
        ]

    def _mutate(
        self,
        evaluation_id: str,
        change: Callable[[ScaledEvalsEvaluation, datetime], tuple[T, bool]],
        *,
        touch: bool = True,
        include_deleted: bool = False,
    ) -> T | None:
        """Apply `change` to a fresh read of the evaluation and save it when it reports a write.

        Returns None when the evaluation is absent. A concurrent write reruns
        `change` against the newer version. `touch` stamps `changed_at`, which
        child-only writes such as telemetry leave alone.
        """
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(evaluation_id, include_deleted=include_deleted)
            if entity is None or not self._access.can_write(entity.owner_id):
                return None
            now = _now()
            result, write = change(entity, now)
            if not write:
                return result
            if touch:
                entity.changed_at = now
            try:
                self._client.update(entity)
            except EntityConflictError:
                continue
            return result
        raise Conflict("evaluation_changed", "evaluation changed concurrently; retry")

    def _claim(
        self,
        candidates: Callable[[datetime], builtins.list[ScaledEvalsEvaluation]],
        claim: Callable[[ScaledEvalsEvaluation, datetime], T],
    ) -> T | None:
        """Claim the first candidate whose write lands, rescanning after a lost race."""
        for _ in range(_MAX_WRITE_ATTEMPTS):
            now = _now()
            ordered = [entity for entity in candidates(now) if self._access.can_write(entity.owner_id)]
            if not ordered:
                return None
            entity = ordered[0]
            result = claim(entity, now)
            entity.changed_at = now
            try:
                self._client.update(entity)
            except EntityConflictError:
                continue
            return result
        return None

    @staticmethod
    def _run_available(benchmark_run_id: str | None) -> bool:
        """Return whether the evaluation's benchmark run, if any, is still live and not cancelled."""
        from scaled_evals.api.benchmark_run_store import BenchmarkRunStore  # noqa: PLC0415 - imports this module

        if benchmark_run_id is None:
            return True
        run = BenchmarkRunStore().get(benchmark_run_id)
        return run is not None and run["cancelled_at"] is None

    # Creation

    def task_in_active_use(self, task_id: str) -> bool:
        """Return whether an unfinished evaluation runs the task."""
        return bool(self._scan(task_id=[task_id], status=_CANCELLABLE))

    def profile_in_active_use(self, profile_id: str) -> bool:
        """Return whether an unfinished evaluation references the config profile."""
        return any(
            profile_id in (entity.framework_profile_id, entity.harbor_profile_id, entity.intake_profile_id)
            for entity in self._scan(status=_CANCELLABLE)
        )

    @staticmethod
    def _benchmark_variant_snapshot(benchmark_run_id: str | None) -> dict[str, Any] | None:
        """Freeze the run's benchmark-variant lineage and operational policy.

        Dispatch and provenance read the frozen copy, so editing a variant later
        cannot change how an already-submitted evaluation runs or reports.
        """
        from scaled_evals.api.benchmark_run_store import BenchmarkRunStore  # noqa: PLC0415 - imports this module

        if not benchmark_run_id:
            return None
        run = BenchmarkRunStore().get(benchmark_run_id)
        if run is None:
            return None
        meta = BenchmarkStore().get_revision_meta(run["benchmark_id"], run["benchmark_revision"])
        if meta is None:
            return None
        policy = meta["operational_policy"] if isinstance(meta["operational_policy"], dict) else {}
        if not policy and meta["derived_from_benchmark_id"] is None:
            return None
        return {
            "derived_from": (
                None
                if meta["derived_from_benchmark_id"] is None
                else {"benchmark_id": meta["derived_from_benchmark_id"], "revision": meta["derived_from_revision"]}
            ),
            "operational_policy": dict(policy),
        }

    @staticmethod
    def _capture_execution_snapshot(
        *,
        evaluation: dict[str, Any],
        task_id: str,
        task_revision: int,
        profile_ids: dict[str, str | None],
        credentials: dict[str, str],
        profiles_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Copy the mutable execution inputs as they are at submission.

        `profiles_by_id` lets a caller creating many evaluations read the
        profiles once.
        """
        revision = TaskStore().revision(task_id, task_revision, include_deleted=False)
        if revision is None:
            raise InvalidReference(f"task revision not found: {task_id} rev {task_revision}")
        task = {
            "id": task_id,
            "name": revision["task_name"],
            "slug": revision["task_slug"],
            **{
                field: revision[field]
                for field in (
                    "revision",
                    "status",
                    "image_ref",
                    "image_digest",
                    "tarball_object_key",
                    "tarball_size_bytes",
                    "tarball_sha256",
                    "build_backend",
                    "build_payload",
                    "build_completed_at",
                    "created_at",
                )
            },
        }
        wanted_profiles = {role: value for role, value in profile_ids.items() if value}
        if profiles_by_id is None:
            profiles_by_id = config_profile_store.ConfigProfileStore().get_many(wanted_profiles.values())
        profiles: dict[str, dict[str, Any]] = {}
        for role, profile_id in wanted_profiles.items():
            if profile_id not in profiles_by_id:
                raise InvalidReference(f"config profile not found: {profile_id}")
            profile = profiles_by_id[profile_id]
            profiles[role] = {key: profile[key] for key in ("id", "name", "type", "config", "updated_at")}
        variant = EvaluationStore._benchmark_variant_snapshot(evaluation.get("benchmark_run_id"))
        if variant is not None:
            evaluation = {**evaluation, "benchmark_variant": variant}
        return build_execution_snapshot(
            captured_at=_now().isoformat(),
            evaluation=evaluation,
            task=task,
            profiles=profiles,
            credentials=credentials,
            submission_identity=current_process_identity(),
        )

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
        profiles_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict:
        """Create a queued task run, standalone or one member of a benchmark run.

        Raises:
            InvalidReference: when the task revision or a config profile is missing.
            Conflict: when the evaluation id is already taken.

        """
        fields: dict[str, Any] = {
            "framework": framework,
            "requested_framework_version": requested_framework_version,
            "framework_version": framework_version,
            "runner_image_ref": runner_image_ref,
            "runner_image_digest": runner_image_digest,
            "framework_adapter_version": framework_adapter_version,
            "sandbox_k8s_version": sandbox_k8s_version,
            "runner_metadata": runner_metadata or {},
            "benchmark_run_id": benchmark_run_id,
            "runtime": runtime,
            "network_policy": network_policy,
            "network_policy_config": network_policy_config or {},
            "n_attempts": n_attempts,
            "parallelism": parallelism,
            "visibility": visibility,
            "extra_skill_object_keys": extra_skill_object_keys,
            "instruction_prefix": instruction_prefix,
            "instruction_postfix": instruction_postfix,
            "initial_user_turns": initial_user_turns or [],
        }
        snapshot = self._capture_execution_snapshot(
            evaluation={"id": evaluation_id, "name": name, **fields},
            task_id=task_id,
            task_revision=task_revision,
            profile_ids={"framework": framework_profile_id, "harbor": harbor_profile_id, "intake": intake_profile_id},
            credentials=credentials,
            profiles_by_id=profiles_by_id,
        )
        entity = ScaledEvalsEvaluation(
            name=evaluation_id,
            workspace=entity_store.private_workspace(),
            owner_id=owner_id,
            display_name=name,
            task_id=task_id,
            task_revision=task_revision,
            framework_profile_id=framework_profile_id,
            harbor_profile_id=harbor_profile_id,
            intake_profile_id=intake_profile_id,
            credentials=credentials,
            status="queued",
            execution_snapshot=snapshot,
            changed_at=_now(),
            **fields,
        )
        _add_event(entity, "queued")
        try:
            created = self._client.create(entity)
        except EntityConflictError:
            raise Conflict("evaluation_exists", f"evaluation already exists: {evaluation_id}") from None
        return row(created)

    # Reads

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
        """Return up to `limit + 1` evaluations after `cursor`.

        Without `benchmark_run_id` only standalone evaluations are listed; the
        members of a benchmark run are listed through their run.
        """
        where: dict[str, Any] = {}
        for field, value in (("status", status), ("task_id", task_id), ("owner_id", owner_id)):
            if value is not None:
                where[field] = [value]
        if benchmark_run_id is not None:
            where["benchmark_run_id"] = [benchmark_run_id]
        else:
            where["standalone"] = [True]
        rows = [
            row(entity)
            for entity in self._scan(**where)
            if (not shared or entity.visibility != "private")
            and entity_store.contains(
                q, entity.name, entity.display_name, entity.task_id, entity.status, entity.framework, entity.runtime
            )
        ]
        return entity_store.keyset_page(rows, cursor=cursor, order=order, limit=limit)

    def list_for_runs(self, run_ids: builtins.list[str]) -> builtins.list[dict]:
        """Return the live member evaluations of the given benchmark runs."""
        if not run_ids:
            return []
        return [row(entity) for entity in self._scan(benchmark_run_id=run_ids)]

    def get(self, evaluation_id: str) -> dict | None:
        """Return one live evaluation with its task revision's image, None when absent."""
        entity = self._get(evaluation_id)
        if entity is None:
            return None
        found = row(entity)
        revision = TaskStore().revision(entity.task_id, entity.task_revision) or {}
        found["image_ref"] = revision.get("image_ref")
        found["image_digest"] = revision.get("image_digest")
        return found

    def exists(self, evaluation_id: str) -> bool:
        return self._get(evaluation_id) is not None

    def get_for_mutation(self, evaluation_id: str) -> dict | None:
        entity = self._get(evaluation_id)
        if entity is None or not self._access.can_write(entity.owner_id):
            return None
        return row(entity)

    def load_observability_row(self, evaluation_id: str) -> dict | None:
        entity = self._get(evaluation_id)
        return None if entity is None else row(entity)

    def load_status_runtime(self, evaluation_id: str) -> dict | None:
        entity = self._get(evaluation_id, include_deleted=True)
        return None if entity is None else row(entity)

    def load_archive_row(self, evaluation_id: str) -> dict | None:
        entity = self._get(evaluation_id)
        return None if entity is None else row(entity)

    def load_stream_status(self, evaluation_id: str) -> str | None:
        entity = self._get(evaluation_id)
        return None if entity is None else entity.status

    def load_runtime_status(self, evaluation_id: str, *, expected_execution_number: int | None = None) -> str | None:
        entity = self._get(evaluation_id)
        if entity is None or expected_execution_number not in (None, entity.current_execution):
            return None
        return entity.status

    def current_execution_is(self, evaluation_id: str, *, expected_execution_number: int) -> bool:
        """Return whether the evaluation is live and still on `expected_execution_number`."""
        # ponytail: a check, not a lock. A retry landing between this check and
        # the caller's publish lets a superseded execution publish once more.
        # Upgrade path: publish under a compare-and-swap on the evaluation.
        entity = self._get(evaluation_id)
        return entity is not None and entity.current_execution == expected_execution_number

    def load_for_dispatch(self, evaluation_id: str) -> dict | None:
        """Return the evaluation with the task revision fields dispatch launches from."""
        entity = self._get(evaluation_id)
        if entity is None:
            return None
        revision = TaskStore().revision(entity.task_id, entity.task_revision)
        if not revision:
            return None
        return {
            **row(entity),
            "task_slug": revision["task_slug"],
            **{field: revision[field] for field in _DISPATCH_REVISION_FIELDS},
        }

    def load_dispatch_state(self, evaluation_id: str) -> dict | None:
        """Return only the fields of the dispatch row that change during a run."""
        entity = self._get(evaluation_id)
        if entity is None:
            return None
        found = row(entity)
        return {column: found[column] for column in DISPATCH_STATE_COLUMNS}

    def list_cancelled_platform_jobs(self, *, limit: int = 100) -> builtins.list[dict[str, Any]]:
        """List cancelled evaluations whose outer Platform Job may still run."""
        entities = [
            entity
            for entity in self._scan(status=["cancelled"], cancel_teardown_status=["pending"])
            if (entity.dispatch_job_name or "").startswith("scaled-evals-evaluation-")
        ]
        entities.sort(key=lambda entity: (entity.cancel_teardown_updated_at or entity.changed_at, entity.name))
        return [row(entity) for entity in entities[:limit]]

    def list_events(self, evaluation_id: str, *, limit: int, cursor: str | None, offset: int) -> builtins.list[dict]:
        """Return up to `limit + 1` status-history entries after `cursor`, skipping `offset`."""
        entity = self._get(evaluation_id, include_deleted=True)
        if entity is None:
            return []
        events = _event_rows(entity)
        if (position := decode_cursor(cursor)) is not None:
            try:
                after = int(position.id)
            except ValueError:
                raise ValueError("invalid cursor") from None
            events = [event for event in events if event["id"] > after]
        return events[offset : offset + limit + 1]

    def load_event_batch(
        self,
        evaluation_id: str,
        *,
        after_created_at: datetime | None = None,
        after_id: int | None = None,
        limit: int,
    ) -> builtins.list[dict]:
        entity = self._get(evaluation_id, include_deleted=True)
        if entity is None:
            return []
        events = _event_rows(entity)
        if after_id is not None:
            events = [event for event in events if event["id"] > after_id]
        return events[:limit]

    # Lifecycle

    def retry_failed(self, evaluation_id: str) -> dict | None:
        """Queue the next execution of a failed logical evaluation."""
        if self.retry_block_reason(evaluation_id) is not None:
            return None

        def change(entity: ScaledEvalsEvaluation, _now: datetime) -> tuple[dict | None, bool]:
            if (
                entity.status != "failed"
                or entity.evidence_claimed_by is not None
                or entity.archive_claimed_by is not None
            ):
                return None, False
            entity.max_executions = max(
                entity.max_executions, entity.current_execution - entity.infrastructure_retries + 1
            )
            _requeue(entity, f"manual retry scheduled; execution {entity.current_execution + 1}")
            entity.next_retry_at = None
            entity.evidence_status = "missing"
            entity.evidence_requested_at = entity.evidence_built_at = entity.evidence_error = None
            entity.evidence_claimed_at = entity.evidence_claimed_by = None
            entity.evidence_build_attempts = 0
            entity.archive_status = "missing"
            entity.archive_object_key = entity.archive_size_bytes = None
            entity.archive_requested_at = entity.archive_built_at = entity.archive_error = None
            entity.archive_claimed_at = entity.archive_claimed_by = None
            entity.archive_build_attempts = 0
            _add_event(
                entity,
                "queued",
                f"manual retry {entity.current_execution}/{entity.max_executions} scheduled",
                type="retry",
            )
            return row(entity), True

        return self._mutate(evaluation_id, change)

    def retry_block_reason(self, evaluation_id: str) -> RetryBlockReason | None:
        """Explain why a failed evaluation cannot transition to a retry."""
        entity = self._get(evaluation_id)
        if entity is None or not self._access.can_write(entity.owner_id) or entity.status != "failed":
            return None
        if entity.evidence_claimed_by is not None or entity.archive_claimed_by is not None:
            return "terminal_artifacts_finalizing"
        if not self._run_available(entity.benchmark_run_id):
            return "benchmark_unavailable"
        return None

    def cancel(self, evaluation_id: str, *, detail: str = "cancelled") -> tuple[dict | None, bool]:
        """Cancel an unfinished evaluation; return its row and whether this call cancelled it."""

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[tuple[dict, bool], bool]:
            if entity.status not in _CANCELLABLE:
                return (row(entity), False), False
            entity.status = "cancelled"
            entity.status_detail = detail
            entity.cancel_teardown_status = "pending"
            entity.cancel_teardown_error = None
            entity.cancel_teardown_updated_at = now
            _start_terminal_artifacts(entity, now)
            _add_event(entity, "cancelled", detail)
            return (row(entity), True), True

        result = self._mutate(evaluation_id, change)
        return result if result is not None else (None, False)

    def record_cancel_teardown_failure(self, evaluation_id: str, detail: str) -> dict | None:
        detail = redact_secret_text(detail)

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict | None, bool]:
            if entity.status != "cancelled":
                return None, False
            entity.status_detail = detail
            entity.cancel_teardown_status = "failed"
            entity.cancel_teardown_error = detail
            entity.cancel_teardown_updated_at = now
            # A running Job finalizes a cancelled evaluation only after its
            # teardown, so the evidence backstop's grace window restarts here.
            if entity.evidence_status == "building":
                entity.evidence_requested_at = now
            _add_event(entity, "cancelled", detail)
            return row(entity), True

        return self._mutate(evaluation_id, change)

    def record_cancel_teardown_succeeded(self, evaluation_id: str) -> dict | None:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict | None, bool]:
            if entity.status != "cancelled" or entity.cancel_teardown_status != "pending":
                return None, False
            entity.cancel_teardown_status = "succeeded"
            entity.cancel_teardown_error = None
            entity.cancel_teardown_updated_at = now
            if entity.evidence_status == "building":
                entity.evidence_requested_at = now
            _add_event(entity, "cancelled", "cancellation teardown succeeded")
            return row(entity), True

        return self._mutate(evaluation_id, change)

    def soft_delete(self, evaluation_id: str) -> bool:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[bool, bool]:
            entity.deleted_at, entity.deleted = now, True
            return True, True

        return bool(self._mutate(evaluation_id, change))

    def request_archive_build(self, evaluation_id: str, *, force: bool) -> dict | None:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict, bool]:
            entity.archive_status = "building"
            entity.archive_requested_at = now
            entity.archive_claimed_at = entity.archive_claimed_by = entity.archive_error = None
            if force:
                entity.archive_object_key = entity.archive_size_bytes = entity.archive_built_at = None
            return row(entity), True

        return self._mutate(evaluation_id, change)

    # Dispatch

    def claim_next(
        self,
        *,
        claim_timeout: float,
        worker_id: str,
        cluster_slot_limit: int = 500,
        per_user_slot_limit: int = 50,
        evaluation_id: str | None = None,
    ) -> dict | None:
        """Claim the next evaluation to drive, admitting a queued one only within the slot caps.

        Recovery work comes first (running, then provisioning), then the queued
        evaluation of the owner using the fewest sandbox slots, oldest first, so
        one large benchmark cannot monopolize the cluster.
        """
        from scaled_evals.api.benchmark_run_store import BenchmarkRunStore  # noqa: PLC0415 - imports this module

        # ponytail: admission counts active slots without a lock, so claimants
        # running at the same time can each admit one queued evaluation past a
        # cap. Upgrade path: a platform core quota on sandboxes, then delete
        # these caps.
        def candidates(now: datetime) -> builtins.list[ScaledEvalsEvaluation]:
            live = self._scan(status=_CLAIMABLE)
            active = [entity for entity in live if entity.status in _ACTIVE]
            cluster_slots = sum(entity.parallelism for entity in active)
            owner_slots: Counter[str | None] = Counter()
            run_members: Counter[str] = Counter()
            for entity in active:
                owner_slots[entity.owner_id] += entity.parallelism
                if entity.benchmark_run_id is not None:
                    run_members[entity.benchmark_run_id] += 1
            caps: dict[str, int | None] = {}

            def admitted(entity: ScaledEvalsEvaluation) -> bool:
                if entity.status != "queued":
                    return True
                if (
                    entity.parallelism > per_user_slot_limit
                    or cluster_slots + entity.parallelism > cluster_slot_limit
                    or owner_slots[entity.owner_id] + entity.parallelism > per_user_slot_limit
                ):
                    return False
                run_id = entity.benchmark_run_id
                if run_id is None:
                    return True
                if run_id not in caps:
                    run = BenchmarkRunStore().get(run_id)
                    caps[run_id] = None if run is None else run["max_concurrent_members"]
                cap = caps[run_id]
                return cap is None or run_members[run_id] < cap

            rank = {"running": 0, "provisioning": 1, "queued": 2}
            eligible = [
                entity
                for entity in live
                if (evaluation_id is None or entity.name == evaluation_id)
                and entity.dispatch_job_name is None
                and (entity.next_retry_at is None or entity.next_retry_at <= now)
                and _stale(entity.dispatch_claimed_at, now, claim_timeout)
                and admitted(entity)
            ]
            return sorted(
                eligible,
                key=lambda entity: (
                    rank[entity.status],
                    owner_slots[entity.owner_id] if entity.status == "queued" else 0,
                    entity.created_at or now,
                    entity.name,
                ),
            )

        def claim(entity: ScaledEvalsEvaluation, now: datetime) -> dict:
            previous = entity.status
            if previous == "queued":
                entity.status = "provisioning"
                entity.status_detail = "claimed by dispatch worker"
                _add_event(entity, "provisioning", "claimed by dispatch worker")
                record = _telemetry(entity, entity.current_execution, now)
                record.provisioning_started_at = record.provisioning_started_at or now
            entity.next_retry_at = None
            entity.dispatch_claimed_at = now
            entity.dispatch_claimed_by = worker_id
            entity.dispatch_attempts += 1
            return {"id": entity.name, "previous_status": previous, "status": entity.status}

        return self._claim(candidates, claim)

    def record_dispatch_job(self, evaluation_id: str, *, execution_number: int, name: str, uid: str) -> None:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[None, bool]:
            if (
                entity.status not in _ACTIVE
                or entity.current_execution != execution_number
                or entity.dispatch_job_name not in (None, name)
            ):
                return None, False
            entity.dispatch_job_name = name
            entity.dispatch_job_uid = uid or None
            entity.dispatch_claimed_at = now
            return None, True

        self._mutate(evaluation_id, change, include_deleted=True)

    def claim_stale_dispatch_job(self, *, stale_seconds: float, claim_timeout: float, worker_id: str) -> dict | None:
        """Lease one outer Job whose dispatcher went quiet, for bounded Kubernetes reconciliation."""

        def candidates(now: datetime) -> builtins.list[ScaledEvalsEvaluation]:
            stale = [
                entity
                for entity in self._scan(status=_ACTIVE)
                if entity.dispatch_job_name is not None
                and _stale(entity.dispatch_claimed_at, now, stale_seconds)
                and _stale(entity.dispatch_reconcile_claimed_at, now, claim_timeout)
                and not any(
                    cleanup.execution_number == entity.current_execution and cleanup.status != "deleted"
                    for cleanup in entity.cleanups
                )
            ]
            return sorted(
                stale,
                key=lambda entity: (
                    entity.dispatch_claimed_at is not None,
                    entity.dispatch_claimed_at or now,
                    entity.created_at or now,
                    entity.name,
                ),
            )

        def claim(entity: ScaledEvalsEvaluation, now: datetime) -> dict:
            entity.dispatch_reconcile_claimed_at = now
            entity.dispatch_reconcile_claimed_by = worker_id
            return {
                "id": entity.name,
                "status": entity.status,
                "current_execution": entity.current_execution,
                "dispatch_job_name": entity.dispatch_job_name,
                "dispatch_job_uid": entity.dispatch_job_uid,
            }

        return self._claim(candidates, claim)

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
        """Own cleanup, then retry or terminalize one failed outer Job."""
        detail = redact_secret_text(detail)
        entity = self._get(evaluation_id)
        benchmark_available = entity is not None and self._run_available(entity.benchmark_run_id)

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict | None, bool]:
            if (
                entity.status not in _ACTIVE
                or entity.current_execution != execution_number
                or entity.dispatch_job_name != dispatch_job_name
                or entity.dispatch_reconcile_claimed_by != reconcile_worker_id
            ):
                return None, False
            retry_allowed = benchmark_available and entity.infrastructure_retries < entity.max_infrastructure_retries
            if entity.backend_handle:
                existing = _cleanup(entity, execution_number)
                if existing is not None and existing.status != "deleted":
                    return None, False
                if existing is not None:
                    entity.cleanups.remove(existing)
                entity.cleanups.append(
                    ScaledEvalsExecutionCleanup(
                        execution_number=execution_number,
                        runtime=entity.runtime,
                        backend_handle=entity.backend_handle,
                        dispatch_job_name=dispatch_job_name,
                        failure_code=failure_code,
                        failure_detail=detail,
                        retry_after_cleanup=retry_allowed,
                        next_attempt_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
                entity.cleanup_open = True
                entity.status_detail = f"{detail}; evaluation-runtime cleanup pending"
                entity.last_failure_code = failure_code
                entity.last_failure_category = "infrastructure"
                entity.infrastructure_retries += 1 if retry_allowed else 0
                entity.dispatch_reconcile_claimed_at = entity.dispatch_reconcile_claimed_by = None
                _add_event(
                    entity,
                    entity.status,
                    f"{failure_code}; runtime cleanup pending; infrastructure retries "
                    f"{entity.infrastructure_retries}/{entity.max_infrastructure_retries}",
                    type="infrastructure",
                )
                return {"action": "cleanup", "retry": retry_allowed}, True
            if retry_allowed:
                _queue_infrastructure_retry(
                    entity,
                    execution_number=execution_number,
                    failure_code=failure_code,
                    detail=detail,
                    retry_delay_seconds=retry_delay_seconds,
                    retry_already_reserved=False,
                    now=now,
                )
                return {"action": "retry", "retry": True}, True
            _terminalize_infrastructure_failure(entity, failure_code=failure_code, detail=detail, now=now)
            return {"action": "failed", "retry": False}, True

        return self._mutate(evaluation_id, change)

    def claim_cleanup(self, *, worker_id: str, claim_timeout: float) -> dict | None:
        """Claim one due runtime cleanup; a worker that died mid-teardown loses its claim after `claim_timeout`."""

        def due(entity: ScaledEvalsEvaluation, now: datetime) -> builtins.list[ScaledEvalsExecutionCleanup]:
            return sorted(
                (
                    cleanup
                    for cleanup in entity.cleanups
                    if cleanup.status in ("pending", "deleting", "delete_failed")
                    and cleanup.next_attempt_at <= now
                    and _stale(cleanup.teardown_claimed_at, now, claim_timeout)
                ),
                key=lambda cleanup: cleanup.next_attempt_at,
            )

        def candidates(now: datetime) -> builtins.list[ScaledEvalsEvaluation]:
            ready = [(due(entity, now), entity) for entity in self._scan(include_deleted=True, cleanup_open=[True])]
            ready = [(cleanups, entity) for cleanups, entity in ready if cleanups]
            ready.sort(key=lambda item: (item[0][0].next_attempt_at, item[1].name))
            return [entity for _cleanups, entity in ready]

        def claim(entity: ScaledEvalsEvaluation, now: datetime) -> dict:
            cleanup = due(entity, now)[0]
            cleanup.status = "deleting"
            cleanup.teardown_claimed_at = now
            cleanup.teardown_claimed_by = worker_id
            cleanup.teardown_attempts += 1
            cleanup.updated_at = now
            return _cleanup_row(entity, cleanup)

        return self._claim(candidates, claim)

    def mark_cleanup_failed(self, cleanup_ref: str, *, worker_id: str, detail: str) -> None:
        evaluation_id, _, execution = cleanup_ref.rpartition("/")

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[None, bool]:
            cleanup = _cleanup(entity, int(execution))
            if cleanup is None or cleanup.status != "deleting" or cleanup.teardown_claimed_by != worker_id:
                return None, False
            cleanup.status = "delete_failed"
            cleanup.delete_error = detail
            cleanup.next_attempt_at = now + timedelta(seconds=min(300, 2 ** min(cleanup.teardown_attempts, 8)))
            cleanup.teardown_claimed_at = cleanup.teardown_claimed_by = None
            cleanup.updated_at = now
            return None, True

        self._mutate(evaluation_id, change, touch=False, include_deleted=True)

    def complete_execution_cleanup(
        self, cleanup_ref: str, *, worker_id: str, retry_delay_seconds: float
    ) -> dict | None:
        """Finish a claimed cleanup and perform its fenced retry or terminal transition."""
        evaluation_id, _, execution = cleanup_ref.rpartition("/")
        execution_number = int(execution)
        entity = self._get(evaluation_id, include_deleted=True)
        benchmark_available = entity is not None and self._run_available(entity.benchmark_run_id)

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict | None, bool]:
            cleanup = _cleanup(entity, execution_number)
            if cleanup is None or cleanup.status != "deleting" or cleanup.teardown_claimed_by != worker_id:
                return None, False
            cleanup.status = "deleted"
            cleanup.deleted_at = cleanup.updated_at = now
            cleanup.delete_error = None
            cleanup.teardown_claimed_at = cleanup.teardown_claimed_by = None
            entity.cleanup_open = any(item.status != "deleted" for item in entity.cleanups)
            if entity.status not in _ACTIVE:
                return {"action": "cleaned", "retry": False}, True
            if entity.current_execution != execution_number or entity.dispatch_job_name != cleanup.dispatch_job_name:
                return {"action": "stale", "retry": False}, True
            if cleanup.retry_after_cleanup and benchmark_available:
                _queue_infrastructure_retry(
                    entity,
                    execution_number=execution_number,
                    failure_code=cleanup.failure_code,
                    detail=cleanup.failure_detail,
                    retry_delay_seconds=retry_delay_seconds,
                    retry_already_reserved=True,
                    now=now,
                )
                return {"action": "retry", "retry": True}, True
            _terminalize_infrastructure_failure(
                entity, failure_code=cleanup.failure_code, detail=cleanup.failure_detail, now=now
            )
            return {"action": "failed", "retry": False}, True

        return self._mutate(evaluation_id, change, include_deleted=True)

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
        """Queue an automatic retry when the failure's retry budget allows one."""
        infrastructure = failure_category == "infrastructure"

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[dict | None, bool]:
            within_budget = (
                entity.infrastructure_retries < entity.max_infrastructure_retries
                if infrastructure
                else entity.current_execution - entity.infrastructure_retries < entity.max_executions
            )
            if (
                entity.status not in _ACTIVE
                or entity.current_execution != execution_number
                or not within_budget
                or expected_dispatch_owner not in (None, entity.dispatch_claimed_by)
            ):
                return None, False
            _requeue(entity, f"automatic retry scheduled after {failure_code}; execution {execution_number + 1}")
            entity.infrastructure_retries += 1 if infrastructure else 0
            entity.next_retry_at = now + timedelta(seconds=delay_seconds)
            entity.last_failure_code = failure_code
            entity.last_failure_category = failure_category
            budget = (
                f"infrastructure retries {entity.infrastructure_retries}/{entity.max_infrastructure_retries}"
                if infrastructure
                else "non-infrastructure executions "
                f"{entity.current_execution - entity.infrastructure_retries}/{entity.max_executions}"
            )
            _add_event(
                entity,
                "queued",
                f"automatic retry execution {entity.current_execution} scheduled after {failure_code}; {budget}",
                type="retry",
            )
            return {
                field: getattr(entity, field)
                for field in (
                    "current_execution",
                    "max_executions",
                    "infrastructure_retries",
                    "max_infrastructure_retries",
                    "next_retry_at",
                )
            }, True

        return self._mutate(evaluation_id, change)

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
        """Move the evaluation to `status`; return False when the transition is not allowed.

        Status never moves backward: a terminal evaluation only accepts another
        terminal status, so a stale observation cannot return it to `running`,
        and a run finishing after its cancel cannot replace the cancel.
        """
        detail = redact_secret_text(detail) if detail is not None else None
        failure_category = failure_category_for_code(failure_code, detail) if status == "failed" else None
        terminal = status in TERMINAL_STATUSES

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[bool, bool]:
            if (
                expected_dispatch_owner not in (None, entity.dispatch_claimed_by)
                or expected_execution_number not in (None, entity.current_execution)
                or (entity.status in TERMINAL_STATUSES and not terminal)
                or (entity.status == "cancelled" and status != "cancelled")
            ):
                return False, False
            entity.status = status
            entity.status_detail = detail
            entity.backend_handle = handle or entity.backend_handle
            if status == "failed":
                entity.last_failure_code = failure_code or "unknown"
                entity.last_failure_category = failure_category
            if terminal:
                entity.next_retry_at = None
                _start_terminal_artifacts(entity, now)
            _add_event(entity, status, detail)
            return True, True

        return bool(self._mutate(evaluation_id, change, include_deleted=True))

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
        """Write a terminal result envelope and its backend-neutral summary."""
        if terminal_status not in {"succeeded", "failed"}:
            raise ValueError(f"unsupported result terminal status: {terminal_status}")
        detail = status_detail if status_detail is not None else result.status_detail()
        detail = redact_secret_text(detail) if detail is not None else None
        failure_category = (
            failure_category_for_code(failure_code, detail, default="task") if terminal_status == "failed" else None
        )
        summary = result.summary

        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[bool, bool]:
            if (
                expected_dispatch_owner not in (None, entity.dispatch_claimed_by)
                or expected_execution_number not in (None, entity.current_execution)
                or entity.status == "cancelled"
            ):
                return False, False
            entity.status = terminal_status
            entity.status_detail = detail
            entity.result = result.result
            entity.reward_value = summary.reward
            entity.reward = summary.legacy_numeric_reward()
            entity.n_trials = summary.n_trials
            entity.n_errored = summary.n_errored
            entity.n_completed = summary.n_completed
            entity.n_failed_solve = summary.n_failed_solve
            entity.exception_counts = dict(summary.exception_counts)
            if terminal_status == "failed":
                entity.last_failure_code = failure_code or "unknown"
                entity.last_failure_category = failure_category
            entity.next_retry_at = None
            _start_terminal_artifacts(entity, now)
            # The summary's own finish time wins over the moment it was written.
            if summary.finished_at:
                finished = datetime.fromisoformat(summary.finished_at)
                entity.finished_at = finished if finished.tzinfo else finished.replace(tzinfo=UTC)
            _add_event(entity, terminal_status, detail)
            return True, True

        return bool(self._mutate(evaluation_id, change, include_deleted=True))

    # Terminal artifacts

    def _claim_artifact(
        self,
        kind: Literal["archive", "evidence"],
        *,
        claim_timeout: float,
        worker_id: str,
        evaluation_id: str | None,
        min_age: float,
    ) -> dict | None:
        def candidates(now: datetime) -> builtins.list[ScaledEvalsEvaluation]:
            def ready(entity: ScaledEvalsEvaluation) -> bool:
                claimed_at = getattr(entity, f"{kind}_claimed_at")
                requested_at = getattr(entity, f"{kind}_requested_at") or entity.changed_at
                if kind == "archive":
                    precondition = entity.evidence_status in ("ready", "missing") or entity.execution_snapshot is None
                else:
                    precondition = entity.evidence_build_attempts < 5
                return (
                    entity.status in TERMINAL_STATUSES
                    and precondition
                    and (evaluation_id is None or entity.name == evaluation_id)
                    and (
                        (claimed_at is not None and _stale(claimed_at, now, claim_timeout))
                        or (claimed_at is None and requested_at <= now - timedelta(seconds=min_age))
                    )
                )

            building: dict[str, Any] = {f"{kind}_status": ["building"]}
            found = [entity for entity in self._scan(**building) if ready(entity)]
            return sorted(
                found,
                key=lambda entity: (getattr(entity, f"{kind}_requested_at") or entity.changed_at, entity.name),
            )

        def claim(entity: ScaledEvalsEvaluation, now: datetime) -> dict:
            setattr(entity, f"{kind}_claimed_at", now)
            setattr(entity, f"{kind}_claimed_by", worker_id)
            setattr(entity, f"{kind}_build_attempts", getattr(entity, f"{kind}_build_attempts") + 1)
            return {"id": entity.name}

        return self._claim(candidates, claim)

    def claim_next_archive(
        self, *, claim_timeout: float, worker_id: str, evaluation_id: str | None = None, min_age: float = 0.0
    ) -> dict | None:
        """Claim an archive build; an unclaimed one must have waited `min_age` seconds."""
        return self._claim_artifact(
            "archive", claim_timeout=claim_timeout, worker_id=worker_id, evaluation_id=evaluation_id, min_age=min_age
        )

    def claim_next_evidence(
        self, *, claim_timeout: float, worker_id: str, evaluation_id: str | None = None, min_age: float = 0.0
    ) -> dict | None:
        """Claim an evidence build; an unclaimed one must have waited `min_age` seconds."""
        return self._claim_artifact(
            "evidence", claim_timeout=claim_timeout, worker_id=worker_id, evaluation_id=evaluation_id, min_age=min_age
        )

    def _on_execution(
        self,
        evaluation_id: str,
        expected_execution_number: int | None,
        apply: Callable[[ScaledEvalsEvaluation, datetime], bool],
        *,
        include_deleted: bool = True,
    ) -> None:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[None, bool]:
            if expected_execution_number not in (None, entity.current_execution):
                return None, False
            return None, apply(entity, now)

        self._mutate(evaluation_id, change, include_deleted=include_deleted)

    def mark_evidence_ready(self, evaluation_id: str, *, expected_execution_number: int | None = None) -> None:
        def apply(entity: ScaledEvalsEvaluation, now: datetime) -> bool:
            entity.evidence_status = "ready"
            entity.evidence_built_at = now
            entity.evidence_error = entity.evidence_claimed_at = entity.evidence_claimed_by = None
            entity.archive_status = "building"
            entity.archive_requested_at = now
            return True

        self._on_execution(evaluation_id, expected_execution_number, apply)

    def mark_evidence_failed(
        self, evaluation_id: str, detail: str, *, expected_execution_number: int | None = None
    ) -> None:
        def apply(entity: ScaledEvalsEvaluation, _now: datetime) -> bool:
            if entity.evidence_status != "building":
                return False
            if entity.evidence_build_attempts >= 5:
                entity.evidence_status = "missing"
            entity.evidence_error = redact_secret_text(detail)
            entity.evidence_claimed_at = entity.evidence_claimed_by = None
            return True

        self._on_execution(evaluation_id, expected_execution_number, apply)

    def mark_archive_building(
        self, evaluation_id: str, *, worker_id: str, expected_execution_number: int | None = None
    ) -> None:
        def apply(entity: ScaledEvalsEvaluation, now: datetime) -> bool:
            entity.archive_status = "building"
            entity.archive_requested_at = entity.archive_requested_at or now
            entity.archive_claimed_at = now
            entity.archive_claimed_by = worker_id
            entity.archive_build_attempts += 1
            entity.archive_error = None
            return True

        self._on_execution(evaluation_id, expected_execution_number, apply, include_deleted=False)

    def mark_archive_ready(
        self, evaluation_id: str, *, object_key: str, size_bytes: int, expected_execution_number: int | None = None
    ) -> None:
        def apply(entity: ScaledEvalsEvaluation, now: datetime) -> bool:
            entity.archive_status = "ready"
            entity.archive_object_key = object_key
            entity.archive_size_bytes = size_bytes
            entity.archive_built_at = now
            entity.archive_error = entity.archive_claimed_at = entity.archive_claimed_by = None
            return True

        self._on_execution(evaluation_id, expected_execution_number, apply, include_deleted=False)

    def mark_archive_missing(
        self, evaluation_id: str, *, detail: str, expected_execution_number: int | None = None
    ) -> None:
        def apply(entity: ScaledEvalsEvaluation, _now: datetime) -> bool:
            entity.archive_status = "missing"
            entity.archive_error = detail
            entity.archive_claimed_at = entity.archive_claimed_by = None
            return True

        self._on_execution(evaluation_id, expected_execution_number, apply, include_deleted=False)

    # Per-execution telemetry

    def _record_telemetry(
        self, evaluation_id: str, execution_number: int, apply: Callable[[ScaledEvalsExecutionTelemetry], None]
    ) -> None:
        def change(entity: ScaledEvalsEvaluation, now: datetime) -> tuple[None, bool]:
            apply(_telemetry(entity, execution_number, now))
            return None, True

        self._mutate(evaluation_id, change, touch=False, include_deleted=True)

    def record_phase(
        self, evaluation_id: str, *, execution_number: int, phase: str, terminal_status: str | None = None
    ) -> None:
        if phase == "terminal":
            if terminal_status not in TERMINAL_STATUSES:
                raise ValueError("terminal phase requires a terminal status")

            def terminal(record: ScaledEvalsExecutionTelemetry) -> None:
                record.terminal_at = record.terminal_at or record.updated_at
                record.terminal_status = terminal_status
                if terminal_status == "failed":
                    record.failure_phase = "provisioning" if record.running_started_at is None else "runtime"

            self._record_telemetry(evaluation_id, execution_number, terminal)
            return
        field = _TELEMETRY_PHASES.get(phase)
        if field is None:
            raise ValueError(f"unsupported execution phase: {phase}")

        def started(record: ScaledEvalsExecutionTelemetry) -> None:
            setattr(record, field, getattr(record, field) or record.updated_at)

        self._record_telemetry(evaluation_id, execution_number, started)

    def record_summary(self, evaluation_id: str, *, execution_number: int, summary: Mapping[str, Any]) -> None:
        def apply(record: ScaledEvalsExecutionTelemetry) -> None:
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_creation_tokens",
                "turn_count",
                "tool_call_count",
                "cost_usd",
            ):
                setattr(record, field, summary.get(field))
            record.usage_source = summary.get("usage_source") or "unknown"
            record.cost_source = summary.get("cost_source") or "unknown"
            record.raw_artifact_refs = list(summary.get("raw_artifact_refs") or [])

        self._record_telemetry(evaluation_id, execution_number, apply)

    def record_intake(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        experiment_ref: str | None,
        run_refs: builtins.list[str],
        status: str,
        expected_records: int | None,
        uploaded_records: int | None,
        error: str | None,
    ) -> None:
        def apply(record: ScaledEvalsExecutionTelemetry) -> None:
            record.intake_experiment_ref = experiment_ref
            record.intake_run_refs = list(run_refs)
            record.intake_status = status
            record.intake_expected_records = expected_records
            record.intake_uploaded_records = uploaded_records
            record.intake_error = redact_secret_text(error)[:2000] if error else None

        self._record_telemetry(evaluation_id, execution_number, apply)

    def record_artifact_sync(
        self,
        evaluation_id: str,
        *,
        execution_number: int,
        status: str,
        file_count: int | None = None,
        error: str | None = None,
    ) -> None:
        def apply(record: ScaledEvalsExecutionTelemetry) -> None:
            record.artifact_sync_status = status
            record.artifact_sync_file_count = file_count
            record.artifact_sync_error = redact_secret_text(error)[:2000] if error else None

        self._record_telemetry(evaluation_id, execution_number, apply)

    def telemetry_for(self, evaluation_id: str) -> builtins.list[dict]:
        entity = self._get(evaluation_id, include_deleted=True)
        if entity is None:
            return []
        return [record.model_dump() for record in sorted(entity.telemetry, key=lambda item: item.execution_number)]

    def cleanups_for(self, evaluation_id: str) -> builtins.list[dict]:
        """Return attempt-aware cleanup state without exposing backend handles."""
        entity = self._get(evaluation_id, include_deleted=True)
        if entity is None:
            return []
        return [
            cleanup.model_dump(
                include={
                    "execution_number",
                    "runtime",
                    "failure_code",
                    "retry_after_cleanup",
                    "status",
                    "teardown_attempts",
                    "delete_error",
                    "created_at",
                    "updated_at",
                    "deleted_at",
                }
            )
            for cleanup in sorted(entity.cleanups, key=lambda item: item.execution_number)
        ]

    def record_resource_samples(
        self, evaluation_id: str, *, execution_number: int, samples: builtins.list[ResourceUsageSample]
    ) -> None:
        """Fold samples into the per-execution, per-component running sums and peaks."""
        if not samples:
            return

        def change(entity: ScaledEvalsEvaluation, _now: datetime) -> tuple[None, bool]:
            for sample in samples:
                usage = next(
                    (
                        item
                        for item in entity.resource_usage
                        if item.execution_number == execution_number and item.component == sample.component
                    ),
                    None,
                )
                if usage is None:
                    usage = ScaledEvalsResourceUsage(
                        execution_number=execution_number,
                        component=sample.component,
                        source=sample.source,
                        first_observed_at=sample.observed_at,
                        last_observed_at=sample.observed_at,
                    )
                    entity.resource_usage.append(usage)
                usage.source = sample.source
                usage.collection_status = sample.collection_status
                usage.collection_error = sample.collection_error
                usage.sample_count += 1
                usage.first_observed_at = min(usage.first_observed_at, sample.observed_at)
                usage.last_observed_at = max(usage.last_observed_at, sample.observed_at)
                for prefix, measure in _MEASURES:
                    value = getattr(sample, measure)
                    if value is None:
                        continue
                    setattr(usage, f"{prefix}_sample_count", getattr(usage, f"{prefix}_sample_count") + 1)
                    setattr(usage, f"{measure}_sum", getattr(usage, f"{measure}_sum") + value)
                    setattr(usage, f"{measure}_max", max(getattr(usage, f"{measure}_max"), value))
                for field in (
                    "cpu_request_cores",
                    "cpu_limit_cores",
                    "memory_request_bytes",
                    "memory_limit_bytes",
                    "gpu_request",
                ):
                    if (value := getattr(sample, field)) is not None:
                        setattr(usage, field, value)
            return None, True

        self._mutate(evaluation_id, change, touch=False, include_deleted=True)

    def resource_usage_for(self, evaluation_id: str) -> builtins.list[dict]:
        """Return bounded, attempt-aware aggregates for the public telemetry API."""
        entity = self._get(evaluation_id, include_deleted=True)
        if entity is None:
            return []

        def average(total: float, count: int) -> float | None:
            return total / count if count else None

        rows = []
        for usage in sorted(entity.resource_usage, key=lambda item: (item.execution_number, item.component)):
            rows.append(
                {
                    **usage.model_dump(
                        include={
                            "execution_number",
                            "component",
                            "source",
                            "collection_status",
                            "collection_error",
                            "sample_count",
                            "first_observed_at",
                            "last_observed_at",
                            "cpu_sample_count",
                            "memory_sample_count",
                            "cpu_request_cores",
                            "cpu_limit_cores",
                            "memory_request_bytes",
                            "memory_limit_bytes",
                            "gpu_request",
                            "gpu_sample_count",
                            "gpu_memory_sample_count",
                        }
                    ),
                    "avg_cpu_cores": average(usage.cpu_usage_cores_sum, usage.cpu_sample_count),
                    "peak_cpu_cores": usage.cpu_usage_cores_max if usage.cpu_sample_count else None,
                    "avg_memory_bytes": average(usage.memory_usage_bytes_sum, usage.memory_sample_count),
                    "peak_memory_bytes": usage.memory_usage_bytes_max if usage.memory_sample_count else None,
                    "avg_gpu_usage_percent": average(usage.gpu_usage_percent_sum, usage.gpu_sample_count),
                    "peak_gpu_usage_percent": usage.gpu_usage_percent_max if usage.gpu_sample_count else None,
                    "avg_gpu_memory_usage_bytes": average(
                        usage.gpu_memory_usage_bytes_sum, usage.gpu_memory_sample_count
                    ),
                    "peak_gpu_memory_usage_bytes": (
                        usage.gpu_memory_usage_bytes_max if usage.gpu_memory_sample_count else None
                    ),
                }
            )
        return rows

    def resource_totals_for(self, evaluation_id: str) -> dict[str, float | None]:
        """Return one evaluation's usage summed over executions and components."""
        entity = self._get(evaluation_id, include_deleted=True)
        usage = [] if entity is None else entity.resource_usage

        def peak(values: Iterable[float | None]) -> float | None:
            present = [float(value) for value in values if value is not None]
            return max(present) if present else None

        return {
            "cpu_samples": float(sum(item.cpu_sample_count for item in usage)),
            "cpu_sum": float(sum(item.cpu_usage_cores_sum for item in usage)),
            "peak_cpu_cores": peak(item.cpu_usage_cores_max for item in usage if item.cpu_sample_count),
            "cpu_request_cores": peak(item.cpu_request_cores for item in usage),
            "cpu_limit_cores": peak(item.cpu_limit_cores for item in usage),
            "memory_samples": float(sum(item.memory_sample_count for item in usage)),
            "memory_sum": float(sum(item.memory_usage_bytes_sum for item in usage)),
            "peak_memory_bytes": peak(item.memory_usage_bytes_max for item in usage if item.memory_sample_count),
            "memory_request_bytes": peak(item.memory_request_bytes for item in usage),
            "memory_limit_bytes": peak(item.memory_limit_bytes for item in usage),
            "gpu_request": peak(item.gpu_request for item in usage),
            "gpu_samples": float(sum(item.gpu_sample_count for item in usage)),
        }

    # Owner and fleet views

    def summary(self, owner_id: str) -> dict[str, Any]:
        """Return the owner's evaluation counts and task counts."""
        cutoff = _now() - timedelta(hours=24)
        owned = self._scan(owner_id=[owner_id])
        evaluations = {
            "active": sum(entity.status in ("blocked", *_ACTIVE) for entity in owned),
            "queued": sum(entity.status == "queued" for entity in owned),
            "failed_24h": sum(entity.status == "failed" and entity.changed_at >= cutoff for entity in owned),
            "succeeded_24h": sum(entity.status == "succeeded" and entity.changed_at >= cutoff for entity in owned),
        }
        return {"evaluations": evaluations, "tasks": TaskStore().owner_summary(owner_id)}

    def capacity(self) -> dict[str, int]:
        live = self._scan(status=_CANCELLABLE)
        return {
            "active_runs": sum(entity.status in ("blocked", *_ACTIVE) for entity in live),
            "queued_runs": sum(entity.status == "queued" for entity in live),
            "active_slots": sum(entity.parallelism for entity in live if entity.status in _ACTIVE),
        }

    def live_usage(self) -> dict:
        """Return queued and active run counts overall and per owner, plus the task total."""
        live = self._scan(status=_CANCELLABLE)

        def counts(entities: builtins.list[ScaledEvalsEvaluation]) -> dict[str, int]:
            return {
                "queued_runs": sum(entity.status == "queued" for entity in entities),
                "active_runs": sum(entity.status != "queued" for entity in entities),
                "total_parallelism": sum(entity.parallelism for entity in entities),
            }

        by_owner: dict[str | None, builtins.list[ScaledEvalsEvaluation]] = {}
        for entity in live:
            by_owner.setdefault(entity.owner_id, []).append(entity)
        actors = [
            {
                "owner_id": owner_id,
                **counts(owned),
                "last_run_at": max(entity.created_at for entity in owned if entity.created_at is not None),
            }
            for owner_id, owned in by_owner.items()
        ]
        return {**counts(live), "total_tasks": TaskStore().count(), "actors": actors}

    def quota_usage(self, owner_id: str) -> dict[str, int]:
        owned = self._scan(owner_id=[owner_id])
        return {
            "evaluations_active": sum(entity.status in _CLAIMABLE for entity in owned),
            "sandbox_slots_active": sum(entity.parallelism for entity in owned if entity.status in _ACTIVE),
            "tasks_owned": TaskStore().count(owner_id=owner_id),
        }

    def recent_activity(self, owner_id: str, *, limit: int) -> builtins.list[dict]:
        rows = [
            {
                "id": entity.name,
                "kind": "evaluation",
                "name": entity.display_name,
                "status": entity.status,
                "created_at": entity.created_at,
            }
            for entity in self._scan(owner_id=[owner_id])
        ]
        rows += TaskStore().owner_activity(owner_id)
        rows.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
        return rows[:limit]

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(entity.status for entity in self._scan()))

    def fleet_totals(self) -> dict[str, int]:
        """Return all-time evaluation totals, deleted evaluations included."""
        everything = self._scan(include_deleted=True)
        return {
            "tasks_run": len({entity.task_id for entity in everything}),
            "evaluation_jobs": len(everything),
            "evaluation_executions": sum(entity.current_execution for entity in everything),
            "completed_trials": sum(entity.n_trials or 0 for entity in everything),
        }

    def dispatch_snapshot(
        self,
        *,
        stuck_queued_seconds: float,
        stuck_provisioning_seconds: float,
        stuck_running_seconds: float,
        stale_worker_seconds: float,
    ) -> dict:
        """Return low-cardinality queue, worker and stuck-job diagnostics for metrics."""
        now = _now()
        active = self._scan(status=_CLAIMABLE)
        stuck_after = {
            "queued": stuck_queued_seconds,
            "provisioning": stuck_provisioning_seconds,
            "running": stuck_running_seconds,
        }

        def stuck(entity: ScaledEvalsEvaluation) -> bool:
            since = (entity.created_at or now) if entity.status == "queued" else entity.changed_at
            return since < now - timedelta(seconds=stuck_after[entity.status])

        claimed = [entity for entity in active if entity.dispatch_claimed_by is not None]
        fresh_cutoff = now - timedelta(seconds=stale_worker_seconds)
        queued_ages = [
            (now - entity.created_at).total_seconds()
            for entity in active
            if entity.status == "queued" and entity.created_at is not None
        ]
        lease_ages = [
            (now - entity.dispatch_claimed_at).total_seconds()
            for entity in claimed
            if entity.dispatch_claimed_at is not None
        ]
        stuck_jobs = Counter((entity.status, entity.runtime) for entity in active if stuck(entity))
        hour_ago = now - timedelta(hours=1)
        backend_failures = Counter(
            entity.runtime
            for entity in self._scan(status=["failed"])
            if entity.changed_at >= hour_ago
            and any(
                marker in (entity.status_detail or "").lower()
                for marker in ("backend", "launch", "status read", "timed out")
            )
        )
        return {
            "oldest_queued_seconds": float(max(queued_ages, default=0)),
            "unclaimed_queued": sum(
                entity.status == "queued" and entity.dispatch_claimed_by is None for entity in active
            ),
            "live_workers": len(
                {
                    entity.dispatch_claimed_by
                    for entity in claimed
                    if entity.dispatch_claimed_at is not None and entity.dispatch_claimed_at >= fresh_cutoff
                }
            ),
            "stale_workers": len(
                {
                    entity.dispatch_claimed_by
                    for entity in claimed
                    if entity.dispatch_claimed_at is not None and entity.dispatch_claimed_at < fresh_cutoff
                }
            ),
            "oldest_worker_lease_seconds": float(max(lease_ages, default=0)),
            "stuck_jobs": [
                {"status": status, "runtime": runtime, "count": count}
                for (status, runtime), count in stuck_jobs.items()
            ],
            "backend_failures": [{"runtime": runtime, "count": count} for runtime, count in backend_failures.items()],
        }
