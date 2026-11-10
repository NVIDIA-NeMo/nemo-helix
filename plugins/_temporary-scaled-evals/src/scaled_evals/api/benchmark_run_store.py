# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runs, stored in Entity Store as one entity per run.

A run stores only its identity, config and the explicit-cancel marker. Its
members are ordinary evaluations carrying `benchmark_run_id`; status, reward and
the per-task breakdown are derived on read from them (`derive_run_view`).
"""

from __future__ import annotations

# `BenchmarkRunRequestRepository.list` shadows the builtin inside the class body.
import builtins
import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import batched
from types import MappingProxyType
from typing import Any, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmarkRun, ScaledEvalsEvaluation

from scaled_evals.api import config_profile_store, entity_store
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.evaluation_store import TERMINAL_STATUSES as _TERMINAL_STATUSES
from scaled_evals.api.evaluation_store import EvaluationGlobalQueries, EvaluationWorkerRepository
from scaled_evals.api.evaluation_store import row as evaluation_row
from scaled_evals.api.failure_diagnostics import failure_category_for_code, is_retryable_failure
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.resource_access import INTERNAL_ACCESS, ResourceAccess
from scaled_evals.api.task_store import TaskStore

T = TypeVar("T")

_MAX_WRITE_ATTEMPTS = 5
_RUN_FIELDS = tuple(
    field
    for field in ScaledEvalsBenchmarkRun.model_fields
    if field not in set(NemoEntity.model_fields) | {"display_name", "archive", "archive_status"}
)


def derive_run_status(run: dict[str, Any], members: Sequence[dict[str, Any]]) -> str:
    """Derive status without building task diagnostics or reward aggregates."""
    if run.get("cancelled_at") is not None:
        return "cancelled"
    if run.get("initialization_status", "complete") == "created":
        return "initializing"
    if not members or any(member.get("status") not in _TERMINAL_STATUSES for member in members):
        return "running"
    if all(member.get("status") == "succeeded" and member.get("reward") is not None for member in members):
        return "succeeded"
    return "failed"


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

    for m in members:
        status = m.get("status")
        reward = m.get("reward")
        terminal = status in _TERMINAL_STATUSES
        teardown_status = m.get("cancel_teardown_status") or "not_requested"
        if teardown_status == "pending":
            n_teardown_pending += 1
        elif teardown_status == "failed":
            n_teardown_failed += 1
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

    status = derive_run_status(run, members)

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


def _row(entity: ScaledEvalsBenchmarkRun) -> dict[str, Any]:
    return {
        "id": entity.name,
        "name": entity.display_name,
        **{field: getattr(entity, field) for field in _RUN_FIELDS},
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


@dataclass(frozen=True)
class BenchmarkRunRead:
    """A live, eligible parent and its authorized members from one batched read."""

    run: dict[str, Any]
    members: list[dict[str, Any]]

    def view(self) -> dict[str, Any]:
        """Roll up only the members paired with this eligible parent."""
        return derive_run_view(self.run, self.members)


def enrich_run_reads(reads: Sequence[BenchmarkRunRead]) -> list[BenchmarkRunRead]:
    """Hydrate task metadata while retaining each eligible parent/member snapshot."""
    tasks = TaskStore().tasks(member["task_id"] for read in reads for member in read.members)
    return _with_task_metadata(reads, tasks)


def _with_task_metadata(
    reads: Sequence[BenchmarkRunRead], tasks: Mapping[str, dict[str, Any]]
) -> list[BenchmarkRunRead]:
    """Attach known task metadata using the same missing-task behavior for every read path."""
    return [
        BenchmarkRunRead(
            read.run,
            [
                {**member, "task_slug": task["slug"], "task_name": task["name"]}
                for member in read.members
                if (task := tasks.get(member["task_id"]))
            ],
        )
        for read in reads
    ]


@dataclass(frozen=True)
class BenchmarkRunCancellation:
    """An accepted cancellation limited to the parent and member IDs authorized at entry."""

    _client: SyncEntityClient = field(repr=False)
    run_id: str
    member_ids: tuple[str, ...]
    cancelled_now: bool
    cancelled_members: Mapping[str, dict[str, Any]]

    def record_teardown(self, evaluation_id: str, failure: str | None) -> None:
        """Persist cleanup only for a member this operation actually cancelled."""
        member = self.cancelled_members[evaluation_id]
        evaluations = EvaluationWorkerRepository(self._client)
        if failure is not None:
            evaluations.record_cancel_teardown_failure(evaluation_id, failure)
        elif not member.get("dispatch_job_name"):
            evaluations.record_cancel_teardown_succeeded(evaluation_id)

    def read(self) -> BenchmarkRunRead | None:
        """Refresh response data for the accepted work set without rediscovering members."""
        try:
            parent = self._client.get(ScaledEvalsBenchmarkRun, self.run_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        members = [
            evaluation_row(entity)
            for entity in entity_store.list_by_names(self._client, ScaledEvalsEvaluation, self.member_ids)
            if not entity.deleted
        ]
        return BenchmarkRunRead(_row(parent), sorted(members, key=lambda member: (member["task_id"], member["id"])))


class _BenchmarkRunRepositoryBase:
    """Shared parent persistence and authorization helpers."""

    def __init__(self, client: SyncEntityClient | None = None, *, access: ResourceAccess) -> None:
        """Use service storage with explicit request access or privileged internal access."""
        self._client = client or entity_store.entity_client()
        self._access = access

    def _can_access(self, entity: ScaledEvalsBenchmarkRun, *, write: bool = False) -> bool:
        """Require parent access and the same access to every live member."""
        if self._access.include_all:
            return True
        return self._authorized_members(entity, write=write) is not None

    def _authorized_members(
        self, entity: ScaledEvalsBenchmarkRun, *, write: bool
    ) -> builtins.list[dict[str, Any]] | None:
        """Keep the member rows used to authorize a parent operation."""
        if not self._access.can_access(entity.owner_id, entity.visibility, write=write):
            return None
        members = EvaluationGlobalQueries(self._client).members_for_runs([entity.name])
        if not self._access.include_all and any(
            not self._access.can_access(member["owner_id"], member["visibility"], write=write) for member in members
        ):
            return None
        return members

    def _get(
        self, run_id: str, *, include_deleted: bool = False, write: bool = False
    ) -> ScaledEvalsBenchmarkRun | None:
        """Authorize a fresh parent and its members once for the requested access mode."""
        entity = self._load_parent(run_id, include_deleted=include_deleted)
        return entity if entity is not None and self._can_access(entity, write=write) else None

    def _load_parent(self, run_id: str, *, include_deleted: bool = False) -> ScaledEvalsBenchmarkRun | None:
        """Load parent state without granting access; callers must authorize before starting work."""
        try:
            entity = self._client.get(ScaledEvalsBenchmarkRun, run_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        if entity.deleted_at is not None and not (include_deleted and self._access.internal):
            return None
        return entity

    def _mutate(
        self,
        run_id: str,
        change: Callable[[ScaledEvalsBenchmarkRun], tuple[T, bool]],
        *,
        include_deleted: bool = False,
    ) -> T | None:
        """Recheck parent and member write access on every compare-and-swap attempt."""
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(run_id, include_deleted=include_deleted, write=True)
            if entity is None:
                return None
            result, write = change(entity)
            if not write:
                return result
            try:
                self._client.update(entity)
            except EntityConflictError:
                continue
            return result
        raise Conflict("benchmark_run_changed", "benchmark run changed concurrently; retry")


class BenchmarkRunRequestRepository(_BenchmarkRunRepositoryBase):
    """Parent and member operations constrained by request access."""

    def load_members(self, benchmark_id: str, revision: int) -> builtins.list[dict]:
        """Member tasks of a benchmark revision, each resolved to a task revision.

        `revision_status` and the image fields are None when the resolved
        revision does not exist, and `revision_status` is ``deleted`` for a
        soft-deleted task, which can't be run; the router validates `ready`.
        """
        rows = BenchmarkStore().members(benchmark_id, revision)
        return [
            {
                "task_id": member["task_id"],
                "position": member["position"],
                "task_revision": revision["revision"] if revision else task["current_revision"],
                "task_slug": task["slug"],
                "revision_status": "deleted" if task["deleted"] else revision["status"] if revision else None,
                **{
                    field: revision[field] if revision else None
                    for field in ("image_ref", "image_digest", "tarball_object_key")
                },
            }
            for member, task, revision in TaskStore().resolve_members(rows, include_deleted=True)
        ]

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
        max_concurrent_members: int | None,
        visibility: str,
        owner_id: str,
        n_attempts: int = 1,
        extra_skill_object_keys: builtins.list[str] | None = None,
        instruction_prefix: str | None = None,
        instruction_postfix: str | None = None,
        initial_user_turns: builtins.list[str] | None = None,
    ) -> dict:
        """Create the run plus one queued member evaluation per member task.

        Each entry in `members` is `{id, task_id, task_revision, task_slug}`.
        Members inherit the run's profiles, credentials, runtime and parallelism.
        Membership stays open while initialization is `created` and closes only
        after every member write succeeds. Returns the completed run row.
        """
        profiles_by_id = config_profile_store.ConfigProfileStore().get_many(
            profile_id
            for profile_id in (
                framework_profile_id,
                harbor_profile_id,
                intake_profile_id,
                *(member.get("framework_profile_id") for member in members),
            )
            if profile_id
        )
        try:
            self._client.create(
                ScaledEvalsBenchmarkRun(
                    name=run_id,
                    workspace=entity_store.private_workspace(),
                    owner_id=owner_id,
                    display_name=name,
                    framework=framework,
                    requested_framework_version=requested_framework_version,
                    framework_version=framework_version,
                    runner_image_ref=runner_image_ref,
                    runner_image_digest=runner_image_digest,
                    framework_adapter_version=framework_adapter_version,
                    sandbox_k8s_version=sandbox_k8s_version,
                    runner_metadata=runner_metadata,
                    benchmark_id=benchmark_id,
                    benchmark_revision=benchmark_revision,
                    framework_profile_id=framework_profile_id,
                    harbor_profile_id=harbor_profile_id,
                    intake_profile_id=intake_profile_id,
                    credentials=credentials,
                    runtime=runtime,
                    network_policy=network_policy,
                    network_policy_config=network_policy_config,
                    parallelism=parallelism,
                    max_concurrent_members=max_concurrent_members,
                    visibility=visibility,
                    initialization_status="created",
                )
            )
        except EntityConflictError:
            raise Conflict("benchmark_run_exists", f"benchmark run already exists: {run_id}") from None
        evaluations = EvaluationWorkerRepository(self._client)
        created: builtins.list[str] = []
        # ponytail: the run and its members are separate writes. A failure part
        # way deletes the run and the members already created, but a crash in
        # between leaves a deleted-looking run with some queued members.
        # Upgrade path: a reconciler that cancels members of a missing run.
        try:
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
                    profiles_by_id=profiles_by_id,
                )
                created.append(member["id"])

            def finish_initialization(entity: ScaledEvalsBenchmarkRun) -> tuple[dict, bool]:
                """Close membership only after every member write has succeeded."""
                if entity.cancelled_at is not None:
                    raise Conflict("benchmark_run_cancelled", "benchmark run was cancelled during initialization")
                entity.initialization_status = "complete"
                return _row(entity), True

            # Creation owns this new run; completion must preserve concurrent parent edits.
            completed = BenchmarkRunWorkerRepository(self._client)._mutate(run_id, finish_initialization)
            if completed is None:
                raise Conflict("benchmark_run_missing", "benchmark run disappeared during initialization")
        except BaseException:
            # Members first, each on its own: one failed delete must not leave
            # the rest dispatchable or replace the original error.
            for evaluation_id in created:
                with contextlib.suppress(Exception):
                    evaluations.soft_delete(evaluation_id)
            with contextlib.suppress(Exception):
                self.soft_delete(run_id)
            raise
        return completed

    def get(self, run_id: str) -> dict | None:
        entity = self._get(run_id)
        return None if entity is None else _row(entity)

    def exists(self, run_id: str) -> bool:
        return self._get(run_id) is not None

    def get_for_mutation(self, run_id: str) -> dict | None:
        """Return a run only when the caller may mutate it and all its live members."""
        entity = self._get(run_id, write=True)
        if entity is None:
            return None
        return _row(entity)

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
        """Return up to `limit + 1` authorized matches after `cursor`.

        Read parents and members in ordered candidate batches until the page and
        lookahead are filled or candidates are exhausted. Search also matches
        the derived status; task metadata is reused across search batches.
        """
        # ponytail: parent discovery still scans all matching runs. Member
        # reads stop once the requested page and lookahead are filled.
        # Upgrade path: push ordered parent discovery into Entity Store.
        entities = [
            entity
            for entity in entity_store.list_all(
                self._client,
                ScaledEvalsBenchmarkRun,
                {"benchmark_id": [benchmark_id]} if benchmark_id is not None else None,
            )
            if entity.deleted_at is None
            and self._access.can_read(entity.owner_id, entity.visibility)
            and (not shared or entity.visibility != "private")
        ]
        # Creation time and ID are immutable. Apply the cursor to candidates,
        # then fill the actual page only from fresh, authorized parent/member reads.
        candidates = entity_store.keyset_page(
            [{"id": entity.name, "created_at": entity.created_at} for entity in entities],
            cursor=cursor,
            order=order,
            limit=len(entities),
        )
        selected: builtins.list[BenchmarkRunRead] = []
        tasks: dict[str, dict[str, Any]] = {}
        queried_task_ids: set[str] = set()
        for batch in batched(candidates, entity_store.FILTER_BATCH_SIZE):
            reads = self.read_for_runs([run["id"] for run in batch])
            eligible = [
                read
                for run in batch
                if (read := reads.get(run["id"])) is not None
                and (benchmark_id is None or read.run["benchmark_id"] == benchmark_id)
                and (not shared or read.run["visibility"] != "private")
            ]
            if q:
                task_ids = {member["task_id"] for read in eligible for member in read.members} - queried_task_ids
                if task_ids:
                    tasks.update(TaskStore().tasks(task_ids))
                    queried_task_ids.update(task_ids)
                eligible = _with_task_metadata(eligible, tasks)
            for read in eligible:
                run = read.run
                if q and not entity_store.contains(
                    q,
                    run["id"],
                    run["name"],
                    run["benchmark_id"],
                    run["framework"],
                    run["runtime"],
                    derive_run_status(run, read.members),
                ):
                    continue
                selected.append(read)
                if len(selected) == limit + 1:
                    break
            if len(selected) == limit + 1:
                break
        if not q:
            selected = enrich_run_reads(selected)
        return [read.view() for read in selected]

    def read_for_runs(self, run_ids: builtins.list[str], *, write: bool = False) -> dict[str, BenchmarkRunRead]:
        """Return fresh eligible parents paired with their members, including empty runs.

        Missing, deleted and denied parents are absent from the result, so callers
        never mistake a failed authorization check for an empty member list.
        Mutation callers request write access to both the parent and every live member.
        """
        if not run_ids:
            return {}
        parents = {
            entity.name: entity
            for entity in entity_store.list_by_names(self._client, ScaledEvalsBenchmarkRun, sorted(set(run_ids)))
            if entity.deleted_at is None and self._access.can_access(entity.owner_id, entity.visibility, write=write)
        }
        rows = EvaluationGlobalQueries(self._client).members_for_runs(list(parents))
        if not self._access.include_all:
            inaccessible_runs = {
                row["benchmark_run_id"]
                for row in rows
                if not self._access.can_access(row["owner_id"], row["visibility"], write=write)
            }
            parents = {run_id: entity for run_id, entity in parents.items() if run_id not in inaccessible_runs}
            rows = [row for row in rows if row["benchmark_run_id"] in parents]
        members: dict[str, builtins.list[dict[str, Any]]] = {run_id: [] for run_id in parents}
        for row in rows:
            members[row["benchmark_run_id"]].append(row)
        return {
            run_id: BenchmarkRunRead(
                _row(entity), sorted(members[run_id], key=lambda member: (member["task_id"], member["id"]))
            )
            for run_id, entity in parents.items()
        }

    def cancel(self, run_id: str) -> BenchmarkRunCancellation | None:
        """Authorize once, then cancel the fixed member set using current record versions.

        The accepted operation may finish after ownership changes. Only IDs in
        its initial authorized set may be cancelled or cleaned up. Version
        conflicts reload records without repeating authorization or discovering
        additional members. The response is refreshed separately after cleanup.
        """
        parent = self._load_parent(run_id)
        if parent is None:
            return None
        members = self._authorized_members(parent, write=True)
        if members is None:
            return None
        for _ in range(_MAX_WRITE_ATTEMPTS):
            if parent.initialization_status != "complete":
                raise Conflict("benchmark_run_initializing", "benchmark run member creation is still in progress")
            if parent.cancelled_at is not None:
                cancelled_now = False
                break
            parent.cancelled_at = datetime.now(tz=UTC)
            try:
                self._client.update(parent)
            except EntityConflictError:
                parent = self._load_parent(run_id)
                if parent is None:
                    return None
                continue
            cancelled_now = True
            break
        else:
            raise Conflict("benchmark_run_changed", "benchmark run changed concurrently; retry")

        # These exact IDs were authorized above. Internal execution preserves
        # that grant while each mutation reads and retries against fresh versions.
        evaluations = EvaluationWorkerRepository(self._client)
        # Terminal members normally need no mutation. Refresh that subset in
        # batches to catch in-flight retries without one GET per finished record.
        terminal_members = {
            entity.name: entity
            for entity in entity_store.list_by_names(
                self._client,
                ScaledEvalsEvaluation,
                [member["id"] for member in members if member["status"] in _TERMINAL_STATUSES],
            )
        }
        cancelled_members: dict[str, dict[str, Any]] = {}
        # ponytail: parent and member writes are separate. An interrupted
        # cancellation can leave unfinished members until another cancellation.
        # Upgrade path: a reconciler cancelling stragglers.
        for member in members:
            if member["status"] in _TERMINAL_STATUSES:
                current = terminal_members.get(member["id"])
                if current is None or current.deleted or current.status in _TERMINAL_STATUSES:
                    continue
            updated, changed = evaluations.cancel(member["id"], detail="cancelled (benchmark run cancelled)")
            if changed and updated is not None:
                cancelled_members[member["id"]] = updated
        return BenchmarkRunCancellation(
            self._client,
            run_id,
            tuple(member["id"] for member in members),
            cancelled_now,
            MappingProxyType(cancelled_members),
        )

    def soft_delete(self, run_id: str) -> bool:
        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            entity.deleted_at = datetime.now(tz=UTC)
            return True, True

        return bool(self._mutate(run_id, change))


class BenchmarkRunGlobalQueries:
    """Read-only fleet-wide run queries."""

    def __init__(self, client: SyncEntityClient | None = None) -> None:
        self._client = client or entity_store.entity_client()

    def count(self) -> int:
        """Return how many runs were ever created, deleted ones included."""
        return len(entity_store.list_all(self._client, ScaledEvalsBenchmarkRun))


class BenchmarkRunWorkerRepository(BenchmarkRunRequestRepository):
    """Internal run lifecycle and experiment identity operations."""

    def __init__(self, client: SyncEntityClient | None = None) -> None:
        super().__init__(client, access=INTERNAL_ACCESS)

    def load_experiment_identity(self, benchmark_run_id: str) -> dict | None:
        """Resolve a benchmark run to its intake Experiment identity.

        Returns `{benchmark_slug, run_name}`: the benchmark slug groups the
        Experiment (one group per benchmark) and names its dataset. None when
        the run or its benchmark is missing.
        """
        try:
            entity = self._client.get(
                ScaledEvalsBenchmarkRun, benchmark_run_id, workspace=entity_store.private_workspace()
            )
        except EntityNotFoundError:
            return None
        slug = BenchmarkStore().slug(entity.benchmark_id)
        return {"benchmark_slug": slug, "run_name": entity.display_name} if slug else None
