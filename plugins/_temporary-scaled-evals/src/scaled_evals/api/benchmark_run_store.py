# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runs, stored in Entity Store as one entity per run.

A run stores only its identity, config and the explicit-cancel marker. Its
members are ordinary evaluations carrying `benchmark_run_id`; status, reward and
the per-task breakdown are derived on read from them (`derive_run_view`).
"""

from __future__ import annotations

# `BenchmarkRunStore.list` shadows the builtin inside the class body.
import builtins
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from nemo_helix_plugin.entities.base import EntityConflictError, EntityNotFoundError, SyncEntityClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_scaled_evals_plugin.entities import ScaledEvalsBenchmarkRun

from scaled_evals.api import config_profile_store, entity_store
from scaled_evals.api.benchmark_store import BenchmarkStore
from scaled_evals.api.evaluation_store import TERMINAL_STATUSES as _TERMINAL_STATUSES
from scaled_evals.api.evaluation_store import EvaluationStore
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


def _row(entity: ScaledEvalsBenchmarkRun) -> dict[str, Any]:
    return {
        "id": entity.name,
        "name": entity.display_name,
        **{field: getattr(entity, field) for field in _RUN_FIELDS},
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
    }


class BenchmarkRunStore:
    def __init__(self, client: SyncEntityClient | None = None, *, access: ResourceAccess = INTERNAL_ACCESS) -> None:
        self._client = client or entity_store.entity_client()
        self._access = access

    def _can_access(self, entity: ScaledEvalsBenchmarkRun, *, write: bool = False) -> bool:
        if self._access.include_all:
            return True
        if write:
            if not self._access.can_write(entity.owner_id):
                return False
        elif not self._access.can_read(entity.owner_id, entity.visibility):
            return False
        members = EvaluationStore(self._client, access=INTERNAL_ACCESS).list_for_runs([entity.name])
        if write:
            return all(self._access.can_write(member["owner_id"]) for member in members)
        return all(self._access.can_read(member["owner_id"], member["visibility"]) for member in members)

    def _get(self, run_id: str, *, include_deleted: bool = False) -> ScaledEvalsBenchmarkRun | None:
        try:
            entity = self._client.get(ScaledEvalsBenchmarkRun, run_id, workspace=entity_store.private_workspace())
        except EntityNotFoundError:
            return None
        if entity.deleted_at is not None and not (include_deleted and self._access.internal):
            return None
        return entity if self._can_access(entity) else None

    def _mutate(
        self,
        run_id: str,
        change: Callable[[ScaledEvalsBenchmarkRun], tuple[T, bool]],
        *,
        include_deleted: bool = False,
    ) -> T | None:
        for _ in range(_MAX_WRITE_ATTEMPTS):
            entity = self._get(run_id, include_deleted=include_deleted)
            if entity is None or not self._can_access(entity, write=True):
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
        Returns the run row.
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
            run = self._client.create(
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
                )
            )
        except EntityConflictError:
            raise Conflict("benchmark_run_exists", f"benchmark run already exists: {run_id}") from None
        evaluations = EvaluationStore(self._client)
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
        except BaseException:
            # Members first, each on its own: one failed delete must not leave
            # the rest dispatchable or replace the original error.
            for evaluation_id in created:
                with contextlib.suppress(Exception):
                    evaluations.soft_delete(evaluation_id)
            with contextlib.suppress(Exception):
                self.soft_delete(run_id)
            raise
        return _row(run)

    def get(self, run_id: str) -> dict | None:
        entity = self._get(run_id)
        return None if entity is None else _row(entity)

    def exists(self, run_id: str) -> bool:
        return self._get(run_id) is not None

    def get_for_mutation(self, run_id: str) -> dict | None:
        entity = self._get(run_id)
        if entity is None or not self._can_access(entity, write=True):
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
        """Return up to `limit + 1` runs after `cursor`; `q` also matches the derived status."""
        # ponytail: lists every live run, and a search derives each run's
        # status from all of its members. Upgrade path: store a derived status
        # the members update on terminal transitions.
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
        if not self._access.include_all:
            member_rows = EvaluationStore(self._client, access=INTERNAL_ACCESS).list_for_runs(
                [entity.name for entity in entities]
            )
            inaccessible_runs = {
                member["benchmark_run_id"]
                for member in member_rows
                if not self._access.can_read(member["owner_id"], member["visibility"])
            }
            entities = [entity for entity in entities if entity.name not in inaccessible_runs]
        runs = [_row(entity) for entity in entities]
        if q:
            members: dict[str, builtins.list[dict]] = {}
            for member in self.members_for_runs([run["id"] for run in runs]):
                members.setdefault(member["benchmark_run_id"], []).append(member)
            runs = [
                run
                for run in runs
                if entity_store.contains(
                    q,
                    run["id"],
                    run["name"],
                    run["benchmark_id"],
                    run["framework"],
                    run["runtime"],
                    derive_run_view(run, members.get(run["id"], []))["status"],
                )
            ]
        return entity_store.keyset_page(runs, cursor=cursor, order=order, limit=limit)

    def count(self) -> int:
        """Return how many runs were ever created, deleted ones included."""
        return len(entity_store.list_all(self._client, ScaledEvalsBenchmarkRun))

    def members_for_runs(self, run_ids: builtins.list[str]) -> builtins.list[dict]:
        """Return the member evaluations of the given runs with their task slug and name, by task."""
        allowed_runs = (
            run_ids if self._access.include_all else [run_id for run_id in run_ids if self._get(run_id) is not None]
        )
        rows = EvaluationStore(self._client, access=self._access).list_for_runs(allowed_runs)
        tasks = TaskStore().tasks(row["task_id"] for row in rows)
        members = [
            {**row, "task_slug": task["slug"], "task_name": task["name"]}
            for row in rows
            if (task := tasks.get(row["task_id"]))
        ]
        return sorted(members, key=lambda member: (member["task_id"], member["id"]))

    def cancel(self, run_id: str) -> tuple[dict | None, bool, builtins.list[dict]]:
        """Cancel a run and its unfinished members.

        Returns `(run_row, cancelled_now, cancelled_members)`; `cancelled_now`
        is False when the run was already cancelled or is missing. The newly
        cancelled members are returned so the API can tear down their runtimes
        exactly as it does for a single cancelled evaluation.
        """

        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            if entity.cancelled_at is not None:
                return False, False
            entity.cancelled_at = datetime.now(tz=UTC)
            return True, True

        changed = self._mutate(run_id, change)
        if changed is None:
            return None, False, []
        cancelled_now = bool(changed)
        run = self._get(run_id)
        if run is None or not self._can_access(run, write=True):
            return None, False, []
        cancelled_members: builtins.list[dict] = []
        if run.cancelled_at is not None:
            evaluations = EvaluationStore(self._client, access=self._access)
            # ponytail: members are cancelled one write each after the run is
            # marked, so a crash or conflict between them leaves some members
            # queued until the run is cancelled again, which retries them.
            # Upgrade path: a reconciler cancelling stragglers.
            for member in evaluations.list_for_runs([run_id]):
                if member["status"] in _TERMINAL_STATUSES:
                    continue
                row, now = evaluations.cancel(member["id"], detail="cancelled (benchmark run cancelled)")
                if now and row is not None:
                    cancelled_members.append(row)
        return self.get(run_id), cancelled_now, cancelled_members

    def soft_delete(self, run_id: str) -> bool:
        def change(entity: ScaledEvalsBenchmarkRun) -> tuple[bool, bool]:
            entity.deleted_at = datetime.now(tz=UTC)
            return True, True

        return bool(self._mutate(run_id, change))
