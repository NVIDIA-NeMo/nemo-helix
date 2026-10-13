# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic run acceptance and reads over Entity Store and Files."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Self

from nemo_evals.taskset_runs.entities import TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.identifiers import random_run_id
from nemo_evals.taskset_runs.plan_storage import PlanDigestMismatchError as PlanDigestMismatchError
from nemo_evals.taskset_runs.plan_storage import PlanStorageError as PlanStorageError
from nemo_evals.taskset_runs.plan_storage import StoredPlan, delete_fileset_ref, load_plan, store_plan
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.entities import EntityClient, EntityClientProtocol
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation

_ACCEPT_RETRY_POLICY = replace(PLATFORM_DEFAULT_RETRY_POLICY, retryable_status_codes=(408, 429))


@dataclass(frozen=True)
class RunAcceptance:
    """Resolved submit inputs; discard refs must be request-owned, uniquely named filesets."""

    plan: TasksetRunPlan
    idempotency_key: str | None
    request_digest: str
    accepted_at: datetime
    execution_deadline_at: datetime
    finalization_deadline_at: datetime
    discard_on_loss: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcceptedRun:
    """Authoritative run and whether this request's named create committed."""

    run: TasksetRunEntity
    created: bool


class RunRequestConflict(RuntimeError):
    """The idempotency key already belongs to a different logical request."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id!r} already exists with a different request digest")


class RunIdExhausted(RuntimeError):
    """Both attempts to allocate a keyless random run name collided."""


class TasksetRunStore:
    """Own the named-create acceptance point and immutable plan binding."""

    def __init__(self, entity_client: EntityClientProtocol[TasksetRunEntity], files_client: AsyncFilesClient) -> None:
        """Test seam accepting an entity adapter; applies no retry policy.

        Production callers must use from_clients so conflict settlement owns retries.
        """
        self.entity_client = entity_client
        self.files_client = files_client

    @classmethod
    def from_clients(cls, entities_client: AsyncEntitiesClient, files_client: AsyncFilesClient) -> Self:
        """Production constructor: clone retries on the raw client before wrapping it.

        Keeps 5xx/transport retries and suppresses ordinary 409 retries. Authorization
        and the matching caller identity are supplied by the submit layer.
        """
        return cls(EntityClient(entities_client.with_retry(_ACCEPT_RETRY_POLICY)), files_client)

    async def accept(self, acceptance: RunAcceptance) -> AcceptedRun:
        """Upload a fresh plan, named-create its run and settle proven conflicts.

        Own-plan equality recovers a committed create whose reply was lost. Keyed
        losers discard only request-owned resources and replay by request digest.
        Keyless losers retry once with a new plan/run ID. Unknown outcomes keep
        all resources because a committed run may already reference them.
        """
        plan = acceptance.plan
        for attempt in range(2):
            stored = await store_plan(self.files_client, plan)
            try:
                run = await self.entity_client.create(self._new_run(acceptance, plan, stored))
                return AcceptedRun(run=run, created=True)
            except NemoEntityConflictError:
                existing = await self.entity_client.get(TasksetRunEntity, name=plan.run_id, workspace=plan.workspace)
                if existing.plan_ref == stored.ref:
                    return AcceptedRun(run=existing, created=True)
                await delete_fileset_ref(self.files_client, stored.ref)
                if acceptance.idempotency_key is None:
                    if attempt == 0:
                        plan = plan.model_copy(update={"run_id": random_run_id()})
                    continue
                await self._discard(acceptance.discard_on_loss)
                if existing.request_digest != acceptance.request_digest:
                    raise RunRequestConflict(existing.name) from None
                return AcceptedRun(run=existing, created=False)
        await self._discard(acceptance.discard_on_loss)
        raise RunIdExhausted("could not allocate a random run name after two collisions")

    async def _discard(self, refs: tuple[str, ...]) -> None:
        for ref in refs:
            await delete_fileset_ref(self.files_client, ref)

    @staticmethod
    def _new_run(acceptance: RunAcceptance, plan: TasksetRunPlan, stored: StoredPlan) -> TasksetRunEntity:
        return TasksetRunEntity(
            name=plan.run_id,
            workspace=plan.workspace,
            plan_ref=stored.ref,
            plan_digest=stored.digest,
            taskset_ref=plan.taskset_ref,
            idempotency_key=acceptance.idempotency_key,
            request_digest=acceptance.request_digest,
            accepted_at=acceptance.accepted_at,
            execution_deadline_at=acceptance.execution_deadline_at,
            finalization_deadline_at=acceptance.finalization_deadline_at,
        )

    async def load_plan(self, run: TasksetRunEntity) -> TasksetRunPlan:
        """Download, verify and validate the run's plan on every call; nothing is cached."""
        return await load_plan(self.files_client, run.plan_ref, expected_digest=run.plan_digest)

    async def find(self, workspace: str, run_id: str) -> TasksetRunEntity | None:
        """Return the named run in workspace, or None only when it is absent."""
        try:
            return await self.entity_client.get(TasksetRunEntity, name=run_id, workspace=workspace)
        except NemoEntityNotFoundError:
            return None

    async def replay(self, workspace: str, run_id: str, request_digest: str) -> TasksetRunEntity | None:
        """Return an identical earlier request's run in any phase, without touching Files."""
        run = await self.find(workspace, run_id)
        if run is not None and run.request_digest != request_digest:
            raise RunRequestConflict(run.name)
        return run

    async def list_active(self, workspace: str, caller: str, taskset_ref: str | None = None) -> list[TasksetRunEntity]:
        """Return caller's accepted/running runs, optionally on one pinned taskset revision.

        Filter and sort every page server-side. Complete over stable data; best-effort
        when concurrent changes shift Entity Store's offset-based pagination.
        """
        conditions: list[FilterOperation] = [
            ComparisonOperation(field="created_by", operator=FilterOperator.EQ, value=caller),
            ComparisonOperation(
                field="data.phase",
                operator=FilterOperator.IN,
                value=[TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING],
            ),
        ]
        if taskset_ref is not None:
            conditions.append(
                ComparisonOperation(field="data.taskset_ref", operator=FilterOperator.EQ, value=taskset_ref)
            )
        filter_operation = LogicalOperation(operator=FilterOperator.AND, operations=conditions)
        runs: list[TasksetRunEntity] = []
        page = 1
        while True:
            result = await self.entity_client.list(
                TasksetRunEntity,
                workspace=workspace,
                filter_operation=filter_operation,
                sort="name",
                page=page,
            )
            runs.extend(result.data)
            if page >= result.pagination.total_pages:
                return runs
            page += 1
