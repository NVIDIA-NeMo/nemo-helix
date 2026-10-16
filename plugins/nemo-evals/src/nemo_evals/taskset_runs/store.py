# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic run acceptance and reads over Entity Store and Files."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Self

from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
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

# The platform default retries 409. For a named create, 409 means "this name is taken": a final
# answer that no retry can change. Retrying it only adds backoff before ``accept`` settles the
# conflict. 5xx and timeouts are still retried; those retries are what turn a committed create
# whose reply was lost into a 409 for our own record, which ``accept`` recognizes by ``plan_ref``.
_ACCEPT_RETRY_POLICY = replace(PLATFORM_DEFAULT_RETRY_POLICY, retryable_status_codes=(408, 429))


@dataclass(frozen=True)
class RunAcceptance:
    """Everything submit resolved, passed to ``TasksetRunStore.accept`` to create the run.

    ``accept`` is the one step where a run comes into existence: a single named
    create in Entity Store either makes the run or finds the one an earlier
    request made. Submit does all resolution first (pinned plan, idempotency key,
    request digest, deadlines) and hands it over in this object, so ``accept``
    never resolves anything again and a retry cannot extend the deadlines.

    Attributes:
        plan: Resolved plan to store; ``plan.run_id`` comes from ``derive_run_id``
            and ``plan.taskset_ref`` is already pinned.
        idempotency_key: Caller-supplied retry key, or None for a keyless submit.
            Stored on the run; submit already used it to derive ``plan.run_id``.
        request_digest: Digest of the logical submit request. A same-key retry with
            a different digest raises ``RunRequestConflict``.
        accepted_at: When submit accepted the request; copied onto the run.
        execution_deadline_at: ``accepted_at`` plus the effective timeout; unfinished
            tasks expire then. Fixed here so retries do not extend it.
        finalization_deadline_at: ``execution_deadline_at`` plus operator grace; the
            final result must be saved by then.
        discard_on_loss: Qualified refs to uniquely named filesets owned only by this
            request (for example its agent-files snapshot). They are deleted, best
            effort, once another request's run is proven to have won; shared
            filesets must never be included.
    """

    plan: TasksetRunPlan
    idempotency_key: str | None
    request_digest: str
    accepted_at: datetime
    execution_deadline_at: datetime
    finalization_deadline_at: datetime
    discard_on_loss: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcceptedRun:
    """Report the authoritative record and which request committed its create.

    A reused record may be in any phase; acceptance does not restart it.

    Attributes:
        run: The persisted run record, including its original plan binding and deadlines.
        created: True when this request's create made the run (including recovery
            of a lost create reply); False when an earlier request with the same
            key and body made it. Submit uses it to choose the HTTP response.
    """

    run: TasksetRunEntity
    created: bool


class RunRequestConflict(RuntimeError):
    """Reject key reuse for a different logical request.

    Constructed with the conflicting ``run_id``, which is retained as an
    attribute so the submit layer can identify the existing run to callers.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id!r} already exists with a different request digest")


class RunIdExhausted(RuntimeError):
    """Both attempts to allocate a keyless random run name collided."""


class TasksetRunStore:
    """Create taskset runs and read them back; the only code that writes a new run record.

    A run is two things: its record in Entity Store (phase, deadlines, request identity)
    and its frozen plan as ``plan.json`` in Files. The store keeps them paired: each run
    points to exactly one plan, and every plan read is checked against the run's digest.

    - ``accept``: store the plan and create the run, safely under retries and races.
    - ``replay``: return the run an identical earlier request made, before redoing work.
    - ``find``: get one run by name, or None.
    - ``load_plan``: download the run's plan and verify it was not changed.
    - ``list_active``: the caller's accepted or running runs, optionally for one taskset revision.

    ``accept`` is an idempotent get-or-create built on a named create (below).
    Unlike plain get-or-create, it rejects the same key with a different request
    (``RunRequestConflict``) and recognizes its own create after a lost reply by
    matching ``plan_ref``. Files holds the plan bytes the run points to.

    Named create: the store creates each run under a name it chooses (``run_id``)
    instead of one Entity Store generates. Entity Store keeps at most one record per
    ``(workspace, entity type, name)``, enforced by a unique database index, so a
    second create with the same name fails with ``NemoEntityConflictError`` (HTTP 409).
    It never overwrites or duplicates. A keyed submit derives ``run_id`` from
    ``(workspace, caller, idempotency_key)``, so "same request" becomes "same name":

    - The database decides the winner. When two creates race, exactly one wins, so no
      lock, transaction, or "check first, then create" step is needed. That last
      pattern would break with several API replicas: both could check, see nothing,
      and both create.
    - The loser reads the existing run and settles: its own ``plan_ref`` means its
      own create succeeded and only the reply was lost; the same ``request_digest``
      means another copy of the same request won; anything else is
      ``RunRequestConflict``.

    Retries: the constructor keeps the platform's retries for 5xx, timeouts, and 429
    but not for 409. A 409 on a named create is a final answer, and ``accept`` must see
    it immediately to settle the conflict. See ``_ACCEPT_RETRY_POLICY``.

    Example:
        ```python
        run_id = derive_run_id("default", "user:alice", "sort-suite-ci-2026-10-05")  # "run-1a2b3c4d5e6f7a8b"

        await entity_client.create(TasksetRunEntity(name=run_id, ...))  # first request: created
        await entity_client.create(TasksetRunEntity(name=run_id, ...))  # retry: NemoEntityConflictError
        # -> get(run_id), then settle:
        #    existing.plan_ref == our plan ref  -> our own create succeeded, the reply was lost (created=True)
        #    same request_digest                -> another copy of the same request won (created=False)
        #    different request_digest           -> RunRequestConflict (409 to the caller)
        ```

    This lives in ``taskset_runs/``, not ``api/service/``, because it has two callers:
    the submit and read routes (through the API's taskset run service), and the
    Evaluator Run Controller, which is not part of the API. Its interface uses domain
    types (``TasksetRunPlan``, ``TasksetRunEntity``), not API schemas, so both callers
    share one copy of the run invariants and the retry policy.
    """

    def __init__(self, entities_client: AsyncEntitiesClient, files_client: AsyncFilesClient) -> None:
        """Production constructor: applies ``_ACCEPT_RETRY_POLICY`` (no 409 retries).

        The policy is applied to a copy of ``entities_client`` before it is wrapped,
        so 5xx and transport errors are still retried but a 409 reaches ``accept``
        immediately. Authorization and the matching caller identity come from the
        clients the caller passes in.

        Args:
            entities_client: Raw authorized client; its own retry policy is not mutated.
            files_client: Authorized client for the same workspace's plan files.
        """
        self._bind(EntityClient(entities_client.with_retry(_ACCEPT_RETRY_POLICY)), files_client)

    @classmethod
    def _with_entity_adapter(
        cls, entity_client: EntityClientProtocol[TasksetRunEntity], files_client: AsyncFilesClient
    ) -> Self:
        """Test seam: build a store over an injected entity adapter, skipping ``__init__``.

        Applies no retry policy, so it is for tests with an in-memory fake such as
        ``FakeEntityStore``. Production code must use the constructor.

        Args:
            entity_client: Typed run-record adapter, typically an in-memory fake.
            files_client: Files client used for plan writes, reads, and cleanup.

        Returns:
            A store that talks to ``entity_client`` directly.
        """
        store = cls.__new__(cls)
        store._bind(entity_client, files_client)
        return store

    def _bind(self, entity_client: EntityClientProtocol[TasksetRunEntity], files_client: AsyncFilesClient) -> None:
        """Set the two clients every store method uses; shared by both constructors.

        Args:
            entity_client: Typed run-record adapter.
            files_client: Files client for plan writes, reads, and cleanup.
        """
        self.entity_client = entity_client
        self.files_client = files_client

    async def accept(self, acceptance: RunAcceptance) -> AcceptedRun:
        """Upload a fresh plan, named-create its run and settle proven conflicts.

        Own-plan equality recovers a committed create whose reply was lost. Keyed
        losers discard only request-owned resources and replay by request digest.
        Keyless losers retry once with a new plan/run ID. Unknown outcomes keep
        all resources because a committed run may already reference them.

        Args:
            acceptance: Resolved plan, fixed times, request identity, and exclusively
                owned filesets to discard if this request is proven to have lost.

        Returns:
            The authoritative run and whether this attempt committed its create,
            including recovery of a lost reply by matching the stored plan ref.

        Raises:
            PlanStorageError: Creating or uploading the plan failed.
            RunRequestConflict: A keyed winner has a different request digest.
            RunIdExhausted: Both keyless name attempts collided.
            Exception: Entity failures, including a failed conflict lookup, propagate
                without cleanup because they do not prove resource ownership.
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
        """Delete losing-request filesets without masking the acceptance outcome.

        Args:
            refs: Qualified refs to filesets proven to belong only to this loser.

        Returns:
            None. Ordinary failures are logged per ref, allowing later cleanup
            attempts to proceed; cancellation still propagates.
        """
        for ref in refs:
            await delete_fileset_ref(self.files_client, ref)

    @staticmethod
    def _new_run(acceptance: RunAcceptance, plan: TasksetRunPlan, stored: StoredPlan) -> TasksetRunEntity:
        """Build an unsaved run with the exact plan binding and original deadlines.

        Args:
            acceptance: Request identity and precomputed times to preserve on retries.
            plan: Plan for this attempt, possibly renamed after a keyless collision.
            stored: Reference and digest of the bytes uploaded for that plan.

        Returns:
            An accepted/pending entity ready for the atomic named create, without
            writing storage or recalculating any request inputs.
        """
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
            phase=TasksetRunPhase.ACCEPTED,
            result_state=ResultState.PENDING,
        )

    async def load_plan(self, run: TasksetRunEntity) -> TasksetRunPlan:
        """Read the accepted bytes afresh so cached state cannot hide tampering.

        Args:
            run: Record supplying the authoritative plan ref and SHA-256 digest.

        Returns:
            A newly validated plan, with nested models independent of earlier reads.

        Raises:
            PlanDigestMismatchError: Downloaded bytes differ from the accepted bytes.
            ValueError: The ref or plan JSON is invalid, including schema validation.
            Exception: Files download/read failures propagate to the caller.
        """
        return await load_plan(self.files_client, run.plan_ref, expected_digest=run.plan_digest)

    async def find(self, workspace: str, run_id: str) -> TasksetRunEntity | None:
        """Look up a run without confusing storage failures with an absent record.

        Args:
            workspace: Workspace that scopes the lookup.
            run_id: Named run identifier within that workspace.

        Returns:
            The persisted run, or None only for an Entity Store not-found response.
            Other client errors propagate.
        """
        try:
            return await self.entity_client.get(TasksetRunEntity, name=run_id, workspace=workspace)
        except NemoEntityNotFoundError:
            return None

    async def replay(self, workspace: str, run_id: str, request_digest: str) -> TasksetRunEntity | None:
        """Replay an identical request before doing resolution or Files work again.

        Args:
            workspace: Workspace containing the caller-scoped run name.
            run_id: Identifier derived for this request's workspace, caller, and key.
            request_digest: Digest of the logical request to compare with the winner.

        Returns:
            The existing run in any phase, preserving its bindings and deadlines,
            or None when absent. No Files operations are performed.

        Raises:
            RunRequestConflict: The run belongs to a different logical request.
            Exception: Lookup errors other than not-found propagate.
        """
        run = await self.find(workspace, run_id)
        if run is not None and run.request_digest != request_digest:
            raise RunRequestConflict(run.name)
        return run

    async def list_active(self, workspace: str, caller: str, taskset_ref: str | None = None) -> list[TasksetRunEntity]:
        """Return caller's accepted/running runs, optionally on one pinned taskset revision.

        Filter and sort every page server-side. Complete over stable data; best-effort
        when concurrent changes shift Entity Store's offset-based pagination.

        Args:
            workspace: Workspace in which to look for active runs.
            caller: Authenticated identity to match against record ``created_by``.
            taskset_ref: Optional pinned revision ref; None includes every taskset.

        Returns:
            Accepted/running records ordered by name across all returned pages,
            useful for detecting a caller's overlapping submissions. List errors
            propagate instead of returning a misleading partial result.
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
