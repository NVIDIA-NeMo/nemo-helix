# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic run acceptance and reads over Entity Store and Files."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol, Self, cast

from nemo_evals.taskset_runs.entities import (
    TERMINAL_RUN_PHASES,
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
    assert_member_transition,
    assert_run_transition,
)
from nemo_evals.taskset_runs.identifiers import member_name
from nemo_evals.taskset_runs.plan_storage import PlanDigestMismatchError as PlanDigestMismatchError
from nemo_evals.taskset_runs.plan_storage import PlanStorageError as PlanStorageError
from nemo_evals.taskset_runs.plan_storage import StoredPlan, delete_fileset_ref, load_plan, store_plan
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.submission import resolve_deadlines
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY, RetryPolicy
from nemo_helix_plugin.entities import EntityBase, EntityClient, EntityClientProtocol, ListResponse
from nemo_helix_plugin.entities.base import EntityUpdateClientProtocol
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.schema import Page, PaginationData


# The platform default retries 409. For a named create, 409 means "this name is taken": a final
# answer that no retry can change. Retrying it only adds backoff before ``accept`` settles the
# conflict. 5xx and timeouts are still retried; those retries are what turn a committed create
# whose reply was lost into a 409 for our own record, which ``accept`` recognizes by ``plan_ref``.
class RunEntityClient(EntityClientProtocol[TasksetRunEntity], EntityUpdateClientProtocol[TasksetRunEntity], Protocol):
    """Run-record reads and writes, including the run controller's versioned updates."""


_ACCEPT_RETRY_POLICY = replace(PLATFORM_DEFAULT_RETRY_POLICY, retryable_status_codes=(408, 429))


def _to_page[EntityT: EntityBase](result: ListResponse[EntityT], *, sort: str) -> Page[EntityT]:
    """Adapt an Entity Store page without recomputing or changing its totals.

    Args:
        result: Filtered, ordered records and server-provided pagination counts.
        sort: Entity-client sort used for the query.

    Returns:
        API page containing the same records, counts and ordering.
    """
    pagination = result.pagination
    return Page(
        data=result.data,
        pagination=PaginationData(
            page=pagination.page,
            page_size=pagination.page_size,
            current_page_size=pagination.current_page_size,
            total_pages=pagination.total_pages,
            total_results=pagination.total_results,
        ),
        sort=sort,
    )


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
        idempotency_key: Retry key, the caller's or one submit generated. Stored on
            the run; submit already used it to derive ``plan.run_id``.
        request_digest: Digest of the logical submit request. A same-key retry with
            a different digest raises ``RunRequestConflict``.
        accepted_at: When submit accepted the request; copied onto the run.
        timeout_seconds: The effective execution budget, counted from the run's first
            slot (``TasksetRunStore.start``).
        execution_deadline_at: ``accepted_at`` plus the queue timeout; the run expires
            then unless its first task gets a slot. Fixed here so retries do not extend it.
        finalization_deadline_at: ``execution_deadline_at`` plus the operator grace;
            ``start`` keeps that grace, so a later config change doesn't move it.
        discard_on_loss: Qualified refs to uniquely named filesets owned only by this
            request (for example its agent-files snapshot). They are deleted, best
            effort, once another request's run is proven to have won; shared
            filesets must never be included.
    """

    plan: TasksetRunPlan
    idempotency_key: str
    request_digest: str
    accepted_at: datetime
    timeout_seconds: int
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


CANCEL_WRITE_ATTEMPTS = 3
"""Reads and versioned writes ``request_cancel`` tries before giving up to a concurrent writer."""


class RunNotFound(LookupError):
    """No run with this name in the workspace."""

    def __init__(self, workspace: str, run_id: str) -> None:
        super().__init__(f"Run not found: {workspace}/{run_id}")


class RunAlreadyFinished(RuntimeError):
    """The run ended before a cancel was requested; there is nothing to cancel."""

    def __init__(self, run_id: str, phase: TasksetRunPhase) -> None:
        super().__init__(f"run {run_id} already finished: {phase.value}")


class CancelContended(RuntimeError):
    """Every attempt lost to a concurrent write (the run controller's pass); a retry is safe."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} kept changing while the cancel was recorded; retry")


class TasksetRunStore:
    """Create taskset runs and read them back; the only code that writes a new run record.

    A run is two things: its record in Entity Store (phase, deadlines, request identity)
    and its frozen plan as ``plan.json`` in Files. The store keeps them paired: each run
    points to exactly one plan, and every plan read is checked against the run's digest.

    - ``accept``: store the plan and create the run, safely under retries and races.
    - ``replay``: return the run an identical earlier request made, before redoing work.
    - ``find``: get one run by name, or None.
    - ``load_plan``: download the run's plan and verify it was not changed.
    - ``list_runs``: one filtered, newest-first page of workspace run records.
    - ``list_members``: a numerically ordered page of children of a persisted run.
    - ``list_active``: the caller's accepted or running runs, optionally for one taskset revision.
    - ``request_cancel``: record a caller's intent to cancel, for the run controller to act on.

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

    def __init__(
        self,
        entities_client: AsyncEntitiesClient,
        files_client: AsyncFilesClient,
        *,
        retry: RetryPolicy = _ACCEPT_RETRY_POLICY,
    ) -> None:
        """Production constructor: applies ``retry``, by default ``_ACCEPT_RETRY_POLICY`` (no 409 retries).

        The policy is applied to a copy of ``entities_client`` before it is wrapped,
        so 5xx and transport errors are still retried but a 409 reaches ``accept``
        immediately. Authorization and the matching caller identity come from the
        clients the caller passes in.

        Args:
            entities_client: Raw authorized client; its own retry policy is not mutated.
            files_client: Authorized client for the same workspace's plan files.
            retry: Entity Store retries; the run controller passes none, so one hung call can't hold a cycle.
        """
        self._bind(EntityClient(entities_client.with_retry(retry)), files_client)

    @classmethod
    def _with_entity_adapter(cls, entity_client: RunEntityClient, files_client: AsyncFilesClient) -> Self:
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

    def _bind(self, entity_client: RunEntityClient, files_client: AsyncFilesClient) -> None:
        """Set the two clients every store method uses; shared by both constructors.

        Args:
            entity_client: Typed run-record adapter.
            files_client: Files client for plan writes, reads, and cleanup.
        """
        self.entity_client = entity_client
        self.files_client = files_client

    async def accept(self, acceptance: RunAcceptance) -> AcceptedRun:
        """Upload a fresh plan, named-create its run and settle proven conflicts.

        Own-plan equality recovers a committed create whose reply was lost. A loser
        discards only request-owned resources and replays by request digest. Unknown
        outcomes keep all resources because a committed run may already reference them.

        Args:
            acceptance: Resolved plan, fixed times, request identity, and exclusively
                owned filesets to discard if this request is proven to have lost.

        Returns:
            The authoritative run and whether this attempt committed its create,
            including recovery of a lost reply by matching the stored plan ref.

        Raises:
            PlanStorageError: Creating or uploading the plan failed.
            RunRequestConflict: The winner has a different request digest.
            Exception: Entity failures, including a failed conflict lookup, propagate
                without cleanup because they do not prove resource ownership.
        """
        plan = acceptance.plan
        stored = await store_plan(self.files_client, plan)
        try:
            run = await self.entity_client.create(self._new_run(acceptance, plan, stored))
            return AcceptedRun(run=run, created=True)
        except NemoEntityConflictError:
            existing = await self.entity_client.get(TasksetRunEntity, name=plan.run_id, workspace=plan.workspace)
        if existing.plan_ref == stored.ref:
            return AcceptedRun(run=existing, created=True)
        await delete_fileset_ref(self.files_client, stored.ref)
        await self._discard(acceptance.discard_on_loss)
        if existing.request_digest != acceptance.request_digest:
            raise RunRequestConflict(existing.name) from None
        return AcceptedRun(run=existing, created=False)

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
            plan: The plan being accepted.
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
            timeout_seconds=acceptance.timeout_seconds,
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

    async def list_runs(
        self,
        workspace: str,
        *,
        caller: str | None = None,
        active: bool = False,
        taskset_ref: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Page[TasksetRunEntity]:
        """Read one newest-first page with all requested filters applied server-side.

        Equal creation timestamps have no secondary sort and can cross page boundaries.
        Offset pages may repeat rows when concurrent insertions shift earlier pages.

        Args:
            workspace: Workspace containing the records.
            caller: Optional creator principal; None includes every caller.
            active: Restrict to accepted and running phases when True.
            taskset_ref: Optional pinned revision; None includes all tasksets.
            page: One-based page number.
            page_size: Maximum records on the requested page.

        Returns:
            Stored records and the filtered Entity Store pagination counts. No Files
            or Jobs reads are performed; storage failures propagate to the caller.
        """
        conditions: list[FilterOperation] = []
        if caller is not None:
            conditions.append(ComparisonOperation(field="created_by", operator=FilterOperator.EQ, value=caller))
        if active:
            conditions.append(
                ComparisonOperation(
                    field="data.phase",
                    operator=FilterOperator.IN,
                    value=[TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING],
                )
            )
        if taskset_ref is not None:
            conditions.append(
                ComparisonOperation(field="data.taskset_ref", operator=FilterOperator.EQ, value=taskset_ref)
            )
        filter_operation = LogicalOperation(operator=FilterOperator.AND, operations=conditions) if conditions else None
        result = await self.entity_client.list(
            TasksetRunEntity,
            workspace=workspace,
            filter_operation=filter_operation,
            sort="-created_at",
            page=page,
            page_size=page_size,
        )
        return _to_page(result, sort="-created_at")

    async def list_open_runs(self, *, page_size: int = 1000) -> list[TasksetRunEntity]:
        """Every run in every workspace that hasn't finished (accepted or running), oldest first.

        For the run controller, whose client acts as the evals service principal. Reads every page:
        a backlog of old open runs must not hide newer ones. Pages of 1,000 (the Entity Store maximum)
        make a multi-page scan rare; a run that finishes mid-scan shifts later pages, so one may be
        skipped (its held slots go uncounted for a cycle) or repeated, and repeats are dropped.
        """
        runs: list[TasksetRunEntity] = []
        seen: set[tuple[str, str]] = set()
        page = 1
        while True:
            result = await self.entity_client.list(
                TasksetRunEntity,
                workspace="-",
                filter_operation=ComparisonOperation(
                    field="data.phase",
                    operator=FilterOperator.IN,
                    value=[TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING],
                ),
                sort="created_at",
                page=page,
                page_size=page_size,
            )
            for run in result.data:
                if (run.workspace, run.name) not in seen:
                    seen.add((run.workspace, run.name))
                    runs.append(run)
            if not result.data or page >= result.pagination.total_pages:
                return runs
            page += 1

    async def set_state(
        self,
        run: TasksetRunEntity,
        *,
        phase: TasksetRunPhase | None = None,
        result_state: ResultState | None = None,
        error: str | None = None,
    ) -> TasksetRunEntity:
        """Write the run's phase, result state and error; a versioned update of only those fields.

        A terminal phase is never changed, and a phase change must pass ``assert_run_transition``.
        Raises ``NemoEntityConflictError`` when another writer updated the run first; the controller
        re-reads it on its next pass.
        """
        if run.phase in TERMINAL_RUN_PHASES:
            return run
        update: dict[str, object] = {}
        if phase is not None and phase is not run.phase:
            assert_run_transition(run.phase, phase)
            update["phase"] = phase
        if result_state is not None:
            update["result_state"] = result_state
        if error is not None:
            update["error"] = error
        if not update:
            return run
        return await self.entity_client.update(run.model_copy(update=update))

    async def start(self, run: TasksetRunEntity, *, now: datetime) -> TasksetRunEntity:
        """The run's first slot: start its execution clock; one versioned write, before its first create intent.

        Writes ``running``, ``started_at`` and both deadlines, counted from ``now``. Every task Job carries the
        run's execution deadline, so this happens once, before any member has a ``task_job``.
        Raises ``NemoEntityConflictError`` when another writer (a cancel) updated the run first.
        """
        assert run.started_at is None, f"run {run.name} already started"
        assert_run_transition(run.phase, TasksetRunPhase.RUNNING)
        grace = (run.finalization_deadline_at - run.execution_deadline_at).total_seconds()  # submit's grace
        execution, finalization = resolve_deadlines(now, run.timeout_seconds, int(grace))
        update = {
            "phase": TasksetRunPhase.RUNNING,
            "started_at": now,
            "execution_deadline_at": execution,
            "finalization_deadline_at": finalization,
        }
        return await self.entity_client.update(run.model_copy(update=update))

    async def request_cancel(self, workspace: str, run_id: str, *, now: datetime) -> TasksetRunEntity:
        """Record the intent to cancel: one versioned write of ``cancel_requested_at``, and nothing else.

        The run controller turns the intent into an outcome (cancel Jobs, settle members, ``cancelled``); until
        it does, the run keeps its phase. Idempotent: an earlier request's time is kept, even once the run is
        ``cancelled``. The store must act as the evals service, as the run controller does: a run may hold fields
        only a service-principal client reads back (the creator's principal, once runs store it), and writing a
        delegated copy back would erase them.

        Raises:
            RunNotFound: No such run in ``workspace``.
            RunAlreadyFinished: The run ended with no cancel requested.
            CancelContended: Every attempt lost to a concurrent write; the record is unchanged.
        """
        for _ in range(CANCEL_WRITE_ATTEMPTS):
            run = await self.find(workspace, run_id)
            if run is None:
                raise RunNotFound(workspace, run_id)
            if run.cancel_requested_at is not None:
                return run
            if run.phase in TERMINAL_RUN_PHASES:
                raise RunAlreadyFinished(run_id, run.phase)
            try:
                return await self.entity_client.update(run.model_copy(update={"cancel_requested_at": now}))
            except NemoEntityConflictError:
                continue  # the controller wrote first; re-read and keep its fields
        raise CancelContended(run_id)

    async def list_members(
        self,
        run: TasksetRunEntity,
        *,
        phase: MemberPhase | None,
        page: int,
        page_size: int,
    ) -> Page[TasksetRunMemberEntity]:
        """Read one parent-scoped member page in numeric task-index order.

        Do not advance offsets while moving members out of a phase filter: re-query
        page 1 or omit that filter to avoid skipping records as the result set shrinks.

        Args:
            run: Persisted parent record, including its entity ID and workspace.
            phase: Optional member phase; None includes every phase.
            page: One-based page number.
            page_size: Maximum records on the requested page.

        Returns:
            Stored members and filtered pagination counts. No plan or Jobs reads occur;
            storage failures propagate.
        """
        conditions: list[FilterOperation] = [
            ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=run.id),
        ]
        if phase is not None:
            conditions.append(ComparisonOperation(field="data.phase", operator=FilterOperator.EQ, value=phase))
        member_client = cast(EntityClientProtocol[TasksetRunMemberEntity], self.entity_client)
        result = await member_client.list(
            TasksetRunMemberEntity,
            workspace=run.workspace,
            filter_operation=LogicalOperation(operator=FilterOperator.AND, operations=conditions),
            sort="task_index",
            page=page,
            page_size=page_size,
        )
        return _to_page(result, sort="task_index")

    async def list_all_members(self, run: TasksetRunEntity) -> list[TasksetRunMemberEntity]:
        """Every member of ``run`` in task-index order, reading every page; any failed page raises.

        For the run controller, the only member writer: members are created once and never deleted,
        so offset pages are stable.
        """
        members: list[TasksetRunMemberEntity] = []
        page = 1
        while True:
            result = await self.list_members(run, phase=None, page=page, page_size=1000)
            members.extend(result.data)
            assert result.pagination is not None
            if not result.data or page >= result.pagination.total_pages:
                return members
            page += 1

    async def create_member(self, run: TasksetRunEntity, task_index: int) -> TasksetRunMemberEntity | None:
        """Create task ``task_index``'s member under ``run``, queued (``image_ready``): its images are in the plan.

        A named create, so a repeat is harmless.

        Returns:
            The new member, or None when it already exists.
        """
        member = TasksetRunMemberEntity(
            name=member_name(task_index), workspace=run.workspace, task_index=task_index, phase=MemberPhase.IMAGE_READY
        )
        member._parent = run.id
        member_client = cast(EntityClientProtocol[TasksetRunMemberEntity], self.entity_client)
        try:
            return await member_client.create(member)
        except NemoEntityConflictError:
            return None

    async def update_member(self, member: TasksetRunMemberEntity, **changes: object) -> TasksetRunMemberEntity:
        """A versioned write of ``changes`` to ``member``.

        The phase must pass ``assert_member_transition``, ``task_job`` and ``job_id`` are write-once, and a
        registered member has its Job's ``job_id``.

        Raises:
            ValueError: An illegal phase change, ``task_job`` or ``job_id`` rewritten, or registered without a
                ``job_id``.
            NemoEntityConflictError: Another write landed first; the next cycle re-reads the member.
        """
        phase = changes.get("phase", member.phase)
        assert isinstance(phase, MemberPhase)
        assert_member_transition(member.phase, phase)
        for key in ("task_job", "job_id"):
            old = getattr(member, key)
            if old is not None and changes.get(key, old) != old:
                raise ValueError(f"member {member.name}'s {key} is write-once ({old})")
        if phase is MemberPhase.REGISTERED and changes.get("job_id", member.job_id) is None:
            raise ValueError(f"member {member.name} can't be registered without its Job's id")
        member_client = cast(EntityUpdateClientProtocol[TasksetRunMemberEntity], self.entity_client)
        return await member_client.update(member.model_copy(update=changes))

    async def list_active(self, workspace: str, caller: str, taskset_ref: str | None = None) -> list[TasksetRunEntity]:
        """Read all pages of this caller's active runs for overlap warnings.

        Complete over stable data; concurrent inserts or equal creation timestamps
        may repeat or move rows across newest-first offset pages.

        Args:
            workspace: Workspace containing the records.
            caller: Creator principal to match.
            taskset_ref: Optional pinned taskset revision.

        Returns:
            Accepted/running records, newest first. List failures propagate rather
            than returning a misleading partial result.
        """
        runs: list[TasksetRunEntity] = []
        page = 1
        while True:
            result = await self.list_runs(
                workspace,
                caller=caller,
                active=True,
                taskset_ref=taskset_ref,
                page=page,
                page_size=100,
            )
            runs.extend(result.data)
            assert result.pagination is not None
            if page >= result.pagination.total_pages:
                return runs
            page += 1
