# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator run controller: start each run's task Jobs under the caps, settle them, merge their bundles.

Submit only accepts a run (its images are built and frozen in the plan); this
controller does the rest, in cycles of two phases:

- **Phase 1** settles every open run, oldest first: it creates the run's members (one per task, queued in
  ``image_ready``), resolves create intents, settles members whose Job ended, stops a cancelled or expired run,
  and once every member has settled ends the run or finalizes it (bundle Job, result record). Each run reports
  a :class:`RunTally`: the task-Job slots it holds and the tasks it has queued.
- **Phase 2** hands out the free slots (:func:`allocate`): each goes to the creator holding the fewest, and
  within that creator to its oldest run below its caps. It registers in grant order, then retries pending
  creates, which already hold a slot.

A task holds a slot from its create intent (``task_job`` written before the create) until its member settles,
so timeouts and crashes never push the cluster over ``max_cluster_concurrent_jobs``. Jobs that refused to
cancel at the finalization deadline, and intents settled without a Job read, stay counted in an in-memory set
of stuck Jobs until they end.

Ending never depends on the creator: records are written, and Jobs read and cancelled, as the evals service.
The controller acts for the run's creator only to create things (task and bundle Jobs, the Intake stamp).

Task and bundle Job names are predictable, so a Job under one counts only when it runs exactly what this
run would create there: someone else's Job is never adopted, bundled or cancelled. A registered task's Job is
known by its id (``job_id``): a Job under the name with another id took it after ours ended and was deleted,
so it is never read for a result, counted or cancelled.

A run that expires or is cancelled keeps its phase and still bundles the tasks that finished: a partial result
over them, ``result_state=ready``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import ClassVar, Protocol

import httpx
from filesets import AsyncFilesetFileSystem, parse_fileset_ref
from nemo_evals.config import RunControllerConfig
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.jobs.agent_evaluate import DEFAULT_RESULT_NAME, SUMMARY_FILE_NAME
from nemo_evals.jobs.result_persistence import agent_eval_result_entity
from nemo_evals.jobs.taskset_bundle import BUNDLE_JOB_SOURCE, TasksetBundleJob
from nemo_evals.taskset_runs.entities import (
    SETTLED_MEMBER_PHASES,
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
    holds_slot,
    is_queued,
)
from nemo_evals.taskset_runs.identifiers import bundle_job_name
from nemo_evals.taskset_runs.run_evaluation import IntakeRunEvaluations, RunEvaluations, stamp_run_evaluation
from nemo_evals.taskset_runs.slots import RunTally, allocate, run_cap, task_slots
from nemo_evals.taskset_runs.specs import AcceptedTaskOutput, TasksetBundleSpec, TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanDigestMismatchError, TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import (
    Registration,
    adoption,
    past_grace,
    register_member,
    runs_request,
    task_job_request,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError, NotFoundError
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityBase, EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.file_manager import AsyncFilesetFileManager
from nemo_helix_plugin.jobs.schemas import HelixJobResultResponse, HelixJobStatus
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})
NOT_STARTED = " before it started"
"""Ends the error of a member a stop settled while it was still queued."""
FAILED_JOB_STATUSES = frozenset({"error", "cancelled"})
MAX_BUNDLE_ATTEMPTS = 3
STALLED_CREATE_SECONDS = 60
"""A bundle Job still ``created`` without a step this long after it was saved will never run.

Jobs saves a Job, then its attempt, then the attempt's first step, with no rollback, and schedules only steps.
A create that failed after the first write leaves a Job that either shows no step or isn't shown at all.
"""
DEVELOPMENT_CALLER = "user:development"
READ_TIMEOUT_SECONDS = 10.0
"""Client timeout for Jobs reads and cancels, Intake, Entity Store and Files: one hung call can't hold a cycle."""
CREATE_TIMEOUT_SECONDS = 30.0
"""Client timeout for Job creates (the SDK default is 60 s)."""
NO_RETRY = RetryPolicy(max_retries=0)
"""The controller's clients never retry: the platform's default retries timeouts, 5xx and 409, so a hung Jobs
call would hold a cycle for three timeouts, and a create retried after an ambiguous attempt could be refused
and read as a refusal of the first POST."""


class RunJobs(Protocol):
    """The Jobs, Files and record operations one run needs.

    Creates and the Intake stamp act for the run's creator; every read, cancel and record write acts as the
    evals service, so ending a run never depends on the creator's access.
    """

    async def get(self, name: str) -> HelixJobResponse | None: ...

    async def result(self, job: str, name: str) -> HelixJobResultResponse | None:
        """Result ``name`` of the Job now under name ``job``; its ``job`` field is that Job's id."""
        ...

    async def create(self, request: CreateHelixJobRequest) -> HelixJobResponse: ...

    async def has_steps(self, name: str) -> bool:
        """True when Job ``name``'s current attempt has at least one step."""
        ...

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary: ...

    async def save_record(self, entity: EntityBase) -> None: ...

    async def get_record(self, name: str) -> AgentEvalResultEntity | None: ...

    async def cancel(self, name: str) -> None:
        """Cancel Job ``name``; a missing or already finished Job is not an error."""
        ...

    @property
    def evaluations(self) -> RunEvaluations:
        """The run's Intake Evaluation, which the controller stamps once at the end."""
        ...


JobsFor = Callable[[str, TasksetRunEntity | None], AbstractAsyncContextManager[RunJobs]]
"""``jobs_for(workspace, run)``: a run's :class:`RunJobs`, creating for its creator; ``run=None`` reads only."""


class JobsUnavailableError(Exception):
    """A Jobs call was skipped: an earlier one this cycle (or drain) timed out or couldn't connect."""


class JobsBreaker:
    """The first Jobs timeout or connection error ends the Jobs calls it guards, until the next cycle.

    Phase 1 and the stuck-Job reads share one; each phase-2 drain has its own. A fast 5xx is an answer, not a
    timeout, and doesn't trip it.
    """

    def __init__(self) -> None:
        self.tripped = False
        self.timeouts = 0

    def trip(self) -> None:
        self.tripped = True
        self.timeouts += 1


class GuardedRunJobs:
    """:class:`RunJobs` whose Jobs calls stop once ``breaker`` trips; Files, records and Intake pass through."""

    def __init__(self, inner: RunJobs, breaker: JobsBreaker) -> None:
        self._inner = inner
        self._breaker = breaker

    async def _jobs_call[T](self, call: Callable[[], Awaitable[T]]) -> T:
        if self._breaker.tripped:
            raise JobsUnavailableError("an earlier Jobs call timed out this cycle")
        try:
            return await call()
        except (NemoTransportError, httpx.TransportError, TimeoutError):
            self._breaker.trip()
            raise

    @property
    def evaluations(self) -> RunEvaluations:
        return self._inner.evaluations

    async def get(self, name: str) -> HelixJobResponse | None:
        return await self._jobs_call(lambda: self._inner.get(name))

    async def result(self, job: str, name: str) -> HelixJobResultResponse | None:
        return await self._jobs_call(lambda: self._inner.result(job, name))

    async def create(self, request: CreateHelixJobRequest) -> HelixJobResponse:
        return await self._jobs_call(lambda: self._inner.create(request))

    async def has_steps(self, name: str) -> bool:
        return await self._jobs_call(lambda: self._inner.has_steps(name))

    async def cancel(self, name: str) -> None:
        await self._jobs_call(lambda: self._inner.cancel(name))

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary:
        return await self._inner.read_summary(bundle_ref)

    async def save_record(self, entity: EntityBase) -> None:
        await self._inner.save_record(entity)

    async def get_record(self, name: str) -> AgentEvalResultEntity | None:
        return await self._inner.get_record(name)


@dataclass(frozen=True)
class StuckJob:
    """A Job counted against the caps although no open run's member holds it.

    Either a Job that refused to cancel when its run was forced to end at ``finalization_deadline_at``
    (``sent_at`` None, ``job_id`` its id), or a create intent settled without a Job read (``sent_at`` its
    ``create_sent_at``): its Job may exist. The Job is never cancelled here; it only stays counted until it
    ends, until another id shows under the name (ours ended), or until a 404 (an intent's only past
    ``create_grace``) or a stepless read past it.
    """

    workspace: str
    name: str
    creator: str
    sent_at: datetime | None = None
    job_id: str | None = None


@dataclass
class _RunState:
    """One open run through one cycle: what phase 1 learned, and what phase 2 needs to register its tasks."""

    run: TasksetRunEntity
    creator: str
    members: list[TasksetRunMemberEntity]
    plan: TasksetRunPlan | None = None
    cap: int | None = 0
    no_grant: bool = True  # stopping, finalizing, plan unavailable, or an error: phase 2 skips the run
    missing: set[str] = field(default_factory=set)  # intents whose Job read 404 this cycle: phase 2 retries them
    matched: set[str] = field(default_factory=set)  # members whose running Job read as theirs (job_id) this pass
    action: str = "waiting"

    def tally(self) -> RunTally:
        held = sum(holds_slot(m) for m in self.members)
        if self.no_grant:
            return RunTally(run=self.run, creator=self.creator, held=held, queued=0, cap=0)
        queued = sum(is_queued(m) for m in self.members)
        return RunTally(run=self.run, creator=self.creator, held=held, queued=queued, cap=self.cap)

    def replace(self, member: TasksetRunMemberEntity) -> TasksetRunMemberEntity:
        self.members = [member if m.task_index == member.task_index else m for m in self.members]
        return member


@dataclass
class CycleReport:
    """What one cycle did, for its log line and for tests."""

    open_runs: int = 0
    used: int = 0
    cap: int = 0
    queued: int = 0
    held_by_creator: dict[str, int] = field(default_factory=dict)
    oldest_queued_wait: timedelta | None = None
    jobs_timeouts: int = 0
    created: int = 0
    phase2_skipped: str | None = None
    grants: list[str] = field(default_factory=list)  # run names in grant order, as allocate() returned them


class MemberListingIncompleteError(RuntimeError):
    """A member create found its name taken: the run's listing missed a member, so its held count is unknown."""


def _settled(member: TasksetRunMemberEntity) -> bool:
    return member.phase in SETTLED_MEMBER_PHASES


def _rotated[T](items: Sequence[T], offset: int) -> list[T]:
    """``items`` starting at ``offset``: a call that always times out can't keep the ones after it from running."""
    if not items:
        return []
    k = offset % len(items)
    return [*items[k:], *items[:k]]


def _from_cursor[T, K: (int, tuple[str, str])](items: Sequence[T], key: Callable[[T], K], cursor: K | None) -> list[T]:
    """``items`` in ``key`` order, starting at the first whose key is ``cursor`` or after, then wrapping around."""
    ordered = sorted(items, key=key)
    if cursor is None:
        return ordered
    k = next((i for i, item in enumerate(ordered) if key(item) >= cursor), 0)
    return [*ordered[k:], *ordered[:k]]


def stop_phase(run: TasksetRunEntity, now: datetime) -> TasksetRunPhase | None:
    """The run's stop, if any: a cancel wins over expiry."""
    if run.cancel_requested_at is not None:
        return TasksetRunPhase.CANCELLED
    if now >= _utc(run.execution_deadline_at):
        return TasksetRunPhase.EXPIRED
    return None


def _stopped(run: TasksetRunEntity, members: list[TasksetRunMemberEntity]) -> tuple[TasksetRunPhase, str] | None:
    """The stop that ended ``members``, with the run's error; None when no member was stopped.

    Only a stop writes a member ``cancelled`` or ``expired``, so a stop that crashed before the run's write still
    ends ``cancelled`` (cancel wins) or ``expired``; a late cancel leaves none, and the run completes.
    """
    for phase, member_phase, what in (
        (TasksetRunPhase.CANCELLED, MemberPhase.CANCELLED, "cancelled"),
        (TasksetRunPhase.EXPIRED, MemberPhase.EXPIRED, "expired at the execution deadline"),
    ):
        stopped = sum(m.phase is member_phase for m in members)
        if not stopped:
            continue
        if phase is TasksetRunPhase.EXPIRED and run.started_at is None:
            # Its deadline was the queue timeout from acceptance: no task ever got a slot.
            waited = _utc(run.execution_deadline_at) - _utc(run.accepted_at)
            return phase, f"no task started within the queue timeout ({_duration(waited)})"
        # Say how many never got a slot before the stop.
        queued = sum(m.phase is member_phase and (m.error or "").endswith(NOT_STARTED) for m in members)
        return phase, f"{stopped} of {len(members)} tasks {what}" + (
            f", {queued} of them still queued" if queued else ""
        )
    return None


def _join(*parts: str | None) -> str | None:
    """The non-empty parts of a run's error, joined with ``"; "``; None when there are none."""
    return "; ".join(part for part in parts if part) or None


def _member_phase(stop: TasksetRunPhase | None) -> MemberPhase:
    """A settle that isn't ``succeeded`` takes the stop's phase; without a stop, ``failed``."""
    if stop is TasksetRunPhase.CANCELLED:
        return MemberPhase.CANCELLED
    if stop is TasksetRunPhase.EXPIRED:
        return MemberPhase.EXPIRED
    return MemberPhase.FAILED


def _ended_phase(stop: TasksetRunPhase | None, status: str) -> MemberPhase:
    """A Job that ended without a result, mapped by the run's stop.

    Under a cancel, ``cancelled``: a Job that fails while being torn down must not fail a cancelled run.
    Past the deadline with no cancel, only a cancelled Job is ``expired``: a Job that failed before the
    deadline, seen after it, must not expire a run whose tasks all finished.
    """
    if stop is TasksetRunPhase.CANCELLED:
        return MemberPhase.CANCELLED
    if stop is TasksetRunPhase.EXPIRED and status == "cancelled":
        return MemberPhase.EXPIRED
    return MemberPhase.FAILED


def _ours(job: HelixJobResponse | None, member: TasksetRunMemberEntity) -> HelixJobResponse | None:
    """``job`` when it is the one ``member`` registered (its ``job_id``); None for a missing or later Job."""
    return job if job is not None and job.id == member.job_id else None


def _job_error(job: HelixJobResponse) -> str:
    details = job.error_details or {}
    message = details.get("message") or details.get("detail") or ""
    return f"job_failed: {str(message)[:500]}" if message else "job_failed"


class TasksetRunReconciler:
    """The controller's cycle (see module docstring), over injected records, Jobs and clock.

    One instance per controller process: its stuck-Job set lives in memory, and the caps assume a single
    writer (one replica, ``Recreate``). ``reconcile`` is sequential.
    """

    def __init__(
        self,
        store: TasksetRunStore,
        jobs_for: JobsFor,
        config: RunControllerConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        stop_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.store = store
        self._jobs_for = jobs_for
        self.config = config
        self._clock = clock
        self._stop_requested = stop_requested
        self.stuck: dict[tuple[str, str], StuckJob] = {}
        self._cycle = 0
        # Where each run's Job reads (and the stuck-Job reads) start next cycle: just past the read that last
        # tripped the cutoff, so one Job whose GET always hangs can't keep the others from ever being read.
        self._read_cursor: dict[str, int] = {}  # run id -> task index
        self._stuck_cursor: tuple[str, str] | None = None

    @property
    def grace(self) -> timedelta:
        return timedelta(seconds=self.config.create_grace_seconds)

    async def reconcile(self) -> CycleReport:
        """One cycle: settle every open run, then hand out the free slots. Logs one line."""
        started = time.monotonic()
        report = CycleReport()
        self._cycle += 1
        try:
            runs = await self.store.list_open_runs()
        except Exception:
            logger.exception("Listing open taskset runs failed; skipping this cycle")
            report.phase2_skipped = "open runs couldn't be listed"
            return report
        now = self._clock()
        breaker = JobsBreaker()
        states: dict[str, _RunState] = {}
        listing_failed: list[str] = []
        # The stuck-Job reads take a turn in the rotation, so a hanging read can't always come first or last.
        for run in _rotated([*runs, None], self._cycle):
            if self._stop_requested():
                return report
            if run is None:
                await self._refresh_stuck(breaker)
                continue
            try:
                state = await self._settle(run, now, breaker)
            except Exception:
                logger.exception("Run %s/%s: listing its members failed; no grants this cycle", run.workspace, run.name)
                listing_failed.append(run.name)
                continue
            states[run.id] = state
            if state.action not in {"waiting", "bundling"}:
                logger.info("Run %s/%s: %s", run.workspace, run.name, state.action)
        ordered = [states[run.id] for run in runs if run.id in states]  # oldest first, as listed
        tallies = [state.tally() for state in ordered]
        stuck_by_creator = Counter(entry.creator for entry in self.stuck.values())
        free = task_slots(tallies, stuck_by_creator, self.config)
        if listing_failed:
            report.phase2_skipped = f"members of {', '.join(listing_failed)} couldn't be listed"
        # A tripped breaker doesn't skip phase 2: it bounds itself (each drain ends at its first ambiguous create),
        # and a single Job whose GET hangs must not stop every grant in the cluster.
        elif self._stop_requested():
            report.phase2_skipped = "shutting down"
        else:
            await self._hand_out(ordered, tallies, free, stuck_by_creator, report)
        self._report(report, ordered, stuck_by_creator, breaker, now, started, runs)
        return report

    # ---- phase 1 ---------------------------------------------------------------------------------------

    async def _settle(self, run: TasksetRunEntity, now: datetime, breaker: JobsBreaker) -> _RunState:
        """Phase 1 for one run. Raises only when its members can't all be listed; every other exit reports a tally."""
        members = await self.store.list_all_members(run)
        state = _RunState(run=run, creator=run.created_by or DEVELOPMENT_CALLER, members=members)
        try:
            await self._settle_run(state, now, breaker)
        except MemberListingIncompleteError:
            raise
        except NemoEntityConflictError:
            logger.debug("Run %s changed during the pass; next pass re-reads it", run.name)
            state.no_grant, state.action = True, "conflict"
        except Exception as exc:
            logger.exception("Run %s/%s: settling failed; no grants this cycle", run.workspace, run.name)
            state.no_grant, state.action = True, f"error: {type(exc).__name__}"
        return state

    async def _settle_run(self, state: _RunState, now: datetime, breaker: JobsBreaker) -> None:
        run = state.run
        try:
            plan = await self.store.load_plan(run)
        except (PlanDigestMismatchError, ValueError, NotFoundError) as exc:
            await self._fail_unreadable(state, now, breaker, reason=str(exc) or type(exc).__name__)
            return
        except Exception as exc:
            if now >= _utc(run.finalization_deadline_at):
                await self._fail_unreadable(state, now, breaker, reason=f"still unreadable at the deadline: {exc}")
                return
            logger.warning("Run %s: loading its plan failed (%s); counted, no grant this cycle", run.name, exc)
            state.action = "plan_unavailable"
            return
        state.plan, state.cap = plan, run_cap(plan, self.config)
        await self._ensure_members(state, plan)
        stop = stop_phase(run, now)
        async with self._jobs_for(run.workspace, run) as inner:
            jobs = GuardedRunJobs(inner, breaker)
            await self._read_jobs(state, plan, jobs, breaker, now, stop)
            if stop is not None and not all(_settled(m) for m in state.members):
                await self._stop(state, jobs, now, stop)
            if all(_settled(m) for m in state.members):
                state.action = await self._end(state, plan, jobs, now)
                return
        if stop is not None:
            state.action = f"stopping ({stop.value})"
            return
        state.no_grant = False

    async def _ensure_members(self, state: _RunState, plan: TasksetRunPlan) -> None:
        """Create the run's missing members, queued in ``image_ready``; only when fewer than its tasks exist."""
        if len(state.members) >= len(plan.spec.tasks):
            return
        have = {m.task_index for m in state.members}
        for index in range(len(plan.spec.tasks)):
            if index in have:
                continue
            created = await self.store.create_member(state.run, index)
            if created is None:
                raise MemberListingIncompleteError(
                    f"run {state.run.name}: member task-{index} exists but wasn't listed"
                )
            state.members.append(created)
        state.members.sort(key=lambda m: m.task_index)

    async def _read_jobs(
        self,
        state: _RunState,
        plan: TasksetRunPlan,
        jobs: RunJobs,
        breaker: JobsBreaker,
        now: datetime,
        stop: TasksetRunPhase | None,
    ) -> None:
        """P3 and P4 in one pass over the run's members with a Job name, starting at the run's read cursor.

        One loop, so a Job whose GET always times out can't keep the run's other Jobs from being read: the
        first timeout ends the cycle's Jobs calls, and the next cycle starts just past the read that tripped it.
        """
        named = [m for m in state.members if m.task_job is not None and not _settled(m)]
        order = _from_cursor(named, lambda m: m.task_index, self._read_cursor.get(state.run.id))
        for member in order:
            if breaker.tripped:
                self._read_cursor[state.run.id] = member.task_index
                break
            job: HelixJobResponse | None = None
            if member.phase is MemberPhase.IMAGE_READY:
                member, job = await self._resolve_intent(state, plan, member, jobs, now, stop)
            if member.phase is MemberPhase.REGISTERED:
                await self._settle_registered(state, member, job, jobs, stop)

    async def _resolve_intent(
        self,
        state: _RunState,
        plan: TasksetRunPlan,
        member: TasksetRunMemberEntity,
        jobs: RunJobs,
        now: datetime,
        stop: TasksetRunPhase | None,
    ) -> tuple[TasksetRunMemberEntity, HelixJobResponse | None]:
        """P3: read a create intent's Job as the evals service: ours → registered; foreign → failed.

        Returns the member as last written and, when it was adopted, the Job just read.
        """
        index = member.task_index
        request = await task_job_request(plan, state.run, index, plan.task_environments[index])
        assert request.name is not None
        try:
            job = await jobs.get(request.name)
            found = await adoption(job, request, jobs) if job is not None else None
        except Exception as exc:
            logger.debug("Run %s: reading intent %s failed: %s", state.run.name, request.name, exc)
            return member, None
        if job is None:
            state.missing.add(member.name)  # pending; phase 2 retries the create, a stop may settle it
        elif found == "ours":
            registered = await self.store.update_member(
                member, phase=MemberPhase.REGISTERED, job_id=job.id, job_status=job.status
            )
            return state.replace(registered), job
        elif found == "foreign":
            error = f"job_name_conflict: Job {request.name} exists and runs something else"
            return state.replace(await self.store.update_member(member, phase=_member_phase(stop), error=error)), None
        elif past_grace(member.create_sent_at, now, self.grace):  # stepless
            error = f"job_name_conflict: Job {request.name} was saved without its first step and will never run"
            return state.replace(await self.store.update_member(member, phase=_member_phase(stop), error=error)), None
        return member, None

    async def _settle_registered(
        self,
        state: _RunState,
        member: TasksetRunMemberEntity,
        job: HelixJobResponse | None,
        jobs: RunJobs,
        stop: TasksetRunPhase | None,
    ) -> None:
        """P4: settle a registered member whose Job ended; record a changed ``job_status`` otherwise.

        ``job`` is the Job when P3 just read it. Ours only when its id is ``member.job_id``: Jobs deletes only
        ended Jobs, so another id under the name means ours ended and was deleted (``job_missing``). A result
        counts only when it names that id too.
        """
        assert member.task_job is not None
        try:
            job = _ours(job or await jobs.get(member.task_job), member)
            status = _status(job) if job is not None else None
            result = await jobs.result(member.task_job, DEFAULT_RESULT_NAME) if status == "completed" else None
        except Exception as exc:
            logger.debug("Run %s: reading Job %s failed: %s", state.run.name, member.task_job, exc)
            return  # keeps its slot and its last job_status
        if job is None:
            phase = MemberPhase.CANCELLED if stop is TasksetRunPhase.CANCELLED else MemberPhase.FAILED
            state.replace(await self.store.update_member(member, phase=phase, error="job_missing"))
        elif status not in TERMINAL_JOB_STATUSES:
            state.matched.add(member.name)
            if job.status != member.job_status:
                state.replace(await self.store.update_member(member, job_status=job.status))
        elif result is not None and result.job != member.job_id:
            # The name changed hands between the two reads; the next read sees the other id and settles it.
            logger.info("Run %s: Job %s was replaced before its result was read", state.run.name, member.task_job)
        elif result is not None:
            state.replace(
                await self.store.update_member(
                    member, phase=MemberPhase.SUCCEEDED, bundle_ref=result.artifact_url, job_status=job.status
                )
            )
        else:
            assert status is not None
            error = {"completed": "no_result", "cancelled": "job_cancelled"}.get(status) or _job_error(job)
            state.replace(
                await self.store.update_member(
                    member, phase=_ended_phase(stop, status), error=error, job_status=job.status
                )
            )

    async def _stop(self, state: _RunState, jobs: RunJobs, now: datetime, stop: TasksetRunPhase) -> None:
        """Cancel the task Jobs this pass proved ours and settle the members that hold no Job.

        Runs after this pass settled the Jobs that already finished. Jobs are read and cancelled as the evals
        service. Registered members keep their slots: P4 settles each one when its Job ends, mapped to the
        stop's reason, and the all-settled branch then ends the run. Stop never writes the run's phase.
        An unreadable plan never reaches _stop: _fail_unreadable ends that run without the plan.
        A Job still refusing to cancel at finalization_deadline_at is logged and its member settled anyway.

        TODO: delete the run's sandboxes by their ``nemo-run`` label. A dispatcher that is
        cancelled, OOM-killed or evicted leaves its sandboxes until OpenSandbox's ``max_sandbox_lifetime_s`` TTL.
        """
        phase = _member_phase(stop)
        forced = now >= _utc(state.run.finalization_deadline_at)
        reason = "run cancelled" if stop is TasksetRunPhase.CANCELLED else "execution deadline passed"
        for member in _rotated([m for m in state.members if not _settled(m)], self._cycle):
            if is_queued(member):
                state.replace(await self.store.update_member(member, phase=phase, error=f"{reason}{NOT_STARTED}"))
            elif member.phase is MemberPhase.IMAGE_READY:  # an intent: its Job wasn't proven ours this pass
                if member.name in state.missing and past_grace(member.create_sent_at, now, self.grace):
                    state.replace(await self.store.update_member(member, phase=phase, error=f"{reason}; no Job"))
                elif forced:
                    await self._force_settle(state, member, phase, f"{reason}; its create's outcome is unknown")
            else:
                await self._cancel(state, member, jobs, phase, reason, forced=forced)

    async def _cancel(
        self,
        state: _RunState,
        member: TasksetRunMemberEntity,
        jobs: RunJobs,
        phase: MemberPhase,
        reason: str,
        *,
        forced: bool,
    ) -> None:
        """Cancel a registered member's Job; it keeps its slot until the Job ends.

        Only a Job this pass read as the member's (``matched``) is cancelled, unless already ``cancelling``: a
        Job that took the name since must never be. At the finalization deadline the member is settled anyway,
        and its Job stays counted in the stuck set.
        """
        assert member.task_job is not None
        if member.name in state.matched and member.job_status is not HelixJobStatus.CANCELLING:
            try:
                await jobs.cancel(member.task_job)
            except Exception as exc:
                logger.info("Run %s: cancelling Job %s failed: %s", state.run.name, member.task_job, exc)
        if forced:
            await self._force_settle(state, member, phase, f"{reason}; its Job didn't end by the deadline")

    async def _force_settle(
        self, state: _RunState, member: TasksetRunMemberEntity, phase: MemberPhase, error: str
    ) -> None:
        """Settle ``member`` without a Job read at the finalization deadline; its Job, if any, stays counted."""
        assert member.task_job is not None
        logger.warning(
            "Run %s/%s: settling %s at the finalization deadline; Job %s stays counted until it ends",
            state.run.workspace,
            state.run.name,
            member.name,
            member.task_job,
        )
        # Counted first: a write that commits and then errors must not drop the Job from the count.
        self._add_stuck(state, member)
        state.replace(await self.store.update_member(member, phase=phase, error=error))

    def _add_stuck(self, state: _RunState, member: TasksetRunMemberEntity) -> None:
        """Count ``member``'s Job until it ends; call before the member's settle is written."""
        assert member.task_job is not None
        # An intent's entry keeps its window: past it, a 404 or a stepless Job drops it. A refusing Job's has none.
        window = (member.create_sent_at or self._clock()) if member.phase is MemberPhase.IMAGE_READY else None
        entry = StuckJob(state.run.workspace, member.task_job, state.creator, window, member.job_id)
        self.stuck[(entry.workspace, entry.name)] = entry

    async def _end(self, state: _RunState, plan: TasksetRunPlan, jobs: RunJobs, now: datetime) -> str:
        """Every member has settled: finalize the run (bundle Job, result record), keeping a stop's phase."""
        stopped = _stopped(state.run, state.members)
        outcome = await finalize_run(state.run, plan, state.members, jobs, self.store, now=now, stopped=stopped)
        return f"{outcome.action} {outcome.detail}".strip()

    async def _fail_unreadable(self, state: _RunState, now: datetime, breaker: JobsBreaker, *, reason: str) -> None:
        """The plan-free path for a run whose plan can't be read: end it ``failed`` (or by its stop).

        Each pass, until the run ends: settle members with no Job and intents at once (intents are never
        cancelled: their names are unverified; they join the stuck set). Cancel registered members' Jobs,
        proven ours at create or adoption; each settles when its Job ends, and holds its slot until then, or
        until ``finalization_deadline_at``. Once every member has settled, write the run. No bundle, no Intake
        stamp: both need the plan.
        """
        run = state.run
        stop = stop_phase(run, now)
        phase = _member_phase(stop)
        error = f"plan_unreadable: {reason}"[:1000]
        forced = now >= _utc(run.finalization_deadline_at)
        logger.warning("Run %s/%s: plan unreadable (%s)", run.workspace, run.name, reason)
        async with self._jobs_for(run.workspace, run) as inner:
            jobs = GuardedRunJobs(inner, breaker)
            for member in _rotated([m for m in state.members if not _settled(m)], self._cycle):
                if is_queued(member):
                    state.replace(await self.store.update_member(member, phase=phase, error=error))
                elif member.phase is MemberPhase.IMAGE_READY:
                    logger.warning(
                        "Run %s: intent %s settled without a Job read; never cancelled", run.name, member.task_job
                    )
                    self._add_stuck(state, member)
                    state.replace(await self.store.update_member(member, phase=phase, error=error))
                else:
                    await self._settle_unreadable_registered(state, member, jobs, phase, error, forced)
        if all(_settled(m) for m in state.members):
            run_phase = stop or TasksetRunPhase.FAILED
            await self.store.set_state(run, phase=run_phase, result_state=ResultState.NONE, error=error)
            state.action = f"{run_phase.value}: {error}"
        else:
            state.action = "plan_unreadable: waiting for its Jobs to end"

    async def _settle_unreadable_registered(
        self,
        state: _RunState,
        member: TasksetRunMemberEntity,
        jobs: RunJobs,
        phase: MemberPhase,
        error: str,
        forced: bool,
    ) -> None:
        assert member.task_job is not None
        try:
            job = _ours(await jobs.get(member.task_job), member)
        except Exception as exc:
            logger.debug("Run %s: reading Job %s failed: %s", state.run.name, member.task_job, exc)
        else:
            if job is None:
                state.replace(await self.store.update_member(member, phase=phase, error=f"{error}; job_missing"))
                return
            if _status(job) in TERMINAL_JOB_STATUSES:
                ended = f"{error}; its Job ended {_status(job)}"
                state.replace(await self.store.update_member(member, phase=phase, error=ended, job_status=job.status))
                return
            state.matched.add(member.name)
            if job.status != member.job_status:
                member = state.replace(await self.store.update_member(member, job_status=job.status))
        await self._cancel(state, member, jobs, phase, error, forced=forced)

    async def _refresh_stuck(self, breaker: JobsBreaker) -> None:
        """Read each stuck Job as the evals service; drop it once it ended, or when a read proves there is none."""
        now = self._clock()
        for key, entry in _from_cursor(list(self.stuck.items()), lambda item: item[0], self._stuck_cursor):
            if breaker.tripped:
                self._stuck_cursor = key  # next cycle starts just past the read that tripped the cutoff
                break
            try:
                async with self._jobs_for(entry.workspace, None) as inner:
                    jobs = GuardedRunJobs(inner, breaker)
                    job = await jobs.get(entry.name)
                    conclusive = past_grace(entry.sent_at, now, self.grace)  # a refusing Job's entry: always
                    if job is not None and entry.job_id is not None and job.id != entry.job_id:
                        gone = True  # ours ended and was deleted; the Job under the name now isn't ours
                    elif job is None:
                        gone = conclusive
                    elif _status(job) in TERMINAL_JOB_STATUSES:
                        gone = True
                    else:
                        gone = entry.sent_at is not None and conclusive and not await jobs.has_steps(entry.name)
            except Exception as exc:
                logger.debug("Reading stuck Job %s/%s failed: %s", entry.workspace, entry.name, exc)
                continue
            if gone:
                logger.info("Stuck Job %s/%s ended or is gone; no longer counted", entry.workspace, entry.name)
                del self.stuck[key]

    # ---- phase 2 ---------------------------------------------------------------------------------------

    async def _hand_out(
        self,
        ordered: list[_RunState],
        tallies: list[RunTally],
        free: int,
        stuck_by_creator: Counter[str],
        report: CycleReport,
    ) -> None:
        """Phase 2: register in ``allocate()``'s grant order, then retry pending creates.

        Two drains, each ending at its first ambiguous create (a timeout or 5xx may still be saving): an
        ambiguous grant doesn't stop the retries, and an ambiguous retry doesn't stop the grants. The clock is
        read again before each create, and a run gets nothing past its ``execution_deadline_at``. An error for
        one run (a 409 on its ``running`` write after a cancel landed) ends only that run's creates this cycle.
        """
        by_id = {state.run.id: state for state in ordered}
        grants = allocate(tallies, free, self.config.max_concurrent_task_jobs_per_creator, stuck_by_creator)
        report.grants = [by_id[run_id].run.name for run_id in grants]
        granted = ((by_id[run_id], next((m for m in by_id[run_id].members if is_queued(m)), None)) for run_id in grants)
        pending = (
            (state, member)
            for state in ordered
            for member in [m for m in state.members if m.name in state.missing and m.phase is MemberPhase.IMAGE_READY]
        )
        async with AsyncExitStack() as stack:
            opened: dict[str, RunJobs] = {}
            for drain in (granted, pending):  # both lazy: each pair is read after the creates before it
                breaker = JobsBreaker()
                for state, member in drain:
                    if state.no_grant or member is None:
                        continue
                    if state.run.id not in opened:
                        try:
                            opened[state.run.id] = await stack.enter_async_context(
                                self._jobs_for(state.run.workspace, state.run)
                            )
                        except Exception:
                            # Like a failed registration: this run gets no creates this cycle; the others still do.
                            logger.exception(
                                "Run %s/%s: opening its Jobs client failed", state.run.workspace, state.run.name
                            )
                            state.no_grant = True
                            continue
                    if not await self._register(state, member, GuardedRunJobs(opened[state.run.id], breaker), report):
                        break
                report.jobs_timeouts += breaker.timeouts

    async def _register(
        self, state: _RunState, member: TasksetRunMemberEntity, jobs: RunJobs, report: CycleReport
    ) -> bool:
        """Create ``member``'s task Job; False when its drain must end (an ambiguous create).

        The run's first grant starts its execution clock (``TasksetRunStore.start``) before the create intent, so
        every create, retry and adoption of its Jobs carries the same deadline. A run past its execution deadline,
        or one whose start or registration failed (a 409 after a cancel landed), gets no more creates this cycle.
        """
        now = self._clock()
        if now >= _utc(state.run.execution_deadline_at):  # the queue timeout until it starts, its budget after
            state.no_grant = True
            return True
        assert state.plan is not None
        try:
            if state.run.started_at is None:
                state.run = await self.store.start(state.run, now=now)
                now = self._clock()  # a slow write must not send a Job already past the deadline it just set
                if now >= _utc(state.run.execution_deadline_at):
                    state.no_grant = True
                    return True
            outcome, written = await register_member(
                member, state.run, state.plan, jobs, self.store, now=now, grace=self.grace
            )
            state.replace(written)
            if outcome is Registration.REGISTERED:
                report.created += 1
        except Exception:
            logger.exception("Run %s/%s: registering %s failed", state.run.workspace, state.run.name, member.name)
            state.no_grant = True
            return True
        return outcome is not Registration.AMBIGUOUS

    def _report(
        self,
        report: CycleReport,
        ordered: list[_RunState],
        stuck_by_creator: Counter[str],
        breaker: JobsBreaker,
        now: datetime,
        started: float,
        runs: list[TasksetRunEntity],
    ) -> None:
        """Log the cycle's line: task slots used/cap, queued tasks, open runs, wait, per creator, duration."""
        held_by_creator = Counter(stuck_by_creator)
        queued, oldest = 0, None
        for state in ordered:
            held_by_creator[state.creator] += sum(holds_slot(m) for m in state.members)
            waiting = sum(is_queued(m) for m in state.members)
            queued += waiting
            if waiting:
                wait = now - _utc(state.run.accepted_at)
                oldest = wait if oldest is None else max(oldest, wait)
        report.open_runs = len(runs)
        report.used = held_by_creator.total()
        report.cap = self.config.max_cluster_concurrent_jobs
        report.queued = queued
        report.held_by_creator = dict(held_by_creator)
        report.oldest_queued_wait = oldest
        report.jobs_timeouts += breaker.timeouts
        logger.info(
            "Taskset runs: task slots %d/%d, queued tasks %d, open runs %d, oldest queued run waited %s, "
            "held per creator %s, stuck Jobs %d, created %d, cycle %.1fs, Jobs timeouts %d%s",
            report.used,
            report.cap,
            report.queued,
            report.open_runs,
            f"{oldest.total_seconds():.0f}s" if oldest is not None else "-",
            dict(sorted(report.held_by_creator.items())),
            len(self.stuck),
            report.created,
            time.monotonic() - started,
            report.jobs_timeouts,
            f", phase 2 skipped: {report.phase2_skipped}" if report.phase2_skipped else "",
        )


@dataclass(frozen=True)
class PassOutcome:
    """What finalization did, for logs and tests."""

    action: str  # failed | bundling | completed | cancelled | expired | bundle_failed | finalization_missed
    detail: str = ""


async def finalize_run(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    members: list[TasksetRunMemberEntity],
    jobs: RunJobs,
    store: TasksetRunStore,
    *,
    now: datetime,
    stopped: tuple[TasksetRunPhase, str] | None = None,
) -> PassOutcome:
    """Every member has settled: bundle the succeeded tasks and save the run's result; safe to repeat.

    The run ends ``completed``, or with its stop's phase and error (``stopped``, from ``_stopped``):

    - none succeeded → ``failed`` (a stopped run: its stop's phase), ``result_state=none``;
    - otherwise → create (or adopt) the bundle Job; once it completes, save the run's result record
      (name = run id) and end the run, ``result_state=ready``. A stopped run's result covers the tasks that
      finished, and its error says how many;
    - no result saved by ``finalization_deadline_at`` → cancel the bundle Job; ``result_state=failed``.

    Runs even after ``execution_deadline_at``: a run whose tasks all finished isn't expired while it bundles.
    A bundle Job whose create failed midway never runs, so the next attempt's name is used (see
    ``STALLED_CREATE_SECONDS``).
    """
    ending, error = stopped or (TasksetRunPhase.COMPLETED, None)
    succeeded = [m for m in members if m.phase is MemberPhase.SUCCEEDED]
    accepted = [
        AcceptedTaskOutput(task_index=m.task_index, job=m.task_job, bundle_ref=m.bundle_ref)
        for m in succeeded
        if m.task_job is not None and m.bundle_ref is not None
    ]
    if not accepted:
        phase, why = stopped or (TasksetRunPhase.FAILED, "no task succeeded")
        await _finish(run, plan, jobs, store, now, phase, ResultState.NONE, error=why)
        return PassOutcome(phase.value, why)
    if stopped is not None:
        error = f"{error}; the result covers {len(accepted)} of {len(members)} tasks"
    if now >= _utc(run.finalization_deadline_at):
        return await _miss_finalization(run, plan, accepted, jobs, store, now, ending=ending, error=error)
    # Fixed once the members settle, so the bundle request is the same on every pass and adoption still matches.
    finished_at = max(_utc(m.updated_at or run.accepted_at) for m in succeeded)
    try:
        bundle = await _ensure_bundle_job(run, plan, accepted, finished_at, jobs, now)
    except ForeignBundleJobError as exc:
        # Left open, not failed: deleting the Job lets the next pass bundle; the finalization deadline caps the wait.
        logger.warning("Run %s: %s", run.name, exc)
        return PassOutcome("bundling", str(exc))
    if bundle is None or _status(bundle) not in TERMINAL_JOB_STATUSES:
        return PassOutcome("bundling", bundle.name if bundle else "")
    if _status(bundle) == "completed":
        result = await jobs.result(bundle.name, DEFAULT_RESULT_NAME)
        if result is not None and result.job != bundle.id:
            # The name changed hands between the two reads; the next pass sees the other Job and refuses it.
            return PassOutcome("bundling", f"{bundle.name} was replaced before its result was read")
        bundle_ref = result.artifact_url if result is not None else None
        if bundle_ref is None:
            return PassOutcome("bundling", f"{bundle.name} completed without {DEFAULT_RESULT_NAME}")
        summary = await jobs.read_summary(bundle_ref)
        record = agent_eval_result_entity(
            summary,
            name=run.name,
            workspace=run.workspace,
            job_id=bundle.name,
            target=plan.spec.target,
            bundle_ref=bundle_ref,
        )
        try:
            await jobs.save_record(record)
        except NemoEntityConflictError:
            # An earlier pass saved this bundle's record, then failed before marking the run done.
            existing = await jobs.get_record(run.name)
            if existing is None or existing.job_id != bundle.name:
                raise
        await _finish(run, plan, jobs, store, now, ending, ResultState.READY, error=error)
        return PassOutcome(ending.value, bundle_ref)
    # The newest attempt failed and no further attempt is allowed.
    failed = f"bundle Job {bundle.name} ended {_status(bundle)}"
    await _finish(run, plan, jobs, store, now, ending, ResultState.FAILED, error=_join(error, failed))
    return PassOutcome("bundle_failed", failed)


async def _finish(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    jobs: RunJobs,
    store: TasksetRunStore,
    now: datetime,
    phase: TasksetRunPhase,
    result_state: ResultState,
    *,
    error: str | None = None,
) -> None:
    """End the run: stamp its Intake Evaluation (when it publishes), then write its final state.

    Stamps first: no pass runs after the final state, so a stamp after it that failed would never be retried.
    A stamp is best effort and never stops the state write.

    Relies on this controller being the only writer of a terminal phase, one pass at a time (one replica,
    ``Recreate``; ``reconcile`` is sequential). If ``set_state`` loses to another write (the cancel route's
    ``cancel_requested_at``), the run stays open and the next pass stamps again before its own write, so the
    last stamp matches the state that wins.
    """
    if plan.intake is not None:
        await stamp_run_evaluation(jobs.evaluations, run, plan, phase=phase, result_state=result_state, now=now)
    await store.set_state(run, phase=phase, result_state=result_state, error=error)


async def _miss_finalization(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    jobs: RunJobs,
    store: TasksetRunStore,
    now: datetime,
    *,
    ending: TasksetRunPhase,
    error: str | None,
) -> PassOutcome:
    """Past the finalization deadline without a saved result: cancel the bundle Job and fail the result.

    The run ends ``ending`` (``completed``, or a stop's phase) with ``error`` first, in both branches.

    The bundle Job GETs and cancels are best effort, so a Jobs outage that spans the deadline can't keep the
    run open. Residual: a bundle Job that couldn't be cancelled may still finish, with no one to save its record.
    """
    names = [bundle_job_name(run.name, n) for n in range(1, MAX_BUNDLE_ATTEMPTS + 1)]
    record = await jobs.get_record(run.name)
    if record is not None and record.job_id in names:
        # An earlier pass saved the result, then failed before marking the run done.
        await _finish(run, plan, jobs, store, now, ending, ResultState.READY, error=error)
        return PassOutcome(ending.value, record.bundle_ref)
    for name in names:
        try:
            bundle = await jobs.get(name)
            if bundle is None or _status(bundle) in TERMINAL_JOB_STATUSES:
                continue
            await _own_bundle_job(bundle, run, plan, accepted)
            await jobs.cancel(name)
        except ForeignBundleJobError as exc:
            logger.warning("Run %s: not cancelling: %s", run.name, exc)
        except Exception as exc:
            logger.warning("Run %s: bundle Job %s couldn't be read or cancelled: %s", run.name, name, exc)
    missed = "finalization deadline passed before the result was saved"
    await _finish(run, plan, jobs, store, now, ending, ResultState.FAILED, error=_join(error, missed))
    return PassOutcome("finalization_missed", missed)


class ForeignBundleJobError(Exception):
    """A Job under a bundle Job's name runs something other than this run's bundle."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Job {name} exists and runs something else; delete it with `nemo jobs delete`")


async def _ensure_bundle_job(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    finished_at: datetime,
    jobs: RunJobs,
    now: datetime,
) -> HelixJobResponse | None:
    """The newest bundle Job; a new attempt starts only when there is none or the newest failed or never runs.

    Bundle Job names are predictable, so a Job found under one is adopted only when it runs this run's
    bundle: anyone who can create Jobs in the workspace could otherwise supply the run's result.

    A name whose create failed midway is skipped: its Job is stalled without a step, or Jobs answers its
    create with 409 yet doesn't show it.

    Why: Jobs saves a Job, its attempt and the attempt's first step as separate writes, with no rollback,
    and schedules only steps. A failure after the first write leaves the name taken by a Job that never runs:

    - attempt not saved or not linked: GET answers 404, yet every create of the name answers 409;
    - first step not saved: the Job stays ``created`` forever.

    Adopting such a Job would keep the run ``bundling`` until its finalization deadline and fail its result
    although every task completed, so the next attempt's name is used instead.

    TODO: AIRCORE-1379 makes Jobs create atomic. Once it ships, remove these workarounds: the stalled and
    unreadable-name skips here, ``_stalled``, ``STALLED_CREATE_SECONDS``, and the stepless checks in
    ``task_jobs.adoption`` and ``create_grace``.

    Raises:
        ForeignBundleJobError: A Job under a bundle Job's name runs something else.
    """
    newest: HelixJobResponse | None = None
    for n in range(1, MAX_BUNDLE_ATTEMPTS + 1):
        name = bundle_job_name(run.name, n)
        found = await jobs.get(name)
        if found is None:
            found = await _create_bundle_job(run, plan, accepted, finished_at, jobs, name)
            if found is not None:
                return await _own_bundle_job(found, run, plan, accepted)
            # 409 on create, 404 on GET: Jobs saved the Job but not its attempt, so the name is lost for good.
            logger.warning("Run %s: bundle Job %s exists but can't be read; trying the next attempt", run.name, name)
            continue
        newest = await _own_bundle_job(found, run, plan, accepted)
        # Saved without its first step, so nothing will ever schedule it: treat it like a failed attempt.
        if await _stalled(newest, jobs, now):
            logger.warning("Run %s: bundle Job %s was saved without a step; trying the next attempt", run.name, name)
        elif _status(newest) not in FAILED_JOB_STATUSES:
            return newest
    return newest


async def _create_bundle_job(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    finished_at: datetime,
    jobs: RunJobs,
    name: str,
) -> HelixJobResponse | None:
    """Create bundle Job ``name``; return the Job now under that name, if Jobs shows one."""
    request = await _bundle_request(run, plan, accepted, name, finished_at)
    try:
        await jobs.create(request)
        logger.info("Created bundle Job %s for run %s (%d tasks)", name, run.name, len(accepted))
    except ConflictError:
        pass  # another pass created it, or someone else did; the caller checks
    return await jobs.get(name)


async def _stalled(job: HelixJobResponse, jobs: RunJobs, now: datetime) -> bool:
    """True when ``job`` is still ``created`` without a step ``STALLED_CREATE_SECONDS`` after it was saved."""
    return (
        _status(job) == "created"
        and now - _utc(job.created_at) >= timedelta(seconds=STALLED_CREATE_SECONDS)
        and not await jobs.has_steps(job.name)
    )


async def _bundle_request(
    run: TasksetRunEntity,
    plan: TasksetRunPlan,
    accepted: list[AcceptedTaskOutput],
    name: str,
    finished_at: datetime,
) -> CreateHelixJobRequest:
    """The create request for bundle Job ``name``."""
    spec = TasksetBundleSpec(
        run_id=run.name,
        workspace=run.workspace,
        plan_ref=run.plan_ref,
        plan_digest=run.plan_digest,
        accepted=accepted,
        started_at=run.accepted_at,
        finished_at=finished_at,
    )
    return CreateHelixJobRequest(
        name=name,
        description=f"Merged result bundle of taskset run {run.name}",
        spec=spec.model_dump(mode="json"),
        platform_spec=await TasksetBundleJob.compile(
            spec=spec, profile=plan.dispatcher_profile, image=plan.dispatcher_image
        ),
        source=BUNDLE_JOB_SOURCE,
    )


async def _own_bundle_job(
    job: HelixJobResponse, run: TasksetRunEntity, plan: TasksetRunPlan, accepted: list[AcceptedTaskOutput]
) -> HelixJobResponse:
    """``job`` when it is the bundle Job this run would create under its name.

    ``finished_at`` comes from the Job itself, so a Job differing only there still merges exactly this run's
    accepted tasks.

    Raises:
        ForeignBundleJobError: ``job`` runs something else.
    """
    finished_at = job.spec.get("finished_at")
    if not isinstance(finished_at, str):
        raise ForeignBundleJobError(job.name)
    try:
        expected = await _bundle_request(run, plan, accepted, job.name, datetime.fromisoformat(finished_at))
    except ValueError as exc:
        raise ForeignBundleJobError(job.name) from exc
    if not runs_request(job, expected):
        raise ForeignBundleJobError(job.name)
    return job


def _duration(value: timedelta) -> str:
    """``value`` in whole hours when it is one ("48 h"), else in seconds ("90 s")."""
    seconds = int(value.total_seconds())
    return f"{seconds // 3600} h" if seconds % 3600 == 0 else f"{seconds} s"


def _utc(value: datetime) -> datetime:
    """Jobs returns naive timestamps; they are UTC."""
    return value if value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)


def _status(job: HelixJobResponse) -> str:
    status = job.status
    return status.value if hasattr(status, "value") else str(status)


class JobsServiceRunJobs:
    """:class:`RunJobs` over the platform: reads, cancels and records as the evals service; creates for the creator.

    ``creating`` is None for a read-only adapter (stuck-Job reads), whose ``create`` raises.
    """

    def __init__(self, service: AsyncNemoClient, creating: AsyncNemoClient | None, workspace: str) -> None:
        self._jobs = AsyncJobsClient.from_client(service.with_options(timeout=READ_TIMEOUT_SECONDS, retry=NO_RETRY))
        self._files = AsyncFilesClient.from_client(service.with_options(timeout=READ_TIMEOUT_SECONDS, retry=NO_RETRY))
        self._entities = EntityClient(
            AsyncEntitiesClient.from_client(service.with_options(timeout=READ_TIMEOUT_SECONDS, retry=NO_RETRY))
        )
        self._creating_jobs = (
            AsyncJobsClient.from_client(creating.with_options(timeout=CREATE_TIMEOUT_SECONDS, retry=NO_RETRY))
            if creating is not None
            else None
        )
        self._evaluations = IntakeRunEvaluations(
            AsyncIntakeClient.from_client(
                (creating or service).with_options(timeout=READ_TIMEOUT_SECONDS, retry=NO_RETRY)
            )
        )
        self._workspace = workspace

    @property
    def evaluations(self) -> RunEvaluations:
        return self._evaluations

    async def get(self, name: str) -> HelixJobResponse | None:
        try:
            return (await self._jobs.get_job(workspace=self._workspace, name=name)).data()
        except NotFoundError:
            return None

    async def result(self, job: str, name: str) -> HelixJobResultResponse | None:
        try:
            return (await self._jobs.get_job_result(workspace=self._workspace, job=job, name=name)).data()
        except NotFoundError:
            return None

    async def create(self, request: CreateHelixJobRequest) -> HelixJobResponse:
        if self._creating_jobs is None:
            raise RuntimeError("this Jobs adapter only reads")
        return (await self._creating_jobs.create_job(workspace=self._workspace, body=request)).data()

    async def has_steps(self, name: str) -> bool:
        response = await self._jobs.list_steps(workspace=self._workspace, name=name, query_params={"page_size": 1})
        return bool(response.page().items)

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary:
        import tempfile

        workspace, fileset, _ = parse_fileset_ref(bundle_ref, workspace_fallback=self._workspace)
        manager = AsyncFilesetFileManager(
            workspace=workspace or self._workspace,
            fileset_name=fileset or "",
            filesystem=AsyncFilesetFileSystem(client=self._files),
            ensure_fileset_exists=False,
        )
        with tempfile.TemporaryDirectory(prefix="run-summary-") as scratch:
            path = (await manager.download_from_url(f"{bundle_ref}/{SUMMARY_FILE_NAME}", local_dir=scratch)).path
            return AgentEvalSummary.model_validate_json(path.read_text(encoding="utf-8"))

    async def save_record(self, entity: EntityBase) -> None:
        await self._entities.save(entity)

    async def get_record(self, name: str) -> AgentEvalResultEntity | None:
        try:
            return await self._entities.get(AgentEvalResultEntity, name=name, workspace=self._workspace)
        except NemoEntityNotFoundError:
            return None

    async def cancel(self, name: str) -> None:
        try:
            await self._jobs.cancel_job(workspace=self._workspace, name=name)
        except NotFoundError:
            pass
        except ConflictError:
            # 409 also covers a Job still running: a rerun replaced its step, or it is pausing. Raise for
            # those so the run stays open and the next pass cancels again; a Job that finished is done.
            job = await self.get(name)
            if job is not None and _status(job) not in TERMINAL_JOB_STATUSES:
                raise


ClientFactory = Callable[[str | None], AbstractAsyncContextManager[AsyncNemoClient]]


@asynccontextmanager
async def _service_client(on_behalf_of: str | None) -> AsyncGenerator[AsyncNemoClient]:
    from nemo_helix_plugin.client_provider import get_async_nemo_client

    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=on_behalf_of) as client:
        yield client


class EvaluatorRunController(NemoController):
    """Start, throttle and settle Kubernetes taskset runs' task Jobs (see module docstring)."""

    name: ClassVar[str] = "evals-taskset-runs"
    dependencies: ClassVar[list[str]] = ["entities", "files", "jobs"]

    def __init__(
        self, client_factory: ClientFactory = _service_client, config: RunControllerConfig | None = None
    ) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._config = config
        self._reconciler: TasksetRunReconciler | None = None
        self._service_context: AbstractAsyncContextManager[AsyncNemoClient] | None = None
        self._service: AsyncNemoClient | None = None

    @property
    def config(self) -> RunControllerConfig:
        if self._config is None:
            from nemo_evals.config import config

            self._config = config.runs.controller
        return self._config

    @property
    def interval_seconds(self) -> float:
        return self.config.interval_seconds

    async def on_startup(self) -> None:
        self._service_context = self._client_factory(None)
        self._service = (await self._service_context.__aenter__()).with_options(
            timeout=READ_TIMEOUT_SECONDS, retry=NO_RETRY
        )
        store = TasksetRunStore(
            AsyncEntitiesClient.from_client(self._service), AsyncFilesClient.from_client(self._service), retry=NO_RETRY
        )
        self._reconciler = TasksetRunReconciler(store, self._jobs_for, self.config, stop_requested=self.stop_requested)

    async def on_shutdown(self) -> None:
        if self._service_context is not None:
            await self._service_context.__aexit__(None, None, None)

    @asynccontextmanager
    async def _jobs_for(self, workspace: str, run: TasksetRunEntity | None) -> AsyncGenerator[RunJobs]:
        """Reads, cancels and records as the evals service; creates for ``run``'s creator."""
        assert self._service is not None
        if run is None:
            yield JobsServiceRunJobs(self._service, None, workspace)
            return
        async with self._client_factory(run.created_by or DEVELOPMENT_CALLER) as creating:
            yield JobsServiceRunJobs(self._service, creating, workspace)

    @property
    def reconciler(self) -> TasksetRunReconciler:
        if self._reconciler is None:
            raise RuntimeError("EvaluatorRunController used before on_startup()")
        return self._reconciler

    @property
    def store(self) -> TasksetRunStore:
        return self.reconciler.store

    async def reconcile(self) -> None:
        await self.reconciler.reconcile()

    async def list_objects(self) -> list:
        raise NotImplementedError("reconcile() runs both phases")

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError("reconcile() runs both phases")
