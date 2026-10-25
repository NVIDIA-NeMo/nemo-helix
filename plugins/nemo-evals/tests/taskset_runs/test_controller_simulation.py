# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run controller over a simulated Jobs cluster.

``TasksetRunReconciler.reconcile()`` runs in a loop over a :class:`FakeJobsCluster`, a fake clock that moves
15 s per cycle, and the in-memory Entity Store. Every expectation comes from the controller's specified behavior, not from the
controller's own tests:

- The 9 **invariants** are checked after every cycle, and invariants 1, 3 and 9 also at the instant each Job
  is saved and each slot is taken (a create intent written), so an overload shows where it happens.
- Invariant 8 is checked on ``allocate()``'s inputs and output (the module global is wrapped), and on the
  order of the create intents phase 2 then writes.
- The two accepted exceptions to invariants 1 and 9 are honored: a stuck Job forgotten by a controller
  restart is marked exempt (and logged), and creates that outlive ``create_grace`` are never injected.

Jobs semantics follow the Jobs service: Jobs answers 409 before saving anything; GET is 404 until the attempt is
linked; a Job saved without its first step never runs; cancel moves a Job to ``cancelling``; only ended Jobs
can be deleted. Member listings bypass the fake store's filter evaluation (far too slow for 1,130 members)
but read the same records.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import httpx
import pytest
from nemo_evals.config import RunControllerConfig
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.jobs.agent_spec import AgentEvalSpec, HarborBuiltinAgentSource, HarborRunnerTarget
from nemo_evals.jobs.harbor_task_dispatch import TASK_JOB_SOURCE
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.jobs.taskset_bundle import BUNDLE_JOB_SOURCE
from nemo_evals.taskset_runs import controller as controller_module
from nemo_evals.taskset_runs.controller import DEVELOPMENT_CALLER, CycleReport, RunJobs, TasksetRunReconciler
from nemo_evals.taskset_runs.entities import (
    SETTLED_MEMBER_PHASES,
    TERMINAL_RUN_PHASES,
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
)
from nemo_evals.taskset_runs.run_evaluation import RunEvaluations
from nemo_evals.taskset_runs.slots import RunTally, allocate, run_cap
from nemo_evals.taskset_runs.specs import TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanDigestMismatchError, TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NemoHTTPError,
    NemoTransportError,
    NotFoundError,
    PermissionDeniedError,
)
from nemo_helix_plugin.entities import EntityBase
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.schemas import FileStorageType, HelixJobResultResponse
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatus
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

logger = logging.getLogger(__name__)

SIM_START = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
CYCLE = timedelta(seconds=15)
TERMINAL_JOB = frozenset({HelixJobStatus.COMPLETED, HelixJobStatus.ERROR, HelixJobStatus.CANCELLED})
OPEN_RUN = frozenset({TasksetRunPhase.ACCEPTED, TasksetRunPhase.RUNNING})
MEMBER_TYPE = TasksetRunMemberEntity.__entity_type__
RUN_TYPE = TasksetRunEntity.__entity_type__


class SimCrash(BaseException):
    """The controller process dies mid-cycle; a BaseException so the controller's own handlers don't catch it."""


def _http_error[ErrorT: NemoHTTPError](cls: type[ErrorT], status: int, detail: str, method: str = "POST") -> ErrorT:
    request = httpx.Request(method, "http://jobs.test/apis/jobs/v2/workspaces/w/jobs")
    return cls(httpx.Response(status, json={"detail": detail}, request=request))


def _timeout(what: str) -> NemoTransportError:
    request = httpx.Request("POST" if what == "create" else "GET", "http://jobs.test/apis/jobs/v2/workspaces/w/jobs")
    return NemoTransportError(httpx.ReadTimeout(f"Jobs {what} timed out", request=request))


@dataclass
class Faults:
    """Per-call (or per-Job, per-cycle) probabilities of each injected fault."""

    create_5xx_before: float = 0.0  # fast 5xx, nothing saved
    create_409_transient: float = 0.0  # 409 before saving (a secret-check blip); no Job under the name
    create_5xx_after: float = 0.0  # saved, then the reply is a 5xx
    create_timeout_unsaved: float = 0.0
    create_timeout_saved: float = 0.0  # saved, visible at once, the client timed out
    create_late_save: float = 0.0  # the client timed out; Jobs links the attempt later, within create_grace
    create_lost_name: float = 0.0  # Job row saved, attempt never linked: 409 and 404 forever (AIRCORE-1379)
    stepless_first: float = 0.0  # the first step is saved a little after the Job shows up
    read_5xx: float = 0.0  # GET, has_steps, result, cancel
    read_timeout: float = 0.0
    refuse_cancel: float = 0.0  # per Job: cancel answers 409 while it runs
    job_error: float = 0.0
    no_result: float = 0.0
    operator_cancel: float = 0.0  # per tick, per running task Job
    delete_ended: float = 0.0  # per tick, per ended task Job
    squat: float = 0.0  # per task at submit: a foreign Job already holds the task Job's name
    replace_deleted: float = 0.0  # per deleted task Job: someone else's Job takes its name at once, same work
    member_list_fail: float = 0.0  # per run listing
    crash: float = 0.0  # per Jobs call or Entity Store write during a cycle
    restart: float = 0.0  # per cycle, between cycles
    mid_cycle_cancel: float = 0.0  # per create: the cancel route writes this run's cancel_requested_at


_JOB_IDS = itertools.count(1)


@dataclass
class SimJob:
    """One Job as the Jobs service holds it."""

    workspace: str
    name: str
    request: CreateHelixJobRequest
    creator: str
    saved_at: datetime
    visible_at: datetime | None  # GET answers 404 until the attempt is linked; None: never linked
    step_at: datetime | None  # None: saved without its first step, never runs
    run_ticks: int
    outcome: HelixJobStatus
    with_result: bool = True
    by_controller: bool = True
    refuse_cancel: bool = False
    status: HelixJobStatus = HelixJobStatus.CREATED
    cancel_ticks: int = 0
    updated_at: datetime = SIM_START
    exempt: str | None = None  # why this Job no longer counts against invariants 1/9 (an accepted exception)
    error_details: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: f"job-{next(_JOB_IDS)}")  # a new one per save, as in Jobs

    @property
    def is_task(self) -> bool:
        return self.request.source == TASK_JOB_SOURCE

    @property
    def will_run(self) -> bool:
        return self.visible_at is not None and self.step_at is not None

    @property
    def active(self) -> bool:
        return self.will_run and self.status not in TERMINAL_JOB

    def visible(self, now: datetime) -> bool:
        return self.visible_at is not None and now >= self.visible_at

    def response(self) -> HelixJobResponse:
        return HelixJobResponse(
            id=self.id,
            attempt_id=f"attempt-{self.name}",
            name=self.name,
            workspace=self.workspace,
            source=self.request.source,
            spec=self.request.spec,
            platform_spec=self.request.platform_spec,
            fileset=f"{self.workspace}/{self.name}",
            status=self.status,
            error_details=self.error_details,
            created_at=self.saved_at,
            updated_at=self.updated_at,
        )


class FakeJobsCluster:
    """The Jobs service across every workspace, with seeded faults; also the Evals records the bundle path saves."""

    def __init__(self, sim: Sim) -> None:
        self.sim = sim
        self.jobs: dict[tuple[str, str], SimJob] = {}
        self.by_id: dict[str, SimJob] = {}  # every Job ever saved, deleted ones included
        self.records: dict[tuple[str, str], EntityBase] = {}
        self.outage: str | None = None  # "5xx" or "hang": every Jobs call fails that way
        self.scripted_create: dict[str, list[str]] = defaultdict(list)  # job name -> outcomes, one per POST
        self.refusing: set[str] = set()  # job names whose cancel answers 409 while they run
        self.hanging: set[str] = set()  # job names whose every call times out (one bad Job)
        self.forbidden: set[str] = set()  # creators who lost access: their creates answer 403 before saving
        self.read_latency = timedelta(0)  # how long each answered read or cancel takes
        self.create_latency = timedelta(0)  # how long each answered create takes
        self.duration: Callable[[str], int] = lambda name: sim.rng.randint(2, 6)
        self.cancel_ticks: tuple[int, int] = (1, 3)
        self.calls: Counter[int] = Counter()  # Jobs calls that reached the service, per cycle
        self.timeouts: Counter[int] = Counter()
        self.create_posts: Counter[int] = Counter()
        self.posts_by_name: Counter[tuple[str, str]] = Counter()
        self.saves: Counter[tuple[str, str]] = Counter()
        self.saved_in_cycle: Counter[int] = Counter()
        self.cancels: list[tuple[int, str, str]] = []  # (cycle, workspace, name)
        self.peak_active = 0

    # ---- the service -------------------------------------------------------------------------------------

    def _enter(self, kind: str, name: str) -> None:
        sim = self.sim
        self.calls[sim.cycle] += 1
        # Phase 1's Jobs calls stop at its first timeout; phase 2 still runs (each drain ends at its first
        # ambiguous create), so a hanging read can't stop every grant in the cluster.
        if sim.phase1_timed_out and sim.grant_order is None:
            sim.violate("phase-1 breaker", f"Jobs {kind} of {name} after a phase-1 Jobs timeout, still in phase 1")
        if name in self.hanging:
            self._time_out(kind)
        if self.outage == "5xx":
            raise _http_error(InternalServerError, 503, "jobs service unavailable", "GET")
        if self.outage == "hang":
            self._time_out(kind)
        if sim.crash_roll():
            raise SimCrash(f"controller died during a Jobs {kind}")
        sim.clock.advance(self.create_latency if kind == "create" else self.read_latency)

    def _time_out(self, kind: str) -> None:
        self.timeouts[self.sim.cycle] += 1
        if self.sim.in_reconcile and self.sim.grant_order is None and kind != "create":
            self.sim.phase1_timed_out = True
        self.sim.clock.advance(timedelta(seconds=30 if kind == "create" else 10))
        raise _timeout(kind)

    def _read_faults(self, kind: str) -> None:
        f, rng = self.sim.faults, self.sim.rng
        if f.read_5xx and rng.random() < f.read_5xx:
            raise _http_error(InternalServerError, 500, f"jobs {kind} failed", "GET")
        if f.read_timeout and rng.random() < f.read_timeout:
            self._time_out(kind)

    async def get(self, workspace: str, name: str) -> HelixJobResponse | None:
        self._enter("get", name)
        self._read_faults("get")
        job = self.jobs.get((workspace, name))
        return job.response() if job is not None and job.visible(self.sim.clock.now) else None

    async def has_steps(self, workspace: str, name: str) -> bool:
        self._enter("steps", name)
        self._read_faults("steps")
        job = self.jobs.get((workspace, name))
        now = self.sim.clock.now
        if job is None or not job.visible(now):
            raise _http_error(NotFoundError, 404, f"Job {name} not found", "GET")
        return job.step_at is not None and now >= job.step_at

    async def result(self, workspace: str, job_name: str, name: str) -> HelixJobResultResponse | None:
        self._enter("result", job_name)
        self._read_faults("result")
        job = self.jobs.get((workspace, job_name))
        if job is None or job.status is not HelixJobStatus.COMPLETED or not job.with_result:
            return None
        return HelixJobResultResponse(
            name=name,
            job=job.id,
            workspace=workspace,
            artifact_url=f"{workspace}/{job_name}-out#results/{name}",
            artifact_storage_type=FileStorageType.FILESET,
        )

    async def cancel(self, workspace: str, name: str) -> None:
        self._enter("cancel", name)
        self._read_faults("cancel")
        self.cancels.append((self.sim.cycle, workspace, name))
        job = self.jobs.get((workspace, name))
        if job is not None and not job.by_controller:
            self.sim.violate("foreign Job", f"the controller cancelled foreign Job {workspace}/{name}")
        if job is None or not job.visible(self.sim.clock.now) or job.status in TERMINAL_JOB:
            return  # missing or already ended: not an error (RunJobs contract)
        if job.refuse_cancel:
            # Jobs answers 409 and the Job still runs; the platform adapter re-raises it then.
            raise _http_error(ConflictError, 409, f"Job {name} can't be cancelled while its step is replaced")
        if job.status is not HelixJobStatus.CANCELLING:
            job.status = HelixJobStatus.CANCELLING
            job.cancel_ticks = self.sim.rng.randint(*self.cancel_ticks)
            job.updated_at = self.sim.clock.now
            self.sim.note(name, "cancelled by the controller -> cancelling")

    async def create(self, workspace: str, run: TasksetRunEntity, request: CreateHelixJobRequest) -> HelixJobResponse:
        """A create POST; a 5xx or timeout is ambiguous and must end its drain (grants or retries) for the cycle."""
        sim = self.sim
        assert request.name is not None
        drain = sim.post_kind.get(request.name, "bundle")
        if drain in sim.drains_ended:
            sim.violate("P1", f"{drain} drain sent {request.name} after an ambiguous create this cycle")
        try:
            return await self._create(workspace, run, request)
        except (InternalServerError, NemoTransportError):
            if drain != "bundle":  # phase 1's bundle creates stop only at a timeout (the phase-1 rule)
                sim.drains_ended.add(drain)
            raise

    async def _create(self, workspace: str, run: TasksetRunEntity, request: CreateHelixJobRequest) -> HelixJobResponse:
        sim = self.sim
        assert request.name is not None
        key = (workspace, request.name)
        sent_at = sim.clock.now
        self._enter("create", request.name)
        self.create_posts[sim.cycle] += 1
        self.posts_by_name[key] += 1
        await sim.on_create_post(workspace, run, request, sent_at)
        if (run.created_by or DEVELOPMENT_CALLER) in self.forbidden:
            sim.note(request.name, "create POST -> 403")
            raise _http_error(PermissionDeniedError, 403, f"{run.created_by} may not create Jobs in {workspace}")
        outcome = self._create_outcome(request.name)
        sim.note(request.name, f"create POST -> {outcome}" + (" (name taken: 409)" if key in self.jobs else ""))
        if outcome == "5xx_before":
            raise _http_error(InternalServerError, 503, "jobs create failed")
        if outcome == "timeout_unsaved":
            self._time_out("create")
        if key in self.jobs:
            raise _http_error(ConflictError, 409, f"Job {request.name} already exists in workspace {workspace}")
        if outcome == "409_transient":
            raise _http_error(ConflictError, 409, "secret 'hf-token' couldn't be read")
        now = sim.clock.now
        visible: datetime | None = now
        if outcome == "late_save":
            grace = sim.config.create_grace_seconds
            visible = now + timedelta(seconds=sim.rng.randint(20, max(20, min(240, grace - 60))))
        elif outcome == "lost_name":
            visible = None
        step = visible
        if visible is not None and sim.faults.stepless_first and sim.rng.random() < sim.faults.stepless_first:
            step = visible + timedelta(seconds=sim.rng.randint(10, 40))
        outcome_status, with_result = HelixJobStatus.COMPLETED, True
        roll = sim.rng.random()
        if roll < sim.faults.job_error:
            outcome_status = HelixJobStatus.ERROR
        elif roll < sim.faults.job_error + sim.faults.no_result:
            with_result = False
        is_task = request.source == TASK_JOB_SOURCE
        job = SimJob(
            workspace=workspace,
            name=request.name,
            request=request,
            creator=sim.creator_of_job(workspace, request.name, run),
            saved_at=now,
            visible_at=visible,
            step_at=step,
            run_ticks=self.duration(request.name) if is_task else 1,
            outcome=outcome_status,
            with_result=with_result,
            refuse_cancel=request.name in self.refusing
            or bool(is_task and sim.faults.refuse_cancel and sim.rng.random() < sim.faults.refuse_cancel),
            updated_at=now,
            error_details={"message": "agent crashed"} if outcome_status is HelixJobStatus.ERROR else None,
        )
        self.jobs[key] = job
        self.by_id[job.id] = job
        self.saves[key] += 1
        self.saved_in_cycle[sim.cycle] += 1
        sim.on_job_saved(job)
        if outcome in ("timeout_saved", "late_save"):
            self._time_out("create")
        if outcome in ("5xx_after", "lost_name"):
            raise _http_error(InternalServerError, 502, "jobs create failed after saving")
        if outcome == "crash_after":
            raise SimCrash("controller died right after Jobs saved a create")
        return job.response()

    def _create_outcome(self, name: str) -> str:
        if self.scripted_create.get(name):
            return self.scripted_create[name].pop(0)
        f, sim = self.sim.faults, self.sim
        odds = (
            ("5xx_before", f.create_5xx_before),
            ("409_transient", f.create_409_transient),
            ("5xx_after", f.create_5xx_after),
            ("timeout_unsaved", f.create_timeout_unsaved),
            ("timeout_saved", f.create_timeout_saved),
            ("late_save", f.create_late_save),
            ("lost_name", f.create_lost_name),
            ("crash_after", f.crash if sim.in_reconcile else 0.0),
        )
        if not any(p for _, p in odds):
            return "ok"
        roll, acc = sim.rng.random(), 0.0
        for outcome, p in odds:
            acc += p
            if roll < acc:
                return outcome
        return "ok"

    def delete(self, workspace: str, name: str) -> None:
        job = self.jobs[(workspace, name)]
        assert job.status in TERMINAL_JOB, f"Jobs refuses to delete Job {name}: it hasn't ended ({job.status})"
        del self.jobs[(workspace, name)]
        self.sim.note(name, f"deleted after it ended {job.status.value}")

    def squat(self, workspace: str, request: CreateHelixJobRequest, *, same_work: bool = False) -> None:
        """A foreign Job under a task Job's name: same source, a different spec unless ``same_work``; never the
        controller's."""
        assert request.name is not None
        now = self.sim.clock.now
        foreign = (
            request if same_work else request.model_copy(update={"spec": {**request.spec, "squatter": "user:mallory"}})
        )
        job = self.jobs[(workspace, request.name)] = SimJob(
            workspace=workspace,
            name=request.name,
            request=foreign,
            creator="user:mallory",
            saved_at=now,
            visible_at=now,
            step_at=now,
            run_ticks=self.sim.rng.randint(3, 30),
            outcome=HelixJobStatus.COMPLETED,
            by_controller=False,
            status=HelixJobStatus.ACTIVE,
            updated_at=now,
        )
        self.by_id[job.id] = job

    # ---- time ----------------------------------------------------------------------------------------------

    def tick(self) -> None:
        sim, now = self.sim, self.sim.clock.now
        for key, job in list(self.jobs.items()):
            before = job.status
            if job.status is HelixJobStatus.CREATED:
                if job.will_run and job.visible(now) and job.step_at is not None and now >= job.step_at:
                    job.status = HelixJobStatus.ACTIVE
            elif job.status is HelixJobStatus.ACTIVE:
                job.run_ticks -= 1
                if job.run_ticks <= 0:
                    job.status = job.outcome
                elif job.is_task and sim.faults.operator_cancel and sim.rng.random() < sim.faults.operator_cancel:
                    job.status, job.cancel_ticks = HelixJobStatus.CANCELLING, sim.rng.randint(*self.cancel_ticks)
            elif job.status is HelixJobStatus.CANCELLING:
                job.cancel_ticks -= 1
                if job.cancel_ticks <= 0:
                    job.status = HelixJobStatus.CANCELLED
            elif (
                job.status in TERMINAL_JOB
                and job.is_task
                and sim.faults.delete_ended
                and not sim.unresolved_intent(job)  # see test_spec_gap_deleted_intent_job_runs_twice
                and sim.rng.random() < sim.faults.delete_ended
            ):
                self.delete(*key)
                if sim.faults.replace_deleted and sim.rng.random() < sim.faults.replace_deleted:
                    self.squat(job.workspace, job.request, same_work=True)
                    sim.note(job.name, "taken by someone else's Job running the same work")
                continue
            if job.status is not before:
                job.updated_at = now
                sim.note(job.name, f"{before.value} -> {job.status.value}")

    def active_task_jobs(self) -> list[SimJob]:
        return [j for j in self.jobs.values() if j.is_task and j.by_controller and j.active and j.exempt is None]

    # ---- records (bundle path) ------------------------------------------------------------------------------

    def save_record(self, workspace: str, entity: EntityBase) -> None:
        key = (workspace, entity.name)
        if key in self.records:
            raise NemoEntityConflictError(f"record {key} exists")
        self.records[key] = entity

    def get_record(self, workspace: str, name: str) -> AgentEvalResultEntity | None:
        return cast(AgentEvalResultEntity | None, self.records.get((workspace, name)))


class ClusterRunJobs:
    """One run's :class:`RunJobs` view of the cluster; ``run=None`` reads only (stuck-Job reads)."""

    def __init__(self, cluster: FakeJobsCluster, workspace: str, run: TasksetRunEntity | None) -> None:
        self._cluster, self._workspace, self._run = cluster, workspace, run

    @property
    def evaluations(self) -> RunEvaluations:
        return cast(RunEvaluations, None)  # plans carry no Intake publication: never stamped

    async def get(self, name: str) -> HelixJobResponse | None:
        return await self._cluster.get(self._workspace, name)

    async def result(self, job: str, name: str) -> HelixJobResultResponse | None:
        return await self._cluster.result(self._workspace, job, name)

    async def create(self, request: CreateHelixJobRequest) -> HelixJobResponse:
        if self._run is None:
            raise RuntimeError("this Jobs adapter only reads")
        return await self._cluster.create(self._workspace, self._run, request)

    async def has_steps(self, name: str) -> bool:
        return await self._cluster.has_steps(self._workspace, name)

    async def read_summary(self, bundle_ref: str) -> AgentEvalSummary:
        return AgentEvalSummary()

    async def save_record(self, entity: EntityBase) -> None:
        self._cluster.save_record(self._workspace, entity)

    async def get_record(self, name: str) -> AgentEvalResultEntity | None:
        return self._cluster.get_record(self._workspace, name)

    async def cancel(self, name: str) -> None:
        await self._cluster.cancel(self._workspace, name)


class SimClock:
    def __init__(self) -> None:
        self.now = SIM_START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@dataclass
class SimRun:
    label: str
    entity: TasksetRunEntity
    plan: TasksetRunPlan
    creator: str
    tasks: int
    started_at: datetime | None = None  # the clock start the first create saw; it must never change
    unreadable: bool = False
    files_5xx: bool = False
    ever_loaded: bool = False
    cancel_from_cycle: int | None = None  # creates in this cycle or later break invariant 5
    baseline: float = 0.0  # active Jobs when the operator lowered the cap; decays as they end

    @property
    def id(self) -> str:
        return self.entity.id

    @property
    def key(self) -> tuple[str, str]:
        return (self.entity.workspace, self.entity.name)


@dataclass
class Truth:
    """Ground truth read from the Entity Store and the cluster, not from the controller."""

    open_ids: set[str]
    held: Counter[str]  # run id -> members holding a slot
    queued: Counter[str]
    held_names: set[tuple[str, str]]
    active: list[SimJob]
    unheld: list[SimJob]  # active, non-exempt controller Jobs no open-run member holds


class Sim:
    """The cluster, the clock, the store and one controller process; checks the invariants as it goes."""

    def __init__(
        self,
        *,
        entity_store: Any,
        files_client: Any,
        make_task: Callable[..., Any],
        sandbox_policy: Any,
        harbor_config: Any,
        monkeypatch: pytest.MonkeyPatch,
        config: RunControllerConfig,
        seed: int,
        faults: Faults | None,
        check_job_status: bool,
    ) -> None:
        self.rng = random.Random(seed)
        self.clock = SimClock()
        self.config = config
        self.faults = faults or Faults()
        self.check_job_status = check_job_status
        self.entity_store = entity_store
        self._make_task, self._sandbox_policy, self._harbor_config = make_task, sandbox_policy, harbor_config
        self.cluster = FakeJobsCluster(self)
        self.runs: list[SimRun] = []
        self.by_id: dict[str, SimRun] = {}
        self.by_key: dict[tuple[str, str], SimRun] = {}
        self.cycle = 0
        self.cycle_start = SIM_START
        self.in_reconcile = False
        self.violations: list[str] = []
        self.notes: list[str] = []
        self.history: list[str] = []
        self.restarts: list[tuple[int, str]] = []
        self.creator_baseline: dict[str, float] = {}
        self.files_down = False
        self.fail_listing: set[str] = set()  # run ids whose member listing raises
        self.cycle_grants: list[str] = []  # run ids, in the order phase 2 wrote create intents
        self.grant_order: list[str] | None = None  # allocate()'s output this cycle
        self.disrupted: set[str] = set()  # run ids whose phase 1 a fault disrupted this cycle
        self.crashed = False
        self.task_jobs: dict[tuple[str, str], str] = {}  # (run id, member) -> task_job ever written
        self._settled: dict[tuple[str, str], tuple[MemberPhase, int]] = {}
        self._final: dict[str, TasksetRunPhase] = {}
        self.reports: list[CycleReport | None] = []
        self.trace: list[tuple[int, str, str]] = []  # (cycle, job name, event): evidence for violations
        self.phase1_timed_out = False
        self.drains_ended: set[str] = set()
        self.retrying = False
        self.post_kind: dict[str, str] = {}  # task Job name -> "grant" or "retry", per the last intent write

        self._orig_entity_update = entity_store.update
        monkeypatch.setattr(entity_store, "update", self._entity_update)
        self.store = TasksetRunStore._with_entity_adapter(entity_store, files_client)
        self._orig_update_member = self.store.update_member
        monkeypatch.setattr(self.store, "update_member", self._update_member)
        monkeypatch.setattr(self.store, "load_plan", self._load_plan)
        monkeypatch.setattr(self.store, "list_all_members", self._list_all_members)
        monkeypatch.setattr(controller_module, "allocate", self._allocate)
        self.reconciler = self._new_reconciler()

    # ---- setup -----------------------------------------------------------------------------------------------

    def _new_reconciler(self) -> TasksetRunReconciler:
        return TasksetRunReconciler(self.store, self._jobs_for, self.config, clock=self.clock)

    @asynccontextmanager
    async def _jobs_for(self, workspace: str, run: TasksetRunEntity | None) -> AsyncGenerator[RunJobs]:
        yield ClusterRunJobs(self.cluster, workspace, run)

    async def submit(
        self,
        label: str,
        *,
        tasks: int,
        creator: str | None = "user:alice",
        workspace: str = "default",
        request_cap: int | None = None,
        timeout: timedelta = timedelta(hours=2),
        finalization_grace: timedelta = timedelta(minutes=30),
        queue_timeout: timedelta = timedelta(hours=48),
        squat: set[int] | None = None,
    ) -> SimRun:
        """Accept a run now, as submit would: the run record, its frozen plan (served by the fake Files).

        Its deadlines count ``queue_timeout`` from now until its first slot, then ``timeout`` from that slot.
        """
        name = f"run-{label}"
        spec = AgentEvalSpec(
            tasks=[self._make_task(f"{label}-t{i}") for i in range(tasks)],
            target=HarborRunnerTarget(source=HarborBuiltinAgentSource(name="oracle")),
        )
        environments = [
            PreparedHarborEnvironment(
                task_revision_digest=_hex(f"{label}-{i}-rev"),
                archive_digest=_hex(f"{label}-{i}-archive"),
                agent=PreparedImage(
                    image_ref=f"registry.invalid/sim/{label}-t{i}@sha256:{_hex(f'{label}-{i}-img')}",
                    preparation_key=_hex(f"{label}-{i}-img"),
                ),
            )
            for i in range(tasks)
        ]
        plan = TasksetRunPlan(
            run_id=name,
            workspace=workspace,
            taskset_ref=f"{workspace}/sim#1",
            spec=spec,
            max_concurrent_task_jobs=request_cap,
            sandbox_policy=self._sandbox_policy,
            dispatcher_image=self._harbor_config.dispatcher_image,
            dispatcher_profile="k8s-cpu",
            task_environments=environments,
        )
        now = self.clock.now
        self.entity_store.caller = creator
        entity = await self.entity_store.create(
            TasksetRunEntity(
                name=name,
                workspace=workspace,
                plan_ref=f"{workspace}/{name}-plan#plan.json",
                plan_digest=_hex(f"{label}-plan"),
                request_digest=_hex(f"{label}-request"),
                taskset_ref=f"{workspace}/sim#1",
                accepted_at=now,
                timeout_seconds=int(timeout.total_seconds()),
                execution_deadline_at=now + queue_timeout,
                finalization_deadline_at=now + queue_timeout + finalization_grace,
            )
        )
        sim_run = SimRun(label, entity, plan, creator or DEVELOPMENT_CALLER, tasks)
        self.runs.append(sim_run)
        self.by_id[entity.id] = sim_run
        self.by_key[sim_run.key] = sim_run
        squatted = set(squat or ())
        if self.faults.squat:
            squatted |= {i for i in range(tasks) if self.rng.random() < self.faults.squat}
        for index in sorted(squatted):
            request = await task_job_request(plan, entity, index, environments[index])
            self.cluster.squat(workspace, request)
        return sim_run

    async def cancel(self, sim_run: SimRun, *, mid_cycle: bool = False) -> None:
        """The cancel route: a versioned write of cancel_requested_at."""
        stored = await self.entity_store.get(TasksetRunEntity, sim_run.entity.name, workspace=sim_run.entity.workspace)
        if stored.phase in TERMINAL_RUN_PHASES or stored.cancel_requested_at is not None:
            return
        await self._orig_entity_update(stored.model_copy(update={"cancel_requested_at": self.clock.now}))
        sim_run.cancel_from_cycle = self.cycle + 1 if mid_cycle else self.cycle
        if mid_cycle:
            self.disrupted.add(sim_run.id)  # its writes this cycle may lose to this one

    def restart(self, reason: str, config: RunControllerConfig | None = None) -> None:
        """A new controller process over the same records and cluster: its stuck set starts empty."""
        for key in self.reconciler.stuck:
            job = self.cluster.jobs.get(key)
            if job is not None and job.active and job.exempt is None:
                job.exempt = f"stuck entry forgotten by a restart in cycle {self.cycle} ({reason})"
                logger.warning("Job %s/%s: %s", key[0], key[1], job.exempt)
                self.notes.append(f"cycle {self.cycle}: {key[1]}: {job.exempt}")
        if config is not None:
            self.config = config
        self.reconciler = self._new_reconciler()
        self.restarts.append((self.cycle, reason))

    def _exempt_interrupted_stuck(self, crash: str) -> None:
        """A crash between a forced settle's member write and its stuck entry: the restart forgets it anyway."""
        truth = self.truth()
        for job in truth.unheld:
            if (job.workspace, job.name) not in self.reconciler.stuck:
                job.exempt = f"settled in cycle {self.cycle}, whose crash ({crash}) came before its stuck entry"
                self.notes.append(f"cycle {self.cycle}: {job.name}: {job.exempt}")
                self.note(job.name, job.exempt)

    def lower_caps(self, config: RunControllerConfig) -> None:
        """The operator lowers caps and restarts the controller: Jobs above the new caps aren't cancelled."""
        active = self.cluster.active_task_jobs()
        by_run = Counter(r.id for r in map(self._run_of, active) if r is not None)
        for sim_run in self.runs:
            sim_run.baseline = by_run[sim_run.id]
        self.creator_baseline = dict(Counter(j.creator for j in active))
        self.restart("operator lowered caps", config)

    # ---- the loop --------------------------------------------------------------------------------------------

    async def step(self, before: Callable[[Sim], Awaitable[None]] | None = None) -> CycleReport | None:
        self.cycle += 1
        self.cycle_start = self.clock.now
        self.cycle_grants, self.grant_order, self.disrupted, self.crashed = [], None, set(), False
        self.phase1_timed_out, self.drains_ended, self.retrying = False, set(), False
        if before is not None:
            await before(self)
        if self.faults.restart and self.rng.random() < self.faults.restart:
            self.restart("random restart")
        self.cluster.tick()
        stuck_before = set(self.reconciler.stuck)
        report: CycleReport | None = None
        self.in_reconcile = True
        try:
            report = await self.reconciler.reconcile()
        except SimCrash as crash:
            self.crashed = True
            self.note("controller", f"crash: {crash}")
            stuck_before = set()
            self._exempt_interrupted_stuck(str(crash))
            self.restart(f"crash: {crash}")
        finally:
            self.in_reconcile = False
        self.reports.append(report)
        for key in set(self.reconciler.stuck) - stuck_before:
            self.note(key[1], f"joined the stuck set ({self.reconciler.stuck[key]})")
        for key in stuck_before - set(self.reconciler.stuck):
            self.note(key[1], "left the stuck set")
        self._check_cycle(report)
        self.clock.now = max(self.cycle_start + CYCLE, self.clock.now)
        return report

    async def run_until_final(self, max_cycles: int, before: Callable[[Sim], Awaitable[None]] | None = None) -> None:
        for _ in range(max_cycles):
            await self.step(before)
            if all(self.phase(r) in TERMINAL_RUN_PHASES for r in self.runs):
                return
        self.violate(7, f"not every run was final after {max_cycles} cycles")

    # ---- reads -----------------------------------------------------------------------------------------------

    def stored(self, sim_run: SimRun) -> TasksetRunEntity:
        return cast(TasksetRunEntity, self.entity_store.entities[(RUN_TYPE, *sim_run.key, None)])

    def phase(self, sim_run: SimRun) -> TasksetRunPhase:
        return self.stored(sim_run).phase

    def members(self, sim_run: SimRun) -> list[TasksetRunMemberEntity]:
        found = []
        for index in range(sim_run.tasks):
            member = self.entity_store.entities.get(
                (MEMBER_TYPE, sim_run.entity.workspace, f"task-{index}", sim_run.id)
            )
            if member is not None:
                found.append(cast(TasksetRunMemberEntity, member))
        return found

    def held(self, sim_run: SimRun) -> int:
        return sum(_holds(m) for m in self.members(sim_run))

    def queued(self, sim_run: SimRun) -> int:
        return sum(m.phase is MemberPhase.IMAGE_READY and m.task_job is None for m in self.members(sim_run))

    def held_by_creator(self) -> Counter[str]:
        held: Counter[str] = Counter()
        for sim_run in self.runs:
            if self.phase(sim_run) in OPEN_RUN:
                held[sim_run.creator] += self.held(sim_run)
        return held

    def _run_of(self, job: SimJob) -> SimRun | None:
        run_name, sep, _ = job.name.rpartition("-task-")
        return self.by_key.get((job.workspace, run_name)) if sep else None

    def unresolved_intent(self, job: SimJob) -> bool:
        """True while a member's create intent names ``job`` and the controller hasn't read it yet."""
        sim_run = self._run_of(job)
        index = job.name.rpartition("-task-")[2]
        member = (
            None
            if sim_run is None
            else self.entity_store.entities.get((MEMBER_TYPE, job.workspace, f"task-{index}", sim_run.id))
        )
        return member is not None and member.phase is MemberPhase.IMAGE_READY and member.task_job == job.name

    def creator_of_job(self, workspace: str, name: str, run: TasksetRunEntity) -> str:
        run_name = name.rpartition("-task-")[0] or name.rpartition("-bundle-")[0]
        sim_run = self.by_key.get((workspace, run_name))
        return sim_run.creator if sim_run is not None else (run.created_by or DEVELOPMENT_CALLER)

    def truth(self) -> Truth:
        open_ids: set[str] = set()
        held: Counter[str] = Counter()
        queued: Counter[str] = Counter()
        held_names: set[tuple[str, str]] = set()
        for sim_run in self.runs:
            is_open = self.phase(sim_run) in OPEN_RUN
            if is_open:
                open_ids.add(sim_run.id)
            for member in self.members(sim_run):
                if _holds(member):
                    held[sim_run.id] += 1
                    if is_open and member.task_job is not None:
                        held_names.add((sim_run.entity.workspace, member.task_job))
                elif member.phase is MemberPhase.IMAGE_READY:
                    queued[sim_run.id] += 1
        active = self.cluster.active_task_jobs()
        unheld = [j for j in active if (j.workspace, j.name) not in held_names]
        return Truth(open_ids, held, queued, held_names, active, unheld)

    # ---- fault seams -----------------------------------------------------------------------------------------

    def crash_roll(self) -> bool:
        return bool(self.in_reconcile and self.faults.crash and self.rng.random() < self.faults.crash)

    async def _entity_update(self, entity: EntityBase, *, original_name: str | None = None) -> EntityBase:
        if self.crash_roll():
            if self.rng.random() < 0.5:
                raise SimCrash(f"controller died before writing {entity.name}")
            await self._orig_entity_update(entity, original_name=original_name)
            raise SimCrash(f"controller died after writing {entity.name}")
        return await self._orig_entity_update(entity, original_name=original_name)

    async def _load_plan(self, run: TasksetRunEntity) -> TasksetRunPlan:
        sim_run = self.by_id[run.id]
        if self.files_down or sim_run.files_5xx:
            raise _http_error(InternalServerError, 503, "files service unavailable", "GET")
        if sim_run.unreadable:
            raise PlanDigestMismatchError(run.plan_ref, run.plan_digest, "0" * 64)
        sim_run.ever_loaded = True
        return sim_run.plan

    async def _list_all_members(self, run: TasksetRunEntity) -> list[TasksetRunMemberEntity]:
        f = self.faults
        if run.id in self.fail_listing or (f.member_list_fail and self.rng.random() < f.member_list_fail):
            self.disrupted.add(run.id)
            raise RuntimeError(f"Entity Store 503 listing the members of {run.name}")
        return [m.model_copy(deep=True) for m in self.members(self.by_id[run.id])]

    async def _update_member(self, member: TasksetRunMemberEntity, **changes: object) -> TasksetRunMemberEntity:
        grant = "task_job" in changes and member.task_job is None
        retry = "task_job" in changes and member.task_job is not None
        assert member.parent is not None
        sim_run = self.by_id[member.parent]
        if grant:
            lower = [m.name for m in self.members(sim_run) if _queued(m) and m.task_index < member.task_index]
            if lower:
                self.violate(8, f"{sim_run.label}: {member.name} granted before lower-index queued {lower[:3]}")
            if self.retrying:
                self.violate(8, f"{sim_run.label}: a grant ({member.name}) after phase 2 started retrying creates")
        if retry:
            self.retrying = True
        if grant or retry:
            self.post_kind[str(changes["task_job"])] = "grant" if grant else "retry"
        self._check_stop_table(sim_run, member, changes)
        try:
            written = await self._orig_update_member(member, **changes)
        except ValueError as exc:
            self.notes.append(f"cycle {self.cycle}: refused member write {member.name} {changes}: {exc}")
            raise
        key = (member.parent, member.name)
        if written.phase is not member.phase and written.phase in (MemberPhase.REGISTERED, MemberPhase.SUCCEEDED):
            job = self.cluster.by_id.get(written.job_id or "")
            if job is None or not job.by_controller:
                self.violate(
                    "foreign settle", f"{written.task_job} {written.phase.value} on a Job that isn't the controller's"
                )
        if written.task_job is not None:
            shown = {k: (v.value if hasattr(v, "value") else v) for k, v in changes.items() if k != "bundle_ref"}
            self.note(written.task_job, f"member {member.phase.value} -> {written.phase.value} {shown}")
            first = self.task_jobs.setdefault(key, written.task_job)
            if first != written.task_job:
                self.violate(4, f"{key} task_job changed {first} -> {written.task_job}")
        if grant:
            self.cycle_grants.append(member.parent)
            self._check_grant(self.by_id[member.parent])
        return written

    def _check_stop_table(self, sim_run: SimRun, member: TasksetRunMemberEntity, changes: dict[str, object]) -> None:
        """A registered member settled from its ended Job: the phase its stop's reason gives."""
        phase, status = changes.get("phase"), changes.get("job_status")
        if member.phase is not MemberPhase.REGISTERED or phase not in SETTLED_MEMBER_PHASES:
            return
        if not isinstance(status, HelixJobStatus) or status not in TERMINAL_JOB:
            return
        if str(changes.get("error") or "").startswith("plan_unreadable"):
            return  # the plan-free path: the stop's phase, else failed (checked by the run's outcome)
        job = self.cluster.jobs.get((sim_run.entity.workspace, member.task_job or ""))
        cancel = sim_run.cancel_from_cycle is not None and sim_run.cancel_from_cycle <= self.cycle
        expired = self.cycle_start >= self.stored(sim_run).execution_deadline_at
        if status is HelixJobStatus.COMPLETED and (job is None or job.with_result):
            expected = MemberPhase.SUCCEEDED
        elif cancel:
            expected = MemberPhase.CANCELLED
        elif expired and status is HelixJobStatus.CANCELLED:
            expected = MemberPhase.EXPIRED
        else:
            expected = MemberPhase.FAILED
        if phase is not expected:
            self.violate(
                "stop table",
                f"{member.task_job} ended {status.value} (cancel={cancel}, past deadline={expired}): "
                f"settled {phase}, the table gives {expected.value}",
            )

    def _check_outcome(self, sim_run: SimRun, stored: TasksetRunEntity, members: list[TasksetRunMemberEntity]) -> None:
        """A finished run's phase follows its members: any cancelled, else any expired, else bundled or failed."""
        if "plan_unreadable" in (stored.error or ""):
            return
        phases = {m.phase for m in members}
        if MemberPhase.CANCELLED in phases:
            expected = TasksetRunPhase.CANCELLED
        elif MemberPhase.EXPIRED in phases:
            expected = TasksetRunPhase.EXPIRED
        elif MemberPhase.SUCCEEDED in phases:
            expected = TasksetRunPhase.COMPLETED
        else:
            expected = TasksetRunPhase.FAILED
        if stored.phase is not expected:
            self.violate("run outcome", f"{sim_run.label} ended {stored.phase}; its members give {expected.value}")
        ready = stored.result_state in (ResultState.READY, ResultState.FAILED)
        if ready != (stored.phase is TasksetRunPhase.COMPLETED):
            self.violate("run outcome", f"{sim_run.label} ended {stored.phase} with result_state {stored.result_state}")

    def _allocate(
        self, tallies: list[RunTally], available_slots: int, creator_cap: int | None, stuck_by_creator: Counter[str]
    ) -> list[str]:
        order = allocate(tallies, available_slots, creator_cap, stuck_by_creator)
        self.grant_order = list(order)
        self._check_allocation(tallies, available_slots, creator_cap, stuck_by_creator, order)
        return order

    # ---- invariants ------------------------------------------------------------------------------------------

    def note(self, name: str, event: str) -> None:
        seconds = (self.clock.now - SIM_START).total_seconds()
        self.trace.append((self.cycle, name, f"c{self.cycle} t+{seconds:.0f}s {name}: {event}"))

    def violate(self, invariant: int | str, message: str) -> None:
        seconds = (self.clock.now - SIM_START).total_seconds()
        self.violations.append(f"[cycle {self.cycle} t+{seconds:.0f}s] invariant {invariant}: {message}")

    async def on_create_post(
        self, workspace: str, run: TasksetRunEntity, request: CreateHelixJobRequest, sent_at: datetime
    ) -> None:
        """Invariants 2 and 5 on every create POST that reaches Jobs."""
        assert request.name is not None
        if request.source not in (TASK_JOB_SOURCE, BUNDLE_JOB_SOURCE):
            self.violate(2, f"create of {request.name} with source {request.source}: only task and bundle Jobs")
            return
        run_name = request.name.rpartition("-task-" if request.source == TASK_JOB_SOURCE else "-bundle-")[0]
        sim_run = self.by_key.get((workspace, run_name))
        if sim_run is None:
            self.violate(5, f"create of {request.name} for no known run")
            return
        if (run.created_by or DEVELOPMENT_CALLER) != sim_run.creator:
            self.violate("auth", f"{request.name} created for {run.created_by}, not the run's creator")
        stored = self.stored(sim_run)
        if stored.phase not in OPEN_RUN:
            self.violate(5, f"create of {request.name} for finished run {sim_run.label} ({stored.phase})")
        if request.source == BUNDLE_JOB_SOURCE:
            if any(m.phase not in SETTLED_MEMBER_PHASES for m in self.members(sim_run)):
                self.violate(6, f"bundle Job {request.name} created while a member is unsettled")
            return
        index = int(request.name.rpartition("-task-")[2])
        expected = sim_run.plan.task_environments[index].model_dump(mode="json")
        if request.spec.get("environment") != expected or request.spec.get("task_index") != index:
            self.violate(2, f"{request.name} isn't created with plan.task_environments[{index}]")
        if sent_at >= stored.execution_deadline_at:
            self.violate(5, f"{request.name} created after execution_deadline_at")
        # The clock: a run starts once, before its first create, and every Job carries the run's deadline.
        if stored.started_at is None:
            self.violate("clock", f"{request.name} created before run {sim_run.label} started its clock")
        elif sim_run.started_at not in (None, stored.started_at):
            self.violate("clock", f"{sim_run.label} started again: {sim_run.started_at}, then {stored.started_at}")
        sim_run.started_at = stored.started_at
        deadline = datetime.fromisoformat(request.spec["execution_deadline_at"])
        if deadline.timestamp() != stored.execution_deadline_at.timestamp():
            self.violate(
                "clock", f"{request.name} carries {deadline}, its run's deadline is {stored.execution_deadline_at}"
            )
        if sim_run.cancel_from_cycle is not None and self.cycle >= sim_run.cancel_from_cycle:
            self.violate(5, f"{request.name} created in a cycle that started after cancel_requested_at")
        member = self.entity_store.entities.get((MEMBER_TYPE, workspace, f"task-{index}", sim_run.id))
        if member is None or member.phase is not MemberPhase.IMAGE_READY or member.task_job != request.name:
            phase = getattr(member, "phase", None)
            self.violate(5, f"{request.name} created for a member that isn't a create intent ({phase})")
        if self.faults.mid_cycle_cancel and self.rng.random() < self.faults.mid_cycle_cancel:
            await self.cancel(sim_run, mid_cycle=True)

    def on_job_saved(self, job: SimJob) -> None:
        if not job.is_task:
            return
        if self.cluster.saves[(job.workspace, job.name)] > 1:
            self.violate(4, f"a second Job saved under {job.name}: its member got two Jobs")
        self._check_caps(f"Job {job.name} saved")

    def _ceiling(self, cap: int | None, baseline: float) -> float:
        return max(math.inf if cap is None else cap, baseline)

    def _check_caps(self, where: str) -> None:
        """Invariants 1, 3 and 9 on the cluster's truth: active (non-exempt) task Jobs of the controller."""
        active = self.cluster.active_task_jobs()
        self.cluster.peak_active = max(self.cluster.peak_active, len(active))
        if len(active) > self.config.max_cluster_concurrent_jobs:
            self.violate(1, f"{where}: {len(active)} active task Jobs > {self.config.max_cluster_concurrent_jobs}")
        by_run: Counter[str] = Counter()
        for job in active:
            sim_run = self._run_of(job)
            if sim_run is not None:
                by_run[sim_run.id] += 1
        for sim_run in self.runs:
            count = by_run[sim_run.id]
            sim_run.baseline = min(sim_run.baseline, count)
            ceiling = self._ceiling(run_cap(sim_run.plan, self.config), sim_run.baseline)
            if count > ceiling:
                self.violate(3, f"{where}: run {sim_run.label} has {count} active task Jobs > its cap {ceiling}")
        by_creator = Counter(j.creator for j in active)
        for creator in {*by_creator, *self.creator_baseline}:
            count = by_creator[creator]
            baseline = min(self.creator_baseline.get(creator, 0), count)
            self.creator_baseline[creator] = baseline
            ceiling = self._ceiling(self.config.max_concurrent_task_jobs_per_creator, baseline)
            if count > ceiling:
                self.violate(9, f"{where}: creator {creator} has {count} active task Jobs > {ceiling}")

    def _check_grant(self, sim_run: SimRun) -> None:
        """A slot was just taken (a create intent written): invariants 1, 3 and 9 by the slots held."""
        truth = self.truth()
        used = sum(truth.held[i] for i in truth.open_ids) + len(truth.unheld)
        if used > self.config.max_cluster_concurrent_jobs:
            self.violate(1, f"grant to {sim_run.label}: {used} slots used > {self.config.max_cluster_concurrent_jobs}")
        cap = run_cap(sim_run.plan, self.config)
        if cap is not None and truth.held[sim_run.id] > cap:
            self.violate(3, f"grant to {sim_run.label}: it holds {truth.held[sim_run.id]} > its cap {cap}")
        creator_cap = self.config.max_concurrent_task_jobs_per_creator
        if creator_cap is not None:
            mine = sum(truth.held[r.id] for r in self.runs if r.creator == sim_run.creator and r.id in truth.open_ids)
            mine += sum(j.creator == sim_run.creator for j in truth.unheld)
            if mine > creator_cap:
                self.violate(9, f"grant to {sim_run.label}: creator {sim_run.creator} holds {mine} > {creator_cap}")

    def _check_allocation(
        self,
        tallies: list[RunTally],
        available: int,
        creator_cap: int | None,
        stuck: Counter[str],
        order: list[str],
    ) -> None:
        """Invariant 8 (and the slot count behind 1 and 9) on allocate()'s inputs and output."""
        truth = self.truth()
        created = [t.run.created_at for t in tallies]
        if created != sorted(c for c in created if c is not None):
            self.violate(8, "tallies aren't oldest run first")
        missing = truth.open_ids - {t.run.id for t in tallies}
        if missing:
            self.violate(8, f"open runs missing from phase 2: {[self.by_id[i].label for i in missing]}")
        for t in tallies:
            sim_run = self.by_id[t.run.id]
            if t.held != truth.held[t.run.id]:
                self.violate(8, f"{sim_run.label} tallied holding {t.held}, holds {truth.held[t.run.id]}")
            if t.creator != sim_run.creator:
                self.violate(8, f"{sim_run.label} tallied for {t.creator}, created by {sim_run.creator}")
            if not t.queued:
                continue
            stored = self.stored(sim_run)
            stopping = (
                sim_run.cancel_from_cycle is not None and sim_run.cancel_from_cycle <= self.cycle
            ) or self.cycle_start >= stored.execution_deadline_at
            if stopping:
                self.violate(5, f"stopping run {sim_run.label} tallied with {t.queued} queued tasks")
            if t.queued != truth.queued[t.run.id]:
                self.violate(8, f"{sim_run.label} tallied {t.queued} queued, has {truth.queued[t.run.id]}")
            if t.cap != run_cap(sim_run.plan, self.config):
                self.violate(
                    3, f"{sim_run.label} tallied with cap {t.cap}, run_cap is {run_cap(sim_run.plan, self.config)}"
                )
        used = sum(t.held for t in tallies) + stuck.total()
        if available != max(0, self.config.max_cluster_concurrent_jobs - used):
            self.violate(1, f"free slots {available} != cap - used ({used})")
        unheld_by_creator = Counter(j.creator for j in truth.unheld)
        for creator, count in unheld_by_creator.items():
            if stuck[creator] < count:
                self.violate(
                    1, f"{count} running Jobs of {creator} held by no open run, {stuck[creator]} counted stuck"
                )
        # Replay the grant order against the rule: least-loaded creator, then its oldest eligible run.
        grants: Counter[str] = Counter()
        run_held = {t.run.id: t.held for t in tallies}
        creator_held: Counter[str] = Counter(stuck)
        for t in tallies:
            creator_held[t.creator] += t.held
        position = {t.run.id: i for i, t in enumerate(tallies)}

        def eligible() -> list[RunTally]:
            return [
                t
                for t in tallies
                if grants[t.run.id] < t.queued
                and (t.cap is None or run_held[t.run.id] < t.cap)
                and (creator_cap is None or creator_held[t.creator] < creator_cap)
            ]

        if len(order) > available:
            self.violate(1, f"allocate() handed out {len(order)} slots of {available}")
        by_id = {t.run.id: t for t in tallies}
        for run_id in order:
            candidates = eligible()
            chosen = by_id[run_id]
            label = self.by_id[run_id].label
            if chosen not in candidates:
                self.violate(8, f"slot to ineligible run {label}")
            elif any(creator_held[t.creator] < creator_held[chosen.creator] for t in candidates):
                self.violate(8, f"slot to {label} while a creator holding fewer had an eligible run")
            elif any(t.creator == chosen.creator and position[t.run.id] < position[run_id] for t in candidates):
                self.violate(8, f"slot to {label} while an older run of {chosen.creator} was eligible")
            grants[run_id] += 1
            run_held[run_id] += 1
            creator_held[chosen.creator] += 1
        if len(order) < available and eligible():
            self.violate(8, f"{available - len(order)} slots left idle while {eligible()[0].run.name} could take one")

    def _check_cycle(self, report: CycleReport | None) -> None:
        truth = self.truth()
        self._check_caps("after the cycle")
        for job in truth.unheld:
            if (job.workspace, job.name) not in self.reconciler.stuck:
                self.violate(1, f"Job {job.name} runs with no open run's member holding it and isn't counted stuck")
        disrupted_cycle = self.crashed or report is None or report.phase2_skipped == "open runs couldn't be listed"
        for sim_run in self.runs:
            stored = self.stored(sim_run)
            members = self.members(sim_run)
            if len(members) > sim_run.tasks:
                self.violate(4, f"{sim_run.label} has {len(members)} members for {sim_run.tasks} tasks")
            previous = self._final.get(sim_run.id)
            if previous is not None and stored.phase is not previous:
                self.violate(6, f"finished run {sim_run.label} changed {previous} -> {stored.phase}")
            if stored.phase in TERMINAL_RUN_PHASES:
                if sim_run.id not in self._final:
                    self._check_outcome(sim_run, stored, members)
                self._final.setdefault(sim_run.id, stored.phase)
                unsettled = [m.name for m in members if m.phase not in SETTLED_MEMBER_PHASES]
                if unsettled:
                    self.violate(6, f"finished run {sim_run.label} ({stored.phase}) has unsettled {unsettled[:5]}")
            elif (
                self.cycle_start >= stored.finalization_deadline_at
                and not disrupted_cycle
                and sim_run.id not in self.disrupted
            ):
                self.violate(7, f"{sim_run.label} still {stored.phase} in a cycle after its finalization deadline")
            for member in members:
                key = (sim_run.id, member.name)
                before = self._settled.get(key)
                if before is not None and (member.phase, member.db_version) != before:
                    self.violate(6, f"settled {sim_run.label}/{member.name} changed {before} -> {member.phase}")
                if member.phase in SETTLED_MEMBER_PHASES:
                    self._settled.setdefault(key, (member.phase, member.db_version))
                elif self.check_job_status and member.phase is MemberPhase.REGISTERED:
                    job = self.cluster.jobs.get((sim_run.entity.workspace, member.task_job or ""))
                    if job is not None and member.job_status != job.status:
                        self.violate("job_status", f"{member.task_job} shows {member.job_status}, Job is {job.status}")
        self._check_grant_order()
        self._record(report, truth)

    def _check_grant_order(self) -> None:
        """Invariant 8, second half: the create intents follow allocate()'s order, minus runs skipped mid-grant."""
        order = self.grant_order or []
        position, skipped = 0, set()
        for run_id in self.cycle_grants:
            while position < len(order) and order[position] != run_id:
                skipped.add(order[position])
                position += 1
            if position == len(order):
                self.violate(8, f"a create intent for {self.by_id[run_id].label} outside allocate()'s order")
                return
            if run_id in skipped:
                self.violate(8, f"{self.by_id[run_id].label} granted after one of its grants was skipped")
            position += 1

    def _record(self, report: CycleReport | None, truth: Truth) -> None:
        runs = " ".join(
            f"{r.label}:{self.phase(r).value[:4]}/h{truth.held[r.id]}q{truth.queued[r.id]}" for r in self.runs
        )
        summary = (
            "crashed"
            if report is None
            else f"used={report.used} grants={len(report.grants)} created={report.created} "
            f"timeouts={report.jobs_timeouts} skip={report.phase2_skipped}"
        )
        seconds = (self.cycle_start - SIM_START).total_seconds()
        self.history.append(
            f"c{self.cycle} t+{seconds:.0f}s active={len(truth.active)} stuck={len(self.reconciler.stuck)} "
            f"{summary} | {runs}"
        )

    def assert_clean(self) -> None:
        if not self.violations:
            return
        first = int(self.violations[0].split()[1])
        evidence = "\n".join(self.history[max(0, first - 4) : first + 2])
        details = "\n".join(self.violations[:15])
        names = {word.rstrip(":,") for word in self.violations[0].split() if "-task-" in word or "-bundle-" in word}
        events = "\n".join(line for _, name, line in self.trace if name in names or name == "controller")
        restarts = ", ".join(f"c{c}: {why}" for c, why in self.restarts)
        pytest.fail(
            f"{len(self.violations)} invariant violations:\n{details}\n--- cycles ---\n{evidence}\n"
            f"--- events ---\n{events}\n--- restarts ---\n{restarts}"
        )


def _queued(member: TasksetRunMemberEntity) -> bool:
    return member.phase is MemberPhase.IMAGE_READY and member.task_job is None


def _holds(member: TasksetRunMemberEntity) -> bool:
    return member.phase is MemberPhase.REGISTERED or (
        member.phase is MemberPhase.IMAGE_READY and member.task_job is not None
    )


def _hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


MakeSim = Callable[..., Sim]


@pytest.fixture
def make_sim(entity_store, files_client, harbor_resolved_task, sandbox_policy, harbor_config, monkeypatch) -> MakeSim:
    def make(
        config: RunControllerConfig,
        *,
        seed: int = 0,
        faults: Faults | None = None,
        check_job_status: bool = False,
    ) -> Sim:
        return Sim(
            entity_store=entity_store,
            files_client=files_client,
            make_task=harbor_resolved_task,
            sandbox_policy=sandbox_policy,
            harbor_config=harbor_config,
            monkeypatch=monkeypatch,
            config=config,
            seed=seed,
            faults=faults,
            check_job_status=check_job_status,
        )

    return make


def caps(cluster: int, per_creator: int | None = None, per_run: int | None = None, **extra: Any) -> RunControllerConfig:
    return RunControllerConfig(
        max_cluster_concurrent_jobs=cluster,
        max_concurrent_task_jobs_per_creator=per_creator,
        default_max_concurrent_task_jobs=per_run,
        **extra,
    )


def fixed(ticks: dict[str, int], default: int) -> Callable[[str], int]:
    """Job durations by name suffix (``-task-3``), else ``default``."""

    def duration(name: str) -> int:
        return ticks.get("task-" + name.rpartition("-task-")[2], default)

    return duration


# ---- scenarios ---------------------------------------------------------------------------------------------


async def test_two_run_example(make_sim: MakeSim) -> None:
    """Cap 50, no other caps: A starts 45 Jobs, B submits 50; B gets 5, then each slot A frees."""
    sim = make_sim(caps(50), check_job_status=True)
    a = await sim.submit("a", tasks=45, creator="user:a")
    sim.cluster.duration = fixed({f"task-{i}": 1 for i in range(10)}, default=6)
    await sim.step()
    assert (sim.held(a), sim.queued(a)) == (45, 0)
    b = await sim.submit("b", tasks=50, creator="user:b")
    await sim.step()
    assert (sim.held(a), sim.held(b), sim.queued(b)) == (45, 5, 45)
    await sim.step()  # 10 of A's Jobs finished
    assert (sim.held(a), sim.held(b), sim.queued(b)) == (35, 15, 35)
    while sim.queued(b) and sim.cycle < 60:
        await sim.step()
        assert sim.held(a) + sim.held(b) == 50 or not sim.queued(b), sim.history[-1]
    await sim.run_until_final(60)
    assert sim.phase(a) is sim.phase(b) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def test_ac9_replay(make_sim: MakeSim) -> None:
    """3 workspaces, A runs 1-5 then B runs 6-10, 113 tasks each, caps 50/30/20: 25/25, every run completes."""
    sim = make_sim(caps(50, per_creator=30, per_run=20), seed=9, check_job_status=True)
    sim.cluster.duration = lambda name: sim.rng.randint(3, 5)
    runs = [await sim.submit(f"ac9-{i}", tasks=113, creator="user:alice", workspace=f"ws{i % 3}") for i in range(1, 6)]
    await sim.step()  # A alone
    assert [sim.held(r) for r in runs] == [20, 10, 0, 0, 0]
    runs += [await sim.submit(f"ac9-{i}", tasks=113, creator="user:bob", workspace=f"ws{i % 3}") for i in range(6, 11)]
    await sim.step()  # B submits: it holds the fewest, so it gets the 20 free slots; run 6 is at its cap
    assert [sim.held(r) for r in runs] == [20, 10, 0, 0, 0, 20, 0, 0, 0, 0]

    balanced = False
    while sim.cycle < 400:
        await sim.step()
        held = sim.held_by_creator()
        capacity = {
            creator: min(
                30,
                sum(
                    min(20, sim.held(r) + sim.queued(r))
                    for r in runs
                    if r.creator == creator and sim.phase(r) in OPEN_RUN
                ),
            )
            for creator in ("user:alice", "user:bob")
        }
        if held["user:alice"] == held["user:bob"] == 25:
            balanced = True
        elif balanced and min(capacity.values()) >= 25:
            sim.violate("full use", f"25/25 lost while both creators have work: {dict(held)}, capacity {capacity}")
        for r in runs:
            members = sim.members(r)
            assert len(members) == 113 or sim.cycle == 1
        if all(sim.phase(r) in TERMINAL_RUN_PHASES for r in runs):
            break
    assert balanced, "the creators never reached 25/25"
    assert [sim.phase(r) for r in runs] == [TasksetRunPhase.COMPLETED] * 10
    assert all(sim.stored(r).result_state is ResultState.READY for r in runs)
    assert sim.cluster.peak_active == 50
    sim.assert_clean()


async def test_request_cap_vs_operator_cap(make_sim: MakeSim) -> None:
    """A request for 40 under an operator cap of 20 gets 20; a request for 5 gets 5."""
    sim = make_sim(caps(50, per_run=20))
    sim.cluster.duration = lambda name: 4
    big = await sim.submit("big", tasks=50, request_cap=40)
    small = await sim.submit("small", tasks=20, request_cap=5)
    peak = Counter[str]()

    async def track(s: Sim) -> None:
        peak["big"] = max(peak["big"], s.held(big))
        peak["small"] = max(peak["small"], s.held(small))

    await sim.step()
    assert (sim.held(big), sim.held(small)) == (20, 5)
    await sim.run_until_final(80, before=track)
    assert dict(peak) == {"big": 20, "small": 5}
    assert sim.phase(big) is sim.phase(small) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def test_operator_lowers_per_run_cap_then_restarts(make_sim: MakeSim) -> None:
    """Per-run cap 20 -> 5 and a restart: nothing is cancelled; no grant until the run holds fewer than 5."""
    sim = make_sim(caps(50, per_run=20))
    sim.cluster.duration = lambda name: sim.rng.randint(3, 9)
    run = await sim.submit("lowered", tasks=40)
    await sim.step()
    await sim.step()
    assert sim.held(run) == 20
    sim.lower_caps(caps(50, per_run=5))
    while sim.phase(run) in OPEN_RUN and sim.cycle < 120:
        report = await sim.step()  # the grant-time check sees each slot taken against the new cap
        if report is not None and run.entity.name in report.grants:
            assert sim.held(run) <= 5, sim.history[-1]
    assert sim.cluster.cancels == []
    assert sim.phase(run) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def test_operator_lowers_per_creator_cap_then_restarts(make_sim: MakeSim) -> None:
    """Per-creator cap 30 -> 10 and a restart: nothing is cancelled; that creator waits until it is below 10."""
    sim = make_sim(caps(50, per_creator=30, per_run=20))
    sim.cluster.duration = lambda name: sim.rng.randint(3, 9)
    runs = [await sim.submit(f"pc{i}", tasks=30, creator="user:alice") for i in range(3)]
    await sim.step()
    assert sim.held_by_creator()["user:alice"] == 30
    sim.lower_caps(caps(50, per_creator=10, per_run=20))
    await sim.run_until_final(200)
    assert sim.cluster.cancels == []
    assert all(sim.phase(r) is TasksetRunPhase.COMPLETED for r in runs)
    sim.assert_clean()


async def test_same_creator_older_run_fills_first(make_sim: MakeSim) -> None:
    """An older run at 10 of 20 (its plan was unavailable): it fills to 20 before the younger run gets a slot."""
    sim = make_sim(caps(30, per_run=20))
    sim.cluster.duration = lambda name: sim.rng.randint(2, 6)
    older = await sim.submit("older", tasks=40)
    younger = await sim.submit("younger", tasks=40)
    older.files_5xx = True
    await sim.step()
    await sim.step()
    assert (sim.held(older), sim.held(younger)) == (0, 20)
    older.files_5xx = False
    await sim.step()
    assert sim.held(older) == 10
    while sim.phase(older) in OPEN_RUN and sim.cycle < 100:
        report = await sim.step()
        if report is not None and younger.entity.name in report.grants:
            assert sim.held(older) == 20 or sim.queued(older) == 0, sim.history[-1]
    await sim.run_until_final(100)
    sim.assert_clean()


async def test_two_creators_converge_to_25_25(make_sim: MakeSim) -> None:
    """A holds 30, B arrives with 2 runs: B gets 20 at once, then each freed slot goes to the creator holding fewer."""
    sim = make_sim(caps(50, per_creator=30, per_run=20))
    sim.cluster.duration = lambda name: sim.rng.randint(3, 7)
    a_runs = [await sim.submit(f"a{i}", tasks=40, creator="user:a") for i in range(2)]
    await sim.step()
    assert [sim.held(r) for r in a_runs] == [20, 10]
    b_runs = [await sim.submit(f"b{i}", tasks=40, creator="user:b") for i in range(2)]
    await sim.step()
    assert [sim.held(r) for r in b_runs] == [20, 0]
    reached = False
    while sim.cycle < 40:
        await sim.step()
        held = sim.held_by_creator()
        if held["user:a"] == held["user:b"] == 25:
            reached = True
        elif reached and all(sim.queued(r) for r in a_runs + b_runs):
            sim.violate("fairness", f"left 25/25 while every run has queued tasks: {dict(held)}")
    assert reached, sim.history[-5:]
    await sim.run_until_final(200)
    sim.assert_clean()


async def _overloaded(sim: Sim, queue_timeout: timedelta = timedelta(hours=48)) -> list[SimRun]:
    """Six 10-task runs from 2 creators, 3 per creator, on a cluster of 10 slots: far more than one budget's work."""
    return [
        await sim.submit(
            f"{creator[-1]}{i}",
            tasks=10,
            creator=creator,
            timeout=timedelta(seconds=300),
            finalization_grace=timedelta(seconds=120),
            queue_timeout=queue_timeout,
        )
        for i in range(3)
        for creator in ("user:a", "user:b")
    ]


async def test_overloaded_runs_wait_then_finish_within_their_budget(make_sim: MakeSim) -> None:
    """Work above capacity: runs queue, and each one's 300 s budget counts from its first slot, so none expires."""
    sim = make_sim(caps(10, per_run=5))
    sim.cluster.duration = lambda name: 3
    runs = await _overloaded(sim)
    await sim.run_until_final(120)
    for run in runs:
        stored = sim.stored(run)
        assert stored.phase is TasksetRunPhase.COMPLETED, (run.label, stored.error)
        assert stored.started_at is not None and stored.started_at >= stored.accepted_at
        assert stored.execution_deadline_at == stored.started_at + timedelta(seconds=300)
    waited = [stored.started_at - stored.accepted_at for stored in map(sim.stored, runs) if stored.started_at]
    # The youngest runs queued 240 s: counted from acceptance, that left 60 s of a 300 s budget and they expired.
    assert max(waited) >= timedelta(seconds=240), waited
    sim.assert_clean()


async def test_overloaded_youngest_runs_expire_at_the_queue_timeout(make_sim: MakeSim) -> None:
    """The same work with a 120 s queue timeout: each creator's youngest runs never start and expire, oldest first
    keeps completing, and no creator loses all its runs."""
    sim = make_sim(caps(10, per_run=5))
    sim.cluster.duration = lambda name: 3
    runs = await _overloaded(sim, queue_timeout=timedelta(seconds=120))
    await sim.run_until_final(60)
    for creator in ("user:a", "user:b"):
        mine = [sim.stored(r) for r in runs if r.creator == creator]  # oldest first
        phases = [r.phase for r in mine]
        assert TasksetRunPhase.COMPLETED in phases and TasksetRunPhase.EXPIRED in phases, phases
        first_expired = phases.index(TasksetRunPhase.EXPIRED)
        assert all(p is TasksetRunPhase.EXPIRED for p in phases[first_expired:]), phases
        for expired in mine[first_expired:]:
            assert expired.started_at is None
            assert expired.error == "no task started within the queue timeout (120 s)"
    sim.assert_clean()


async def test_auth_off_single_creator_is_fifo(make_sim: MakeSim) -> None:
    """Auth off: no created_by, so every run is user:development; grants are first in, first out."""
    sim = make_sim(caps(50, per_run=20))
    sim.cluster.duration = lambda name: 4
    runs = [await sim.submit(f"dev{i}", tasks=30, creator=None) for i in range(3)]
    await sim.step()
    assert [sim.held(r) for r in runs] == [20, 20, 10]
    await sim.run_until_final(100)
    assert {j.creator for j in sim.cluster.jobs.values()} == {DEVELOPMENT_CALLER}
    assert all(sim.phase(r) is TasksetRunPhase.COMPLETED for r in runs)
    sim.assert_clean()


async def test_create_timeout_saved_holds_the_slot(make_sim: MakeSim) -> None:
    """Cap 1: a create times out but Jobs saved it; next cycle the older run has work, yet the slot stays held."""
    sim = make_sim(caps(1))
    sim.cluster.duration = lambda name: 3
    older = await sim.submit("older", tasks=2)
    younger = await sim.submit("younger", tasks=2)
    older.files_5xx = True
    sim.cluster.scripted_create["run-younger-task-0"] = ["timeout_saved"]
    await sim.step()
    assert sim.held(younger) == 1 and sim.members(younger)[0].phase is MemberPhase.IMAGE_READY
    older.files_5xx = False
    await sim.step()
    assert sim.members(younger)[0].phase is MemberPhase.REGISTERED  # adopted, not created again
    assert sim.held(older) == 0
    assert sim.cluster.posts_by_name[("default", "run-younger-task-0")] == 1
    await sim.run_until_final(60)
    sim.assert_clean()


async def test_create_timeout_unsaved_is_retried_without_a_new_slot(make_sim: MakeSim) -> None:
    """A create times out unsaved: its intent keeps the slot; phase 2 retries it after its new grants."""
    sim = make_sim(caps(2))
    sim.cluster.duration = lambda name: 5
    run = await sim.submit("retry", tasks=3)
    sim.cluster.scripted_create["run-retry-task-0"] = ["timeout_unsaved"]
    report = await sim.step()
    assert report is not None and len(report.grants) == 2
    # The grant drain ended at the ambiguous create: task-1 stays queued, without an intent.
    phases = [(m.phase, m.task_job is not None) for m in sim.members(run)]
    assert phases == [
        (MemberPhase.IMAGE_READY, True),
        (MemberPhase.IMAGE_READY, False),
        (MemberPhase.IMAGE_READY, False),
    ]
    report = await sim.step()
    assert report is not None and len(report.grants) == 1
    assert [m.phase for m in sim.members(run)] == [
        MemberPhase.REGISTERED,
        MemberPhase.REGISTERED,
        MemberPhase.IMAGE_READY,
    ]
    assert sim.cluster.posts_by_name[("default", "run-retry-task-0")] == 2
    await sim.run_until_final(60)
    assert sim.phase(run) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def _stop_scenario(make_sim: MakeSim, *, cancel: bool) -> None:
    sim = make_sim(caps(20, per_run=20))
    sim.cluster.duration = lambda name: 100 if name.startswith("run-stop-") else 2
    sim.cluster.cancel_ticks = (2, 2)
    run = await sim.submit("stop", tasks=80, timeout=timedelta(hours=1) if cancel else timedelta(seconds=45))
    await sim.step()
    waiting = await sim.submit("waiting", tasks=5, creator="user:bob")
    await sim.step()
    if cancel:
        await sim.step()
        await sim.cancel(run)
    else:
        await sim.step()
    assert (sim.held(run), sim.queued(run)) == (20, 60)
    await sim.step()  # cycle 4: the deadline (t+45 s) or the cancel
    stopped = MemberPhase.CANCELLED if cancel else MemberPhase.EXPIRED
    assert sum(m.phase is stopped for m in sim.members(run)) == 60
    assert sim.held(run) == 20 and sim.held(waiting) == 0
    assert len(sim.cluster.cancels) == 20
    await sim.step()  # cancelling
    assert sim.held(run) == 20 and sim.held(waiting) == 0
    await sim.step()  # the cancelled Jobs ended: members settle, the run ends, the slots go to the next run
    assert sim.phase(run) is (TasksetRunPhase.CANCELLED if cancel else TasksetRunPhase.EXPIRED)
    assert [m.phase for m in sim.members(run)] == [stopped] * 80
    assert sim.held(waiting) == 5
    assert len(sim.cluster.cancels) == 20  # a Job already cancelling isn't cancelled again
    await sim.run_until_final(30)
    sim.assert_clean()


async def test_deadline_with_queued_and_registered(make_sim: MakeSim) -> None:
    """60 queued and 20 registered at the deadline: 20 cancels, 60 expired at once, 20 as their Jobs end."""
    await _stop_scenario(make_sim, cancel=False)


async def test_cancel_mid_run(make_sim: MakeSim) -> None:
    """cancel_requested_at mid-run: the same as the deadline, with phase cancelled."""
    await _stop_scenario(make_sim, cancel=True)


async def test_jobs_5xx_for_five_cycles(make_sim: MakeSim) -> None:
    """Jobs answers 5xx for 5 cycles: no new Job, nothing settled, at most one create a cycle; then it recovers."""
    sim = make_sim(caps(15, per_run=10))
    sim.cluster.duration = lambda name: sim.rng.randint(3, 5)
    first = await sim.submit("first", tasks=30)
    outage = range(4, 9)
    second: list[SimRun] = []

    async def before(s: Sim) -> None:
        s.cluster.outage = "5xx" if s.cycle in outage else None
        if s.cycle == 5:
            second.append(await s.submit("second", tasks=10, creator="user:bob"))

    settled_before = 0
    for _ in range(9):
        await sim.step(before)
        settled = sum(m.phase in SETTLED_MEMBER_PHASES for r in sim.runs for m in sim.members(r))
        if sim.cycle in outage:
            assert settled == settled_before, sim.history[-1]
            assert sim.cluster.saved_in_cycle[sim.cycle] == 0
            assert sim.cluster.create_posts[sim.cycle] <= 1
        settled_before = settled
    await sim.run_until_final(100, before)
    assert sim.phase(first) is sim.phase(second[0]) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def test_jobs_hang_for_five_cycles(make_sim: MakeSim) -> None:
    """Jobs hangs for 5 cycles: each cycle makes at most three Jobs calls (a read, then one create per drain)."""
    sim = make_sim(caps(15, per_run=10))
    sim.cluster.duration = lambda name: sim.rng.randint(3, 5)
    await sim.submit("first", tasks=30)
    outage = range(4, 9)

    async def before(s: Sim) -> None:
        s.cluster.outage = "hang" if s.cycle in outage else None
        if s.cycle == 5:
            await s.submit("second", tasks=10, creator="user:bob")

    settled_before = 0
    for _ in range(9):
        report = await sim.step(before)
        settled = sum(m.phase in SETTLED_MEMBER_PHASES for r in sim.runs for m in sim.members(r))
        if sim.cycle in outage:
            assert settled == settled_before, sim.history[-1]
            assert 1 <= sim.cluster.calls[sim.cycle] <= 3, sim.history[-1]
            assert sim.cluster.timeouts[sim.cycle] == sim.cluster.calls[sim.cycle]
            assert report is not None and report.created == 0
        settled_before = settled
    await sim.run_until_final(100, before)
    assert all(sim.phase(r) is TasksetRunPhase.COMPLETED for r in sim.runs), sim.history[-1]
    sim.assert_clean()


async def test_plan_unreadable_with_20_jobs_running(make_sim: MakeSim) -> None:
    """The plan breaks with 20 Jobs running: 20 cancels, slots held until they end, then the run fails."""
    sim = make_sim(caps(25, per_creator=25, per_run=20))
    sim.cluster.duration = lambda name: 50 if "-u-" in name else 3
    sim.cluster.cancel_ticks = (2, 2)
    run = await sim.submit("u", tasks=40)
    await sim.step()
    other = await sim.submit("v", tasks=10, creator="user:bob")
    await sim.step()
    assert sim.held(run) == 20
    run.unreadable = True
    await sim.step()
    assert sum(m.phase is MemberPhase.FAILED for m in sim.members(run)) == 20
    assert sim.held(run) == 20
    assert sum(name.startswith("run-u-") for _, _, name in sim.cluster.cancels) == 20
    await sim.run_until_final(30)
    stored = sim.stored(run)
    assert stored.phase is TasksetRunPhase.FAILED and "plan_unreadable" in (stored.error or "")
    assert all(m.phase is MemberPhase.FAILED for m in sim.members(run))
    assert sim.phase(other) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


async def test_plan_digest_mismatch_before_any_job(make_sim: MakeSim) -> None:
    """A plan unreadable on the first pass: the run fails in one pass, with no Jobs call."""
    sim = make_sim(caps(10))
    run = await sim.submit("broken", tasks=5)
    run.unreadable = True
    await sim.step()
    assert sim.phase(run) is TasksetRunPhase.FAILED
    assert sim.cluster.calls[1] == 0
    sim.assert_clean()


async def test_files_5xx_for_five_cycles(make_sim: MakeSim) -> None:
    """Files answers 5xx for 5 cycles while loading plans: no run fails, and runs resume when Files recovers."""
    sim = make_sim(caps(10, per_run=5))
    sim.cluster.duration = lambda name: sim.rng.randint(2, 4)
    runs = [await sim.submit(f"f{i}", tasks=12, creator=f"user:{i}") for i in range(2)]

    async def before(s: Sim) -> None:
        s.files_down = s.cycle in range(3, 8)

    settled_before = 0
    for _ in range(8):
        await sim.step(before)
        settled = sum(m.phase in SETTLED_MEMBER_PHASES for r in runs for m in sim.members(r))
        if sim.files_down:
            assert all(sim.phase(r) in OPEN_RUN for r in runs)
            assert settled == settled_before
        settled_before = settled
    await sim.run_until_final(100, before)
    assert all(sim.phase(r) is TasksetRunPhase.COMPLETED for r in runs)
    sim.assert_clean()


async def test_job_refusing_cancel_past_finalization_then_restart(make_sim: MakeSim) -> None:
    """A Job refuses to cancel: the run ends at its finalization deadline, the Job stays counted until a restart."""
    sim = make_sim(caps(3))
    sim.cluster.duration = fixed({"task-0": 40}, default=30)
    sim.cluster.cancel_ticks = (1, 1)
    sim.cluster.refusing.add("run-stuck-task-0")
    stuck = await sim.submit("stuck", tasks=3, timeout=timedelta(seconds=45), finalization_grace=timedelta(seconds=60))
    waiting = await sim.submit("waiting", tasks=3, creator="user:bob")
    for _ in range(3):
        await sim.step()
    sim.cluster.duration = lambda name: 200  # the waiting run's Jobs outlive the scenario
    while sim.phase(stuck) in OPEN_RUN and sim.cycle < 20:
        await sim.step()
    assert sim.phase(stuck) is TasksetRunPhase.EXPIRED
    finalization = sim.stored(stuck).finalization_deadline_at
    assert sim.cycle_start < finalization + CYCLE
    assert ("default", "run-stuck-task-0") in sim.reconciler.stuck
    await sim.step()
    assert sim.held(waiting) == 2  # the stuck Job still holds the third slot
    sim.restart("operator restart")
    await sim.step()
    assert sim.held(waiting) == 3  # the logged exception: the stuck Job is no longer counted
    assert len([j for j in sim.cluster.jobs.values() if j.is_task and j.active]) == 4
    while sim.cluster.jobs[("default", "run-stuck-task-0")].active:
        await sim.step()
    assert len([j for j in sim.cluster.jobs.values() if j.is_task and j.active]) <= 3
    sim.assert_clean()


async def test_member_listing_fails_for_one_run(make_sim: MakeSim) -> None:
    """One run's member listing raises: phase 2 is skipped for the cycle; every other run still settles."""
    sim = make_sim(caps(10, per_run=5))
    sim.cluster.duration = lambda name: 1
    broken = await sim.submit("broken", tasks=10)
    healthy = await sim.submit("healthy", tasks=10, creator="user:bob")
    await sim.step()
    await sim.step()
    sim.fail_listing.add(broken.id)
    settled = sum(m.phase in SETTLED_MEMBER_PHASES for m in sim.members(healthy))
    report = await sim.step()
    assert report is not None and report.phase2_skipped and report.grants == []
    assert sum(m.phase in SETTLED_MEMBER_PHASES for m in sim.members(healthy)) > settled
    sim.fail_listing.clear()
    await sim.run_until_final(60)
    sim.assert_clean()


async def test_operator_cancels_a_running_job(make_sim: MakeSim) -> None:
    """An operator cancels a running Job: its member fails job_cancelled and the slot goes to the next task."""
    sim = make_sim(caps(2))
    sim.cluster.duration = lambda name: 6
    sim.cluster.cancel_ticks = (1, 1)
    run = await sim.submit("op", tasks=3)
    await sim.step()
    await sim.step()
    job = sim.cluster.jobs[("default", "run-op-task-0")]
    job.status = HelixJobStatus.CANCELLED  # the operator's cancel went through
    with pytest.raises(AssertionError):
        sim.cluster.delete("default", "run-op-task-1")  # Jobs refuses to delete a running Job
    await sim.step()
    member = sim.members(run)[0]
    assert (member.phase, member.error) == (MemberPhase.FAILED, "job_cancelled")
    assert sim.members(run)[2].phase is MemberPhase.REGISTERED
    await sim.run_until_final(60)
    sim.assert_clean()


async def test_job_completing_before_the_deadline_succeeds(make_sim: MakeSim) -> None:
    """A Job that completes between the last cycle and the deadline: succeeded, not expired; bundling finishes."""
    sim = make_sim(caps(5))
    sim.cluster.duration = fixed({"task-1": 50}, default=3)  # task-0 completes after the t+45 cycle
    run = await sim.submit("edge", tasks=2, timeout=timedelta(seconds=50))
    await sim.run_until_final(30)
    phases = [m.phase for m in sim.members(run)]
    assert phases == [MemberPhase.SUCCEEDED, MemberPhase.EXPIRED]
    assert sim.phase(run) is TasksetRunPhase.EXPIRED
    sim.assert_clean()


async def test_bundling_past_the_execution_deadline_completes(make_sim: MakeSim) -> None:
    """Every task finished, then the execution deadline passes while bundling: completed, result ready."""
    sim = make_sim(caps(5))
    sim.cluster.duration = lambda name: 3
    run = await sim.submit("bundling", tasks=2, timeout=timedelta(seconds=50))
    await sim.run_until_final(30)
    assert [m.phase for m in sim.members(run)] == [MemberPhase.SUCCEEDED] * 2
    assert (sim.phase(run), sim.stored(run).result_state) == (TasksetRunPhase.COMPLETED, ResultState.READY)
    sim.assert_clean()


async def test_cancel_after_every_member_settled_completes(make_sim: MakeSim) -> None:
    """A cancel after every member settled: the run finalizes and ends completed; the bundle isn't cancelled."""
    sim = make_sim(caps(5))
    sim.cluster.duration = lambda name: 1
    run = await sim.submit("late", tasks=2)
    await sim.step()
    while not all(m.phase in SETTLED_MEMBER_PHASES for m in sim.members(run)):
        await sim.step()
    await sim.cancel(run)
    await sim.run_until_final(10)
    assert sim.phase(run) is TasksetRunPhase.COMPLETED
    assert not [name for _, _, name in sim.cluster.cancels if "-bundle-" in name]
    sim.assert_clean()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
async def test_fault_heavy_random_cluster(make_sim: MakeSim, seed: int) -> None:
    """Every fault at once, over several creators, workspaces and deadlines; all 9 invariants after every cycle."""
    faults = Faults(
        create_5xx_before=0.04,
        create_409_transient=0.02,
        create_5xx_after=0.03,
        create_timeout_unsaved=0.03,
        create_timeout_saved=0.03,
        create_late_save=0.04,
        create_lost_name=0.01,
        stepless_first=0.05,
        read_5xx=0.03,
        read_timeout=0.01,
        refuse_cancel=0.05,
        job_error=0.05,
        no_result=0.03,
        operator_cancel=0.01,
        delete_ended=0.02,
        replace_deleted=0.5,
        squat=0.02,
        member_list_fail=0.02,
        crash=0.002,
        restart=0.03,
        mid_cycle_cancel=0.002,
    )
    sim = make_sim(caps(12, per_creator=8, per_run=5, create_grace_seconds=180), seed=seed, faults=faults)
    sim.cluster.duration = lambda name: sim.rng.randint(1, 8)
    rng = random.Random(seed * 101)
    plan = [
        (
            rng.randint(0, 6),
            f"r{i}",
            rng.randint(4, 24),
            rng.choice(["user:a", "user:b", "user:c"]),
            rng.choice(["ws1", "ws2"]),
            rng.choice([None, 3, 10]),
            timedelta(seconds=rng.choice([180, 600, 1200])),
        )
        for i in range(6)
    ]
    cancels = {f"r{i}": rng.randint(3, 20) for i in range(6) if rng.random() < 0.3}
    submitted: dict[str, SimRun] = {}
    files_down_until = [0]

    async def before(s: Sim) -> None:
        for at, label, tasks, creator, workspace, request_cap, timeout in plan:
            if at == s.cycle - 1:
                submitted[label] = await s.submit(
                    label,
                    tasks=tasks,
                    creator=creator,
                    workspace=workspace,
                    request_cap=request_cap,
                    timeout=timeout,
                    finalization_grace=timedelta(seconds=120),
                )
        for label, at in cancels.items():
            if at == s.cycle and label in submitted:
                await s.cancel(submitted[label])
        if files_down_until[0] < s.cycle and s.rng.random() < 0.03:
            files_down_until[0] = s.cycle + s.rng.randint(0, 3)  # a Files outage of 1-4 cycles
        s.files_down = s.cycle <= files_down_until[0]

    for _ in range(200):
        await sim.step(before)
        if len(sim.runs) == len(plan) and all(sim.phase(r) in TERMINAL_RUN_PHASES for r in sim.runs):
            break
    else:
        sim.violate(7, "not every run was final after 200 cycles")
    for sim_run in sim.runs:
        if sim_run.ever_loaded:
            assert len(sim.members(sim_run)) == sim_run.tasks, sim_run.label
    sim.assert_clean()


async def test_deadline_mid_grant_stops_the_creates(make_sim: MakeSim) -> None:
    """A run passes its deadline in the middle of its grant: the clock is read again, and creates stop."""
    sim = make_sim(caps(10))
    sim.cluster.duration = lambda name: 10
    sim.cluster.create_latency = timedelta(seconds=5)
    run = await sim.submit("mid", tasks=10, timeout=timedelta(seconds=20))
    report = await sim.step()
    assert report is not None and len(report.grants) == 10
    assert sum(name.startswith("run-mid-") for _, name in sim.cluster.jobs) == 4  # at t+0, 5, 10 and 15 s
    await sim.run_until_final(20)
    assert sim.phase(run) is TasksetRunPhase.EXPIRED
    sim.assert_clean()


async def test_deadline_between_phase_1_and_phase_2_creates_nothing(make_sim: MakeSim) -> None:
    """Phase 1 takes long enough that a queued run passes its queue timeout before phase 2: it gets no Job."""
    sim = make_sim(caps(10, per_run=3))
    sim.cluster.duration = lambda name: 20
    busy = await sim.submit("busy", tasks=3)
    await sim.step()  # busy holds 3 registered Jobs
    sim.cluster.read_latency = timedelta(seconds=8)  # each Jobs read takes 8 s from now on
    late = await sim.submit("late", tasks=3, creator="user:bob", queue_timeout=timedelta(seconds=20))
    report = await sim.step()  # starts at t+15; phase 1 reads 3 Jobs (24 s): phase 2 runs after t+35
    assert report is not None and late.entity.name in report.grants
    assert not [name for _, name in sim.cluster.jobs if name.startswith("run-late-")]
    await sim.run_until_final(60)
    assert sim.phase(late) is TasksetRunPhase.EXPIRED and sim.phase(busy) is TasksetRunPhase.COMPLETED
    assert sim.stored(late).error == "no task started within the queue timeout (20 s)"
    sim.assert_clean()


async def test_creator_loses_access_mid_run(make_sim: MakeSim) -> None:
    """Creates answer 403 for the creator: queued tasks fail job_forbidden, Jobs still settle, the run ends.

    The bundle Job is created for the creator too, so it can't start either: the run ends at its
    finalization deadline, completed with result_state=failed (the bound a bundle Job has).
    """
    sim = make_sim(caps(10, per_run=3))
    sim.cluster.duration = lambda name: 3
    run = await sim.submit("lost", tasks=8, timeout=timedelta(minutes=5), finalization_grace=timedelta(minutes=2))
    await sim.step()
    sim.cluster.forbidden.add("user:alice")
    while sim.held(run) or sim.queued(run):
        await sim.step()
    assert sim.cycle < 12 and sim.held(run) == 0
    phases = Counter(m.phase for m in sim.members(run))
    assert phases == Counter({MemberPhase.SUCCEEDED: 3, MemberPhase.FAILED: 5})
    assert all((m.error or "").startswith("job_forbidden") for m in sim.members(run) if m.phase is MemberPhase.FAILED)
    await sim.run_until_final(40)
    stored = sim.stored(run)
    assert (stored.phase, stored.result_state) == (TasksetRunPhase.COMPLETED, ResultState.FAILED)
    sim.assert_clean()


async def test_files_500_on_one_plan_every_cycle(make_sim: MakeSim) -> None:
    """Files answers 500 for one run's plan every cycle: others keep their grants; it ends by its deadline."""
    sim = make_sim(caps(6, per_run=3))
    sim.cluster.duration = lambda name: 4
    broken = await sim.submit(
        "broken", tasks=6, timeout=timedelta(seconds=60), finalization_grace=timedelta(seconds=60)
    )
    await sim.step()
    healthy = await sim.submit("healthy", tasks=12, creator="user:bob")
    broken.files_5xx = True
    await sim.run_until_final(40)
    stored = sim.stored(broken)
    assert stored.phase in TERMINAL_RUN_PHASES and "plan_unreadable" in (stored.error or "")
    assert sim.phase(healthy) is TasksetRunPhase.COMPLETED
    sim.assert_clean()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Spec gap (invariant 4): P3 retries any intent whose GET is 404, and an ended, deleted Job reads 404 too. "
        "An intent whose Job ran and was deleted before the controller could read it (here: its plan was "
        "unavailable) is created a second time, so the task runs twice"
    ),
)
async def test_spec_gap_deleted_intent_job_runs_twice(make_sim: MakeSim) -> None:
    sim = make_sim(caps(2))
    sim.cluster.duration = lambda name: 1
    run = await sim.submit("twice", tasks=1)
    sim.cluster.scripted_create["run-twice-task-0"] = ["5xx_after"]  # saved; the reply is lost
    await sim.step()
    run.files_5xx = True  # P3 needs the plan to prove the Job ours: it can't read the intent for a while
    await sim.step()
    await sim.step()
    assert sim.cluster.jobs[("default", "run-twice-task-0")].status is HelixJobStatus.COMPLETED
    sim.cluster.delete("default", "run-twice-task-0")  # an operator cleans up ended Jobs
    run.files_5xx = False
    await sim.run_until_final(20)
    sim.assert_clean()


async def test_a_hanging_stuck_job_doesnt_stop_the_grants(make_sim: MakeSim) -> None:
    """A stuck Job whose GET always times out stays counted (nothing proves it ended), but phase 2 still runs."""
    sim = make_sim(caps(3))
    sim.cluster.duration = lambda name: 400 if "-stuck-" in name else 2
    sim.cluster.refusing.add("run-stuck-task-0")
    stuck = await sim.submit("stuck", tasks=1, timeout=timedelta(seconds=15), finalization_grace=timedelta(seconds=15))
    for _ in range(3):
        await sim.step()
    assert sim.phase(stuck) is TasksetRunPhase.EXPIRED and ("default", "run-stuck-task-0") in sim.reconciler.stuck
    sim.cluster.hanging.add("run-stuck-task-0")  # the stuck Job's GET now always times out
    later = await sim.submit("later", tasks=4, creator="user:bob", timeout=timedelta(minutes=10))
    for _ in range(10):
        await sim.step()
    assert all(m.phase is MemberPhase.SUCCEEDED for m in sim.members(later)), "\n".join(sim.history[-3:])
    assert ("default", "run-stuck-task-0") in sim.reconciler.stuck  # still counted: never dropped on a guess
    sim.assert_clean()


async def test_a_hanging_intent_get_doesnt_starve_its_runs_registered_jobs(make_sim: MakeSim) -> None:
    """One Job whose GET always times out can't keep the others from being read.

    One create intent whose GET always times out: intents and registered Jobs are read in one loop that starts
    at a rotating offset, so the run's registered Jobs are read and settle ``succeeded`` in other cycles.
    """
    sim = make_sim(caps(3))
    sim.cluster.duration = lambda name: 2
    run = await sim.submit("starved", tasks=3, timeout=timedelta(seconds=300), finalization_grace=timedelta(seconds=60))
    sim.cluster.scripted_create["run-starved-task-0"] = ["timeout_unsaved", "timeout_unsaved"]
    await sim.step()  # task-0's create times out: the grant drain ends
    await sim.step()  # task-1 and task-2 are created; task-0's retry times out again
    assert [m.phase for m in sim.members(run)] == [MemberPhase.IMAGE_READY] + [MemberPhase.REGISTERED] * 2
    sim.cluster.hanging.add("run-starved-task-0")  # from now on every GET of task-0 times out
    for _ in range(4):
        await sim.step()
    ended = [sim.cluster.jobs[("default", f"run-starved-task-{i}")].status for i in (1, 2)]
    assert ended == [HelixJobStatus.COMPLETED] * 2
    for _ in range(4):
        await sim.step()  # 4 more cycles, still before the execution deadline
    phases = [m.phase for m in sim.members(run)][1:]
    assert phases == [MemberPhase.SUCCEEDED] * 2, (
        f"task-1 and task-2 completed 4+ cycles ago and are still {phases}: "
        f"every cycle's first Jobs call is task-0's hanging GET\n" + "\n".join(sim.history[-6:])
    )


async def test_an_adopted_first_job_finds_the_run_running(make_sim: MakeSim) -> None:
    """The run's first grant writes ``running`` before the create, so a first Job P3 adopts (its create timed
    out, then the Job showed up) never leaves the run ``accepted``."""
    sim = make_sim(caps(5))
    sim.cluster.duration = lambda name: 5
    run = await sim.submit("adopted", tasks=1)
    sim.cluster.scripted_create["run-adopted-task-0"] = ["timeout_saved"]
    await sim.step()
    await sim.step()
    assert sim.members(run)[0].phase is MemberPhase.REGISTERED
    await sim.step()
    assert sim.phase(run) is TasksetRunPhase.RUNNING, f"run is {sim.phase(run)} with its Job registered and active"
