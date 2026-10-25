# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller cycles over in-memory records and Jobs: members, slots, registration, settling, stops, bundling."""

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import nemo_evals.taskset_runs.controller as controller_module
import pytest
from nemo_evals.config import RunControllerConfig
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.taskset_runs.controller import (
    TERMINAL_JOB_STATUSES,
    JobsServiceRunJobs,
    TasksetRunReconciler,
)
from nemo_evals.taskset_runs.entities import (
    MemberPhase,
    ResultState,
    TasksetRunEntity,
    TasksetRunMemberEntity,
    TasksetRunPhase,
)
from nemo_evals.taskset_runs.run_evaluation import QUEUE_DURATION_KEY, RESULT_STATE_KEY, ensure_run_evaluation
from nemo_evals.taskset_runs.specs import RunIntakePublication, TasksetBundleSpec, TasksetRunPlan
from nemo_evals.taskset_runs.store import PlanDigestMismatchError, TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import (
    ConflictError,
    InternalServerError,
    NemoHTTPError,
    NemoTransportError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import FileStorageType, HelixJobResultResponse
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatus
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

# harbor_run is accepted 2026-10-15 12:00, execution deadline 13:00, finalization deadline 14:00.
T0 = datetime(2026, 10, 15, 12, 0, 30, tzinfo=timezone.utc)
BEFORE_DEADLINE = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
AFTER_EXECUTION = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)
AFTER_FINALIZATION = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
CYCLE = timedelta(seconds=15)
GRACE = timedelta(seconds=600)
SUMMARY = AgentEvalSummary.model_validate(
    {
        "scores": {
            "scores": [{"name": "harbor_reward.reward", "score_type": "range", "count": 2, "nan_count": 0, "mean": 0.5}]
        },
        "task_count": 2,
        "trial_count": 2,
        "score_count": 2,
    }
)


def http_error[ErrorT: NemoHTTPError](cls: type[ErrorT], status: int, detail: str = "scripted") -> ErrorT:
    return cls(httpx.Response(status, json={"detail": detail}, request=httpx.Request("POST", "http://jobs.test")))


def timeout() -> NemoTransportError:
    return NemoTransportError(httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://jobs.test")))


class FakeRunJobs:
    """In-memory Jobs (and the run's records) with scripted failures.

    ``script(op, *outcomes)`` queues outcomes for the next calls of ``op`` ("create", "get", "cancel",
    "has_steps", "result"): an exception to raise, or a callable run before the call.
    Cancel moves a running Job to ``cancelling``; ``finish`` ends it, as Jobs does later. Every saved Job gets a
    new id, and a result names the id of the Job it was saved under.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, HelixJobResponse] = {}
        self.results: dict[tuple[str, str], HelixJobResultResponse] = {}
        self.saved = 0
        self.created: list[CreateHelixJobRequest] = []
        self.creates: list[str] = []  # every create call's name, including refused ones
        self.records: list[AgentEvalResultEntity] = []
        self.cancelled: list[str] = []
        self.stepless: set[str] = set()
        self.calls: list[str] = []
        self.scripted: dict[str, list[Any]] = {}
        self.broken: dict[str, Exception] = {}  # Job name -> what every get of it raises
        self.evaluations: Any = None  # a FakeRunEvaluations, for runs that publish to Intake

    def script(self, op: str, *outcomes: Any) -> None:
        self.scripted.setdefault(op, []).extend(outcomes)

    async def _scripted(self, op: str) -> None:
        self.calls.append(op)
        queue = self.scripted.get(op)
        if queue:
            outcome = queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is not None:
                result = outcome()
                if hasattr(result, "__await__"):
                    await result

    def put(self, request: CreateHelixJobRequest, status: str) -> HelixJobResponse:
        assert request.name is not None
        self.saved += 1
        self.jobs[request.name] = HelixJobResponse(
            id=f"id-{request.name}-{self.saved}",
            attempt_id="a",
            name=request.name,
            workspace="default",
            source=request.source,
            spec=request.spec,
            platform_spec=request.platform_spec,
            fileset=f"default/job-fileset-{request.name}",
            status=HelixJobStatus(status),
        )
        return self.jobs[request.name]

    def finish(self, name: str, status: str, *, result: bool = True) -> None:
        # The Jobs API returns naive timestamps, which once broke the bundle Job's duration.
        naive = datetime(2026, 10, 16, 17, 32, 49)
        self.jobs[name] = self.jobs[name].model_copy(update={"status": HelixJobStatus(status), "updated_at": naive})
        if result and status == "completed":
            self.results[(name, "agent-eval-results")] = HelixJobResultResponse(
                name="agent-eval-results",
                job=self.jobs[name].id,
                workspace="default",
                attempt_id="a",
                artifact_url=f"default/job-fileset-{name}#results/a/agent-eval-results",
                artifact_storage_type=FileStorageType.FILESET,
            )

    def replace(self, name: str, status: str = "active") -> HelixJobResponse:
        """Delete ended Job ``name`` and save another under its name, as a later create would: same work, new id."""
        job = self.jobs[name]
        assert job.status.value in TERMINAL_JOB_STATUSES, "Jobs deletes only ended Jobs"
        self.results.pop((name, "agent-eval-results"), None)
        request = CreateHelixJobRequest(name=name, source=job.source, spec=job.spec, platform_spec=job.platform_spec)
        return self.put(request, status)

    def finish_all(self, status: str = "completed", *, prefix: str = "") -> None:
        for name, job in list(self.jobs.items()):
            if name.startswith(prefix) and "-task-" in name and job.status.value not in TERMINAL_JOB_STATUSES:
                self.finish(name, status)

    def active(self, prefix: str = "") -> list[str]:
        return [n for n, j in self.jobs.items() if n.startswith(prefix) and j.status.value not in TERMINAL_JOB_STATUSES]

    async def get(self, name):
        await self._scripted("get")
        if name in self.broken:
            raise self.broken[name]
        return self.jobs.get(name)

    async def result(self, job, name):
        await self._scripted("result")
        return self.results.get((job, name))

    async def create(self, request):
        self.creates.append(request.name)
        await self._scripted("create")
        if request.name in self.jobs:
            raise http_error(ConflictError, 409, f"Job {request.name} already exists")
        self.created.append(request)
        return self.put(request, "created")

    async def has_steps(self, name):
        await self._scripted("has_steps")
        return name not in self.stepless

    async def read_summary(self, bundle_ref):
        return SUMMARY

    async def save_record(self, entity):
        # Like the entity store: a new record whose name exists is a conflict.
        if any(record.name == entity.name for record in self.records):
            raise NemoEntityConflictError(entity.name)
        self.records.append(entity)

    async def get_record(self, name):
        return next((record for record in self.records if record.name == name), None)

    async def cancel(self, name):
        await self._scripted("cancel")
        self.cancelled.append(name)
        if name in self.jobs and self.jobs[name].status.value not in TERMINAL_JOB_STATUSES:
            self.jobs[name] = self.jobs[name].model_copy(update={"status": HelixJobStatus.CANCELLING})


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class Harness:
    """A reconciler over ``FakeEntityStore``, one ``FakeRunJobs`` for every workspace, and a fake clock."""

    def __init__(self, entity_store, store: TasksetRunStore, jobs: FakeRunJobs, config: RunControllerConfig) -> None:
        self.entity_store = entity_store
        self.store = store
        self.jobs = jobs
        self.clock = Clock(T0)
        self.plans: dict[str, TasksetRunPlan] = {}
        self.plan_errors: dict[str, Exception] = {}
        self.reconciler = TasksetRunReconciler(store, self.jobs_for, config, clock=self.clock)

        async def load_plan(run: TasksetRunEntity) -> TasksetRunPlan:
            if run.name in self.plan_errors:
                raise self.plan_errors[run.name]
            return self.plans[run.name]

        store.load_plan = load_plan  # ty: ignore[invalid-assignment]

    @asynccontextmanager
    async def jobs_for(self, workspace: str, run: TasksetRunEntity | None) -> AsyncGenerator[FakeRunJobs]:
        yield self.jobs

    async def submit(
        self, plan: TasksetRunPlan, *, name: str, creator: str = "user:alice", workspace: str = "default", **run: Any
    ) -> TasksetRunEntity:
        accepted = run.pop("accepted_at", datetime(2026, 10, 15, 12, tzinfo=timezone.utc))
        self.entity_store.caller = creator
        record = TasksetRunEntity(
            name=name,
            workspace=workspace,
            plan_ref=f"{workspace}/{name}-plan#plan.json",
            plan_digest="f" * 64,
            request_digest="0" * 64,
            taskset_ref=plan.taskset_ref,
            accepted_at=accepted,
            # Started by the first cycle at T0, the run's deadlines are still AFTER_EXECUTION and AFTER_FINALIZATION.
            timeout_seconds=run.pop("timeout_seconds", int((AFTER_EXECUTION - T0).total_seconds())),
            execution_deadline_at=run.pop("execution_deadline_at", accepted + timedelta(hours=1)),
            finalization_deadline_at=run.pop("finalization_deadline_at", accepted + timedelta(hours=2)),
            **run,
        )
        self.plans[name] = plan.model_copy(update={"run_id": name, "workspace": workspace})
        return await self.entity_store.create(record)

    async def cycle(self, at: datetime | None = None):
        if at is not None:
            self.clock.now = at
        report = await self.reconciler.reconcile()
        self.clock.now += CYCLE
        return report

    async def run(self, run: TasksetRunEntity) -> TasksetRunEntity:
        return await self.entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)

    async def members(self, run: TasksetRunEntity) -> list[TasksetRunMemberEntity]:
        return await self.store.list_all_members(await self.run(run))

    async def phases(self, run: TasksetRunEntity) -> list[str]:
        return [m.phase.value for m in await self.members(run)]

    async def cancel(self, run: TasksetRunEntity, at: datetime) -> None:
        current = await self.run(run)
        await self.entity_store.update(current.model_copy(update={"cancel_requested_at": at}))

    def request(self, run: TasksetRunEntity, index: int):
        plan = self.plans[run.name]
        return task_job_request(plan, run, index, plan.task_environments[index])


def sized(plan: TasksetRunPlan, tasks: int, *, cap: int | None = None) -> TasksetRunPlan:
    """``plan`` with ``tasks`` tasks (copies of its first) and a request cap."""
    spec = plan.spec.model_copy(update={"tasks": [plan.spec.tasks[0]] * tasks})
    return plan.model_copy(
        update={
            "spec": spec,
            "task_environments": [plan.task_environments[0]] * tasks,
            "max_concurrent_task_jobs": cap,
        }
    )


def config(**caps: Any) -> RunControllerConfig:
    return RunControllerConfig.model_validate({"default_max_concurrent_task_jobs": None, **caps})


@pytest.fixture
def store(entity_store, files_client) -> TasksetRunStore:
    return TasksetRunStore._with_entity_adapter(entity_store, files_client)


@pytest.fixture
def jobs() -> FakeRunJobs:
    return FakeRunJobs()


@pytest.fixture
def make_harness(entity_store, store, jobs) -> Callable[..., Harness]:
    return lambda **caps: Harness(entity_store, store, jobs, config(**caps))


@pytest.fixture
def harness(make_harness) -> Harness:
    return make_harness()


def task(run: TasksetRunEntity, index: int) -> str:
    return f"{run.name}-task-{index}"


# ---- the happy path and the bundle ----------------------------------------------------------------------


async def test_first_cycle_creates_members_and_registers_the_run_running(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    report = await harness.cycle()
    assert report.created == 2 and [r.name for r in jobs.created] == [task(run, 0), task(run, 1)]
    members = await harness.members(run)
    assert [(m.phase, m.task_job, m.job_status) for m in members] == [
        (MemberPhase.REGISTERED, task(run, 0), HelixJobStatus.CREATED),
        (MemberPhase.REGISTERED, task(run, 1), HelixJobStatus.CREATED),
    ]
    assert all(m.create_sent_at == T0 for m in members)
    assert (await harness.run(run)).phase is TasksetRunPhase.RUNNING
    # Every Job runs the images frozen in the plan.
    assert jobs.created[0] == await harness.request(run, 0)


async def test_job_status_is_written_only_when_it_changes(harness, harbor_plan, jobs, entity_store):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    versions = [m.db_version for m in await harness.members(run)]
    await harness.cycle()
    assert [m.db_version for m in await harness.members(run)] == versions  # unchanged status: no write
    jobs.finish(task(run, 0), "active")
    jobs.broken[task(run, 1)] = http_error(InternalServerError, 500)  # task 1's read fails
    await harness.cycle()
    m0, m1 = await harness.members(run)
    assert m0.job_status is HelixJobStatus.ACTIVE and m0.db_version == versions[0] + 1
    assert m1.job_status is HelixJobStatus.CREATED and m1.phase is MemberPhase.REGISTERED  # last read kept


async def test_finished_tasks_get_one_bundle_then_a_ready_result(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    assert await harness.phases(run) == ["succeeded", "succeeded"]
    await harness.cycle()
    bundle = jobs.created[-1]
    assert bundle.name == f"{run.name}-bundle-1" and bundle.source == "evals.taskset-bundle"
    assert len(jobs.created) == 3  # once, although two cycles saw it
    spec = TasksetBundleSpec.model_validate(bundle.spec)
    assert [(a.task_index, a.job) for a in spec.accepted] == [(0, task(run, 0)), (1, task(run, 1))]
    # The latest succeeded member's write: fixed once the members settle, so every pass sends the same request.
    assert spec.finished_at == max(m.updated_at for m in await harness.members(run) if m.updated_at)
    jobs.finish(bundle.name, "completed")
    await harness.cycle()
    (record,) = jobs.records
    assert record.name == run.name and record.job_id == bundle.name and record.target_kind == "harbor"
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.READY


async def test_errored_tasks_are_left_out_of_the_bundle_and_none_succeeded_fails(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    other = await harness.submit(harbor_plan, name="run-b")
    await harness.cycle()
    jobs.finish(task(run, 0), "completed")
    jobs.finish(task(run, 1), "error")
    jobs.finish(task(other, 0), "error")
    jobs.finish(task(other, 1), "completed", result=False)
    await harness.cycle()
    assert [m.error for m in await harness.members(run)] == [None, "job_failed"]
    assert [m.error for m in await harness.members(other)] == ["job_failed", "no_result"]
    await harness.cycle()
    bundle = next(r for r in jobs.created if r.name == f"{run.name}-bundle-1")
    assert [a.task_index for a in TasksetBundleSpec.model_validate(bundle.spec).accepted] == [0]
    other = await harness.run(other)
    assert other.phase is TasksetRunPhase.FAILED and other.result_state is ResultState.NONE


async def test_failed_bundle_is_retried_then_given_up(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    for attempt in (1, 2, 3):
        await harness.cycle()
        jobs.finish(f"{run.name}-bundle-{attempt}", "error")
    await harness.cycle()
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.FAILED
    assert "bundle-3" in (run.error or "")


async def test_a_bundle_result_from_another_job_under_the_name_is_not_saved(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    await harness.cycle()
    bundle = jobs.created[-1]
    jobs.finish(bundle.name, "completed")
    key = (bundle.name, "agent-eval-results")
    ours = jobs.results[key]
    jobs.results[key] = ours.model_copy(update={"job": "job-someone-else", "artifact_url": "default/elsewhere"})
    await harness.cycle()
    assert not jobs.records and (await harness.run(run)).result_state is ResultState.PENDING
    jobs.results[key] = ours
    await harness.cycle()
    (record,) = jobs.records
    assert record.job_id == bundle.name and (await harness.run(run)).result_state is ResultState.READY


def tampered(request: CreateHelixJobRequest) -> CreateHelixJobRequest:
    """``request`` under its own name, but running another image: someone else's Job squatting the name."""
    platform_spec = request.platform_spec.model_copy(deep=True)
    executor = platform_spec.steps[0].executor
    assert isinstance(executor, CPUExecutionProvider)
    executor.container.image = "registry.invalid/evil@sha256:" + "e" * 64
    return request.model_copy(update={"platform_spec": platform_spec})


async def test_a_foreign_job_under_the_bundle_name_is_never_adopted(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    await harness.cycle()
    bundle = jobs.created[-1]
    jobs.put(tampered(bundle), "completed")
    jobs.finish(bundle.name, "completed")
    await harness.cycle()
    assert not jobs.records and (await harness.run(run)).result_state is ResultState.PENDING
    await harness.cycle(at=AFTER_FINALIZATION)
    assert bundle.name not in jobs.cancelled  # never cancelled either
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.FAILED


async def test_bundle_job_saved_without_its_step_is_replaced(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    await harness.cycle()
    first = f"{run.name}-bundle-1"
    jobs.stepless.add(first)
    jobs.jobs[first] = jobs.jobs[first].model_copy(update={"created_at": harness.clock.now})
    await harness.cycle()
    assert [r.name for r in jobs.created if "bundle" in r.name] == [first]  # may still be saving
    await harness.cycle(at=harness.clock.now + timedelta(seconds=60))
    assert [r.name for r in jobs.created if "bundle" in r.name] == [first, f"{run.name}-bundle-2"]


@pytest.fixture
def publishing(harbor_plan) -> TasksetRunPlan:
    intake = RunIntakePublication(
        experiment="swe-agent-tuning",
        experiment_id="exp-1",
        agent_name="o",
        dataset_name="default/debug",
        dataset_version="e" * 64,
    )
    return harbor_plan.model_copy(update={"intake": intake})


async def test_a_publishing_run_stamps_its_evaluation_once_at_the_end(harness, publishing, jobs, fake_evaluations):
    jobs.evaluations = fake_evaluations
    run = await harness.submit(publishing, name="run-a")
    await ensure_run_evaluation(fake_evaluations, run.workspace, run.name, harness.plans[run.name])
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    await harness.cycle()
    assert not fake_evaluations.patches  # task Jobs publish trials; only the run's end touches the Evaluation
    jobs.finish(f"{run.name}-bundle-1", "completed")
    done = harness.clock.now
    await harness.cycle()
    ((name, patch),) = fake_evaluations.patches
    assert name == run.name and patch.status == "completed" and patch.metadata is not None
    assert patch.metadata[RESULT_STATE_KEY] == "ready"
    assert float(patch.metadata["eval_duration_sec"]) == (done - run.accepted_at).total_seconds()
    assert float(patch.metadata[QUEUE_DURATION_KEY]) == (T0 - run.accepted_at).total_seconds()  # its first slot


async def test_a_publishing_run_that_never_started_stamps_its_whole_life_as_queued(
    harness, publishing, jobs, fake_evaluations
):
    jobs.evaluations = fake_evaluations
    run = await harness.submit(publishing, name="run-a")
    await ensure_run_evaluation(fake_evaluations, run.workspace, run.name, harness.plans[run.name])
    await harness.cycle(at=AFTER_EXECUTION)  # its queue timeout passed before any slot
    ((_, patch),) = fake_evaluations.patches
    assert patch.status == "expired" and patch.metadata is not None
    waited = (AFTER_EXECUTION - run.accepted_at).total_seconds()
    assert float(patch.metadata[QUEUE_DURATION_KEY]) == float(patch.metadata["eval_duration_sec"]) == waited


async def test_an_expired_publishing_run_stamps_expired_and_a_failed_stamp_never_blocks(
    harness, publishing, jobs, fake_evaluations
):
    jobs.evaluations = fake_evaluations
    run = await harness.submit(publishing, name="run-a")
    await ensure_run_evaluation(fake_evaluations, run.workspace, run.name, harness.plans[run.name])
    fake_evaluations.fail = {"patch"}
    await harness.cycle(at=AFTER_EXECUTION)
    assert (await harness.run(run)).phase is TasksetRunPhase.EXPIRED
    fake_evaluations.fail = set()
    other = await harness.submit(publishing, name="run-b")
    await ensure_run_evaluation(fake_evaluations, other.workspace, other.name, harness.plans[other.name])
    await harness.cycle(at=AFTER_EXECUTION)
    ((_, patch),) = fake_evaluations.patches
    assert patch.status == "expired" and patch.metadata is not None and patch.metadata[RESULT_STATE_KEY] == "none"


async def test_an_expired_publishing_run_with_a_partial_result_stamps_expired_and_ready(
    harness, publishing, jobs, fake_evaluations
):
    jobs.evaluations = fake_evaluations
    run = await stopped_with_one_finished(harness, publishing, jobs, "expired")
    await ensure_run_evaluation(fake_evaluations, run.workspace, run.name, harness.plans[run.name])
    jobs.finish(f"{run.name}-bundle-1", "completed")
    await harness.cycle()
    ((_, patch),) = fake_evaluations.patches
    assert patch.status == "expired" and patch.metadata is not None and patch.metadata[RESULT_STATE_KEY] == "ready"


# ---- caps and slot order --------------------------------------------------------------------------------


async def test_a_cancel_between_the_stamp_and_the_state_write_leaves_the_evaluation_matching_the_run(
    harness, store, publishing, jobs, fake_evaluations, monkeypatch
):
    """The cancel route's write lands after the pass stamps ``expired``: its ``set_state`` loses, and the next pass
    stamps again before its own write, so the last stamp matches the run. The members were already written
    ``expired``, so the run ends ``expired``: the expiry came first."""
    jobs.evaluations = fake_evaluations
    run = await harness.submit(publishing, name="run-a")
    await ensure_run_evaluation(fake_evaluations, run.workspace, run.name, harness.plans[run.name])
    stamp = fake_evaluations.patch

    async def cancel_after_the_stamp(workspace, name, body):
        await stamp(workspace, name, body)
        if len(fake_evaluations.patches) == 1:
            await store.request_cancel(workspace, name, now=harness.clock.now)

    monkeypatch.setattr(fake_evaluations, "patch", cancel_after_the_stamp)
    await harness.cycle(at=AFTER_EXECUTION)
    assert (await harness.run(run)).phase is TasksetRunPhase.ACCEPTED  # set_state lost to the cancel's write
    await harness.cycle()
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.EXPIRED and run.cancel_requested_at is not None
    assert [patch.status for _, patch in fake_evaluations.patches] == ["expired", "expired"]
    evaluation = fake_evaluations.evaluations[run.name]
    assert evaluation.status == run.phase.value and evaluation.metadata[RESULT_STATE_KEY] == run.result_state.value


async def test_the_operator_cap_takes_precedence_over_the_request(make_harness, harbor_plan, jobs):
    harness = make_harness(default_max_concurrent_task_jobs=2)
    asks_more = await harness.submit(sized(harbor_plan, 5, cap=4), name="run-a")
    asks_less = await harness.submit(sized(harbor_plan, 5, cap=1), name="run-b")
    await harness.cycle()
    assert len(jobs.active(asks_more.name)) == 2 and len(jobs.active(asks_less.name)) == 1
    await harness.cycle()
    assert len(jobs.created) == 3  # nothing more until a slot frees
    jobs.finish(task(asks_more, 0), "completed")
    await harness.cycle()
    assert len(jobs.active(asks_more.name)) == 2 and len(jobs.created) == 4


async def test_the_two_run_example(make_harness, harbor_plan, jobs):
    harness = make_harness(max_cluster_concurrent_jobs=50)
    a = await harness.submit(sized(harbor_plan, 45), name="run-a", creator="user:a")
    await harness.cycle()
    b = await harness.submit(sized(harbor_plan, 50), name="run-b", creator="user:b")
    await harness.cycle()
    assert len(jobs.active(a.name)) == 45 and len(jobs.active(b.name)) == 5
    for i in range(10):
        jobs.finish(task(a, i), "completed")
    report = await harness.cycle()
    assert len(jobs.active(b.name)) == 15 and report.used == 50 and report.queued == 35


async def test_creators_share_the_cluster_least_loaded_first(make_harness, harbor_plan, jobs):
    harness = make_harness(
        max_cluster_concurrent_jobs=50, max_concurrent_task_jobs_per_creator=30, default_max_concurrent_task_jobs=20
    )
    a = [await harness.submit(sized(harbor_plan, 113), name=f"run-a{i}", creator="user:a") for i in range(3)]
    await harness.cycle()
    assert [len(jobs.active(r.name)) for r in a] == [20, 10, 0]  # A's creator cap; 20 slots stay free
    b = [await harness.submit(sized(harbor_plan, 113), name=f"run-b{i}", creator="user:b") for i in range(2)]
    report = await harness.cycle()
    assert [len(jobs.active(r.name)) for r in b] == [20, 0]  # B holds the fewest: all 20 at once
    assert report.held_by_creator == {"user:a": 30, "user:b": 20}
    for i in range(10):  # 10 of A's run 1 finish
        jobs.finish(task(a[0], i), "completed")
    report = await harness.cycle()
    assert report.held_by_creator == {"user:a": 25, "user:b": 25}
    assert [len(jobs.active(r.name)) for r in (*a, *b)] == [15, 10, 0, 20, 5]


async def test_a_run_whose_jobs_client_wont_open_for_its_creates_doesnt_stop_the_others(
    make_harness, harbor_plan, jobs
):
    harness = make_harness()
    a = await harness.submit(sized(harbor_plan, 2), name="run-a", creator="user:a")
    b = await harness.submit(sized(harbor_plan, 2), name="run-b", creator="user:b")
    opens: dict[str, int] = {}
    real = harness.reconciler._jobs_for

    @asynccontextmanager
    async def failing_for_a_creates(workspace, run):
        if run is not None:
            opens[run.name] = opens.get(run.name, 0) + 1
            if run.name == a.name and opens[a.name] == 2:  # phase 1's open worked; phase 2's fails (no token)
                raise RuntimeError("token unavailable")
        async with real(workspace, run) as inner:
            yield inner

    harness.reconciler._jobs_for = failing_for_a_creates
    report = await harness.cycle()
    assert jobs.active(a.name) == [] and len(jobs.active(b.name)) == 2
    assert report.created == 2  # the cycle finished and reported
    await harness.cycle()
    assert len(jobs.active(a.name)) == 2


# ---- registration ---------------------------------------------------------------------------------------


async def test_a_create_timeout_ends_the_grants_and_keeps_its_slot_then_the_job_is_adopted(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 3), name="run-a")
    request = await harness.request(run, 0)

    def saved_then_timed_out():
        jobs.put(request, "created")  # Jobs saved it; the reply was lost
        raise timeout()

    jobs.script("create", saved_then_timed_out)
    report = await harness.cycle()
    assert jobs.creates == [task(run, 0)] and report.jobs_timeouts == 1  # the other two grants wait
    m0, m1, m2 = await harness.members(run)
    assert m0.phase is MemberPhase.IMAGE_READY and m0.task_job == task(run, 0)  # the intent holds the slot
    assert m1.task_job is None and m2.task_job is None
    assert report.used == 1 and report.queued == 2
    await harness.cycle()
    assert (await harness.members(run))[0].phase is MemberPhase.REGISTERED  # adopted, not created twice
    assert jobs.creates.count(task(run, 0)) == 1 and len(jobs.active(run.name)) == 3


async def test_every_registration_records_the_id_of_the_job_it_registered(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 3), name="run-a")
    second, third = await harness.request(run, 1), await harness.request(run, 2)

    def saved_then_timed_out():
        jobs.put(third, "created")
        raise timeout()

    # Task 0 is created (201); task 1's POST meets the Job an earlier POST saved (409, adopted); task 2's POST
    # times out after Jobs saved it, and the next cycle's read adopts it.
    jobs.script("create", None, lambda: jobs.put(second, "created"), saved_then_timed_out)
    await harness.cycle()
    await harness.cycle()
    members = await harness.members(run)
    assert [m.phase for m in members] == [MemberPhase.REGISTERED] * 3
    assert [m.job_id for m in members] == [jobs.jobs[task(run, i)].id for i in range(3)]


async def test_a_create_whose_job_was_not_saved_is_retried_after_the_grants(make_harness, harbor_plan, jobs):
    harness = make_harness(default_max_concurrent_task_jobs=1)
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle()
    assert (await harness.members(run))[0].task_job == task(run, 0)
    report = await harness.cycle()  # 404 → retried with no new slot (the run cap is 1: no grant)
    assert jobs.creates == [task(run, 0), task(run, 0)] and report.grants == []
    assert (await harness.phases(run)) == ["registered", "image_ready"]


async def test_a_delayed_commit_holds_its_slot_through_a_cancel_until_it_is_adopted_and_cancelled(
    harness, harbor_plan, jobs
):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    request = await harness.request(run, 0)
    jobs.script("create", timeout())  # the create times out at T0 + 30 s; Jobs saves it at T0 + 50 s
    await harness.cycle(at=T0)
    jobs.script("create", http_error(ConflictError, 409, "taken"))  # the retry meets the half-saved name
    await harness.cycle(at=T0 + timedelta(seconds=15))
    await harness.cancel(run, T0 + timedelta(seconds=40))
    report = await harness.cycle(at=T0 + timedelta(seconds=45))  # GET 404, within grace: held, not settled
    assert report.used == 1 and await harness.phases(run) == ["image_ready"]
    jobs.put(request, "created")  # saved at T0 + 50 s
    await harness.cycle(at=T0 + timedelta(seconds=60))
    assert jobs.cancelled == [task(run, 0)] and await harness.phases(run) == ["registered"]
    assert (await harness.run(run)).phase is not TasksetRunPhase.CANCELLED  # waits for the Job to end
    jobs.finish(task(run, 0), "cancelled")
    await harness.cycle()
    assert await harness.phases(run) == ["cancelled"]
    assert (await harness.run(run)).phase is TasksetRunPhase.CANCELLED


async def test_a_delayed_commit_with_a_refused_retry_keeps_its_slot_and_is_adopted(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    request = await harness.request(run, 0)
    jobs.script("create", timeout())
    await harness.cycle(at=T0)
    jobs.script("create", http_error(UnprocessableEntityError, 422, "profile gone"))
    report = await harness.cycle(at=T0 + timedelta(seconds=15))
    (member,) = await harness.members(run)
    assert member.phase is MemberPhase.IMAGE_READY and member.create_sent_at == T0  # restored, still held
    assert report.used == 1
    jobs.put(request, "created")
    await harness.cycle(at=T0 + timedelta(seconds=60))
    assert await harness.phases(run) == ["registered"]


async def test_a_refused_retry_fails_with_its_own_reason_past_grace(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle(at=T0)
    jobs.script("create", *[http_error(PermissionDeniedError, 403, "no access")] * 100)
    await harness.cycle(at=T0 + GRACE - CYCLE)
    assert await harness.phases(run) == ["image_ready"]
    await harness.cycle(at=T0 + GRACE)
    (member,) = await harness.members(run)
    assert member.phase is MemberPhase.FAILED and member.error == "job_forbidden: no access"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (http_error(UnprocessableEntityError, 422, "bad spec"), "job_spec_invalid: bad spec"),
        (http_error(NotFoundError, 404, "no profile"), "job_spec_invalid: no profile"),
        (http_error(PermissionDeniedError, 403, "forbidden"), "job_forbidden: forbidden"),
    ],
)
async def test_a_refused_first_post_fails_at_once_and_releases_its_slot(make_harness, harbor_plan, jobs, error, reason):
    harness = make_harness(max_cluster_concurrent_jobs=1)
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    jobs.script("create", error)
    report = await harness.cycle()
    (m0, m1) = await harness.members(run)
    assert m0.phase is MemberPhase.FAILED and m0.error == reason and report.used == 0
    await harness.cycle()
    assert (await harness.members(run))[1].phase is MemberPhase.REGISTERED  # the slot went to the next task


async def test_a_delayed_first_step_stays_pending_then_registers_and_stepless_past_grace_fails(
    harness, harbor_plan, jobs
):
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    for i in (0, 1):
        jobs.put(await harness.request(run, i), "created")  # e.g. a create that timed out mid-save
        jobs.stepless.add(task(run, i))
    jobs.script("create", http_error(ConflictError, 409, "taken"), http_error(ConflictError, 409, "taken"))
    await harness.cycle(at=T0)
    assert await harness.phases(run) == ["image_ready", "image_ready"]  # 409, ours but stepless: pending
    jobs.stepless.discard(task(run, 0))
    await harness.cycle(at=T0 + CYCLE)
    assert await harness.phases(run) == ["registered", "image_ready"]
    await harness.cycle(at=T0 + GRACE)
    m1 = (await harness.members(run))[1]
    assert m1.phase is MemberPhase.FAILED and m1.error is not None and m1.error.startswith("job_name_conflict")


async def test_a_lost_name_fails_one_cycle_after_grace_and_releases_its_slot(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle(at=T0)
    jobs.script("create", *[http_error(ConflictError, 409, "name taken")] * 100)  # every retry 409, every GET 404
    at = T0 + CYCLE
    while at < T0 + GRACE:
        await harness.cycle(at=at)
        (member,) = await harness.members(run)
        assert member.phase is MemberPhase.IMAGE_READY and member.create_sent_at == T0  # restored after each 409
        at += CYCLE
    report = await harness.cycle(at=at)
    (member,) = await harness.members(run)
    assert member.phase is MemberPhase.FAILED and member.error == "job_name_conflict: name taken"
    assert report.used == 0


async def test_a_409_whose_lookup_fails_keeps_the_earlier_creates_window(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle(at=T0)  # the first create's outcome is unknown: its window opens at T0
    for at in (T0 + CYCLE, T0 + 2 * CYCLE):
        jobs.script("create", http_error(ConflictError, 409, "taken"))
        jobs.script("get", None, timeout())  # P3 reads 404; the lookup after the retry's 409 times out
        await harness.cycle(at=at)
        (member,) = await harness.members(run)
        # The 409 saved nothing: the window stays T0's, so the slot can't outlive create_grace.
        assert member.phase is MemberPhase.IMAGE_READY and member.create_sent_at == T0


async def test_a_first_post_409_with_no_job_is_retried_until_grace_keeping_jobs_detail(harness, harbor_plan, jobs):
    transient = await harness.submit(sized(harbor_plan, 1), name="run-a")
    persistent = await harness.submit(sized(harbor_plan, 1), name="run-b")
    missing = http_error(ConflictError, 409, "secret 'hf-token' not found")
    jobs.script("create", missing, missing, None, missing)  # run-a: 409, then succeeds; run-b: 409 forever
    await harness.cycle(at=T0)
    await harness.cycle(at=T0 + CYCLE)
    assert await harness.phases(transient) == ["registered"]
    jobs.script("create", *[missing] * 100)
    await harness.cycle(at=T0 + GRACE)
    (member,) = await harness.members(persistent)
    assert member.phase is MemberPhase.FAILED and member.error == "job_name_conflict: secret 'hf-token' not found"


async def test_a_foreign_job_under_an_intent_name_fails_the_member_and_is_never_cancelled(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.put(tampered(await harness.request(run, 0)), "active")
    await harness.cycle()
    (member,) = await harness.members(run)
    assert member.phase is MemberPhase.FAILED and (member.error or "").startswith("job_name_conflict")
    await harness.cycle(at=AFTER_FINALIZATION)
    assert not jobs.cancelled


async def test_grant_order_follows_creators_and_ends_at_an_ambiguous_create(make_harness, harbor_plan, jobs):
    harness = make_harness(max_cluster_concurrent_jobs=10)
    a = await harness.submit(sized(harbor_plan, 5), name="run-a", creator="user:a")
    b = await harness.submit(sized(harbor_plan, 5), name="run-b", creator="user:b")
    jobs.script("create", None, None, timeout())  # A, B, then A's second times out
    report = await harness.cycle()
    assert report.grants[:4] == ["run-a", "run-b", "run-a", "run-b"]
    assert jobs.creates == [task(a, 0), task(b, 0), task(a, 1)]
    assert report.held_by_creator == {"user:a": 2, "user:b": 1}
    assert await harness.phases(a) == ["registered", "image_ready", *["image_ready"] * 3]
    assert (await harness.members(a))[2].task_job is None


async def test_an_ambiguous_retry_does_not_stop_the_grants(make_harness, harbor_plan, jobs):
    harness = make_harness()
    old = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle()
    new = await harness.submit(sized(harbor_plan, 2), name="run-b")
    jobs.script("create", None, None, http_error(InternalServerError, 503, "busy"))  # grants ok, the retry 503s
    await harness.cycle()
    assert await harness.phases(new) == ["registered", "registered"]
    assert await harness.phases(old) == ["image_ready"]


# ---- settling ---------------------------------------------------------------------------------------------


async def test_job_outcomes_settle_members_and_release_slots(make_harness, harbor_plan, jobs):
    harness = make_harness(max_cluster_concurrent_jobs=4)
    run = await harness.submit(sized(harbor_plan, 6), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 0), "error")
    jobs.finish(task(run, 1), "cancelled")  # an operator cancelled it
    jobs.finish(task(run, 2), "completed", result=False)
    del jobs.jobs[task(run, 3)]  # deleted after it ended
    report = await harness.cycle()
    assert [m.error for m in (await harness.members(run))[:4]] == [
        "job_failed",
        "job_cancelled",
        "no_result",
        "job_missing",
    ]
    assert report.created == 2 and len(jobs.active(run.name)) == 2


async def test_a_jobs_timeout_in_phase_1_stops_its_reads_but_not_the_grants(make_harness, harbor_plan, jobs):
    harness = make_harness(max_cluster_concurrent_jobs=4)
    run = await harness.submit(sized(harbor_plan, 3), name="run-a")
    await harness.cycle()
    jobs.finish_all()
    later = await harness.submit(sized(harbor_plan, 1), name="run-b")
    jobs.script("get", timeout())
    jobs.calls.clear()
    report = await harness.cycle()
    # One read, then the cutoff; phase 2 bounds itself, so a hanging read doesn't stop the free slot's grant.
    assert jobs.calls == ["get", "create"] and report.jobs_timeouts == 1 and report.phase2_skipped is None
    assert await harness.phases(run) == ["registered"] * 3  # members keep their state
    assert await harness.phases(later) == ["registered"]


async def test_a_hanging_read_doesnt_starve_another_runs_finished_tasks(harness, harbor_plan, jobs):
    a = await harness.submit(sized(harbor_plan, 3), name="run-a")
    b = await harness.submit(sized(harbor_plan, 3), name="run-b")
    await harness.cycle()
    jobs.finish(task(b, 0), "completed")
    jobs.finish(task(b, 2), "completed")
    jobs.broken[task(a, 2)] = timeout()  # every read of these two times out
    jobs.broken[task(b, 1)] = timeout()
    for _ in range(6):  # each run's reads resume just past the one that tripped the cutoff
        await harness.cycle()
    assert await harness.phases(b) == ["succeeded", "registered", "succeeded"]
    assert await harness.phases(a) == ["registered"] * 3


async def test_a_job_that_took_an_ended_jobs_name_is_never_read_for_a_result_counted_or_cancelled(
    make_harness, harbor_plan, jobs
):
    harness = make_harness(max_cluster_concurrent_jobs=1)
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 0), "error")
    jobs.replace(task(run, 0))  # ours ended and was deleted; a new Job runs the same work under its name
    jobs.finish(task(run, 0), "completed")  # ...and has a result
    jobs.calls.clear()
    report = await harness.cycle()
    m0, m1 = await harness.members(run)
    assert (m0.phase, m0.error, m0.bundle_ref) == (MemberPhase.FAILED, "job_missing", None)
    assert "result" not in jobs.calls[: jobs.calls.index("create")] and not jobs.cancelled
    assert m1.phase is MemberPhase.REGISTERED and report.used == 1  # the slot went to task 1, not the newcomer


async def test_a_result_saved_under_another_job_is_never_used(harness, harbor_plan, jobs):
    """The read matched our Job, but the name changed hands before its result was read."""
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 0), "completed")

    def replaced_and_finished():
        jobs.replace(task(run, 0))
        jobs.finish(task(run, 0), "completed")

    jobs.script("result", replaced_and_finished)
    (before,) = await harness.members(run)
    await harness.cycle()
    (member,) = await harness.members(run)
    assert member.db_version == before.db_version and member.bundle_ref is None  # nothing written this cycle
    await harness.cycle()
    (member,) = await harness.members(run)
    assert (member.phase, member.error) == (MemberPhase.FAILED, "job_missing")  # the next read sees the new id


async def test_spec_drift_keeps_registered_jobs_and_settles_them_by_their_end_state(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    await harness.cycle()
    name = task(run, 0)
    jobs.jobs[name] = jobs.jobs[name].model_copy(update={"spec": {"changed": True}})  # a redeploy's compiler
    jobs.finish(name, "completed")
    await harness.cycle()
    assert await harness.phases(run) == ["succeeded"]


# ---- stops ------------------------------------------------------------------------------------------------


async def test_the_deadline_expires_queued_tasks_at_once_and_running_ones_when_their_jobs_end(
    make_harness, harbor_plan, jobs
):
    harness = make_harness(max_cluster_concurrent_jobs=2)
    run = await harness.submit(sized(harbor_plan, 4), name="run-a")
    await harness.cycle()
    report = await harness.cycle(at=AFTER_EXECUTION)
    assert sorted(jobs.cancelled) == [task(run, 0), task(run, 1)]
    assert await harness.phases(run) == ["registered", "registered", "expired", "expired"]
    assert report.used == 2 and report.created == 0  # slots held until the Jobs end
    await harness.cycle()
    assert jobs.cancelled.count(task(run, 0)) == 1  # already cancelling: not cancelled again
    jobs.finish(task(run, 0), "cancelled")
    jobs.finish(task(run, 1), "completed")  # finished just before the deadline
    await harness.cycle()
    assert await harness.phases(run) == ["expired", "succeeded", "expired", "expired"]
    # The task that finished is bundled: a partial result over it, and the run stays expired.
    bundle = jobs.created[-1]
    assert bundle.name == f"{run.name}-bundle-1"
    assert [a.task_index for a in TasksetBundleSpec.model_validate(bundle.spec).accepted] == [1]
    jobs.finish(bundle.name, "completed")
    await harness.cycle()
    (record,) = jobs.records
    assert record.name == run.name and record.job_id == bundle.name
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.EXPIRED and run.result_state is ResultState.READY
    assert run.error == (
        "3 of 4 tasks expired at the execution deadline, 2 of them still queued; the result covers 1 of 4 tasks"
    )


# ---- partial results of a stopped run --------------------------------------------------------------------


async def stopped_with_one_finished(harness, harbor_plan, jobs, stop: str) -> TasksetRunEntity:
    """A 2-task run: task 1 completed, then ``stop`` ("expired" or "cancelled"); its bundle Job is created."""
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 1), "completed")
    if stop == "cancelled":
        await harness.cancel(run, BEFORE_DEADLINE)
        await harness.cycle(at=BEFORE_DEADLINE)
    else:
        await harness.cycle(at=AFTER_EXECUTION)
    jobs.finish(task(run, 0), "cancelled")
    await harness.cycle()  # every member settled: the bundle Job is created
    assert await harness.phases(run) == [stop, "succeeded"]
    assert jobs.created[-1].name == f"{run.name}-bundle-1"
    return run


STOP_ERRORS = {
    "expired": "1 of 2 tasks expired at the execution deadline; the result covers 1 of 2 tasks",
    "cancelled": "1 of 2 tasks cancelled; the result covers 1 of 2 tasks",
}


@pytest.mark.parametrize("stop", ["expired", "cancelled"])
async def test_a_stopped_run_keeps_its_phase_and_a_result_over_the_tasks_that_finished(
    harness, harbor_plan, jobs, stop
):
    run = await stopped_with_one_finished(harness, harbor_plan, jobs, stop)
    assert (await harness.run(run)).phase is TasksetRunPhase.RUNNING  # open while it bundles
    jobs.finish(f"{run.name}-bundle-1", "completed")
    await harness.cycle()
    stored = await harness.run(run)
    assert (stored.phase.value, stored.result_state) == (stop, ResultState.READY)
    assert stored.error == STOP_ERRORS[stop]


async def test_a_stopped_run_with_no_finished_task_has_no_result(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    await harness.cycle()
    await harness.cycle(at=AFTER_EXECUTION)
    jobs.finish(task(run, 0), "cancelled")
    await harness.cycle()
    stored = await harness.run(run)
    assert (stored.phase, stored.result_state) == (TasksetRunPhase.EXPIRED, ResultState.NONE)
    assert stored.error == "1 of 1 tasks expired at the execution deadline"
    assert not any(r.name.startswith(f"{run.name}-bundle") for r in jobs.created)


async def test_a_stopped_runs_failed_bundles_fail_its_result_and_keep_its_stop(harness, harbor_plan, jobs):
    run = await stopped_with_one_finished(harness, harbor_plan, jobs, "expired")
    for attempt in (1, 2, 3):
        jobs.finish(f"{run.name}-bundle-{attempt}", "error")
        await harness.cycle()
    stored = await harness.run(run)
    assert (stored.phase, stored.result_state) == (TasksetRunPhase.EXPIRED, ResultState.FAILED)
    assert stored.error == f"{STOP_ERRORS['expired']}; bundle Job {run.name}-bundle-3 ended error"


async def test_a_stopped_run_past_its_finalization_deadline_fails_its_result_and_keeps_its_stop(
    harness, harbor_plan, jobs
):
    run = await stopped_with_one_finished(harness, harbor_plan, jobs, "expired")
    await harness.cycle(at=AFTER_FINALIZATION)
    stored = await harness.run(run)
    assert (stored.phase, stored.result_state) == (TasksetRunPhase.EXPIRED, ResultState.FAILED)
    assert stored.error == f"{STOP_ERRORS['expired']}; finalization deadline passed before the result was saved"
    assert f"{run.name}-bundle-1" in jobs.cancelled


@pytest.mark.parametrize("stop", ["expired", "cancelled"])
@pytest.mark.parametrize("retry_after_the_deadline", [False, True])
async def test_a_saved_result_whose_terminal_write_failed_keeps_the_stop(
    harness, harbor_plan, jobs, monkeypatch, stop, retry_after_the_deadline
):
    """The record is saved, then the run's write fails; the next pass ends the run with the stop, ready, its error.

    Before the finalization deadline the record's create conflicts and is adopted; after it, ``_miss_finalization``
    finds it. Both write the same phase and error as the first attempt.
    """
    run = await stopped_with_one_finished(harness, harbor_plan, jobs, stop)
    jobs.finish(f"{run.name}-bundle-1", "completed")
    real = harness.store.set_state

    async def lost(*args, **kwargs):
        raise RuntimeError("the run's write failed")

    monkeypatch.setattr(harness.store, "set_state", lost)
    await harness.cycle()
    assert len(jobs.records) == 1 and (await harness.run(run)).result_state is ResultState.PENDING
    monkeypatch.setattr(harness.store, "set_state", real)
    await harness.cycle(at=AFTER_FINALIZATION if retry_after_the_deadline else None)
    stored = await harness.run(run)
    assert (stored.phase.value, stored.result_state, stored.error) == (stop, ResultState.READY, STOP_ERRORS[stop])
    assert len(jobs.records) == 1


async def test_an_error_while_cancelling_is_cancelled_and_past_the_deadline_stays_failed(harness, harbor_plan, jobs):
    cancelled = await harness.submit(sized(harbor_plan, 1), name="run-a")
    expired = await harness.submit(sized(harbor_plan, 1), name="run-b")
    await harness.cycle()
    await harness.cancel(cancelled, BEFORE_DEADLINE)
    await harness.cycle(at=AFTER_EXECUTION)
    jobs.finish_all("error")
    await harness.cycle()
    assert await harness.phases(cancelled) == ["cancelled"] and await harness.phases(expired) == ["failed"]
    assert (await harness.run(cancelled)).phase is TasksetRunPhase.CANCELLED
    assert (await harness.run(expired)).phase is TasksetRunPhase.FAILED


async def test_a_stop_cancels_only_jobs_this_pass_read_as_its_own(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 3), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 0), "error")
    jobs.replace(task(run, 0))  # a new Job under task 0's name, running
    jobs.broken[task(run, 1)] = http_error(InternalServerError, 500)  # task 1's Job can't be read this pass
    await harness.cancel(run, BEFORE_DEADLINE)
    await harness.cycle(at=BEFORE_DEADLINE)
    assert jobs.cancelled == [task(run, 2)]  # never the newcomer, nor a Job this pass didn't read
    m0, m1, _ = await harness.members(run)
    assert (m0.phase, m0.error) == (MemberPhase.CANCELLED, "job_missing") and m1.phase is MemberPhase.REGISTERED
    del jobs.broken[task(run, 1)]
    await harness.cycle()
    assert sorted(jobs.cancelled) == [task(run, 1), task(run, 2)]
    assert jobs.jobs[task(run, 0)].status is HelixJobStatus.ACTIVE


async def test_a_cancel_after_every_member_settled_still_finalizes(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle()
    await harness.cancel(run, harness.clock.now)
    await harness.cycle()
    jobs.finish(f"{run.name}-bundle-1", "completed")
    await harness.cycle()
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.cancel_requested_at is not None
    assert f"{run.name}-bundle-1" not in jobs.cancelled


async def test_a_stop_that_crashed_before_the_runs_write_keeps_its_outcome(harness, harbor_plan, jobs, monkeypatch):
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    late = await harness.submit(harbor_plan, name="run-b")
    await harness.cycle()
    jobs.finish(task(late, 0), "completed")
    jobs.finish(task(late, 1), "completed")
    await harness.cycle()  # late: every member settled
    await harness.cancel(run, BEFORE_DEADLINE)
    await harness.cancel(late, BEFORE_DEADLINE)
    await harness.cycle()
    jobs.finish_all("cancelled")
    real = harness.store.set_state

    async def lost(*args, **kwargs):
        raise RuntimeError("entity store unavailable")

    monkeypatch.setattr(harness.store, "set_state", lost)
    await harness.cycle()
    assert await harness.phases(run) == ["cancelled", "cancelled"]
    monkeypatch.setattr(harness.store, "set_state", real)
    await harness.cycle()
    assert (await harness.run(run)).phase is TasksetRunPhase.CANCELLED
    assert (await harness.run(late)).phase is TasksetRunPhase.RUNNING  # a late cancel finalizes (bundling)


async def test_stop_settles_an_intent_with_no_job_only_past_grace(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    jobs.script("create", timeout())
    await harness.cycle(at=T0)
    await harness.cancel(run, T0)
    await harness.cycle(at=T0 + CYCLE)
    assert await harness.phases(run) == ["image_ready"]  # 404 within grace: may still be saving
    assert jobs.creates == [task(run, 0)]  # a stopping run gets no retry
    await harness.cycle(at=T0 + GRACE)
    assert await harness.phases(run) == ["cancelled"] and not jobs.cancelled
    assert (await harness.run(run)).phase is TasksetRunPhase.CANCELLED


async def test_a_job_refusing_to_cancel_is_settled_at_the_finalization_deadline_and_stays_counted(
    make_harness, harbor_plan, jobs
):
    harness = make_harness(max_cluster_concurrent_jobs=1)
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    await harness.cycle()
    jobs.script("cancel", *[http_error(ConflictError, 409, "rerun")] * 100)
    await harness.cycle(at=AFTER_EXECUTION)
    assert await harness.phases(run) == ["registered"]
    await harness.cycle(at=AFTER_FINALIZATION)
    assert await harness.phases(run) == ["expired"]
    assert (await harness.run(run)).phase is TasksetRunPhase.EXPIRED
    later = await harness.submit(sized(harbor_plan, 1), name="run-b", accepted_at=AFTER_FINALIZATION)
    report = await harness.cycle()
    assert report.used == 1 and report.created == 0  # the stuck Job still holds the only slot
    jobs.finish(task(run, 0), "completed")
    report = await harness.cycle()
    assert report.created == 1 and not harness.reconciler.stuck and await harness.phases(later) == ["registered"]


async def test_a_jobs_outage_spanning_the_finalization_deadline_still_ends_every_run(harness, harbor_plan, jobs):
    bundling = await harness.submit(sized(harbor_plan, 1), name="run-a")
    running = await harness.submit(sized(harbor_plan, 1), name="run-b")
    await harness.cycle()
    jobs.finish(task(bundling, 0), "completed")
    await harness.cycle()
    await harness.cycle()  # run-a's bundle Job created
    jobs.script("get", *[timeout()] * 100)
    jobs.script("cancel", *[timeout()] * 100)
    await harness.cycle(at=AFTER_FINALIZATION)
    await harness.cycle()
    bundling, running = await harness.run(bundling), await harness.run(running)
    assert (bundling.phase, bundling.result_state) == (TasksetRunPhase.COMPLETED, ResultState.FAILED)
    assert running.phase is TasksetRunPhase.EXPIRED and await harness.phases(running) == ["expired"]
    assert set(harness.reconciler.stuck) == {("default", task(running, 0))}


async def test_finished_tasks_are_bundled_after_the_execution_deadline(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    await harness.cycle()
    jobs.finish_all()
    await harness.cycle(at=AFTER_EXECUTION)
    await harness.cycle()
    assert not jobs.cancelled and any(r.name == f"{run.name}-bundle-1" for r in jobs.created)
    assert (await harness.run(run)).phase is TasksetRunPhase.RUNNING


# ---- the execution clock -------------------------------------------------------------------------------


async def test_the_first_grant_starts_the_clock_and_later_grants_keep_it(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 2, cap=1), name="run-a")
    assert run.started_at is None and run.execution_deadline_at == run.accepted_at + timedelta(hours=1)
    started = T0 + timedelta(minutes=20)  # queued until now: none of it comes out of the budget
    await harness.cycle(at=started)
    stored = await harness.run(run)
    assert stored.phase is TasksetRunPhase.RUNNING and stored.started_at == started
    assert stored.execution_deadline_at == started + timedelta(seconds=stored.timeout_seconds)
    assert stored.finalization_deadline_at == stored.execution_deadline_at + timedelta(hours=1)  # submit's grace
    assert jobs.created == [await harness.request(stored, 0)]  # the Job carries the started deadline
    jobs.finish(task(run, 0), "completed")
    await harness.cycle()
    await harness.cycle()
    later = await harness.run(run)
    assert jobs.created[1] == await harness.request(later, 1)
    assert (later.started_at, later.execution_deadline_at) == (started, stored.execution_deadline_at)


async def test_the_queue_timeout_expires_a_run_that_never_started(harness, harbor_plan, jobs):
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")  # its queue timeout ends at 13:00
    await harness.cycle(at=AFTER_EXECUTION)
    stored = await harness.run(run)
    assert not jobs.created and await harness.phases(run) == ["expired", "expired"]
    assert stored.phase is TasksetRunPhase.EXPIRED and stored.result_state is ResultState.NONE
    assert stored.started_at is None
    assert stored.error == "no task started within the queue timeout (1 h)"


async def test_a_failed_start_write_creates_nothing_and_the_next_cycle_starts_the_run(
    harness, harbor_plan, jobs, monkeypatch
):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    real = harness.store.start

    async def crash(*args, **kwargs):
        raise RuntimeError("crashed")

    monkeypatch.setattr(harness.store, "start", crash)
    await harness.cycle()
    assert not jobs.created and [m.task_job for m in await harness.members(run)] == [None]
    assert (await harness.run(run)).phase is TasksetRunPhase.ACCEPTED
    monkeypatch.setattr(harness.store, "start", real)
    await harness.cycle()
    stored = await harness.run(run)
    assert stored.started_at == T0 + CYCLE and jobs.created == [await harness.request(stored, 0)]


async def test_a_start_whose_reply_was_lost_is_never_written_again(harness, harbor_plan, jobs, monkeypatch):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")
    real = harness.store.start
    starts = []

    async def commit_then_lose_the_reply(run, **kwargs):
        starts.append(run.name)
        await real(run, **kwargs)
        raise httpx.ReadTimeout("reply lost")

    monkeypatch.setattr(harness.store, "start", commit_then_lose_the_reply)
    await harness.cycle()
    assert not jobs.created
    committed = await harness.run(run)
    assert committed.started_at == T0
    await harness.cycle()
    stored = await harness.run(run)
    assert starts == [run.name]  # the next cycle read started_at: no second start
    assert (stored.started_at, stored.execution_deadline_at) == (T0, committed.execution_deadline_at)
    assert jobs.created == [await harness.request(stored, 0)]


async def test_a_crash_after_the_start_creates_the_job_with_the_same_deadline(harness, harbor_plan, jobs, monkeypatch):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a")

    async def crash(*args, **kwargs):
        raise RuntimeError("crashed")

    monkeypatch.setattr(controller_module, "register_member", crash)
    await harness.cycle()
    started = await harness.run(run)
    assert started.phase is TasksetRunPhase.RUNNING and await harness.phases(run) == ["image_ready"]
    monkeypatch.undo()
    await harness.cycle()
    stored = await harness.run(run)
    assert stored.execution_deadline_at == started.execution_deadline_at
    assert jobs.created == [await harness.request(stored, 0)]


async def test_a_start_write_slower_than_the_budget_sends_no_job(harness, harbor_plan, jobs, monkeypatch):
    run = await harness.submit(sized(harbor_plan, 1), name="run-a", timeout_seconds=5)
    real = harness.store.start

    async def slow(run, **kwargs):
        written = await real(run, **kwargs)
        harness.clock.now += timedelta(seconds=10)  # the write outlasts the 5 s budget it just started
        return written

    monkeypatch.setattr(harness.store, "start", slow)
    await harness.cycle()
    assert not jobs.created and [m.task_job for m in await harness.members(run)] == [None]
    await harness.cycle()
    stored = await harness.run(run)
    assert stored.phase is TasksetRunPhase.EXPIRED
    assert stored.error == "1 of 1 tasks expired at the execution deadline, 1 of them still queued"


# ---- unreadable plans and isolation ---------------------------------------------------------------------


async def test_an_unreadable_plan_with_no_jobs_fails_in_one_pass_without_a_jobs_call(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    harness.plan_errors[run.name] = PlanDigestMismatchError("ref", "a", "b")
    await harness.cycle()
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.FAILED and run.error is not None and run.error.startswith("plan_unreadable")
    assert not jobs.calls


async def test_an_unreadable_plan_cancels_registered_jobs_and_never_an_intent(make_harness, harbor_plan, jobs):
    harness = make_harness(max_cluster_concurrent_jobs=3)
    run = await harness.submit(sized(harbor_plan, 4), name="run-a")
    jobs.script("create", None, None, timeout())
    await harness.cycle(at=T0)  # task 0, 1 registered; task 2 an intent; task 3 queued
    harness.plan_errors[run.name] = ValueError("invalid plan")
    report = await harness.cycle()
    assert sorted(jobs.cancelled) == [task(run, 0), task(run, 1)]
    assert await harness.phases(run) == ["registered", "registered", "failed", "failed"]
    assert report.used == 3  # two Jobs held by their members, the intent by the stuck set
    jobs.finish_all("cancelled")
    report = await harness.cycle()
    assert (await harness.run(run)).phase is TasksetRunPhase.FAILED
    assert report.used == 1  # the intent's entry, until create_grace proves no Job is coming
    await harness.cycle(at=T0 + GRACE)
    assert not harness.reconciler.stuck


async def test_an_unreadable_plan_never_cancels_or_counts_a_job_that_took_a_members_name(
    make_harness, harbor_plan, jobs
):
    harness = make_harness(max_cluster_concurrent_jobs=2)
    run = await harness.submit(sized(harbor_plan, 2), name="run-a")
    await harness.cycle()
    jobs.finish(task(run, 0), "error")
    jobs.replace(task(run, 0))
    harness.plan_errors[run.name] = ValueError("invalid plan")
    report = await harness.cycle()
    assert jobs.cancelled == [task(run, 1)]
    m0, m1 = await harness.members(run)
    assert m0.phase is MemberPhase.FAILED and m0.error == "plan_unreadable: invalid plan; job_missing"
    assert m1.phase is MemberPhase.REGISTERED and report.used == 1
    # Task 1's Job doesn't end by the finalization deadline: settled anyway, and counted until it ends...
    report = await harness.cycle(at=AFTER_FINALIZATION)
    assert await harness.phases(run) == ["failed", "expired"]
    assert (await harness.run(run)).phase is TasksetRunPhase.EXPIRED
    assert report.used == 1 and set(harness.reconciler.stuck) == {("default", task(run, 1))}
    # ...or until another Job shows under its name: then ours ended, and the newcomer is never counted.
    jobs.finish(task(run, 1), "cancelled")
    jobs.replace(task(run, 1))
    report = await harness.cycle()
    assert not harness.reconciler.stuck and report.used == 0
    assert jobs.cancelled == [task(run, 1)]


async def test_a_files_outage_never_fails_a_run_until_its_finalization_deadline(harness, harbor_plan, jobs):
    run = await harness.submit(harbor_plan, name="run-a")
    other = await harness.submit(harbor_plan, name="run-b")
    await harness.cycle()
    harness.plan_errors[run.name] = http_error(InternalServerError, 500, "files down")
    report = await harness.cycle()
    assert report.used == 4 and (await harness.run(run)).phase is TasksetRunPhase.RUNNING
    jobs.finish_all("completed", prefix=other.name)
    jobs.finish_all("cancelled", prefix=run.name)
    await harness.cycle(at=AFTER_FINALIZATION)
    run = await harness.run(run)
    assert run.phase is TasksetRunPhase.EXPIRED and (run.error or "").startswith("plan_unreadable")
    gone = await harness.submit(harbor_plan, name="run-c", accepted_at=AFTER_FINALIZATION)
    harness.plan_errors[gone.name] = http_error(NotFoundError, 404, "plan gone")
    await harness.cycle()
    assert (await harness.run(gone)).phase is TasksetRunPhase.FAILED  # a Files 404: unreadable at once


async def test_a_409_on_one_runs_start_skips_only_its_grants(harness, harbor_plan, jobs, monkeypatch):
    first = await harness.submit(sized(harbor_plan, 2), name="run-a")
    second = await harness.submit(sized(harbor_plan, 2), name="run-b")
    real = harness.store.start

    async def lose_first(run, **kwargs):
        if run.name == first.name:
            raise NemoEntityConflictError("a cancel landed")
        return await real(run, **kwargs)

    monkeypatch.setattr(harness.store, "start", lose_first)
    report = await harness.cycle()
    # The start comes before the first intent, so the run that lost it has no Job at all.
    assert not jobs.active(first.name) and len(jobs.active(second.name)) == 2
    assert report.created == 2
    assert (await harness.run(first)).started_at is None


async def test_a_member_listing_failure_skips_phase_2(harness, harbor_plan, jobs, monkeypatch):
    run = await harness.submit(harbor_plan, name="run-a")
    other = await harness.submit(harbor_plan, name="run-b")
    real = harness.store.list_all_members

    async def broken(r):
        if r.name == run.name:
            raise RuntimeError("entity store")
        return await real(r)

    monkeypatch.setattr(harness.store, "list_all_members", broken)
    report = await harness.cycle()
    assert report.phase2_skipped and not jobs.created
    assert len(await real(await harness.run(other))) == 2  # phase 1 still settled the other run


# ---- the real Jobs adapter ------------------------------------------------------------------------------

JOBS_PATH = "/apis/jobs/v2/workspaces/default/jobs/"


@pytest.fixture
async def conflicting_jobs(harbor_plan, harbor_run) -> AsyncIterator[tuple[JobsServiceRunJobs, FakeRunJobs, list[str]]]:
    """JobsServiceRunJobs over a Jobs API that answers every cancel with 409, as for a finished, rerun or pausing Job."""
    fake = FakeRunJobs()
    name = fake.put(await task_job_request(harbor_plan, harbor_run, 0, harbor_plan.task_environments[0]), "active").name
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert path.startswith(JOBS_PATH) and name in path, path
        requests.append(f"{request.method} {path}")
        if request.method == "POST":
            return httpx.Response(409, json={"detail": f"Cannot cancel job '{name}' from its current state."})
        return httpx.Response(200, json=fake.jobs[name].model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        async with AsyncNemoClient(base_url="http://platform.test", http_client=http_client) as client:
            yield JobsServiceRunJobs(client, client, "default"), fake, requests


async def test_a_cancel_conflict_on_a_running_job_is_raised_and_on_a_finished_one_ignored(conflicting_jobs):
    adapter, fake, requests = conflicting_jobs
    name = next(iter(fake.jobs))
    with pytest.raises(ConflictError):
        await adapter.cancel(name)  # e.g. a rerun replaced its step mid-cancel
    assert requests == [f"POST {JOBS_PATH}{name}/cancel", f"GET {JOBS_PATH}{name}"]
    fake.finish(name, "completed")
    await adapter.cancel(name)


async def test_the_controllers_store_never_retries_an_entity_store_call():
    from contextlib import asynccontextmanager as context

    from nemo_evals.taskset_runs.controller import EvaluatorRunController
    from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))

    @context
    async def provider(on_behalf_of):  # the platform's provider sets a retrying policy
        yield AsyncNemoClient(
            base_url="http://platform.test", http_client=http_client, retry=PLATFORM_DEFAULT_RETRY_POLICY
        )

    controller = EvaluatorRunController(provider, config=RunControllerConfig())
    await controller.on_startup()
    retry = controller.store.entity_client._client.retry  # ty: ignore[unresolved-attribute]
    assert retry is not None and retry.max_retries == 0
    await controller.on_shutdown()
    await http_client.aclose()


async def test_the_controllers_clients_never_retry_a_jobs_call(harbor_plan, harbor_run):
    from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY

    posts: list[str] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        posts.append(request.method)
        return httpx.Response(503, json={"detail": "busy"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    # The platform's client provider sets a retry policy that retries 5xx, timeouts and 409.
    client = AsyncNemoClient(
        base_url="http://platform.test", http_client=http_client, retry=PLATFORM_DEFAULT_RETRY_POLICY
    )
    request = await task_job_request(harbor_plan, harbor_run, 0, harbor_plan.task_environments[0])
    with pytest.raises(InternalServerError):
        await JobsServiceRunJobs(client, client, "default").create(request)
    assert posts == ["POST"]  # one attempt: a retry's refusal could be mistaken for the first POST's
    await http_client.aclose()
