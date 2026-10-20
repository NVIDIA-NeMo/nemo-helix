# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One controller pass: notice finished task Jobs, start the bundle Job, save the run's result, enforce deadlines."""

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.taskset_runs.controller import TERMINAL_JOB_STATUSES, JobsServiceRunJobs, PassOutcome, complete_run
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.specs import TasksetBundleSpec
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatus
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

# harbor_run is accepted 2026-10-15 12:00, execution deadline 13:00, finalization deadline 14:00.
BEFORE_DEADLINE = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
AFTER_EXECUTION = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)
AFTER_FINALIZATION = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
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


class FakeRunJobs:
    def __init__(self) -> None:
        self.jobs: dict[str, HelixJobResponse] = {}
        self.results: dict[tuple[str, str], str] = {}
        self.created: list[CreateHelixJobRequest] = []
        self.records: list[AgentEvalResultEntity] = []
        self.cancelled: list[str] = []
        self.stepless: set[str] = set()

    def put(self, request: CreateHelixJobRequest, status: str) -> None:
        assert request.name is not None
        self.jobs[request.name] = HelixJobResponse(
            id=f"id-{request.name}",
            attempt_id="a",
            name=request.name,
            workspace="default",
            source=request.source,
            spec=request.spec,
            platform_spec=request.platform_spec,
            fileset=f"default/job-fileset-{request.name}",
            status=HelixJobStatus(status),
        )

    def finish(self, name: str, status: str, *, result: bool = True) -> None:
        # The Jobs API returns naive timestamps, which once broke the bundle Job's duration.
        naive = datetime(2026, 10, 16, 17, 32, 49)
        self.jobs[name] = self.jobs[name].model_copy(update={"status": HelixJobStatus(status), "updated_at": naive})
        if result and status == "completed":
            self.results[(name, "agent-eval-results")] = f"default/job-fileset-{name}#results/a/agent-eval-results"

    async def get(self, name):
        return self.jobs.get(name)

    async def result_url(self, job, result):
        return self.results.get((job, result))

    async def create(self, request):
        if request.name in self.jobs:
            raise ConflictError.__new__(ConflictError)
        self.created.append(request)
        self.put(request, "created")

    async def has_steps(self, name):
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
        self.cancelled.append(name)
        if name in self.jobs and self.jobs[name].status.value not in TERMINAL_JOB_STATUSES:
            self.finish(name, "cancelled")


@pytest.fixture
async def stored_run(entity_store, harbor_run) -> TasksetRunEntity:
    entity_store.caller = "user:alice"
    return await entity_store.create(harbor_run)


@pytest.fixture
def store(entity_store, files_client) -> TasksetRunStore:
    return TasksetRunStore._with_entity_adapter(entity_store, files_client)


@pytest.fixture
async def jobs(harbor_plan, stored_run) -> FakeRunJobs:
    fake = FakeRunJobs()
    for index in range(len(harbor_plan.spec.tasks)):
        fake.put(await task_job_request(harbor_plan, stored_run, index, harbor_plan.task_environments[index]), "active")
    return fake


async def run_pass(run, plan, jobs, store, *, now=BEFORE_DEADLINE):
    return await complete_run(run, plan, jobs, store, now=now)


async def reread(entity_store, run):
    return await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)


async def test_running_task_jobs_mark_the_run_running_once(stored_run, harbor_plan, jobs, store, entity_store):
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action == "running"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.RUNNING
    assert (await run_pass(run, harbor_plan, jobs, store)).action == "waiting"
    assert not jobs.created


async def test_finished_tasks_get_one_bundle_then_a_ready_result(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    first = await run_pass(stored_run, harbor_plan, jobs, store)
    again = await run_pass(stored_run, harbor_plan, jobs, store)
    assert first.action == again.action == "bundling"
    (bundle,) = jobs.created
    assert bundle.name == f"{stored_run.name}-bundle-1" and bundle.source == "evals.taskset-bundle"
    spec = TasksetBundleSpec.model_validate(bundle.spec)
    assert [a.task_index for a in spec.accepted] == [0, 1]
    assert spec.finished_at.utcoffset() is not None and spec.finished_at - spec.started_at > timedelta(0)
    assert bundle.platform_spec.steps[0].executor.container.image == harbor_plan.dispatcher_image

    jobs.finish(bundle.name, "completed")
    done = await run_pass(stored_run, harbor_plan, jobs, store)
    assert done.action == "completed"
    (record,) = jobs.records
    assert record.name == stored_run.name and record.job_id == bundle.name
    assert record.bundle_ref.endswith(f"{bundle.name}#results/a/agent-eval-results")
    assert record.scores.score("harbor_reward.reward").mean == 0.5 and record.target_kind == "harbor"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.READY


async def test_a_saved_record_is_adopted_when_marking_the_run_done_failed(
    stored_run, harbor_plan, jobs, store, entity_store, monkeypatch
):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    jobs.finish(f"{stored_run.name}-bundle-1", "completed")

    async def lost_update(*args, **kwargs):
        raise RuntimeError("entity store unavailable")

    real_set_state = store.set_state
    monkeypatch.setattr(store, "set_state", lost_update)
    with pytest.raises(RuntimeError):
        await run_pass(stored_run, harbor_plan, jobs, store)
    assert len(jobs.records) == 1  # saved before the update failed

    monkeypatch.setattr(store, "set_state", real_set_state)
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action == "completed"
    assert len(jobs.records) == 1
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.READY


async def test_a_record_from_another_job_is_not_adopted(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    jobs.finish(f"{stored_run.name}-bundle-1", "completed")
    jobs.records.append(AgentEvalResultEntity.model_construct(name=stored_run.name, job_id="someone-else"))
    with pytest.raises(NemoEntityConflictError):
        await run_pass(stored_run, harbor_plan, jobs, store)


def tampered(request: CreateHelixJobRequest) -> CreateHelixJobRequest:
    """``request`` under its own name, but running another image: someone else's Job squatting the name."""
    platform_spec = request.platform_spec.model_copy(deep=True)
    executor = platform_spec.steps[0].executor
    assert isinstance(executor, CPUExecutionProvider)
    executor.container.image = "registry.invalid/evil@sha256:" + "e" * 64
    return request.model_copy(update={"platform_spec": platform_spec})


async def test_a_foreign_job_under_the_bundle_name_is_never_adopted(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    (bundle,) = jobs.created
    jobs.put(tampered(bundle), "created")
    jobs.finish(bundle.name, "completed")

    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    assert outcome.action == "bundling" and "runs something else" in outcome.detail
    assert not jobs.records and len(jobs.created) == 1
    assert (await reread(entity_store, stored_run)).result_state is not ResultState.READY


async def test_a_foreign_job_behind_a_create_conflict_is_never_adopted(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")

    async def squatted(request):
        jobs.put(tampered(request), "completed")  # created by someone else between our get and create
        raise ConflictError.__new__(ConflictError)

    jobs.create = squatted
    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    assert outcome.action == "bundling" and "runs something else" in outcome.detail
    assert not jobs.records


async def test_own_bundle_job_is_adopted_after_task_job_timestamps_move(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    (bundle,) = jobs.created
    for name in list(jobs.jobs):
        if name != bundle.name:
            jobs.jobs[name] = jobs.jobs[name].model_copy(update={"updated_at": datetime(2026, 10, 16, 18, 0, 0)})
    jobs.finish(bundle.name, "completed")
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action == "completed"
    assert len(jobs.created) == 1


async def test_errored_tasks_are_left_out_of_the_bundle(stored_run, harbor_plan, jobs, store):
    names = sorted(jobs.jobs)
    jobs.finish(names[0], "completed")
    jobs.finish(names[1], "error")
    await run_pass(stored_run, harbor_plan, jobs, store)
    spec = TasksetBundleSpec.model_validate(jobs.created[0].spec)
    assert [a.task_index for a in spec.accepted] == [0]


async def test_no_completed_task_fails_the_run_without_a_bundle(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "error")
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action == "failed"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.FAILED and run.result_state is ResultState.NONE
    assert not jobs.created


async def test_failed_bundle_is_retried_then_given_up(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    for attempt in (1, 2, 3):
        await run_pass(stored_run, harbor_plan, jobs, store)
        jobs.finish(f"{stored_run.name}-bundle-{attempt}", "error")
    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    assert outcome.action == "bundle_failed" and len(jobs.created) == 3
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.FAILED
    assert "bundle-3" in (run.error or "")


def saved_without_a_step(jobs: FakeRunJobs, name: str, *, at: datetime) -> None:
    """Jobs saved the Job and its attempt, then failed to save the first step: nothing ever schedules it."""
    jobs.stepless.add(name)
    jobs.jobs[name] = jobs.jobs[name].model_copy(update={"created_at": at})


async def test_a_bundle_job_saved_without_its_step_is_replaced_by_the_next_attempt(
    stored_run, harbor_plan, jobs, store, entity_store
):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    first = f"{stored_run.name}-bundle-1"
    saved_without_a_step(jobs, first, at=BEFORE_DEADLINE)

    # Still within STALLED_CREATE_SECONDS: the create may be in flight, so the Job is left alone.
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=BEFORE_DEADLINE + timedelta(seconds=59))
    assert outcome == PassOutcome("bundling", first) and len(jobs.created) == 1

    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=BEFORE_DEADLINE + timedelta(seconds=60))
    second = f"{stored_run.name}-bundle-2"
    assert outcome == PassOutcome("bundling", second) and [job.name for job in jobs.created] == [first, second]
    jobs.finish(second, "completed")
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=BEFORE_DEADLINE + timedelta(seconds=90))
    assert outcome.action == "completed" and jobs.records[0].job_id == second
    assert (await reread(entity_store, stored_run)).result_state is ResultState.READY


async def test_a_queued_bundle_job_with_its_step_is_never_replaced(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    first = f"{stored_run.name}-bundle-1"
    jobs.jobs[first] = jobs.jobs[first].model_copy(update={"created_at": BEFORE_DEADLINE - timedelta(minutes=20)})
    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    assert outcome == PassOutcome("bundling", first) and len(jobs.created) == 1


async def test_a_bundle_name_jobs_holds_but_cannot_show_is_skipped(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    first = f"{stored_run.name}-bundle-1"
    create = jobs.create

    async def saved_without_its_attempt(request):
        # Jobs saved the Job but not its attempt: GET answers 404, and every create of the name 409.
        if request.name == first:
            raise ConflictError.__new__(ConflictError)
        await create(request)

    jobs.create = saved_without_its_attempt
    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    second = f"{stored_run.name}-bundle-2"
    assert outcome == PassOutcome("bundling", second) and [job.name for job in jobs.created] == [second]
    # The next pass finds the second attempt again instead of starting a third.
    outcome = await run_pass(stored_run, harbor_plan, jobs, store)
    assert outcome == PassOutcome("bundling", second) and len(jobs.created) == 1


async def test_missing_task_job_waits(stored_run, harbor_plan, jobs, store):
    jobs.jobs.pop(sorted(jobs.jobs)[0])
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action in {"waiting", "running"}
    assert not jobs.created


async def test_a_foreign_job_under_a_task_name_is_never_bundled(stored_run, harbor_plan, jobs, store):
    first, second = sorted(jobs.jobs)
    jobs.put(tampered(await task_job_request(harbor_plan, stored_run, 0, harbor_plan.task_environments[0])), "active")
    jobs.finish(first, "completed")
    jobs.finish(second, "completed")
    assert (await run_pass(stored_run, harbor_plan, jobs, store)).action in {"waiting", "running"}
    assert not jobs.created


async def test_a_foreign_job_under_a_task_name_is_not_cancelled_at_the_execution_deadline(
    stored_run, harbor_plan, jobs, store, entity_store
):
    first, second = sorted(jobs.jobs)
    jobs.put(tampered(await task_job_request(harbor_plan, stored_run, 0, harbor_plan.task_environments[0])), "active")
    jobs.finish(second, "completed")
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)
    assert outcome.action == "expired" and not jobs.cancelled
    assert (await reread(entity_store, stored_run)).phase is TasksetRunPhase.EXPIRED


async def test_unfinished_task_jobs_are_cancelled_at_the_execution_deadline(
    stored_run, harbor_plan, jobs, store, entity_store
):
    first, second = sorted(jobs.jobs)
    jobs.finish(first, "completed")  # finished just before the deadline; simple policy: no partial result
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)
    assert outcome.action == "expired" and jobs.cancelled == [second]
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.EXPIRED and run.result_state is ResultState.NONE
    assert run.error == "execution deadline passed with 1 of 2 task Jobs unfinished"
    assert not jobs.created and not jobs.records


async def test_a_missing_task_job_does_not_keep_the_run_open_past_the_deadline(
    stored_run, harbor_plan, jobs, store, entity_store
):
    first, second = sorted(jobs.jobs)
    jobs.jobs.pop(first)
    jobs.finish(second, "completed")
    assert (await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)).action == "expired"
    assert not jobs.cancelled
    assert (await reread(entity_store, stored_run)).phase is TasksetRunPhase.EXPIRED


async def test_an_expiry_whose_state_write_failed_still_ends_the_run(
    stored_run, harbor_plan, jobs, store, entity_store, monkeypatch
):
    async def lost_update(*args, **kwargs):
        raise RuntimeError("entity store unavailable")

    real_set_state = store.set_state
    monkeypatch.setattr(store, "set_state", lost_update)
    with pytest.raises(RuntimeError):
        await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)
    assert sorted(jobs.cancelled) == sorted(jobs.jobs)  # cancelled before the write

    # Every task Job has ended (cancelled), so the next pass can't tell expiry from failure; M4's members can.
    monkeypatch.setattr(store, "set_state", real_set_state)
    assert (await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)).action == "failed"
    assert (await reread(entity_store, stored_run)).phase is TasksetRunPhase.FAILED


async def test_finished_task_jobs_are_bundled_after_the_execution_deadline(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    assert (await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_EXECUTION)).action == "bundling"
    assert not jobs.cancelled and len(jobs.created) == 1


async def test_a_bundle_still_running_at_the_finalization_deadline_fails_the_result(
    stored_run, harbor_plan, jobs, store, entity_store
):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    (bundle,) = jobs.created
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_FINALIZATION)
    assert outcome.action == "finalization_missed" and jobs.cancelled == [bundle.name]
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.FAILED
    assert run.error == "finalization deadline passed before the result was saved"
    assert not jobs.records and len(jobs.created) == 1  # no new attempt after the deadline


async def test_a_foreign_job_under_the_bundle_name_is_not_cancelled_at_the_finalization_deadline(
    stored_run, harbor_plan, jobs, store, entity_store
):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    (bundle,) = jobs.created
    jobs.put(tampered(bundle), "active")
    outcome = await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_FINALIZATION)
    assert outcome.action == "finalization_missed" and not jobs.cancelled
    assert (await reread(entity_store, stored_run)).result_state is ResultState.FAILED


async def test_a_result_saved_before_a_lost_update_is_adopted_after_the_finalization_deadline(
    stored_run, harbor_plan, jobs, store, entity_store, monkeypatch
):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await run_pass(stored_run, harbor_plan, jobs, store)
    jobs.finish(f"{stored_run.name}-bundle-1", "completed")

    async def lost_update(*args, **kwargs):
        raise RuntimeError("entity store unavailable")

    real_set_state = store.set_state
    monkeypatch.setattr(store, "set_state", lost_update)
    with pytest.raises(RuntimeError):
        await run_pass(stored_run, harbor_plan, jobs, store)

    monkeypatch.setattr(store, "set_state", real_set_state)
    assert (await run_pass(stored_run, harbor_plan, jobs, store, now=AFTER_FINALIZATION)).action == "completed"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.READY


JOBS_PATH = "/apis/jobs/v2/workspaces/default/jobs/"


@pytest.fixture
async def conflicting_jobs(jobs: FakeRunJobs) -> AsyncIterator[tuple[JobsServiceRunJobs, list[str]]]:
    """JobsServiceRunJobs over a Jobs API that answers every cancel with 409, as for a finished, rerun or pausing Job."""
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        name, _, action = path.removeprefix(JOBS_PATH).partition("/")
        assert path.startswith(JOBS_PATH) and name in jobs.jobs, path
        requests.append(f"{request.method} {path}")
        if (request.method, action) == ("POST", "cancel"):
            return httpx.Response(409, json={"detail": f"Cannot cancel job '{name}' from its current state."})
        assert (request.method, action) == ("GET", ""), requests[-1]
        return httpx.Response(200, json=jobs.jobs[name].model_dump(mode="json"))  # read now: finish() replaces it

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        async with AsyncNemoClient(base_url="http://platform.test", http_client=http_client) as client:
            yield JobsServiceRunJobs(client, "default"), requests


async def test_a_cancel_conflict_on_a_running_job_is_raised_so_the_next_pass_retries(jobs, conflicting_jobs):
    adapter, requests = conflicting_jobs
    name = next(iter(jobs.jobs))
    with pytest.raises(ConflictError):
        await adapter.cancel(name)  # e.g. a rerun replaced its step mid-cancel
    assert requests == [f"POST {JOBS_PATH}{name}/cancel", f"GET {JOBS_PATH}{name}"]


async def test_a_cancel_conflict_on_a_finished_job_is_ignored(jobs, conflicting_jobs):
    adapter, requests = conflicting_jobs
    name = next(iter(jobs.jobs))
    jobs.finish(name, "completed")
    await adapter.cancel(name)
    assert requests == [f"POST {JOBS_PATH}{name}/cancel", f"GET {JOBS_PATH}{name}"]


async def test_open_runs_are_listed_across_workspaces_oldest_first(store, entity_store, harbor_run):
    entity_store.caller = "user:alice"
    first = await entity_store.create(harbor_run)
    other = await entity_store.create(harbor_run.model_copy(update={"name": "run-other", "workspace": "team"}))
    done = await entity_store.create(
        harbor_run.model_copy(update={"name": "run-done", "phase": TasksetRunPhase.COMPLETED})
    )
    names = [run.name for run in await store.list_open_runs()]
    assert set(names) == {first.name, other.name} and done.name not in names


async def test_open_runs_are_read_past_the_first_page(store, entity_store, harbor_run):
    entity_store.caller = "user:alice"
    for i in range(5):
        await entity_store.create(harbor_run.model_copy(update={"name": f"run-{i:016x}"}))
    names = [run.name for run in await store.list_open_runs(page_size=2)]
    assert sorted(names) == [f"run-{i:016x}" for i in range(5)]


async def test_set_state_never_changes_a_terminal_run(store, stored_run, entity_store):
    failed = await store.set_state(stored_run, phase=TasksetRunPhase.FAILED, error="x")
    again = await store.set_state(failed, phase=TasksetRunPhase.COMPLETED)
    assert again.phase is TasksetRunPhase.FAILED
    assert (await reread(entity_store, stored_run)).error == "x"
