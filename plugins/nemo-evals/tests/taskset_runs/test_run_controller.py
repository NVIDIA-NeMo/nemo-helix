# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One controller pass: notice finished task Jobs, start the bundle Job, save the run's result."""

from datetime import datetime, timedelta

import pytest
from nemo_evals.entities import AgentEvalResultEntity
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from nemo_evals.taskset_runs.controller import complete_run
from nemo_evals.taskset_runs.entities import ResultState, TasksetRunEntity, TasksetRunPhase
from nemo_evals.taskset_runs.specs import TasksetBundleSpec
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_evals.taskset_runs.task_jobs import task_job_request
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatus
from nhx_evals_sdk.agent_eval.results import AgentEvalSummary

IMAGE = PreparedImage(image_ref="registry.invalid/a@sha256:" + "a" * 64, preparation_key="k")
ENV = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="d", agent=IMAGE)
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

    async def read_summary(self, bundle_ref):
        return SUMMARY

    async def save_record(self, entity):
        # Like the entity store: a new record whose name exists is a conflict.
        if any(record.name == entity.name for record in self.records):
            raise NemoEntityConflictError(entity.name)
        self.records.append(entity)

    async def get_record(self, name):
        return next((record for record in self.records if record.name == name), None)


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
        fake.put(await task_job_request(harbor_plan, stored_run, index, ENV), "active")
    return fake


async def reread(entity_store, run):
    return await entity_store.get(TasksetRunEntity, name=run.name, workspace=run.workspace)


async def test_running_task_jobs_mark_the_run_running_once(stored_run, harbor_plan, jobs, store, entity_store):
    assert (await complete_run(stored_run, harbor_plan, jobs, store)).action == "running"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.RUNNING
    assert (await complete_run(run, harbor_plan, jobs, store)).action == "waiting"
    assert not jobs.created


async def test_finished_tasks_get_one_bundle_then_a_ready_result(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    first = await complete_run(stored_run, harbor_plan, jobs, store)
    again = await complete_run(stored_run, harbor_plan, jobs, store)
    assert first.action == again.action == "bundling"
    (bundle,) = jobs.created
    assert bundle.name == f"{stored_run.name}-bundle-1" and bundle.source == "evals.taskset-bundle"
    spec = TasksetBundleSpec.model_validate(bundle.spec)
    assert [a.task_index for a in spec.accepted] == [0, 1]
    assert spec.finished_at.utcoffset() is not None and spec.finished_at - spec.started_at > timedelta(0)
    assert bundle.platform_spec.steps[0].executor.container.image == harbor_plan.dispatcher_image

    jobs.finish(bundle.name, "completed")
    done = await complete_run(stored_run, harbor_plan, jobs, store)
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
    await complete_run(stored_run, harbor_plan, jobs, store)
    jobs.finish(f"{stored_run.name}-bundle-1", "completed")

    async def lost_update(*args, **kwargs):
        raise RuntimeError("entity store unavailable")

    real_set_state = store.set_state
    monkeypatch.setattr(store, "set_state", lost_update)
    with pytest.raises(RuntimeError):
        await complete_run(stored_run, harbor_plan, jobs, store)
    assert len(jobs.records) == 1  # saved before the update failed

    monkeypatch.setattr(store, "set_state", real_set_state)
    assert (await complete_run(stored_run, harbor_plan, jobs, store)).action == "completed"
    assert len(jobs.records) == 1
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.READY


async def test_a_record_from_another_job_is_not_adopted(stored_run, harbor_plan, jobs, store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    await complete_run(stored_run, harbor_plan, jobs, store)
    jobs.finish(f"{stored_run.name}-bundle-1", "completed")
    jobs.records.append(AgentEvalResultEntity.model_construct(name=stored_run.name, job_id="someone-else"))
    with pytest.raises(NemoEntityConflictError):
        await complete_run(stored_run, harbor_plan, jobs, store)


async def test_errored_tasks_are_left_out_of_the_bundle(stored_run, harbor_plan, jobs, store):
    names = sorted(jobs.jobs)
    jobs.finish(names[0], "completed")
    jobs.finish(names[1], "error")
    await complete_run(stored_run, harbor_plan, jobs, store)
    spec = TasksetBundleSpec.model_validate(jobs.created[0].spec)
    assert [a.task_index for a in spec.accepted] == [0]


async def test_no_completed_task_fails_the_run_without_a_bundle(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "error")
    assert (await complete_run(stored_run, harbor_plan, jobs, store)).action == "failed"
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.FAILED and run.result_state is ResultState.NONE
    assert not jobs.created


async def test_failed_bundle_is_retried_then_given_up(stored_run, harbor_plan, jobs, store, entity_store):
    for name in list(jobs.jobs):
        jobs.finish(name, "completed")
    for attempt in (1, 2, 3):
        await complete_run(stored_run, harbor_plan, jobs, store)
        jobs.finish(f"{stored_run.name}-bundle-{attempt}", "error")
    outcome = await complete_run(stored_run, harbor_plan, jobs, store)
    assert outcome.action == "bundle_failed" and len(jobs.created) == 3
    run = await reread(entity_store, stored_run)
    assert run.phase is TasksetRunPhase.COMPLETED and run.result_state is ResultState.FAILED
    assert "bundle-3" in (run.error or "")


async def test_missing_task_job_waits(stored_run, harbor_plan, jobs, store):
    jobs.jobs.pop(sorted(jobs.jobs)[0])
    assert (await complete_run(stored_run, harbor_plan, jobs, store)).action in {"waiting", "running"}
    assert not jobs.created


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
