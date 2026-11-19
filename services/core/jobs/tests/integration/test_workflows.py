# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow behavior through real in-process Jobs, Files and Entity APIs."""

from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatusUpdateRequest
from nemo_helix_plugin.jobs.workflows import WorkflowInput, WorkflowMode, WorkflowNodeState, WorkflowSpec
from nhx.common.entities.client import SyncEntityClient
from nhx.core.files.service import FilesService
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJobAttempt, HelixJobStep
from nhx.core.jobs.service import JobsService
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration


@dataclass
class WorkflowHarness:
    context: ClientContext
    jobs: JobsClient
    files: FilesClient

    def fileset(self) -> FilesetOutput:
        return self.files.create_fileset(
            workspace="default", body=CreateFilesetRequest(name="input-" + uuid4().hex[:16])
        ).data()

    def create(
        self,
        dependencies: dict[str, list[str]],
        inputs: dict[str, list[str]] | None = None,
        mode: str = "continuous",
        storage: bool = False,
    ) -> HelixJobResponse:
        spec = HelixJobSpec(
            workflow=WorkflowSpec.model_validate({"mode": mode}),
            steps=[
                HelixJobStepSpec(
                    name=name,
                    depends_on=parents,
                    inputs=[WorkflowInput(fileset_id=identifier) for identifier in (inputs or {}).get(name, [])],
                    executor=TestConstants.TEST_EXECUTOR,
                    environment=[HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]
                    if storage
                    else [],
                )
                for name, parents in dependencies.items()
            ],
        )
        return self.jobs.create_job(
            workspace="default",
            body=CreateHelixJobRequest(
                name="workflow-" + uuid4().hex[:12],
                source="test",
                spec={},
                platform_spec=spec,
            ),
        ).data()

    def tick(self, job: HelixJobResponse) -> HelixJobStatusResponse:
        assert job.attempt_id is not None
        self.jobs.update_attempt_status_details(
            job.id, job.attempt_id, workspace="default", body={"workflow_reconcile": True}
        )
        return self.status(job)

    def status(self, job: HelixJobResponse) -> HelixJobStatusResponse:
        return self.jobs.get_job_status(workspace="default", name=job.id).data()

    def node_state(self, job: HelixJobResponse, name: str) -> WorkflowNodeState:
        assert job.attempt_id is not None
        store = SyncEntityClient(EntitiesClient.from_client(self.context.client))
        attempt = store.get_by_id(HelixJobAttempt, job.attempt_id)
        assert attempt.workflow_state is not None
        return attempt.workflow_state.nodes[name]

    def finish(self, job: HelixJobResponse, name: str, status: HelixJobStatus = HelixJobStatus.COMPLETED) -> None:
        state = self.status(job)
        assert state.workflow is not None and job.attempt_id is not None
        node = next(node for node in state.workflow.nodes if node.name == name)
        assert node.active_execution_id is not None
        for transition in (HelixJobStatus.ACTIVE, status):
            self.jobs.update_attempt_step_status(
                workspace="default",
                job=job.id,
                attempt=job.attempt_id,
                name=node.active_execution_id,
                body=HelixJobStatusUpdateRequest(status=transition),
            )

    def invalidate(self, job: HelixJobResponse, names: list[str], request_id: str | None = None) -> None:
        assert job.attempt_id is not None
        self.jobs.update_attempt_status_details(
            job.id,
            job.attempt_id,
            workspace="default",
            body={"workflow_invalidation": {"steps": names, "request_id": request_id or str(uuid4())}},
        )


@pytest.fixture(scope="module")
def workflow_harness() -> Iterator[WorkflowHarness]:
    with create_test_client(JobsService, FilesService, client_type=ClientContext) as context:
        yield WorkflowHarness(context, JobsClient.from_client(context.client), FilesClient.from_client(context.client))


def test_workflow_roots_join_and_noop_poll(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"left": [], "right": [], "join": ["left", "right"]})
    initial = harness.status(job)
    assert initial.steps == []
    first = harness.tick(job)
    assert first.workflow is not None
    assert len(first.steps) == 2
    assert first.workflow.nodes[2].scheduling_state == "waiting_dependencies"
    harness.finish(job, "left")
    assert len(harness.tick(job).steps) == 2
    harness.finish(job, "right")
    assert len(harness.tick(job).steps) == 3
    harness.finish(job, "join")
    clean = harness.tick(job)
    assert clean.status == HelixJobStatus.ACTIVE
    assert clean.workflow is not None and clean.workflow.waiting_for_changes
    assert all(node.freshness == "clean" for node in clean.workflow.nodes)
    revisions = [node.requested_run_revision for node in clean.workflow.nodes]
    again = harness.tick(job)
    assert len(again.steps) == 3
    assert again.workflow is not None
    assert [node.requested_run_revision for node in again.workflow.nodes] == revisions


def test_only_input_commit_triggers_reexecution(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    source = harness.fileset()
    unrelated = harness.fileset()
    job = harness.create({"work": []}, {"work": [source.id]})
    harness.tick(job)
    harness.finish(job, "work")
    before = harness.tick(job)
    assert before.workflow is not None
    revision = before.workflow.nodes[0].requested_run_revision
    harness.files.upload_file(workspace="default", name=source.name, path="data", content=b"changed")
    harness.files.commit_fileset(workspace="default", name=unrelated.name)
    same = harness.tick(job)
    assert len(same.steps) == 1
    assert same.workflow is not None and same.workflow.nodes[0].requested_run_revision == revision
    harness.files.commit_fileset(workspace="default", name=source.name)
    dirty = harness.tick(job)
    assert len(dirty.steps) == 2
    assert dirty.workflow is not None
    assert dirty.workflow.nodes[0].requested_run_revision != revision
    assert dirty.workflow.nodes[0].last_inputs_checked_at is not None


@pytest.mark.parametrize("trigger", ["manual", "input_commit"])
def test_continuous_failure_waits_then_recovers(workflow_harness: WorkflowHarness, trigger: str) -> None:
    harness = workflow_harness
    source = harness.fileset()
    job = harness.create({"root": [], "child": ["root"], "independent": []}, {"root": [source.id]})
    harness.tick(job)
    harness.finish(job, "root", HelixJobStatus.ERROR)
    harness.finish(job, "independent")
    failed = harness.tick(job)
    assert failed.status == HelixJobStatus.ACTIVE
    assert failed.workflow is not None and failed.workflow.degraded
    assert failed.workflow.waiting_for_changes
    assert len(failed.steps) == 2
    for _ in range(2):
        assert len(harness.tick(job).steps) == 2
    harness.invalidate(job, ["child"])
    assert len(harness.tick(job).steps) == 2
    if trigger == "manual":
        harness.invalidate(job, ["root"])
    else:
        harness.files.commit_fileset(workspace="default", name=source.name)
    assert len(harness.tick(job).steps) == 3
    harness.finish(job, "root")
    recovered = harness.tick(job)
    assert len(recovered.steps) == 4
    harness.finish(job, "child")
    assert harness.tick(job).workflow is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("depends_on", []),
        ("depends_on", None),
        ("inputs", []),
        ("inputs", None),
    ],
)
def test_workflow_fields_require_opt_in(workflow_harness: WorkflowHarness, field: str, value: object) -> None:
    harness = workflow_harness
    payload = {
        "source": "test",
        "spec": {},
        "platform_spec": {
            "steps": [
                {
                    "name": "work",
                    "executor": TestConstants.TEST_EXECUTOR.model_dump(),
                    field: value,
                }
            ]
        },
    }
    before_jobs = len(list(harness.jobs.list_jobs(workspace="default").items()))
    before_files = len(list(harness.files.list_filesets(workspace="default").items()))
    response = harness.context.test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=payload)
    assert response.status_code == 422
    assert len(list(harness.jobs.list_jobs(workspace="default").items())) == before_jobs
    assert len(list(harness.files.list_filesets(workspace="default").items())) == before_files


@pytest.mark.parametrize(
    "dependencies",
    [
        {"aa": ["aa"]},
        {"aa": ["bb"], "bb": ["aa"]},
        {"ok": [], "aa": ["bb"], "bb": ["cc"], "cc": ["aa"]},
        {"aa": ["missing"]},
        {"aa": [], "bb": ["aa", "aa"]},
    ],
)
def test_invalid_graph_is_rejected_before_creation(
    workflow_harness: WorkflowHarness, dependencies: dict[str, list[str]]
) -> None:
    harness = workflow_harness
    payload = {
        "source": "test",
        "spec": {},
        "platform_spec": {
            "workflow": {"mode": "continuous"},
            "steps": [
                {"name": name, "depends_on": parents, "executor": TestConstants.TEST_EXECUTOR.model_dump()}
                for name, parents in dependencies.items()
            ],
        },
    }
    before = len(list(harness.files.list_filesets(workspace="default").items()))
    response = harness.context.test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=payload)
    assert response.status_code == 422
    assert len(list(harness.files.list_filesets(workspace="default").items())) == before


def test_job_edits_do_not_edit_attempt_or_later_step(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"root": [], "later": ["root"]})
    assert job.attempt_id is not None
    before = harness.jobs.get_job_attempt(workspace="default", job=job.id, attempt=job.attempt_id).data()
    harness.tick(job)
    changed = job.platform_spec.model_copy(deep=True)
    changed.steps[1].config["version"] = "new-job-definition"
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}",
        json={"spec": {"new": True}, "platform_spec": changed.model_dump(mode="json")},
    )
    assert response.status_code == 200
    harness.finish(job, "root")
    harness.tick(job)
    steps = list(harness.jobs.list_attempt_steps(workspace="default", job=job.id, attempt=job.attempt_id).items())
    later = next(step.step_spec for step in steps if step.step_spec is not None and step.step_spec.name == "later")
    assert later.config.get("version") is None
    after = harness.jobs.get_job_attempt(workspace="default", job=job.id, attempt=job.attempt_id).data()
    assert after.platform_spec == before.platform_spec
    assert after.spec == before.spec


def test_invalid_update_preserves_job_and_attempt(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"aa": [], "bb": ["aa"]})
    bad = job.platform_spec.model_dump(mode="json")
    bad["steps"][0]["depends_on"] = ["bb"]
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}", json={"spec": {}, "platform_spec": bad}
    )
    assert response.status_code == 422
    current = harness.jobs.get_job(workspace="default", name=job.id).data()
    assert current.platform_spec == job.platform_spec


def test_pause_holds_queued_work_and_manual_request_until_resume(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    first = harness.tick(job)
    assert len(first.steps) == 1
    harness.jobs.pause_job(workspace="default", name=job.id)
    paused = harness.tick(job)
    assert paused.status == HelixJobStatus.PAUSED
    assert paused.workflow is not None and not paused.workflow.waiting_for_changes
    harness.invalidate(job, ["work"])
    assert harness.tick(job).status == HelixJobStatus.PAUSED
    assert len(harness.status(job).steps) == 1
    harness.jobs.resume_job(workspace="default", name=job.id)
    resumed = harness.tick(job)
    assert resumed.status == HelixJobStatus.ACTIVE
    assert len(resumed.steps) == 1 and resumed.steps[0].id == first.steps[0].id
    harness.finish(job, "work")
    harness.tick(job)
    assert len(harness.tick(job).steps) == 2


def test_draining_pause_waits_for_active_step_and_does_not_unlock_child(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"root": [], "child": ["root"]})
    state = harness.tick(job)
    assert job.attempt_id is not None and state.workflow is not None
    harness.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=state.steps[0].name,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    harness.jobs.pause_job(workspace="default", name=job.id)
    assert harness.tick(job).status == HelixJobStatus.PAUSING
    harness.finish(job, "root")
    drained = harness.tick(job)
    assert drained.status == HelixJobStatus.PAUSED
    assert len(drained.steps) == 1
    harness.jobs.resume_job(workspace="default", name=job.id)
    assert len(harness.tick(job).steps) == 2


def test_cancel_queued_work_is_terminal_and_never_reopens(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"root": [], "child": ["root"]})
    harness.tick(job)
    harness.jobs.cancel_job(workspace="default", name=job.id)
    ended = harness.tick(job)
    assert ended.status == HelixJobStatus.CANCELLED
    assert ended.steps[0].status == HelixJobStatus.CANCELLED
    assert harness.tick(job).status == HelixJobStatus.CANCELLED
    assert job.attempt_id is not None
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
        json={"workflow_invalidation": {"steps": ["root"], "request_id": "late"}},
    )
    assert response.status_code == 409


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_mode_completion_and_fresh_attempt(workflow_harness: WorkflowHarness, mode: str) -> None:
    harness = workflow_harness
    job = harness.create({"work": []}, mode=mode)
    harness.tick(job)
    harness.finish(job, "work")
    done = harness.tick(job)
    assert done.status == (HelixJobStatus.ACTIVE if mode == "continuous" else HelixJobStatus.COMPLETED)
    if mode == "run_to_completion":
        assert done.workflow is not None and not done.workflow.waiting_for_changes
        assert job.attempt_id is not None
        again = harness.jobs.rerun_job(workspace="default", job=job.id).data()
        assert again.attempt_id != job.attempt_id
        old = harness.context.test_client.get(
            f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/status",
            params={"attempt_id": job.attempt_id},
        )
        assert old.status_code == 200 and old.json()["status"] == "completed"
        assert (
            harness.jobs.get_job_attempt(workspace="default", job=job.id, attempt=job.attempt_id).data().status
            == HelixJobStatus.COMPLETED
        )


def test_once_failure_ends_after_drain(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"root": [], "child": ["root"]}, mode="run_to_completion")
    harness.tick(job)
    harness.finish(job, "root", HelixJobStatus.ERROR)
    result = harness.tick(job)
    if result.status == HelixJobStatus.ACTIVE:
        result = harness.tick(job)
    assert result.status == HelixJobStatus.ERROR
    assert len(result.steps) == 1


def test_input_commits_during_execution_coalesce_without_overlap(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    source = harness.fileset()
    job = harness.create({"work": []}, {"work": [source.id]})
    first = harness.tick(job)
    initial = harness.node_state(job, "work")
    assert initial.active_inputs == {source.id: source.commit_revision}
    assert job.attempt_id is not None
    harness.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=first.steps[0].name,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    for _ in range(3):
        harness.files.commit_fileset(workspace="default", name=source.name)
        pending = harness.tick(job)
        assert len(pending.steps) == 1
        assert pending.workflow is not None and pending.workflow.nodes[0].has_pending_run
        active = harness.node_state(job, "work")
        assert active.active_inputs == initial.active_inputs
        assert active.active_run_revision == initial.active_run_revision
        assert active.observed_inputs != active.active_inputs
    harness.finish(job, "work")
    settled = harness.tick(job)
    assert settled.workflow is not None and settled.workflow.nodes[0].freshness == "dirty"
    assert not settled.workflow.waiting_for_changes
    finished = harness.node_state(job, "work")
    assert finished.successful_inputs == initial.active_inputs
    assert finished.successful_revision == initial.active_run_revision
    assert finished.active_inputs is None and finished.active_run_revision is None
    next_run = harness.tick(job)
    assert len(next_run.steps) == 2
    latest = harness.node_state(job, "work")
    assert latest.active_inputs == latest.observed_inputs != initial.active_inputs
    assert latest.active_run_revision != initial.active_run_revision


def test_progress_and_usage_details_do_not_replace_controller_state(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    before = harness.tick(job)
    assert job.attempt_id is not None and before.workflow is not None
    harness.jobs.update_attempt_status_details(
        job.id, job.attempt_id, workspace="default", body={"epoch": 4, "percentage_done": 50}
    )
    after = harness.status(job)
    assert after.status_details["epoch"] == 4
    assert after.workflow == before.workflow
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
        json={"workflow_state": {"nodes": {}}},
    )
    assert response.status_code == 409
    assert harness.status(job).workflow == before.workflow


@pytest.mark.parametrize(
    "body",
    [
        {"workflow_invalidation": {"steps": ["unknown"], "request_id": "aa"}},
        {"workflow_invalidation": {"steps": [], "request_id": "aa"}},
        {"workflow_invalidation": {"steps": ["work", "work"], "request_id": "aa"}},
        {"workflow_invalidation": {"steps": ["work"], "request_id": ""}},
    ],
)
def test_invalid_manual_request_is_atomic(workflow_harness: WorkflowHarness, body: dict) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    before = harness.tick(job)
    assert job.attempt_id is not None
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details", json=body
    )
    assert response.status_code == 422
    assert harness.status(job).workflow == before.workflow


def test_missing_input_waits_without_spinning(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    source = harness.fileset()
    job = harness.create({"work": []}, {"work": [source.id]})
    harness.files.delete_fileset_by_id(workspace="default", fileset_id=source.id)
    for _ in range(3):
        state = harness.tick(job)
        assert state.status == HelixJobStatus.ACTIVE
        assert state.steps == []
        assert state.workflow is not None
        assert state.workflow.nodes[0].freshness == "unknown"
        assert state.workflow.nodes[0].last_poll_error


def test_descendants_only_rerun_for_their_own_input_commit(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    source, downstream_input = harness.fileset(), harness.fileset()
    job = harness.create({"root": [], "child": ["root"]}, {"root": [source.id], "child": [downstream_input.id]})
    harness.tick(job)
    harness.finish(job, "root")
    harness.tick(job)
    harness.finish(job, "child")
    harness.tick(job)
    harness.invalidate(job, ["root"])
    harness.tick(job)
    harness.finish(job, "root")
    unchanged = harness.tick(job)
    assert len(unchanged.steps) == 3
    assert unchanged.workflow is not None
    assert all(node.freshness == "clean" for node in unchanged.workflow.nodes)
    harness.files.commit_fileset(workspace="default", name=downstream_input.name)
    changed = harness.tick(job)
    assert len(changed.steps) == 4
    assert changed.workflow is not None
    assert changed.workflow.nodes[0].active_execution_id is None
    assert changed.workflow.nodes[1].active_execution_id is not None


def test_manual_request_during_run_and_repeated_signal(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    first = harness.tick(job)
    assert job.attempt_id is not None
    harness.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=first.steps[0].name,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    harness.invalidate(job, ["work"], "same-request")
    assert len(harness.tick(job).steps) == 1
    harness.finish(job, "work")
    harness.tick(job)
    assert len(harness.tick(job).steps) == 2
    harness.finish(job, "work")
    harness.tick(job)
    harness.invalidate(job, ["work"], "same-request")
    assert len(harness.tick(job).steps) == 2


def test_requests_for_different_nodes_are_not_lost(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"left": [], "right": []})
    harness.tick(job)
    harness.finish(job, "left")
    harness.finish(job, "right")
    harness.tick(job)
    harness.invalidate(job, ["left"], "left-request")
    harness.invalidate(job, ["right"], "right-request")
    assert len(harness.tick(job).steps) == 4


def test_pause_before_first_controller_tick(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    harness.jobs.pause_job(workspace="default", name=job.id)
    paused = harness.tick(job)
    assert paused.status == HelixJobStatus.PAUSED and paused.steps == []
    harness.jobs.resume_job(workspace="default", name=job.id)
    assert len(harness.tick(job).steps) == 1


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("continuous", HelixJobStatus.PAUSED),
        ("run_to_completion", HelixJobStatus.ERROR),
    ],
)
def test_failure_during_draining_pause(
    workflow_harness: WorkflowHarness,
    mode: str,
    expected: HelixJobStatus,
) -> None:
    harness = workflow_harness
    job = harness.create({"work": []}, mode=mode)
    first = harness.tick(job)
    assert job.attempt_id is not None
    harness.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=first.steps[0].name,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    harness.jobs.pause_job(workspace="default", name=job.id)
    harness.tick(job)
    harness.finish(job, "work", HelixJobStatus.ERROR)
    assert harness.tick(job).status == expected


def test_execution_ids_cannot_cross_attempts(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    first = harness.create({"work": []})
    second = harness.create({"work": []})
    state = harness.tick(first)
    assert second.attempt_id is not None
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{second.id}/attempts/{second.attempt_id}/steps/{state.steps[0].id}/status",
        json={"status": "active"},
    )
    assert response.status_code == 404


def test_once_rejects_manual_invalidation(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []}, mode="run_to_completion")
    assert job.attempt_id is not None
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
        json={"workflow_invalidation": {"steps": ["work"], "request_id": "request"}},
    )
    assert response.status_code == 409


def test_mixed_progress_and_trigger_are_rejected_atomically(workflow_harness: WorkflowHarness) -> None:
    harness = workflow_harness
    job = harness.create({"work": []})
    assert job.attempt_id is not None
    response = harness.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
        json={"epoch": 7, "workflow_invalidation": {"steps": ["work"], "request_id": "request"}},
    )
    assert response.status_code == 409
    assert "epoch" not in harness.status(job).status_details


@pytest.mark.parametrize("edit_job", [False, True])
def test_actual_controller_scheduler_and_backend_run_in_process(edit_job: bool) -> None:
    from nemo_helix_plugin.jobs.execution_profiles import E2EJobExecutionProfile
    from nhx.core.jobs.config import JobsServiceConfig
    from nhx.core.jobs.controllers.backends.registry import BackendRegistry
    from nhx.core.jobs.controllers.reconciler import JobReconciler
    from nhx.core.jobs.controllers.scheduler import JobScheduler

    config = JobsServiceConfig(
        executors=[E2EJobExecutionProfile(provider="cpu", profile="default")],
        enable_subprocess_executor=False,
    )
    with create_test_client(
        JobsService,
        FilesService,
        client_type=ClientContext,
        service_configs={JobsService: config},
    ) as context:
        harness = WorkflowHarness(
            context, JobsClient.from_client(context.client), FilesClient.from_client(context.client)
        )
        job = harness.create({"left": [], "right": [], "join": ["left", "right"]})
        if edit_job:
            from nemo_helix_plugin.jobs.types import UpdateHelixJobRequest

            harness.jobs.update_job(
                workspace="default",
                name=job.id,
                body=UpdateHelixJobRequest(
                    spec={"edited": True},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStepSpec(name="future", executor=TestConstants.TEST_EXECUTOR),
                        ]
                    ),
                ),
            )
        backends = BackendRegistry.from_config(nemo_client=context.client, profiles=config.executors)
        scheduler = JobScheduler(backends, context.client)
        reconciler = JobReconciler(backends, context.client)
        for _ in range(12):
            reconciler.step()
            scheduler.step()
            state = harness.status(job)
            assert state.workflow is not None
            if state.workflow.waiting_for_changes:
                break
        assert state.workflow is not None and state.workflow.waiting_for_changes
        assert all(node.freshness == "clean" for node in state.workflow.nodes)
        assert len(state.steps) == 3
        harness.invalidate(job, ["left"])
        for _ in range(12):
            reconciler.step()
            scheduler.step()
            state = harness.status(job)
            assert state.workflow is not None
            if state.workflow.waiting_for_changes and len(state.steps) == 4:
                break
        assert len(state.steps) == 4
        assert state.workflow is not None and all(node.freshness == "clean" for node in state.workflow.nodes)


def test_watch_reports_workflow_progress_while_status_stays_active(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_helix_plugin.client.response import NemoResponse
    from nemo_helix_plugin.jobs.watch_types import JobStatusEvent, JobWorkflowEvent

    harness = workflow_harness
    job = harness.create({"work": []}, mode="run_to_completion")
    first = harness.tick(job)
    original = harness.jobs.get_job_status
    polls = 0
    assert job.attempt_id is not None
    captured_attempt_id = job.attempt_id

    def get_status(
        self: JobsClient, *, workspace: str | None = None, name: str, attempt_id: str | None = None
    ) -> NemoResponse[HelixJobStatusResponse]:
        nonlocal polls
        polls += 1
        if polls in (2, 3):
            harness.jobs.update_attempt_step_status(
                workspace="default",
                job=job.id,
                attempt=captured_attempt_id,
                name=first.steps[0].id,
                body=HelixJobStatusUpdateRequest(
                    status=HelixJobStatus.ACTIVE if polls == 2 else HelixJobStatus.COMPLETED
                ),
            )
            harness.jobs.update_attempt_status_details(
                job.id, captured_attempt_id, workspace="default", body={"workflow_reconcile": True}
            )
        return original(workspace=workspace, name=name, attempt_id=attempt_id)

    monkeypatch.setattr(JobsClient, "get_job_status", get_status)
    events = list(harness.jobs.watch_job(job.id, workspace="default", include_logs=False, poll_interval=0))
    assert [event.status for event in events if isinstance(event, JobStatusEvent)] == ["active", "completed"]
    views = [event.workflow for event in events if isinstance(event, JobWorkflowEvent)]
    assert len(views) == 3
    assert views[0].nodes[0].requested_run_revision == views[1].nodes[0].requested_run_revision
    assert views[1].nodes[0].scheduling_state == "running"
    assert views[2].nodes[0].freshness == "clean"


@pytest.mark.parametrize("interruption", ["before_create", "after_create"])
@pytest.mark.parametrize("trigger", ["unchanged", "input_commit", "manual"])
def test_restart_recovers_execution_snapshot_from_workflow_state(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
    trigger: str,
) -> None:
    from nhx.common.entities.client import EntityBase, EntityClient
    from nhx.core.jobs.entities import HelixJobStep

    harness = workflow_harness
    source = harness.fileset()
    job = harness.create({"work": []}, {"work": [source.id]})
    original = EntityClient.create

    async def create[EntityT: EntityBase](self: EntityClient, entity: EntityT) -> EntityT:
        if isinstance(entity, HelixJobStep):
            if interruption == "after_create":
                await original(self, entity)
            raise RuntimeError("controller interrupted at execution creation")
        return await original(self, entity)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "create", create)
        with pytest.raises(RuntimeError, match="controller interrupted"):
            harness.tick(job)
    before = harness.status(job)
    assert len(before.steps) == int(interruption == "after_create")
    snapshot = harness.node_state(job, "work")
    assert snapshot.active_step_id is None
    assert snapshot.active_run_revision == snapshot.requested_run_revision
    assert snapshot.active_inputs == {source.id: source.commit_revision}
    if before.steps:
        assert job.attempt_id is not None
        contexts = list(
            harness.jobs.list_attempt_steps(workspace="default", job=job.id, attempt=job.attempt_id).items()
        )
        assert len(contexts) == 1 and contexts[0].workflow_held

    if trigger == "input_commit":
        harness.files.commit_fileset(workspace="default", name=source.name)
    elif trigger == "manual":
        harness.invalidate(job, ["work"])
    recovered = harness.tick(job)
    assert len(recovered.steps) == 1
    assert recovered.workflow is not None
    active = harness.node_state(job, "work")
    assert active.active_step_id == recovered.steps[0].id
    assert active.active_run_revision == snapshot.active_run_revision
    assert active.active_inputs == snapshot.active_inputs
    if before.steps:
        assert active.active_step_id == before.steps[0].id
    harness.finish(job, "work")
    completed = harness.tick(job)
    assert completed.workflow is not None
    if trigger != "unchanged":
        assert completed.workflow.nodes[0].freshness == "dirty"
        harness.tick(job)
        next_run = harness.node_state(job, "work")
        assert next_run.active_run_revision == next_run.requested_run_revision != snapshot.active_run_revision
        harness.finish(job, "work")
        completed = harness.tick(job)
    assert completed.workflow is not None and completed.workflow.nodes[0].freshness == "clean"
    finished = harness.node_state(job, "work")
    assert finished.successful_revision == finished.requested_run_revision
    assert finished.active_step_id is None and finished.active_run_revision is None and finished.active_inputs is None


def test_draining_pause_retention_uses_attempt_clock_without_paused_steps(
    workflow_harness: WorkflowHarness,
) -> None:
    from nemo_helix_plugin.jobs.types import JobControl

    harness = workflow_harness
    job = harness.create({"work": []}, storage=True)
    harness.tick(job)
    harness.finish(job, "work")
    harness.tick(job)
    harness.jobs.update_job_control(workspace="default", name=job.id, body=JobControl(pause_ttl_seconds=0))
    harness.jobs.pause_job(workspace="default", name=job.id)
    expired = harness.tick(job)
    assert expired.status == HelixJobStatus.CANCELLING
    assert expired.status_details["storage_reclaimed_at"]
    assert all(step.status == HelixJobStatus.COMPLETED for step in expired.steps)
    response = harness.context.client.http_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/resume")
    assert response.status_code == 409
    assert harness.tick(job).status == HelixJobStatus.CANCELLED


def test_active_continuous_failure_does_not_expire_shared_storage(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nhx.core.jobs.config import config

    monkeypatch.setattr(config.storage, "failed_storage_ttl_seconds", 0)
    harness = workflow_harness
    job = harness.create({"work": []}, storage=True)
    harness.tick(job)
    harness.finish(job, "work", HelixJobStatus.ERROR)
    for _ in range(3):
        active = harness.tick(job)
        assert active.status == HelixJobStatus.ACTIVE
        assert "storage_reclaimed_at" not in active.status_details


def test_once_failure_retention_expires_from_stopped_attempt(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nhx.core.jobs.config import config

    monkeypatch.setattr(config.storage, "failed_storage_ttl_seconds", 0)
    harness = workflow_harness
    job = harness.create({"work": []}, mode="run_to_completion", storage=True)
    harness.tick(job)
    harness.finish(job, "work", HelixJobStatus.ERROR)
    ended = harness.tick(job)
    assert ended.status == HelixJobStatus.ERROR
    assert ended.status_details["storage_reclaimed_at"]
    assert harness.tick(job).status == HelixJobStatus.ERROR


def test_poll_failure_preserves_observation_time_and_clears_after_recovery(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from nemo_helix_plugin.client.errors import NemoHTTPError
    from nemo_helix_plugin.files.client import AsyncFilesClient

    harness = workflow_harness
    source = harness.fileset()
    job = harness.create({"work": []}, {"work": [source.id]})
    harness.tick(job)
    harness.finish(job, "work")
    clean = harness.tick(job)
    assert clean.workflow is not None
    checked_at = clean.workflow.nodes[0].last_inputs_checked_at
    assert checked_at is not None

    async def unavailable(self: AsyncFilesClient, *, workspace: str | None = None, fileset_id: str) -> None:
        raise NemoHTTPError(httpx.Response(503, request=httpx.Request("GET", "http://test/fileset")))

    with monkeypatch.context() as patch:
        patch.setattr(AsyncFilesClient, "get_fileset_by_id", unavailable)
        failed = harness.tick(job)
    assert failed.workflow is not None
    observed = failed.workflow.nodes[0]
    assert observed.freshness == "unknown" and observed.last_poll_error
    assert observed.last_inputs_checked_at == checked_at
    assert observed.last_poll_attempt_at is not None and observed.last_poll_attempt_at > checked_at
    recovered = harness.tick(job)
    assert recovered.workflow is not None
    node = recovered.workflow.nodes[0]
    assert node.freshness == "clean" and node.last_poll_error is None
    assert node.reason != "input_unavailable"
    assert len(recovered.steps) == 1


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
@pytest.mark.parametrize("phase", [HelixJobStatus.CREATED, HelixJobStatus.QUEUED, HelixJobStatus.ACTIVE])
@pytest.mark.parametrize("cancellation", [HelixJobStatus.CANCELLING, HelixJobStatus.CANCELLED])
def test_node_cancellation_prunes_only_descendants(
    workflow_harness: WorkflowHarness, mode: str, phase: HelixJobStatus, cancellation: HelixJobStatus
) -> None:
    h = workflow_harness
    job = h.create(
        {"stop": [], "blocked": ["stop"], "leaf": ["blocked"], "independent": [], "tail": ["independent"]},
        mode=mode,
    )
    first = h.tick(job)
    assert job.attempt_id is not None and first.workflow is not None
    stop_id = next(node.active_execution_id for node in first.workflow.nodes if node.name == "stop")
    assert stop_id is not None
    for status in (phase, cancellation):
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=stop_id,
            body=HelixJobStatusUpdateRequest(status=status),
        )
    pruning = h.tick(job)
    assert pruning.status == HelixJobStatus.ACTIVE
    assert pruning.workflow is not None
    assert len(pruning.steps) == 2
    for name in ("blocked", "leaf"):
        node = next(node for node in pruning.workflow.nodes if node.name == name)
        assert node.last_outcome == "cancelled" and node.active_execution_id is None
    h.finish(job, "independent")
    continued = h.tick(job)
    assert continued.status == HelixJobStatus.ACTIVE and len(continued.steps) == 3
    h.finish(job, "tail")
    ended = h.tick(job)
    if cancellation == HelixJobStatus.CANCELLING:
        assert ended.status == HelixJobStatus.ACTIVE
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=stop_id,
            body=HelixJobStatusUpdateRequest(status=HelixJobStatus.CANCELLED),
        )
        ended = h.tick(job)
    expected = HelixJobStatus.ACTIVE if mode == "continuous" else HelixJobStatus.CANCELLED
    assert ended.status == expected
    assert all(step.status.is_terminal() for step in ended.steps)
    assert ended.workflow is not None
    assert {node.name: node.last_outcome for node in ended.workflow.nodes} == {
        "stop": "cancelled",
        "blocked": "cancelled",
        "leaf": "cancelled",
        "independent": "completed",
        "tail": "completed",
    }
    assert h.tick(job).status == expected
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=stop_id,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.CANCELLED),
    )
    assert h.tick(job).status == expected

    if mode == "continuous":
        assert h.tick(job).workflow is not None
        assert len(h.status(job).steps) == 3


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
@pytest.mark.parametrize("error_first", [False, True])
def test_node_cancellation_outcome_takes_precedence_over_sibling_error(
    workflow_harness: WorkflowHarness, mode: str, error_first: bool
) -> None:
    h = workflow_harness
    graph = {"failed": [], "stop": []} if error_first else {"stop": [], "failed": []}
    job = h.create(graph, mode=mode)
    h.tick(job)
    h.finish(job, "stop", HelixJobStatus.CANCELLED)
    h.finish(job, "failed", HelixJobStatus.ERROR)
    assert h.tick(job).status == (HelixJobStatus.ACTIVE if mode == "continuous" else HelixJobStatus.CANCELLED)


def test_continuous_cancelled_pass_drains_before_pending_inputs_run(workflow_harness: WorkflowHarness) -> None:
    h = workflow_harness
    source = h.fileset()
    job = h.create({"stop": [], "independent": []}, {"independent": [source.id]})
    h.tick(job)
    h.finish(job, "stop", HelixJobStatus.CANCELLED)
    h.finish(job, "independent")
    h.files.commit_fileset(workspace="default", name=source.name)
    ended = h.tick(job)
    assert ended.status == HelixJobStatus.ACTIVE
    assert ended.workflow is not None and ended.workflow.pass_number == 1
    independent = next(node for node in ended.workflow.nodes if node.name == "independent")
    assert independent.has_pending_run
    assert len(ended.steps) == 2
    next_pass = h.tick(job)
    assert next_pass.workflow is not None and next_pass.workflow.pass_number == 2
    assert len(next_pass.steps) == 3
    h.finish(job, "independent")
    waiting = h.tick(job)
    assert waiting.status == HelixJobStatus.ACTIVE
    assert waiting.workflow is not None and waiting.workflow.waiting_for_changes


@pytest.mark.parametrize("trigger", ["input_commit", "manual"])
def test_continuous_cancelled_node_retries_only_after_input_change_or_invalidation(
    workflow_harness: WorkflowHarness, trigger: str
) -> None:
    h = workflow_harness
    source = h.fileset()
    job = h.create({"work": [], "child": ["work"]}, {"work": [source.id]})
    h.tick(job)
    h.finish(job, "work", HelixJobStatus.CANCELLED)
    cancelled = h.tick(job)
    assert cancelled.status == HelixJobStatus.ACTIVE
    assert cancelled.workflow is not None and cancelled.workflow.waiting_for_changes and cancelled.workflow.degraded
    assert [node.last_outcome for node in cancelled.workflow.nodes] == ["cancelled", "cancelled"]
    assert len(h.tick(job).steps) == 1
    if trigger == "input_commit":
        h.files.commit_fileset(workspace="default", name=source.name)
    else:
        h.invalidate(job, ["work"])
    retry = h.tick(job)
    assert len(retry.steps) == 2
    assert retry.workflow is not None and retry.workflow.pass_number == 2
    assert h.node_state(job, "child").active_step_id is None
    h.finish(job, "work")
    assert len(h.tick(job).steps) == 3
    h.finish(job, "child")
    recovered = h.tick(job)
    assert recovered.status == HelixJobStatus.ACTIVE
    assert recovered.workflow is not None and recovered.workflow.waiting_for_changes
    assert not recovered.workflow.degraded
    assert all(node.last_outcome == "completed" for node in recovered.workflow.nodes)


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_explicit_cancel_with_new_inputs_still_cancels_node(workflow_harness: WorkflowHarness, mode: str) -> None:
    h = workflow_harness
    source = h.fileset()
    job = h.create({"work": [], "child": ["work"]}, {"work": [source.id]}, mode=mode)
    first = h.tick(job)
    assert job.attempt_id is not None
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=first.steps[0].id,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    h.files.commit_fileset(workspace="default", name=source.name)
    h.tick(job)
    h.finish(job, "work", HelixJobStatus.CANCELLED)
    ended = h.tick(job)
    assert ended.status == (HelixJobStatus.ACTIVE if mode == "continuous" else HelixJobStatus.CANCELLED)
    assert len(ended.steps) == 1
    assert len(h.tick(job).steps) == (2 if mode == "continuous" else 1)


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
@pytest.mark.parametrize("after_update", [False, True])
def test_restart_preserves_admission_and_current_pass(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch, mode: str, after_update: bool
) -> None:
    from nhx.common.entities.client import EntityBase, EntityClient

    h = workflow_harness
    source = h.fileset()
    job = h.create({"work": [], "child": ["work"]}, {"work": [source.id]}, mode=mode)
    first = h.tick(job)
    assert job.attempt_id is not None
    original = EntityClient.update

    async def update[EntityT: EntityBase](
        self: EntityClient, entity: EntityT, *, original_name: str | None = None
    ) -> EntityT:
        if isinstance(entity, HelixJobAttempt) and entity.id == job.attempt_id:
            assert entity.workflow_state is not None
            if entity.workflow_state.nodes["work"].scheduling_state == "admitted":
                if after_update:
                    await original(self, entity, original_name=original_name)
                raise RuntimeError("interrupted admission")
        return await original(self, entity, original_name=original_name)

    body: dict[str, object] = {"workflow_admission": {"step_id": first.steps[0].id}}
    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", update)
        with pytest.raises(RuntimeError, match="interrupted admission"):
            h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    h.files.commit_fileset(workspace="default", name=source.name)
    recovered = h.tick(job)
    assert len(recovered.steps) == 1 and recovered.steps[0].id == first.steps[0].id
    assert h.node_state(job, "work").active_inputs == {source.id: source.commit_revision}
    h.jobs.pause_job(workspace="default", name=job.id)
    paused = h.tick(job)
    assert paused.status == (HelixJobStatus.PAUSING if after_update else HelixJobStatus.PAUSED)
    if not after_update:
        h.jobs.resume_job(workspace="default", name=job.id)
        h.tick(job)
    h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    h.finish(job, "work")
    h.tick(job)
    if after_update:
        assert h.status(job).status == HelixJobStatus.PAUSED
        h.jobs.resume_job(workspace="default", name=job.id)
        h.tick(job)
    h.finish(job, "child")
    ended = h.tick(job)
    assert ended.status == (HelixJobStatus.ACTIVE if mode == "continuous" else HelixJobStatus.COMPLETED)
    assert len(ended.steps) == 2
    if mode == "continuous":
        assert len(h.tick(job).steps) == 3


def test_restart_after_failure_retires_queued_work_without_turning_into_cancel(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nhx.common.entities.client import EntityBase, EntityClient
    from nhx.core.jobs.entities import HelixJobStep

    h = workflow_harness
    job = h.create({"failed": [], "queued": []}, mode="run_to_completion")
    h.tick(job)
    h.finish(job, "failed", HelixJobStatus.ERROR)
    queued_id = h.node_state(job, "queued").active_step_id
    original = EntityClient.update

    async def update[EntityT: EntityBase](
        self: EntityClient, entity: EntityT, *, original_name: str | None = None
    ) -> EntityT:
        result = await original(self, entity, original_name=original_name)
        if isinstance(entity, HelixJobStep) and entity.id == queued_id and entity.status == HelixJobStatus.CANCELLED:
            raise RuntimeError("interrupted failure drain")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", update)
        with pytest.raises(RuntimeError, match="interrupted failure drain"):
            h.tick(job)
    assert h.tick(job).status == HelixJobStatus.ERROR


def test_cancelled_parent_keeps_dirty_descendant_pending_without_empty_passes(
    workflow_harness: WorkflowHarness,
) -> None:
    h = workflow_harness
    job = h.create({"root": [], "middle": ["root"], "leaf": ["middle"]})
    h.tick(job)
    for name in ("root", "middle", "leaf"):
        h.finish(job, name)
        h.tick(job)
    h.invalidate(job, ["root", "leaf"])
    h.tick(job)
    h.finish(job, "root", HelixJobStatus.CANCELLED)
    cancelled = h.tick(job)
    assert cancelled.workflow is not None and cancelled.workflow.pass_number == 2
    for _ in range(3):
        waiting = h.tick(job)
        assert waiting.workflow is not None and waiting.workflow.pass_number == 2
        assert waiting.workflow.waiting_for_changes
        assert len(waiting.steps) == 4
    h.invalidate(job, ["root"])
    h.tick(job)
    h.finish(job, "root")
    recovered = h.tick(job)
    assert len(recovered.steps) == 6
    h.finish(job, "leaf")
    clean = h.tick(job)
    assert clean.workflow is not None and not clean.workflow.degraded
    assert clean.workflow.waiting_for_changes


@pytest.mark.parametrize("outcome", [HelixJobStatus.ERROR, HelixJobStatus.CANCELLED])
@pytest.mark.parametrize("shape", ["chain", "deep_chain", "join"])
def test_unsuccessful_ancestor_blocks_cached_descendants(
    workflow_harness: WorkflowHarness, outcome: HelixJobStatus, shape: str
) -> None:
    h = workflow_harness
    graph = {"root": [], "middle": ["root"], "leaf": ["middle"], "independent": []}
    initial_order = ["root", "middle", "leaf", "independent"]
    if shape == "deep_chain":
        graph["bridge"] = ["middle"]
        graph["leaf"] = ["bridge"]
        initial_order.insert(2, "bridge")
    elif shape == "join":
        graph["other"] = []
        graph["middle"] = ["root", "other"]
        initial_order.insert(0, "other")
    job = h.create(dict(reversed(list(graph.items()))))
    h.tick(job)
    for name in initial_order:
        h.finish(job, name)
        h.tick(job)
    initial_count = len(h.status(job).steps)
    middle_success = h.node_state(job, "middle").successful_step_id
    h.invalidate(job, ["root", "leaf", "independent"])
    h.tick(job)
    h.finish(job, "root", outcome)
    blocked = h.tick(job)
    assert blocked.workflow is not None and blocked.workflow.pass_number == 2
    assert h.node_state(job, "leaf").active_step_id is None
    assert h.node_state(job, "leaf").blocked_by == graph["leaf"]
    assert len(blocked.steps) == initial_count + 2
    h.finish(job, "independent")
    for _ in range(3):
        waiting = h.tick(job)
        assert waiting.workflow is not None and waiting.workflow.waiting_for_changes
        assert waiting.workflow.pass_number == 2
        assert len(waiting.steps) == initial_count + 2
    h.invalidate(job, ["root"])
    h.tick(job)
    h.finish(job, "root")
    recovered = h.tick(job)
    assert recovered.workflow is not None and recovered.workflow.pass_number == 3
    assert h.node_state(job, "middle").successful_step_id == middle_success
    assert h.node_state(job, "leaf").active_step_id is not None
    assert len(recovered.steps) == initial_count + 4
    h.finish(job, "leaf")
    clean = h.tick(job)
    assert clean.workflow is not None and clean.workflow.waiting_for_changes
    assert not clean.workflow.degraded


def test_current_pass_success_releases_children_with_newer_inputs_pending(workflow_harness: WorkflowHarness) -> None:
    h = workflow_harness
    job = h.create({"leaf": ["middle"], "middle": ["root"], "root": [], "independent": []})
    h.tick(job)
    h.finish(job, "root")
    h.invalidate(job, ["root"])
    released = h.tick(job)
    assert released.workflow is not None and released.workflow.pass_number == 1
    assert h.node_state(job, "root").freshness == "dirty"
    assert h.node_state(job, "middle").active_step_id is not None
    h.finish(job, "middle")
    assert len(h.tick(job).steps) == 4
    h.finish(job, "leaf")
    h.finish(job, "independent")
    drained = h.tick(job)
    assert drained.workflow is not None and drained.workflow.pass_number == 1
    next_pass = h.tick(job)
    assert next_pass.workflow is not None and next_pass.workflow.pass_number == 2
    assert len(next_pass.steps) == 5
    h.finish(job, "root")
    clean = h.tick(job)
    assert clean.workflow is not None and clean.workflow.waiting_for_changes
    assert len(clean.steps) == 5


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
@pytest.mark.parametrize("phase", ["before_first_tick", "queued", "active_pending", "active_draining"])
def test_pause_cannot_override_workflow_cancellation(workflow_harness: WorkflowHarness, mode: str, phase: str) -> None:
    h = workflow_harness
    job = h.create({"root": [], "child": ["root"]}, mode=mode)
    assert job.attempt_id is not None
    execution_id = None
    if phase != "before_first_tick":
        initial = h.tick(job)
        execution_id = initial.steps[0].id
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=execution_id,
            body=HelixJobStatusUpdateRequest(
                status=HelixJobStatus.QUEUED if phase == "queued" else HelixJobStatus.ACTIVE
            ),
        )
    h.jobs.cancel_job(workspace="default", name=job.id)
    if phase == "active_draining":
        assert h.tick(job).status == HelixJobStatus.CANCELLING
    store = SyncEntityClient(EntitiesClient.from_client(h.context.client))
    before = store.get_by_id(HelixJobAttempt, job.attempt_id)
    response = h.context.test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/pause")
    assert response.status_code == 409
    after = store.get_by_id(HelixJobAttempt, job.attempt_id)
    assert after.model_dump() == before.model_dump()
    assert h.status(job).status == HelixJobStatus.CANCELLING
    assert h.context.test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/resume").status_code == 409
    # Another cancel remains valid while cancellation is draining.
    h.jobs.cancel_job(workspace="default", name=job.id)
    draining = h.tick(job)
    if phase.startswith("active"):
        assert draining.status == HelixJobStatus.CANCELLING and execution_id is not None
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=execution_id,
            body=HelixJobStatusUpdateRequest(status=HelixJobStatus.CANCELLED),
        )
    ended = h.tick(job)
    assert ended.status == HelixJobStatus.CANCELLED
    assert len(ended.steps) == (0 if phase == "before_first_tick" else 1)
    assert h.tick(job).status == HelixJobStatus.CANCELLED
    assert h.context.test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/resume").status_code == 409


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
@pytest.mark.parametrize("reverse_roots", [False, True])
@pytest.mark.parametrize(
    "phase", ["error_then_cancel", "cancel_then_error", "cancelling_then_error", "failing_then_cancel", "all_finished"]
)
def test_step_failure_cannot_override_workflow_cancellation(
    workflow_harness: WorkflowHarness, mode: str, reverse_roots: bool, phase: str
) -> None:
    h = workflow_harness
    roots: dict[str, list[str]] = {"failure": [], "other": []}
    if reverse_roots:
        roots = dict(reversed(list(roots.items())))
    job = h.create({**roots, "child": ["failure", "other"]}, mode=mode)
    assert job.attempt_id is not None
    h.tick(job)
    failure_id = h.node_state(job, "failure").active_step_id
    other_id = h.node_state(job, "other").active_step_id
    assert failure_id is not None and other_id is not None
    for execution_id in (failure_id, other_id):
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=execution_id,
            body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
        )
    if phase in ("cancel_then_error", "cancelling_then_error"):
        h.jobs.cancel_job(workspace="default", name=job.id)
        if phase == "cancelling_then_error":
            assert h.tick(job).status == HelixJobStatus.CANCELLING
    diagnostics = {"message": "synthetic step failure", "code": "test_failure"}
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=failure_id,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ERROR, error_details=diagnostics),
    )
    if phase == "failing_then_cancel":
        failed = h.tick(job)
        assert failed.workflow is not None
        assert failed.workflow.control == ("failing" if mode == "run_to_completion" else "running")
    if phase == "all_finished":
        h.finish(job, "other")
    if phase not in ("cancel_then_error", "cancelling_then_error"):
        h.jobs.cancel_job(workspace="default", name=job.id)
    draining = h.tick(job)
    assert draining.workflow is not None and draining.workflow.control == "cancelling"
    assert h.node_state(job, "failure").last_outcome == "error"
    assert h.node_state(job, "child").active_step_id is None
    store = SyncEntityClient(EntitiesClient.from_client(h.context.client))
    error_step = store.get_by_id(HelixJobStep, failure_id)
    assert error_step.status == HelixJobStatus.ERROR and error_step.error_details == diagnostics
    if phase != "all_finished":
        assert draining.status == HelixJobStatus.CANCELLING
        assert store.get_by_id(HelixJobStep, other_id).status == HelixJobStatus.CANCELLING
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=other_id,
            body=HelixJobStatusUpdateRequest(status=HelixJobStatus.CANCELLED),
        )
    ended = h.tick(job)
    assert ended.status == HelixJobStatus.CANCELLED and len(ended.steps) == 2
    assert h.tick(job).status == HelixJobStatus.CANCELLED
    preserved = store.get_by_id(HelixJobStep, failure_id)
    assert preserved.status == HelixJobStatus.ERROR and preserved.error_details == diagnostics


@pytest.mark.parametrize(
    "mode,trigger", [("continuous", "commit"), ("continuous", "manual"), ("run_to_completion", "commit")]
)
def test_pass_keeps_downstream_inputs_fixed(workflow_harness: WorkflowHarness, mode: str, trigger: str) -> None:
    h = workflow_harness
    source, downstream = h.fileset(), h.fileset()
    job = h.create({"root": [], "child": ["root"]}, {"root": [source.id], "child": [downstream.id]}, mode=mode)
    initial = h.tick(job)
    root = h.node_state(job, "root")
    child_revision = h.node_state(job, "child").requested_run_revision
    assert job.attempt_id is not None
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=initial.steps[0].id,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
    )
    if trigger == "commit":
        h.files.commit_fileset(workspace="default", name=source.name)
        h.files.commit_fileset(workspace="default", name=downstream.name)
    else:
        h.invalidate(job, ["root", "child"])
    h.tick(job)
    h.finish(job, "root")
    released = h.tick(job)
    child = h.node_state(job, "child")
    assert len(released.steps) == 2
    assert child.active_run_revision == child_revision
    assert child.active_inputs == {downstream.id: downstream.commit_revision}
    assert h.node_state(job, "root").successful_revision == root.active_run_revision
    h.finish(job, "child")
    drained = h.tick(job)
    assert drained.workflow is not None and drained.workflow.pass_number == 1
    assert all(node.has_pending_run for node in drained.workflow.nodes)
    if mode == "run_to_completion":
        assert drained.status == HelixJobStatus.COMPLETED
        assert len(h.tick(job).steps) == 2
    else:
        again = h.tick(job)
        assert again.workflow is not None and again.workflow.pass_number == 2 and len(again.steps) == 3
        h.finish(job, "root")
        h.tick(job)
        latest = h.node_state(job, "child")
        assert latest.active_run_revision == latest.requested_run_revision != child_revision
        assert latest.active_inputs == latest.observed_inputs
        h.finish(job, "child")
        clean = h.tick(job)
        assert clean.workflow is not None and clean.workflow.waiting_for_changes


@pytest.mark.parametrize("trigger", ["commit", "manual"])
def test_queued_execution_is_fixed_for_its_pass(workflow_harness: WorkflowHarness, trigger: str) -> None:
    h = workflow_harness
    source = h.fileset()
    job = h.create({"work": []}, {"work": [source.id]})
    first = h.tick(job)
    before = h.node_state(job, "work")
    if trigger == "commit":
        h.files.commit_fileset(workspace="default", name=source.name)
    else:
        h.invalidate(job, ["work"])
    pending = h.tick(job)
    current = h.node_state(job, "work")
    assert len(pending.steps) == 1 and pending.steps[0].id == first.steps[0].id
    assert current.active_run_revision == before.active_run_revision
    assert current.active_inputs == before.active_inputs
    assert current.requested_run_revision != current.active_run_revision


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_admitted_work_drains_before_attempt_pauses(workflow_harness: WorkflowHarness, mode: str) -> None:
    h = workflow_harness
    job = h.create({"root": [], "child": ["root"]}, mode=mode)
    first = h.tick(job)
    assert job.attempt_id is not None
    body: dict[str, object] = {"workflow_admission": {"step_id": first.steps[0].id}}
    h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    assert h.node_state(job, "root").scheduling_state == "admitted"
    h.jobs.pause_job(workspace="default", name=job.id)
    draining = h.tick(job)
    assert draining.status == HelixJobStatus.PAUSING
    h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    h.finish(job, "root")
    paused = h.tick(job)
    assert paused.status == HelixJobStatus.PAUSED and len(paused.steps) == 1
    h.jobs.resume_job(workspace="default", name=job.id)
    resumed = h.tick(job)
    assert resumed.workflow is not None and resumed.workflow.pass_number == 1
    assert len(resumed.steps) == 2


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_pause_rejects_unadmitted_work(workflow_harness: WorkflowHarness, mode: str) -> None:
    h = workflow_harness
    job = h.create({"work": []}, mode=mode)
    first = h.tick(job)
    h.jobs.pause_job(workspace="default", name=job.id)
    assert h.tick(job).status == HelixJobStatus.PAUSED
    response = h.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
        json={"workflow_admission": {"step_id": first.steps[0].id}},
    )
    assert response.status_code == 409
    assert h.status(job).status == HelixJobStatus.PAUSED


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_stale_scheduler_cannot_dispatch_after_attempt_pause(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from unittest.mock import Mock

    from nemo_helix_plugin.jobs.execution_profiles import E2EJobExecutionProfile
    from nhx.core.jobs.controllers.backends.exceptions import SchedulingDeferred
    from nhx.core.jobs.controllers.backends.registry import BackendRegistry
    from nhx.core.jobs.controllers.scheduler import JobScheduler

    h = workflow_harness
    job = h.create({"work": []}, mode=mode)
    h.tick(job)
    profile = E2EJobExecutionProfile(provider="cpu", profile="default")
    registry = BackendRegistry.from_config(nemo_client=h.context.client, profiles=[profile])
    scheduler = JobScheduler(registry, h.context.client)
    step = next(step for step in scheduler.get_steps_for_scheduling() if step.job_id == job.id)
    backend = registry.get_backend(provider="cpu", profile="default")
    launch = Mock()
    monkeypatch.setattr(backend, "schedule", launch)
    h.jobs.pause_job(workspace="default", name=job.id)
    assert h.tick(job).status == HelixJobStatus.PAUSED
    with pytest.raises(SchedulingDeferred):
        scheduler.schedule_step(step)
    launch.assert_not_called()


@pytest.mark.parametrize("winner", ["pause", "admission"])
def test_pause_and_admission_share_attempt_version(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch, winner: str
) -> None:
    from nhx.common.entities.client import EntityBase, EntityClient
    from nhx.core.jobs.app.workflow import WORKFLOW_CONTROL

    h = workflow_harness
    job = h.create({"work": []})
    first = h.tick(job)
    assert job.attempt_id is not None
    original = EntityClient.update
    intervened = False

    async def update[EntityT: EntityBase](
        self: EntityClient, entity: EntityT, *, original_name: str | None = None
    ) -> EntityT:
        nonlocal intervened
        if isinstance(entity, HelixJobAttempt) and entity.id == job.attempt_id and not intervened:
            assert entity.workflow_state is not None
            pause_write = entity.status == HelixJobStatus.PAUSING
            admission_write = entity.workflow_state.nodes["work"].scheduling_state == "admitted"
            if (winner == "pause" and admission_write) or (winner == "admission" and pause_write):
                intervened = True
                competing = await self.get_by_id(HelixJobAttempt, entity.id)
                assert competing.workflow_state is not None
                if winner == "pause":
                    competing.status = HelixJobStatus.PAUSING
                    competing.status_details = {**competing.status_details, WORKFLOW_CONTROL: "pause"}
                else:
                    competing.workflow_state.nodes["work"].scheduling_state = "admitted"
                await original(self, competing)
        return await original(self, entity, original_name=original_name)

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", update)
        if winner == "pause":
            response = h.context.test_client.patch(
                f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details",
                json={"workflow_admission": {"step_id": first.steps[0].id}},
            )
        else:
            response = h.context.test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/pause")
    assert intervened and response.status_code == 409
    if winner == "admission":
        h.jobs.pause_job(workspace="default", name=job.id)
        assert h.tick(job).status == HelixJobStatus.PAUSING
        h.finish(job, "work")
    assert h.tick(job).status == HelixJobStatus.PAUSED


def test_backend_deferral_keeps_admitted_work_drainable(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from nemo_helix_plugin.jobs.execution_profiles import E2EJobExecutionProfile
    from nhx.core.jobs.controllers.backends.exceptions import SchedulingDeferred
    from nhx.core.jobs.controllers.backends.registry import BackendRegistry
    from nhx.core.jobs.controllers.reconciler import JobReconciler
    from nhx.core.jobs.controllers.scheduler import JobScheduler

    h = workflow_harness
    job = h.create({"work": [], "child": ["work"]})
    first = h.tick(job)
    registry = BackendRegistry.from_config(
        nemo_client=h.context.client, profiles=[E2EJobExecutionProfile(provider="cpu", profile="default")]
    )
    scheduler = JobScheduler(registry, h.context.client)
    step = next(step for step in scheduler.get_steps_for_scheduling() if step.job_id == job.id)
    backend = registry.get_backend(provider="cpu", profile="default")
    with monkeypatch.context() as patch:
        patch.setattr(backend, "schedule", Mock(side_effect=SchedulingDeferred("backend capacity full")))
        with pytest.raises(SchedulingDeferred):
            scheduler.schedule_step(step)
    assert h.node_state(job, "work").scheduling_state == "admitted"
    h.jobs.pause_job(workspace="default", name=job.id)
    assert h.tick(job).status == HelixJobStatus.PAUSING
    update = scheduler.schedule_step(step)
    assert job.attempt_id is not None
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=step.id,
        body=HelixJobStatusUpdateRequest(status=update.status, status_details=update.status_details),
    )
    reconciler = JobReconciler(registry, h.context.client)
    for _ in range(8):
        paused = h.tick(job)
        if paused.status == HelixJobStatus.PAUSED:
            break
        reconciler.step()
    assert paused.status == HelixJobStatus.PAUSED
    assert len(paused.steps) == 1 and paused.steps[0].id == first.steps[0].id


@pytest.mark.parametrize(
    "body",
    [
        {"workflow_admission": {}},
        {"workflow_admission": {"step_id": ""}},
        {"workflow_admission": {"step_id": "x", "extra": True}},
    ],
)
def test_workflow_admission_validates_request(
    workflow_harness: WorkflowHarness, body: dict[str, dict[str, str | bool]]
) -> None:
    h = workflow_harness
    job = h.create({"work": []})
    h.tick(job)
    response = h.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job.id}/attempts/{job.attempt_id}/status-details", json=body
    )
    assert response.status_code == 422
    assert h.node_state(job, "work").scheduling_state != "admitted"


def test_workflow_admission_rejects_foreign_execution(workflow_harness: WorkflowHarness) -> None:
    h = workflow_harness
    left, right = h.create({"work": []}), h.create({"work": []})
    h.tick(left)
    other = h.tick(right)
    response = h.context.test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{left.id}/attempts/{left.attempt_id}/status-details",
        json={"workflow_admission": {"step_id": other.steps[0].id}},
    )
    assert response.status_code == 409


def test_new_pass_resets_nodes_before_scheduling_and_recovers(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nhx.common.entities.client import EntityBase, EntityClient

    h = workflow_harness
    job = h.create({"root": [], "child": ["root"]})
    h.tick(job)
    for name in ("root", "child"):
        h.finish(job, name)
        h.tick(job)
    child_success = h.node_state(job, "child").successful_step_id
    h.invalidate(job, ["root"])
    original = EntityClient.update

    async def update[EntityT: EntityBase](
        self: EntityClient, entity: EntityT, *, original_name: str | None = None
    ) -> EntityT:
        saved = await original(self, entity, original_name=original_name)
        if isinstance(entity, HelixJobAttempt) and entity.id == job.attempt_id:
            assert entity.workflow_state is not None
            if entity.workflow_state.pass_number == 2:
                raise RuntimeError("interrupted after pass reset")
        return saved

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "update", update)
        with pytest.raises(RuntimeError, match="interrupted after pass reset"):
            h.tick(job)
    for name in ("root", "child"):
        reset = h.node_state(job, name)
        assert reset.decision_pass == 0
        assert reset.scheduling_state == "waiting_dependencies"
        assert reset.reason is None and reset.blocked_by == []
        assert reset.active_step_id is None
    assert h.node_state(job, "child").successful_step_id == child_success
    h.invalidate(job, ["child"])
    resumed = h.tick(job)
    assert resumed.workflow is not None and resumed.workflow.pass_number == 2
    assert len(resumed.steps) == 3
    assert h.node_state(job, "child").scheduling_state == "waiting_dependencies"
    h.finish(job, "root")
    drained = h.tick(job)
    assert len(drained.steps) == 3
    assert h.node_state(job, "child").successful_step_id == child_success
    assert h.node_state(job, "child").decision_pass == 2
    third = h.tick(job)
    assert third.workflow is not None and third.workflow.pass_number == 3
    assert len(third.steps) == 4 and h.node_state(job, "child").active_step_id is not None


def test_new_pass_preserves_unchanged_failure_suppression(workflow_harness: WorkflowHarness) -> None:
    h = workflow_harness
    job = h.create({"bad": [], "good": [], "child": ["bad"]})
    h.tick(job)
    h.finish(job, "bad", HelixJobStatus.ERROR)
    h.finish(job, "good")
    h.tick(job)
    bad_execution = h.node_state(job, "bad").latest_step_id
    h.invalidate(job, ["good"])
    second = h.tick(job)
    assert second.workflow is not None and second.workflow.pass_number == 2 and second.workflow.degraded
    assert len(second.steps) == 3
    bad = h.node_state(job, "bad")
    assert bad.active_step_id is None and bad.latest_step_id == bad_execution
    assert bad.last_outcome == "error" and bad.reason == "awaiting_invalidation"
    assert h.node_state(job, "child").blocked_by == ["bad"]
    h.finish(job, "good")
    waiting = h.tick(job)
    assert waiting.workflow is not None and waiting.workflow.waiting_for_changes


def test_once_failure_drains_already_admitted_work(workflow_harness: WorkflowHarness) -> None:
    from nhx.core.jobs.app.workflow import workflow_step_held

    h = workflow_harness
    job = h.create({"bad": [], "other": [], "child": ["bad", "other"]}, mode="run_to_completion")
    h.tick(job)
    other_id = h.node_state(job, "other").active_step_id
    assert job.attempt_id is not None and other_id is not None
    body: dict[str, object] = {"workflow_admission": {"step_id": other_id}}
    h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    h.finish(job, "bad", HelixJobStatus.ERROR)
    draining = h.tick(job)
    assert draining.status == HelixJobStatus.ACTIVE
    assert h.node_state(job, "other").scheduling_state == "admitted"
    store = SyncEntityClient(EntitiesClient.from_client(h.context.client))
    attempt = store.get_by_id(HelixJobAttempt, job.attempt_id)
    step = store.get_by_id(HelixJobStep, other_id)
    assert not workflow_step_held(attempt, step)
    h.jobs.update_attempt_status_details(job.id, job.attempt_id, workspace="default", body=body)
    h.finish(job, "other")
    finished = h.tick(job)
    assert finished.status == HelixJobStatus.ERROR
    assert len(finished.steps) == 2


def _create_event_workflow(
    h: WorkflowHarness, steps: list[HelixJobStepSpec], mode: WorkflowMode = "continuous"
) -> HelixJobResponse:
    return h.jobs.create_job(
        workspace="default",
        body=CreateHelixJobRequest(
            name="events-" + uuid4().hex[:12],
            source="test",
            spec={},
            platform_spec=HelixJobSpec(workflow=WorkflowSpec(mode=mode), steps=steps),
        ),
    ).data()


def test_event_filesets_are_or_triggers_and_dependencies_are_persistent(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit, WorkflowStepSuccess

    h = workflow_harness
    left, right = h.fileset(), h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(name="prepare", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(name="validate", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(
                name="work",
                depends_on=[WorkflowStepSuccess(step_success="prepare"), WorkflowStepSuccess(step_success="validate")],
                trigger=[WorkflowFilesetCommit(fileset_commit=left.id), WorkflowFilesetCommit(fileset_commit=right.id)],
                executor=TestConstants.TEST_EXECUTOR,
            ),
        ],
    )
    assert len(h.tick(job).steps) == 2
    h.finish(job, "prepare")
    assert len(h.tick(job).steps) == 2
    h.finish(job, "validate")
    ready = h.tick(job)
    assert len(ready.steps) == 3
    assert ready.workflow is not None and len(ready.workflow.nodes[2].trigger) == 2
    h.finish(job, "work")
    waiting = h.tick(job)
    assert waiting.workflow is not None and waiting.workflow.waiting_for_changes
    for fileset in (left, right):
        h.files.commit_fileset(workspace="default", name=fileset.name)
        before = len(h.status(job).steps)
        assert len(h.tick(job).steps) == before + 1
        h.finish(job, "work")
        h.tick(job)
        assert len(h.tick(job).steps) == before + 1
    assert h.node_state(job, "prepare").successful_step_id == ready.steps[0].id


def test_step_success_or_trigger_does_not_wait_for_other_source(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowStepSuccess

    h = workflow_harness
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(
                name="work",
                depends_on=[],
                trigger=[WorkflowStepSuccess(step_success="left"), WorkflowStepSuccess(step_success="right")],
                executor=TestConstants.TEST_EXECUTOR,
            ),
            HelixJobStepSpec(name="left", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(name="right", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
        ],
    )
    assert len(h.tick(job).steps) == 2
    h.finish(job, "left")
    assert len(h.tick(job).steps) == 3
    h.finish(job, "work")
    h.tick(job)
    h.finish(job, "right")
    assert len(h.tick(job).steps) == 3
    second = h.tick(job)
    assert second.workflow is not None and second.workflow.pass_number == 2
    assert len(second.steps) == 4
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 4
    h.invalidate(job, ["left"])
    h.tick(job)
    h.finish(job, "left")
    assert len(h.tick(job).steps) == 6


@pytest.mark.parametrize("outcome", [HelixJobStatus.ERROR, HelixJobStatus.CANCELLED])
def test_only_successful_execution_triggers_dependents(
    workflow_harness: WorkflowHarness, outcome: HelixJobStatus
) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowStepSuccess

    h = workflow_harness
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(name="source", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(
                name="work",
                depends_on=[],
                trigger=[WorkflowStepSuccess(step_success="source")],
                executor=TestConstants.TEST_EXECUTOR,
            ),
        ],
    )
    h.tick(job)
    h.finish(job, "source", outcome)
    h.tick(job)
    assert len(h.tick(job).steps) == 1
    h.invalidate(job, ["source"])
    h.tick(job)
    h.finish(job, "source")
    assert len(h.tick(job).steps) == 3
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 3


def test_fileset_readiness_does_not_trigger_work(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit

    h = workflow_harness
    source, prerequisite = h.fileset(), h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(
                name="work",
                depends_on=[WorkflowFilesetCommit(fileset_commit=prerequisite.id)],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id)],
                executor=TestConstants.TEST_EXECUTOR,
            )
        ],
    )
    h.tick(job)
    h.finish(job, "work")
    h.tick(job)
    h.files.commit_fileset(workspace="default", name=prerequisite.name)
    assert len(h.tick(job).steps) == 1
    h.files.commit_fileset(workspace="default", name=source.name)
    assert len(h.tick(job).steps) == 2


def test_one_missing_or_event_does_not_block_other_event(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit

    h = workflow_harness
    available, missing = h.fileset(), h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(
                name="work",
                depends_on=[],
                trigger=[
                    WorkflowFilesetCommit(fileset_commit=available.id),
                    WorkflowFilesetCommit(fileset_commit=missing.id),
                ],
                executor=TestConstants.TEST_EXECUTOR,
            )
        ],
    )
    h.files.delete_fileset_by_id(workspace="default", fileset_id=missing.id)
    assert len(h.tick(job).steps) == 1
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 1
    h.files.commit_fileset(workspace="default", name=available.name)
    assert len(h.tick(job).steps) == 2


@pytest.mark.parametrize("change", ["input_commit", "manual"])
def test_step_event_keeps_midpass_fileset_change_pending(workflow_harness: WorkflowHarness, change: str) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit, WorkflowStepSuccess

    h = workflow_harness
    source = h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(name="prepare", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(
                name="work",
                depends_on=[WorkflowStepSuccess(step_success="prepare")],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id), WorkflowStepSuccess(step_success="prepare")],
                executor=TestConstants.TEST_EXECUTOR,
            ),
        ],
    )
    h.tick(job)
    if change == "input_commit":
        h.files.commit_fileset(workspace="default", name=source.name)
    else:
        h.invalidate(job, ["work"])
    h.finish(job, "prepare")
    assert len(h.tick(job).steps) == 2
    captured = h.node_state(job, "work")
    assert captured.active_inputs is not None
    assert captured.active_inputs[source.id] == source.commit_revision
    assert captured.active_run_revision != captured.requested_run_revision
    if change == "input_commit":
        assert captured.active_inputs != captured.observed_inputs
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 3
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 3


def test_all_fileset_dependencies_must_be_ready_without_consuming_pending_trigger(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from nemo_helix_plugin.client.errors import NemoHTTPError
    from nemo_helix_plugin.client.response import NemoResponse
    from nemo_helix_plugin.files.client import AsyncFilesClient
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit

    h = workflow_harness
    source, left, right = h.fileset(), h.fileset(), h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(
                name="work",
                depends_on=[
                    WorkflowFilesetCommit(fileset_commit=left.id),
                    WorkflowFilesetCommit(fileset_commit=right.id),
                ],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id)],
                executor=TestConstants.TEST_EXECUTOR,
            )
        ],
    )
    original = AsyncFilesClient.from_client(h.context.async_client).get_fileset_by_id

    async def unavailable(
        self: AsyncFilesClient, *, workspace: str | None = None, fileset_id: str
    ) -> NemoResponse[FilesetOutput]:
        if fileset_id == right.id:
            raise NemoHTTPError(httpx.Response(503, request=httpx.Request("GET", "http://test/fileset")))
        return await original(workspace=workspace, fileset_id=fileset_id)

    with monkeypatch.context() as patch:
        patch.setattr(AsyncFilesClient, "get_fileset_by_id", unavailable)
        blocked = h.tick(job)
        assert blocked.steps == [] and blocked.workflow is not None
        assert blocked.workflow.nodes[0].blocked_by == ["fileset:" + right.id]
        h.invalidate(job, ["work"])
        h.files.commit_fileset(workspace="default", name=source.name)
        for _ in range(2):
            still_blocked = h.tick(job)
            assert still_blocked.steps == [] and still_blocked.workflow is not None
            assert still_blocked.workflow.pass_number == 1
    assert len(h.tick(job).steps) == 1
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 1


def test_prior_success_remains_ready_during_a_later_failed_rerun(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit, WorkflowStepSuccess

    h = workflow_harness
    source = h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(name="prepare", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(
                name="work",
                depends_on=[WorkflowStepSuccess(step_success="prepare")],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id)],
                executor=TestConstants.TEST_EXECUTOR,
            ),
        ],
    )
    h.tick(job)
    h.finish(job, "prepare")
    h.tick(job)
    h.finish(job, "work")
    h.tick(job)
    prior = h.node_state(job, "prepare").successful_step_id
    h.invalidate(job, ["prepare"])
    h.files.commit_fileset(workspace="default", name=source.name)
    assert len(h.tick(job).steps) == 4
    h.finish(job, "prepare", HelixJobStatus.ERROR)
    h.tick(job)
    assert h.node_state(job, "work").active_step_id is not None
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 4
    assert h.node_state(job, "prepare").successful_step_id == prior


def test_event_triggers_wait_while_attempt_is_paused(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit

    h = workflow_harness
    source = h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(
                name="work",
                depends_on=[],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id)],
                executor=TestConstants.TEST_EXECUTOR,
            )
        ],
    )
    h.tick(job)
    h.finish(job, "work")
    h.tick(job)
    h.jobs.pause_job(workspace="default", name=job.id)
    assert h.tick(job).status == HelixJobStatus.PAUSED
    h.files.commit_fileset(workspace="default", name=source.name)
    assert len(h.tick(job).steps) == 1
    h.jobs.resume_job(workspace="default", name=job.id)
    assert len(h.tick(job).steps) == 2
    h.finish(job, "work")
    h.tick(job)
    assert len(h.tick(job).steps) == 2


@pytest.mark.parametrize("condition_kind", ["trigger", "depends_on"])
def test_new_fileset_conditions_can_wait_for_the_initial_commit(
    workflow_harness: WorkflowHarness, monkeypatch: pytest.MonkeyPatch, condition_kind: str
) -> None:
    from nemo_helix_plugin.client.response import NemoResponse
    from nemo_helix_plugin.files.client import AsyncFilesClient
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit

    h = workflow_harness
    source = h.fileset()
    original = AsyncFilesClient.from_client(h.context.async_client).get_fileset_by_id

    async def uncommitted(
        self: AsyncFilesClient, *, workspace: str | None = None, fileset_id: str
    ) -> NemoResponse[FilesetOutput]:
        response = await original(workspace=workspace, fileset_id=fileset_id)
        if fileset_id == source.id:
            response.body.commit_revision = None
        return response

    step = HelixJobStepSpec(name="work", depends_on=[], executor=TestConstants.TEST_EXECUTOR)
    if condition_kind == "trigger":
        step.trigger = [WorkflowFilesetCommit(fileset_commit=source.id)]
    else:
        step.depends_on = [WorkflowFilesetCommit(fileset_commit=source.id)]
    with monkeypatch.context() as patch:
        patch.setattr(AsyncFilesClient, "get_fileset_by_id", uncommitted)
        job = _create_event_workflow(h, [step])
        assert h.tick(job).steps == []
        assert h.tick(job).steps == []
    assert len(h.tick(job).steps) == 1


def test_event_workflow_once_mode_remains_terminal(workflow_harness: WorkflowHarness) -> None:
    from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit, WorkflowStepSuccess

    h = workflow_harness
    source = h.fileset()
    job = _create_event_workflow(
        h,
        [
            HelixJobStepSpec(name="prepare", depends_on=[], executor=TestConstants.TEST_EXECUTOR),
            HelixJobStepSpec(
                name="work",
                depends_on=[WorkflowStepSuccess(step_success="prepare")],
                trigger=[WorkflowFilesetCommit(fileset_commit=source.id)],
                executor=TestConstants.TEST_EXECUTOR,
            ),
        ],
        mode="run_to_completion",
    )
    assert len(h.tick(job).steps) == 1
    h.finish(job, "prepare")
    assert len(h.tick(job).steps) == 2
    h.finish(job, "work")
    assert h.tick(job).status == HelixJobStatus.COMPLETED
    h.files.commit_fileset(workspace="default", name=source.name)
    still_terminal = h.tick(job)
    assert still_terminal.status == HelixJobStatus.COMPLETED and len(still_terminal.steps) == 2
