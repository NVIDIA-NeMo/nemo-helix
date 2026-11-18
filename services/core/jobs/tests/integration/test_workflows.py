# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow behavior through real in-process Jobs, Files and Entity APIs."""

from collections.abc import Iterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatusUpdateRequest
from nemo_helix_plugin.jobs.workflows import WorkflowInput, WorkflowSpec
from nhx.core.files.service import FilesService
from nhx.core.jobs.app.test_helpers import TestConstants
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
    assert len(resumed.steps) == 2
    assert resumed.steps[0].status == HelixJobStatus.CANCELLED


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
    harness.finish(job, "work")
    settled = harness.tick(job)
    assert settled.workflow is not None and settled.workflow.nodes[0].freshness == "dirty"
    assert not settled.workflow.waiting_for_changes
    next_run = harness.tick(job)
    assert len(next_run.steps) == 2


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
    attempt_id = job.attempt_id

    def get_status(
        self: JobsClient, *, workspace: str | None = None, name: str
    ) -> NemoResponse[HelixJobStatusResponse]:
        nonlocal polls
        polls += 1
        if polls in (2, 3):
            harness.jobs.update_attempt_step_status(
                workspace="default",
                job=job.id,
                attempt=attempt_id,
                name=first.steps[0].id,
                body=HelixJobStatusUpdateRequest(
                    status=HelixJobStatus.ACTIVE if polls == 2 else HelixJobStatus.COMPLETED
                ),
            )
            harness.jobs.update_attempt_status_details(
                job.id, attempt_id, workspace="default", body={"workflow_reconcile": True}
            )
        return original(workspace=workspace, name=name)

    monkeypatch.setattr(JobsClient, "get_job_status", get_status)
    events = list(harness.jobs.watch_job(job.id, workspace="default", include_logs=False, poll_interval=0))
    assert [event.status for event in events if isinstance(event, JobStatusEvent)] == ["active", "completed"]
    views = [event.workflow for event in events if isinstance(event, JobWorkflowEvent)]
    assert len(views) == 3
    assert views[0].nodes[0].requested_run_revision == views[1].nodes[0].requested_run_revision
    assert views[1].nodes[0].scheduling_state == "running"
    assert views[2].nodes[0].freshness == "clean"


def test_restart_adopts_execution_created_before_controller_state_save(
    workflow_harness: WorkflowHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nhx.common.entities.client import EntityClient
    from nhx.core.jobs.entities import HelixJobStep

    harness = workflow_harness
    job = harness.create({"work": []})
    original = EntityClient.create
    failed = False

    async def create(self, entity):
        nonlocal failed
        result = await original(self, entity)
        if isinstance(entity, HelixJobStep) and not failed:
            failed = True
            raise RuntimeError("controller interrupted after execution creation")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(EntityClient, "create", create)
        with pytest.raises(RuntimeError, match="controller interrupted"):
            harness.tick(job)
    before = harness.status(job)
    assert len(before.steps) == 1
    recovered = harness.tick(job)
    assert len(recovered.steps) == 1
    assert recovered.workflow is not None
    assert recovered.workflow.nodes[0].active_execution_id == before.steps[0].id
    harness.finish(job, "work")
    completed = harness.tick(job)
    assert completed.workflow is not None and completed.workflow.nodes[0].freshness == "clean"


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
