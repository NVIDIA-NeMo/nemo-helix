# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Docker execution with in-process platform services.

Run with NHX_TEST_WORKFLOW_DOCKER=1; requires Docker and busybox:stable.
"""

import os
import re
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import pytest
from docker.models.containers import Container
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse
from nemo_helix_plugin.jobs.workflows import WorkflowInput, WorkflowMode, WorkflowSpec
from nhx.core.files.service import FilesService
from nhx.core.jobs.app.constants import JOB_ID_LABEL, JOB_STEP_NAME_LABEL
from nhx.core.jobs.config import JobsServiceConfig
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.controllers.scheduler import JobScheduler
from nhx.core.jobs.service import JobsService
from nhx.testing import ClientContext, create_test_client
from pydantic import BaseModel, Field

import docker
from docker import DockerClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("NHX_TEST_WORKFLOW_DOCKER") != "1", reason="opt-in real Docker workflow test"),
]


class DockerRunState(BaseModel):
    started_at: datetime = Field(alias="StartedAt")
    finished_at: datetime = Field(alias="FinishedAt")
    exit_code: int = Field(alias="ExitCode")


def container_state(container: Container) -> DockerRunState:
    assert container.attrs is not None
    return DockerRunState.model_validate(container.attrs["State"])


@dataclass
class DockerWorkflowHarness:
    jobs: JobsClient
    files: FilesClient
    docker: DockerClient
    scheduler: JobScheduler
    reconciler: JobReconciler
    prefix: str
    created: list[HelixJobResponse] = field(default_factory=list)

    def create(
        self,
        graph: dict[str, list[str]],
        *,
        mode: WorkflowMode = "continuous",
        inputs: dict[str, list[str]] | None = None,
        commands: dict[str, str] | None = None,
    ) -> HelixJobResponse:
        job = self.jobs.create_job(
            workspace="default",
            body=CreateHelixJobRequest(
                name=f"{self.prefix}-{len(self.created)}",
                source="docker-workflow-test",
                spec={},
                platform_spec=HelixJobSpec(
                    workflow=WorkflowSpec(mode=mode),
                    steps=[
                        HelixJobStepSpec(
                            name=name,
                            depends_on=parents,
                            inputs=[WorkflowInput(fileset_id=value) for value in (inputs or {}).get(name, [])],
                            environment=[
                                HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/workflow")
                            ],
                            executor=CPUExecutionProvider(
                                provider="cpu",
                                profile=self.prefix,
                                container=ContainerSpec(
                                    image="busybox:stable",
                                    command=["sh", "-c", (commands or {}).get(name, "sleep 0.2; echo success")],
                                ),
                            ),
                        )
                        for name, parents in graph.items()
                    ],
                ),
            ),
        ).data()
        self.created.append(job)
        return job

    def status(self, job: HelixJobResponse) -> HelixJobStatusResponse:
        return self.jobs.get_job_status(workspace="default", name=job.id).data()

    def tick(self, job: HelixJobResponse) -> HelixJobStatusResponse:
        self.reconciler.step()
        self.scheduler.step()
        return self.status(job)

    def wait_for(
        self,
        job: HelixJobResponse,
        predicate: Callable[[HelixJobStatusResponse], bool],
        timeout: float = 60,
    ) -> HelixJobStatusResponse:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.tick(job)
            if predicate(state):
                return state
            time.sleep(0.1)
        raise AssertionError(self.status(job).model_dump_json(indent=2))

    def settled(self, job: HelixJobResponse) -> HelixJobStatusResponse:
        return self.wait_for(job, lambda state: state.workflow is not None and state.workflow.waiting_for_changes)

    def invalidate(self, job: HelixJobResponse, name: str) -> None:
        assert job.attempt_id is not None
        self.jobs.update_attempt_status_details(
            job.id,
            job.attempt_id,
            workspace="default",
            body={
                "workflow_invalidation": {"steps": [name], "request_id": str(uuid4())},
            },
        )

    def containers(self, job: HelixJobResponse) -> list[Container]:
        return self.docker.containers.list(all=True, filters={"label": f"{JOB_ID_LABEL}={job.name}"})


@pytest.fixture
def docker_workflow(monkeypatch: pytest.MonkeyPatch) -> Generator[DockerWorkflowHarness, None, None]:
    prefix = "dag-docker-" + uuid4().hex[:10]
    client = docker.from_env()
    client.ping()
    client.images.get("busybox:stable")
    profile = DockerJobExecutionProfile(
        provider="cpu",
        profile=prefix,
        config=DockerJobExecutionProfileConfig(
            cleanup_completed_jobs_immediately=False,
            networking={"job_container_network": "bridge"},
            storage={"volume_name": prefix + "-storage", "volume_permissions_image": "busybox:stable"},
        ),
    )
    config = JobsServiceConfig(executors=[profile], enable_subprocess_executor=False)
    monkeypatch.setattr("nhx.core.jobs.api.v2.jobs.endpoints.profiles", config.executors)
    try:
        with create_test_client(
            JobsService, FilesService, client_type=ClientContext, service_configs={JobsService: config}
        ) as context:
            registry = BackendRegistry.from_config(nemo_client=context.client, profiles=config.executors)
            harness = DockerWorkflowHarness(
                JobsClient.from_client(context.client),
                FilesClient.from_client(context.client),
                client,
                JobScheduler(registry, context.client),
                JobReconciler(registry, context.client),
                prefix,
            )
            try:
                yield harness
            finally:
                registry.shutdown_all_backends()
    finally:
        for container in client.containers.list(all=True):
            if prefix in container.name:
                container.remove(force=True, v=True)
        for volume in client.volumes.list():
            if prefix in volume.name:
                volume.remove()
        client.close()


def test_docker_diamond_and_once_completion(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.create(
        {"root": [], "left": ["root"], "right": ["root"], "join": ["left", "right"]}, mode="run_to_completion"
    )
    state = h.wait_for(job, lambda value: value.status.is_terminal())
    assert state.status == HelixJobStatus.COMPLETED, state.model_dump_json(indent=2)
    assert state.workflow is not None and all(node.freshness == "clean" for node in state.workflow.nodes)
    containers = h.containers(job)
    assert len(containers) == 4
    by_name = {step.name: step.id for step in state.steps}
    executions: dict[str, str] = {}
    for node in state.workflow.nodes:
        assert node.latest_execution_id is not None
        executions[node.name] = node.latest_execution_id
    actual = {by_name[c.labels[JOB_STEP_NAME_LABEL]]: container_state(c) for c in containers}
    for node, parents in {"left": ["root"], "right": ["root"], "join": ["left", "right"]}.items():
        for parent in parents:
            assert actual[executions[parent]].finished_at <= actual[executions[node]].started_at
    assert all(container_state(c).exit_code == 0 for c in containers)


def test_docker_continuous_commit_manual_rerun_and_drain_pause(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    source = h.files.create_fileset(workspace="default", body=CreateFilesetRequest(name=h.prefix + "-input")).data()
    job = h.create(
        {"root": [], "child": ["root"]}, inputs={"root": [source.id]}, commands={"root": "sleep 2; echo root"}
    )
    first = h.settled(job)
    assert len(first.steps) == 2 and first.status == HelixJobStatus.ACTIVE
    h.files.upload_file(workspace="default", name=source.name, path="input", content=b"changed without commit")
    assert len(h.tick(job).steps) == 2
    h.files.commit_fileset(workspace="default", name=source.name)
    changed = h.settled(job)
    assert len(changed.steps) == 3
    h.invalidate(job, "root")
    running = h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    assert len(running.steps) == 4
    h.jobs.pause_job(workspace="default", name=job.id)
    paused = h.wait_for(job, lambda state: state.status == HelixJobStatus.PAUSED)
    assert len(paused.steps) == 4
    h.invalidate(job, "child")
    assert len(h.tick(job).steps) == 4
    h.jobs.resume_job(workspace="default", name=job.id)
    resumed = h.settled(job)
    assert len(resumed.steps) == 5
    containers = h.containers(job)
    assert len(containers) == 5 and len({c.name for c in containers}) == 5
    assert all(container_state(c).exit_code == 0 for c in containers)
    assert resumed.workflow is not None and all(node.freshness == "clean" for node in resumed.workflow.nodes)


def test_docker_failed_node_recovers_without_reopening_attempt(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.create(
        {"root": [], "child": ["root"], "independent": []},
        commands={
            "root": "if test -f /workflow/first-failure; then echo recovered; else touch /workflow/first-failure; exit 7; fi",
        },
    )
    failed = h.settled(job)
    assert failed.status == HelixJobStatus.ACTIVE and failed.workflow is not None and failed.workflow.degraded
    assert len(failed.steps) == 2
    assert len(h.tick(job).steps) == 2
    h.invalidate(job, "child")
    assert len(h.tick(job).steps) == 2
    h.invalidate(job, "root")
    recovered = h.settled(job)
    assert recovered.attempt_id == failed.attempt_id
    assert len(recovered.steps) == 4
    assert recovered.workflow is not None and all(node.freshness == "clean" for node in recovered.workflow.nodes)
    assert sorted(container_state(c).exit_code for c in h.containers(job)) == [0, 0, 0, 7]


def test_docker_task_callbacks_follow_each_execution(docker_workflow: DockerWorkflowHarness) -> None:
    from pytest_httpserver import HTTPServer
    from werkzeug.wrappers import Request, Response

    h = docker_workflow
    requests: list[str] = []

    def forward(request: Request) -> Response:
        requests.append(request.path)
        response = h.jobs.http_client.request(
            request.method,
            h.jobs.base_url + request.path,
            content=request.data,
            headers={"Content-Type": "application/json"},
        )
        return Response(response.content, status=response.status_code, content_type="application/json")

    server = HTTPServer(host="0.0.0.0", port=0)
    server.expect_request(
        re.compile("/apis/jobs/v2/workspaces/default/jobs/" + h.prefix + ".*"), method="PUT"
    ).respond_with_handler(forward)
    server.start()
    try:
        command = r"""set -eu
report() {
    body=$(printf '{"status":"%s","status_details":{"probe":"container"}}' "$1")
    path="/apis/jobs/v2/workspaces/$NEMO_JOB_WORKSPACE/jobs/$NEMO_JOB_ID/attempts/$NEMO_JOB_ATTEMPT_ID/steps/$NEMO_JOB_STEP/tasks/$NEMO_JOB_TASK"
    printf 'PUT %s HTTP/1.1\r\nHost: host.docker.internal\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' "$path" "${#body}" "$body" |
        nc -w 10 host.docker.internal PORT > /tmp/callback-response
    grep -q '^HTTP/1.[01] 200 ' /tmp/callback-response || { cat /tmp/callback-response; exit 1; }
}
report active
sleep 0.2
report completed
""".replace("PORT", str(server.port))
        job = h.create({"work": []}, commands={"work": command})
        first = h.settled(job)
        assert first.workflow is not None and not first.workflow.degraded, first.model_dump_json(indent=2)
        h.invalidate(job, "work")
        second = h.settled(job)
        assert len(second.steps) == 2
        for step in second.steps:
            assert step.status == HelixJobStatus.COMPLETED
            assert len(step.tasks) == 1
            assert step.tasks[0].status == HelixJobStatus.COMPLETED
            assert step.tasks[0].status_details["probe"] == "container"
        assert len({step.tasks[0].id for step in second.steps}) == 2
        assert len(requests) == 4
        assert len(set(requests)) == 2
    finally:
        server.stop()


def test_docker_cancel_stops_running_node_without_starting_child(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.create({"root": [], "child": ["root"]}, commands={"root": "trap 'exit 0' TERM; sleep 30 & wait"})
    h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    h.jobs.cancel_job(workspace="default", name=job.id)
    cancelled = h.wait_for(job, lambda state: state.status.is_terminal())
    assert cancelled.status == HelixJobStatus.CANCELLED
    assert len(cancelled.steps) == 1
    assert cancelled.steps[0].status == HelixJobStatus.CANCELLED
    assert cancelled.workflow is not None and all(node.active_execution_id is None for node in cancelled.workflow.nodes)
    assert all(container.status != "running" for container in h.containers(job))
    assert h.tick(job).status == HelixJobStatus.CANCELLED
