# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Docker execution with in-process platform services.

Run with NHX_TEST_WORKFLOW_DOCKER=1; requires Docker and busybox:stable.
"""

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from docker.models.containers import Container
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobLog, HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, HelixJobStatusUpdateRequest
from nemo_helix_plugin.jobs.workflows import WorkflowInput, WorkflowMode, WorkflowSpec
from nhx.common.config import HelixConfig
from nhx.core.files.service import FilesService
from nhx.core.jobs.app.constants import (
    JOB_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TASK_ID_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
)
from nhx.core.jobs.config import JobsServiceConfig
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.controllers.scheduler import JobScheduler
from nhx.core.jobs.service import JobsService
from nhx.testing import ClientContext, create_test_client
from pydantic import BaseModel, Field
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

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
    client: NemoClient
    profile: DockerJobExecutionProfile
    registry: BackendRegistry
    requests: list[tuple[str, str, int]]
    platform_host: str
    created: list[HelixJobResponse] = field(default_factory=list)

    def restart_controller(self) -> None:
        self.registry.shutdown_all_backends()
        self.registry = BackendRegistry.from_config(nemo_client=self.client, profiles=[self.profile])
        self.scheduler = JobScheduler(self.registry, self.client)
        self.reconciler = JobReconciler(self.registry, self.client)

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
                            config={"logical_name": name},
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

    def release(self, job: HelixJobResponse, execution_id: str) -> None:
        step = next(step for step in self.status(job).steps if step.id == execution_id)
        container = next(c for c in self.containers(job) if c.labels[JOB_STEP_NAME_LABEL] == step.name)
        result = container.exec_run(["touch", "/workflow/release-" + step.name])
        assert result.exit_code == 0

    def logs(self, job: HelixJobResponse, step_name: str | None = None) -> list[HelixJobLog]:
        assert job.attempt_id is not None
        from nemo_helix_plugin.jobs.types import JobLogsQueryParams

        query = JobLogsQueryParams(attempt_id=job.attempt_id)
        if step_name is not None:
            query["step_id"] = step_name
        return list(self.jobs.list_job_logs(workspace="default", name=job.id, query_params=query).items())

    def storage_exists(self, job: HelixJobResponse) -> bool:
        container = self.docker.containers.create(
            "busybox:stable",
            ["test", "-d", f"/volume/jobs/default/{job.name}"],
            name=self.prefix + "-inspect-" + uuid4().hex[:8],
            volumes={self.prefix + "-storage": {"bind": "/volume", "mode": "ro"}},
        )
        try:
            container.start()
            status = container.wait()["StatusCode"]
            assert status in (0, 1)
            return status == 0
        finally:
            container.remove(force=True)

    def containers(self, job: HelixJobResponse) -> list[Container]:
        containers = self.docker.containers.list(
            all=True, filters={"label": [f"{JOB_ID_LABEL}={job.name}", f"{JOB_TYPE_LABEL}={JOB_TYPE_JOB}"]}
        )
        for container in containers:
            assert container.attrs is not None
            assert container.attrs["Config"]["Entrypoint"][:3] == ["/jobs-launcher", "run", "--"]
        return containers


class DockerImagePlatform(BaseModel):
    architecture: Literal["arm64", "amd64"] = Field(alias="Architecture")


@pytest.fixture(scope="session")
def workflow_launcher(tmp_path_factory: pytest.TempPathFactory) -> Path:
    prebuilt = os.environ.get("NHX_TEST_WORKFLOW_LAUNCHER")
    if prebuilt:
        binary = Path(prebuilt).resolve(strict=True)
    else:
        with closing(docker.from_env()) as client:
            image = client.images.get("busybox:stable")
            platform = DockerImagePlatform.model_validate(image.attrs)
        binary = tmp_path_factory.mktemp("workflow-launcher") / "jobs-launcher"
        build_env = {**os.environ, "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": platform.architecture}
        subprocess.run(
            ["go", "build", "-mod=readonly", "-trimpath", "-o", str(binary), "./main.go"],
            cwd=Path(__file__).resolve().parents[2] / "jobs-launcher",
            env=build_env,
            check=True,
        )
    with binary.open("rb") as stream:
        assert stream.read(4) == b"\x7fELF", "The launcher must be a Linux binary"
    return binary


@pytest.fixture
def docker_workflow(
    monkeypatch: pytest.MonkeyPatch, workflow_launcher: Path
) -> Generator[DockerWorkflowHarness, None, None]:
    prefix = "dag-docker-" + uuid4().hex[:10]
    client = docker.from_env()
    client.ping()
    client.images.get("busybox:stable")
    platform_host = "host.docker.internal"
    if sys.platform == "linux":
        bridge = client.networks.get("bridge")
        assert bridge.attrs is not None
        platform_host = bridge.attrs["IPAM"]["Config"][0]["Gateway"]
        assert isinstance(platform_host, str)
    requests: list[tuple[str, str, int]] = []

    def forward(request: Request) -> Response:
        response = context.client.http_client.request(
            request.method,
            context.client.base_url + request.full_path.rstrip("?"),
            content=request.data,
            headers={"Content-Type": request.content_type or "application/octet-stream"},
        )
        requests.append((request.method, request.path, response.status_code))
        return Response(
            response.content, status=response.status_code, content_type=response.headers.get("content-type")
        )

    proxy = HTTPServer(host="0.0.0.0", port=0)
    proxy.expect_request(re.compile("/apis/.*")).respond_with_handler(forward)
    proxy.start()
    profile = DockerJobExecutionProfile(
        provider="cpu",
        profile=prefix,
        config=DockerJobExecutionProfileConfig(
            cleanup_completed_jobs_immediately=False,
            launcher_tool_path=str(workflow_launcher),
            networking={"job_container_network": "bridge"},
            storage={"volume_name": prefix + "-storage", "volume_permissions_image": "busybox:stable"},
        ),
    )
    config = JobsServiceConfig(executors=[profile], enable_subprocess_executor=False)
    monkeypatch.setattr("nhx.core.jobs.api.v2.jobs.endpoints.profiles", config.executors)
    try:
        with create_test_client(
            JobsService,
            FilesService,
            client_type=ClientContext,
            service_configs={
                JobsService: config,
                HelixConfig: HelixConfig(
                    base_url="http://127.0.0.1",
                    service_discovery={
                        "platform": f"http://{platform_host}:{proxy.port}",
                    },
                ),
            },
        ) as context:
            registry = BackendRegistry.from_config(nemo_client=context.client, profiles=config.executors)
            harness = DockerWorkflowHarness(
                JobsClient.from_client(context.client),
                FilesClient.from_client(context.client),
                client,
                JobScheduler(registry, context.client),
                JobReconciler(registry, context.client),
                prefix,
                context.client,
                profile,
                registry,
                requests,
                platform_host,
            )
            try:
                yield harness
            finally:
                harness.registry.shutdown_all_backends()
    finally:
        for container in client.containers.list(all=True, filters={"name": prefix}, ignore_removed=True):
            if prefix in container.name:
                container.remove(force=True, v=True)
        for volume in client.volumes.list():
            if prefix in volume.name:
                volume.remove()
        client.close()
        proxy.stop()


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
    printf 'PUT %s HTTP/1.1\r\nHost: CALLBACK_HOST\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' "$path" "${#body}" "$body" |
        nc -w 10 CALLBACK_HOST PORT > /tmp/callback-response
    grep -q '^HTTP/1.[01] 200 ' /tmp/callback-response || { cat /tmp/callback-response; exit 1; }
}
report active
sleep 0.2
report completed
""".replace("PORT", str(server.port)).replace("CALLBACK_HOST", h.platform_host)
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


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_docker_cancel_stops_running_node_without_starting_child(
    docker_workflow: DockerWorkflowHarness, mode: WorkflowMode
) -> None:
    h = docker_workflow
    job = h.create({"root": [], "child": ["root"]}, mode=mode, commands={"root": "trap 'exit 0' TERM; sleep 30 & wait"})
    h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    h.jobs.cancel_job(workspace="default", name=job.id)
    with pytest.raises(NemoHTTPError) as conflict:
        h.jobs.pause_job(workspace="default", name=job.id)
    assert conflict.value.status_code == 409
    assert h.status(job).status == HelixJobStatus.CANCELLING
    cancelled = h.wait_for(job, lambda state: state.status.is_terminal())
    assert cancelled.status == HelixJobStatus.CANCELLED
    assert len(cancelled.steps) == 1
    assert cancelled.steps[0].status == HelixJobStatus.CANCELLED
    assert cancelled.workflow is not None and all(node.active_execution_id is None for node in cancelled.workflow.nodes)
    assert all(container.status != "running" for container in h.containers(job))
    assert h.tick(job).status == HelixJobStatus.CANCELLED


GATED_WORK = 'while ! test -f "/workflow/release-$NEMO_JOB_STEP"; do sleep 0.1; done; echo gate-released'


def test_launcher_flushes_stdout_stderr_and_preserves_failed_exit(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.create(
        {"success": [], "failure": []},
        commands={
            "success": "printf success-output; printf stderr-tail >&2",
            "failure": "printf failure-output; printf failure-tail >&2; exit 7",
        },
    )
    state = h.settled(job)
    assert state.workflow is not None and state.workflow.degraded
    assert sorted(container_state(c).exit_code for c in h.containers(job)) == [0, 7]
    logs = h.logs(job)
    assert {"success-output", "stderr-tail", "failure-output", "failure-tail"} <= {row.message for row in logs}
    task_by_step = {c.labels[JOB_STEP_NAME_LABEL]: c.labels[JOB_TASK_ID_LABEL] for c in h.containers(job)}
    assert all(row.job_task == task_by_step[row.job_step] for row in logs)
    assert any(
        method == "POST" and path.endswith("/otlp/v1/logs") and status == 200 for method, path, status in h.requests
    )


def test_controller_replacement_adopts_running_container_and_paused_requests(
    docker_workflow: DockerWorkflowHarness,
) -> None:
    h = docker_workflow
    job = h.create({"root": [], "child": ["root"]}, commands={"root": GATED_WORK})
    running = h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    execution = running.steps[0].id
    container = h.containers(job)[0].id
    h.restart_controller()
    adopted = h.tick(job)
    assert adopted.workflow is not None and adopted.workflow.nodes[0].active_execution_id == execution
    assert [c.id for c in h.containers(job)] == [container]
    h.release(job, execution)
    assert len(h.settled(job).steps) == 2
    h.jobs.pause_job(workspace="default", name=job.id)
    h.wait_for(job, lambda state: state.status == HelixJobStatus.PAUSED)
    h.invalidate(job, "child")
    h.restart_controller()
    assert h.tick(job).status == HelixJobStatus.PAUSED
    assert len(h.status(job).steps) == 2
    h.jobs.resume_job(workspace="default", name=job.id)
    assert len(h.settled(job).steps) == 3


def test_docker_midrun_changes_coalesce_and_old_completion_leaves_work_pending(
    docker_workflow: DockerWorkflowHarness,
) -> None:
    h = docker_workflow
    source = h.files.create_fileset(workspace="default", body=CreateFilesetRequest(name=h.prefix + "-input")).data()
    job = h.create({"root": []}, inputs={"root": [source.id]}, commands={"root": GATED_WORK})
    first = h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    execution = first.steps[0].id
    for _ in range(3):
        h.files.commit_fileset(workspace="default", name=source.name)
        h.invalidate(job, "root")
        pending = h.tick(job)
        assert pending.workflow is not None and pending.workflow.nodes[0].active_execution_id == execution
        assert len(h.containers(job)) == 1
    h.release(job, execution)
    drained = h.wait_for(
        job, lambda state: state.workflow is not None and state.workflow.nodes[0].active_execution_id is None
    )
    assert drained.workflow is not None and drained.workflow.nodes[0].has_pending_run
    assert not drained.workflow.waiting_for_changes
    second = h.wait_for(
        job, lambda state: len(state.steps) == 2 and any(step.status == HelixJobStatus.ACTIVE for step in state.steps)
    )
    latest = second.steps[-1]
    assert second.steps[0].status == HelixJobStatus.COMPLETED
    h.release(job, latest.id)
    final = h.settled(job)
    assert len(final.steps) == 2 and final.workflow is not None and final.workflow.nodes[0].freshness == "clean"
    containers = sorted(h.containers(job), key=lambda c: container_state(c).started_at)
    assert container_state(containers[0]).finished_at <= container_state(containers[1]).started_at


def test_docker_old_and_duplicate_callbacks_cannot_satisfy_new_revision(docker_workflow: DockerWorkflowHarness) -> None:
    from nemo_helix_plugin.client.errors import NemoHTTPError
    from nemo_helix_plugin.jobs.types import HelixJobStatusUpdateRequest, HelixJobTaskUpdate

    h = docker_workflow
    job = h.create({"work": []}, commands={"work": GATED_WORK})
    first = h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    old = first.steps[0]
    task = h.containers(job)[0].labels[JOB_TASK_ID_LABEL]
    h.release(job, old.id)
    h.settled(job)
    h.invalidate(job, "work")
    running = h.wait_for(
        job, lambda state: len(state.steps) == 2 and any(step.status == HelixJobStatus.ACTIVE for step in state.steps)
    )
    assert running.workflow is not None and job.attempt_id is not None
    node = running.workflow.nodes[0]
    revision, active = node.requested_run_revision, node.active_execution_id
    for _ in range(2):
        h.jobs.update_attempt_step_task(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            step=old.id,
            name=task,
            body=HelixJobTaskUpdate(status=HelixJobStatus.COMPLETED, status_details={"late": True}),
        )
    with pytest.raises(NemoHTTPError) as error:
        h.jobs.update_attempt_step_status(
            workspace="default",
            job=job.id,
            attempt=job.attempt_id,
            name=old.id,
            body=HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE),
        )
    assert error.value.status_code == 409
    after = h.tick(job)
    assert after.workflow is not None
    current = after.workflow.nodes[0]
    assert current.requested_run_revision == revision and current.active_execution_id == active
    assert current.freshness == "dirty" and current.successful_execution_id == old.id
    assert active is not None
    h.release(job, active)
    final = h.settled(job)
    assert final.workflow is not None and final.workflow.nodes[0].successful_execution_id == active
    for step in final.steps:
        assert {row.job_step for row in h.logs(job, step.name)} == {step.name}


def test_launcher_cannot_spawn_command_and_docker_does_not_auto_retry(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.jobs.create_job(
        workspace="default",
        body=CreateHelixJobRequest(
            name=h.prefix + "-launch",
            source="docker-workflow-test",
            spec={},
            platform_spec=HelixJobSpec(
                workflow=WorkflowSpec(mode="continuous"),
                steps=[
                    HelixJobStepSpec(
                        name="broken",
                        depends_on=[],
                        executor=CPUExecutionProvider(
                            provider="cpu",
                            profile=h.prefix,
                            container=ContainerSpec(
                                image="busybox:stable", command=["/definitely-missing-dag-command"]
                            ),
                        ),
                    )
                ],
            ),
        ),
    ).data()
    h.created.append(job)
    failed = h.settled(job)
    assert failed.workflow is not None and failed.workflow.degraded and failed.status == HelixJobStatus.ACTIVE
    container = h.containers(job)[0]
    assert container_state(container).exit_code != 0
    assert b"[launcher]" in container.logs() and b"no such file" in container.logs()
    assert len(h.tick(job).steps) == 1
    h.invalidate(job, "broken")
    assert len(h.settled(job).steps) == 2
    assert len(h.tick(job).steps) == 2


def test_docker_active_timeout_drains_once_workflow_to_error(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    h.profile.config.ttl_seconds_active = 1
    h.restart_controller()
    job = h.create({"root": [], "child": ["root"]}, mode="run_to_completion", commands={"root": "sleep 30"})
    ended = h.wait_for(job, lambda state: state.status.is_terminal())
    assert ended.status == HelixJobStatus.ERROR and len(ended.steps) == 1
    assert "TTL" in str(ended.steps[0].error_details)
    assert all(c.status != "running" for c in h.containers(job))
    assert h.tick(job).status == HelixJobStatus.ERROR


def test_docker_pause_retention_reclaims_actual_job_storage(docker_workflow: DockerWorkflowHarness) -> None:
    from nemo_helix_plugin.client.errors import NemoHTTPError
    from nemo_helix_plugin.jobs.types import JobControl

    h = docker_workflow
    job = h.create({"work": []}, commands={"work": "echo saved > /workflow/checkpoint; echo saved"})
    h.settled(job)
    assert h.storage_exists(job)
    h.jobs.update_job_control(workspace="default", name=job.id, body=JobControl(pause_ttl_seconds=0))
    h.jobs.pause_job(workspace="default", name=job.id)
    expired = h.wait_for(job, lambda state: state.status == HelixJobStatus.CANCELLED and not h.storage_exists(job))
    assert expired.status_details["storage_reclaimed_at"]
    with pytest.raises(NemoHTTPError) as error:
        h.jobs.resume_job(workspace="default", name=job.id)
    assert error.value.status_code == 409


def test_launcher_child_uses_captured_config_after_job_definition_edit(docker_workflow: DockerWorkflowHarness) -> None:
    import json

    from nemo_helix_plugin.jobs.types import UpdateHelixJobRequest

    h = docker_workflow
    job = h.create(
        {"root": [], "child": ["root"]},
        mode="run_to_completion",
        commands={
            "root": GATED_WORK,
            "child": 'cat "$NEMO_JOB_STEP_CONFIG_FILE_PATH"; echo original-child',
        },
    )
    active = h.wait_for(job, lambda state: any(step.status == HelixJobStatus.ACTIVE for step in state.steps))
    h.jobs.update_job(
        workspace="default",
        name=job.id,
        body=UpdateHelixJobRequest(
            spec={"revision": 2},
            platform_spec=HelixJobSpec(
                workflow=WorkflowSpec(mode="continuous"),
                steps=[
                    HelixJobStepSpec(
                        name="future",
                        depends_on=[],
                        config={"logical_name": "future"},
                        executor=CPUExecutionProvider(
                            provider="cpu",
                            profile=h.prefix,
                            container=ContainerSpec(image="busybox:stable", command=["false"]),
                        ),
                    ),
                ],
            ),
        ),
    )
    h.release(job, active.steps[0].id)
    ended = h.wait_for(job, lambda state: state.status.is_terminal())
    assert ended.status == HelixJobStatus.COMPLETED and len(ended.steps) == 2
    assert ended.workflow is not None and ended.workflow.mode == "run_to_completion"
    logs = h.logs(job)
    assert "original-child" in {row.message for row in logs}
    assert {"logical_name": "child"} in [json.loads(row.message) for row in logs if row.message.startswith("{")]


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_docker_node_cancel_prunes_branch_and_continuous_inputs_retry(
    docker_workflow: DockerWorkflowHarness, mode: WorkflowMode
) -> None:
    from nemo_helix_plugin.jobs.types import HelixJobStatusUpdateRequest

    h = docker_workflow
    source = h.files.create_fileset(
        workspace="default", body=CreateFilesetRequest(name=h.prefix + "-cancel-input")
    ).data()
    job = h.create(
        {"stop": [], "blocked": ["stop"], "leaf": ["blocked"], "independent": [], "tail": ["independent"]},
        mode=mode,
        inputs={"stop": [source.id]},
        commands={"stop": GATED_WORK, "independent": GATED_WORK},
    )
    running = h.wait_for(
        job, lambda state: len(state.steps) == 2 and all(step.status == HelixJobStatus.ACTIVE for step in state.steps)
    )
    assert job.attempt_id is not None and running.workflow is not None
    stop_id = next(node.active_execution_id for node in running.workflow.nodes if node.name == "stop")
    independent_id = next(node.active_execution_id for node in running.workflow.nodes if node.name == "independent")
    assert stop_id is not None and independent_id is not None
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=stop_id,
        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.CANCELLING),
    )
    pruned = h.wait_for(
        job,
        lambda state: (
            state.workflow is not None
            and all(
                node.last_outcome == "cancelled"
                for node in state.workflow.nodes
                if node.name in ("stop", "blocked", "leaf")
            )
        ),
    )
    assert pruned.status == HelixJobStatus.ACTIVE and len(pruned.steps) == 2
    assert next(step for step in pruned.steps if step.id == independent_id).status == HelixJobStatus.ACTIVE
    if mode == "continuous":
        h.files.commit_fileset(workspace="default", name=source.name)
        assert len(h.tick(job).steps) == 2
    h.release(job, independent_id)
    drained = h.wait_for(
        job,
        lambda state: (
            len(state.steps) == 3
            and all(step.status.is_terminal() for step in state.steps)
            and (mode == "continuous" or state.status.is_terminal())
        ),
    )
    assert drained.workflow is not None and drained.workflow.pass_number == 1
    if mode == "run_to_completion":
        assert drained.status == HelixJobStatus.CANCELLED
        assert h.tick(job).status == HelixJobStatus.CANCELLED
    else:
        assert drained.status == HelixJobStatus.ACTIVE
        retry = h.wait_for(job, lambda state: len(state.steps) == 4 and state.steps[-1].status == HelixJobStatus.ACTIVE)
        assert retry.workflow is not None and retry.workflow.pass_number == 2
        retry_id = next(node.active_execution_id for node in retry.workflow.nodes if node.name == "stop")
        assert retry_id is not None and retry_id != stop_id
        h.release(job, retry_id)
        recovered = h.settled(job)
        assert len(recovered.steps) == 6
        assert recovered.workflow is not None and not recovered.workflow.degraded
        assert all(node.freshness == "clean" for node in recovered.workflow.nodes)
    assert all(container.status != "running" for container in h.containers(job))


def test_docker_failed_ancestor_blocks_cached_middle_and_dirty_leaf(docker_workflow: DockerWorkflowHarness) -> None:
    h = docker_workflow
    job = h.create(
        {"leaf": ["middle"], "middle": ["root"], "root": [], "independent": []},
        commands={
            "root": (
                "if test -f /workflow/fail-next; then rm /workflow/fail-next; exit 7; "
                "else touch /workflow/fail-next; echo success; fi"
            ),
        },
    )
    first = h.settled(job)
    assert len(first.steps) == 4
    assert first.workflow is not None
    middle_id = next(node.successful_execution_id for node in first.workflow.nodes if node.name == "middle")
    for name in ("root", "leaf", "independent"):
        h.invalidate(job, name)
    failed = h.settled(job)
    assert failed.workflow is not None and failed.workflow.degraded and failed.workflow.pass_number == 2
    assert len(failed.steps) == 6
    leaf = next(node for node in failed.workflow.nodes if node.name == "leaf")
    assert leaf.has_pending_run and leaf.blocked_by == ["middle"]
    for _ in range(3):
        waiting = h.tick(job)
        assert waiting.workflow is not None and waiting.workflow.pass_number == 2
        assert waiting.workflow.waiting_for_changes and len(waiting.steps) == 6
    h.invalidate(job, "root")
    recovered = h.settled(job)
    assert recovered.attempt_id == first.attempt_id
    assert len(recovered.steps) == 8
    assert recovered.workflow is not None and not recovered.workflow.degraded
    assert next(node.successful_execution_id for node in recovered.workflow.nodes if node.name == "middle") == middle_id
    assert sum(container_state(container).exit_code == 7 for container in h.containers(job)) == 1


@pytest.mark.parametrize("mode", ["continuous", "run_to_completion"])
def test_docker_failure_during_workflow_cancellation(
    docker_workflow: DockerWorkflowHarness, mode: WorkflowMode
) -> None:
    h = docker_workflow
    job = h.create(
        {"failure": [], "other": [], "child": ["failure", "other"]},
        mode=mode,
        commands={"failure": GATED_WORK + "; echo failed-before-cancel >&2; exit 7", "other": GATED_WORK},
    )
    running = h.wait_for(
        job, lambda state: len(state.steps) == 2 and all(step.status == HelixJobStatus.ACTIVE for step in state.steps)
    )
    assert running.workflow is not None
    failure_id = next(node.active_execution_id for node in running.workflow.nodes if node.name == "failure")
    other_id = next(node.active_execution_id for node in running.workflow.nodes if node.name == "other")
    assert failure_id is not None and other_id is not None
    failure_name = next(step.name for step in running.steps if step.id == failure_id)
    container = next(c for c in h.containers(job) if c.labels[JOB_STEP_NAME_LABEL] == failure_name)
    h.release(job, failure_id)
    assert container.wait(timeout=15)["StatusCode"] == 7
    # Publish the real backend error before the DAG harvest, then accept cancel.
    context = next(
        step for step in h.jobs.list_steps(name=job.name, workspace="default").items() if step.id == failure_id
    )
    update = h.registry.get_backend(provider=h.profile.provider, profile=h.profile.profile).sync(context)
    assert update.status == HelixJobStatus.ERROR and job.attempt_id is not None
    h.jobs.update_attempt_step_status(
        workspace="default",
        job=job.id,
        attempt=job.attempt_id,
        name=failure_id,
        body=HelixJobStatusUpdateRequest(
            status=update.status, status_details=update.status_details, error_details=update.error_details
        ),
    )
    h.jobs.cancel_job(workspace="default", name=job.id)
    ended = h.wait_for(job, lambda state: state.status.is_terminal())
    assert ended.status == HelixJobStatus.CANCELLED and len(ended.steps) == 2
    assert next(step.status for step in ended.steps if step.id == failure_id) == HelixJobStatus.ERROR
    assert next(step.status for step in ended.steps if step.id == other_id) == HelixJobStatus.CANCELLED
    assert ended.workflow is not None
    assert next(node.last_outcome for node in ended.workflow.nodes if node.name == "failure") == "error"
    assert "failed-before-cancel" in {row.message for row in h.logs(job)}
    assert all(c.status != "running" for c in h.containers(job))
    assert h.tick(job).status == HelixJobStatus.CANCELLED
