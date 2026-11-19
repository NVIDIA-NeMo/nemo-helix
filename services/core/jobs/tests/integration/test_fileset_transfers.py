# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real launcher transfers through in-process Jobs and Files services."""

import os
import re
import subprocess
import time
from contextlib import closing
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider, SubprocessExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.transfers import FilesetPull, FilesetPush, StepFilesets
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nhx.common.config import HelixConfig
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.files.app.backends.local import LocalStorageConfig
from nhx.core.files.config import FilesConfig
from nhx.core.files.service import FilesService
from nhx.core.jobs.config import JobsServiceConfig
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.backends.subprocess import (
    SubprocessJobExecutionProfile,
    SubprocessJobExecutionProfileConfig,
)
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.controllers.scheduler import JobScheduler
from nhx.core.jobs.service import JobsService
from nhx.testing import ClientContext, create_test_client
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Request, Response

import docker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NHX_TEST_FILESET_TRANSFERS") != "1", reason="opt-in real launcher transfer test"
    ),
]


@pytest.fixture(scope="session")
def transfer_launchers(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2] / "jobs-launcher"
    directory = tmp_path_factory.mktemp("transfer-launchers")
    host = directory / "jobs-launcher-host"
    linux = directory / "jobs-launcher-linux"
    subprocess.run(["go", "build", "-mod=readonly", "-o", str(host), "./main.go"], cwd=root, check=True)
    with closing(docker.from_env()) as client:
        architecture = client.images.get("busybox:stable").attrs["Architecture"]
    subprocess.run(
        ["go", "build", "-mod=readonly", "-o", str(linux), "./main.go"],
        cwd=root,
        env={**os.environ, "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": architecture},
        check=True,
    )
    return host, linux


@pytest.mark.parametrize("backend", ["docker", "local"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_task_pull_run_upload_commit(
    backend: Literal["docker", "local"],
    exit_code: int,
    transfer_launchers: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "transfer-e2e-" + uuid4().hex[:10]
    proxy = HTTPServer(host="0.0.0.0", port=0)
    requests: list[tuple[str, str]] = []

    def forward(request: Request) -> Response:
        requests.append((request.method, request.path))
        response = context.client.http_client.request(
            request.method,
            context.client.base_url + request.full_path.rstrip("?"),
            content=request.data,
            headers={"Content-Type": request.content_type or "application/octet-stream"},
        )
        return Response(
            response.content, status=response.status_code, content_type=response.headers.get("content-type")
        )

    proxy.expect_request(re.compile("/apis/.*")).respond_with_handler(forward)
    proxy.start()
    if backend == "docker":
        profile = DockerJobExecutionProfile(
            provider="cpu",
            profile=prefix,
            config=DockerJobExecutionProfileConfig(
                cleanup_completed_jobs_immediately=False,
                launcher_tool_path=str(transfer_launchers[1]),
                networking={"job_container_network": "bridge"},
                storage={"volume_name": prefix + "-storage", "volume_permissions_image": "busybox:stable"},
            ),
        )
        input_dir, output_dir = "/input", "/output"
        command = ["sh", "-c", f"mkdir -p /output; cat /input/input.txt > /output/result.txt; exit {exit_code}"]
        executor = CPUExecutionProvider(
            provider="cpu", profile=prefix, container=ContainerSpec(image="busybox:stable", command=command)
        )
        platform_host = "host.docker.internal"
    else:
        profile = SubprocessJobExecutionProfile(
            profile=prefix,
            config=SubprocessJobExecutionProfileConfig(
                working_directory=str(tmp_path / "processes"), launcher_tool_path=str(transfer_launchers[0])
            ),
        )
        input_dir, output_dir = str(tmp_path / "input"), str(tmp_path / "output")
        command = [
            "/bin/sh",
            "-c",
            f"mkdir -p '{output_dir}'; cat '{input_dir}/input.txt' > '{output_dir}/result.txt'; exit {exit_code}",
        ]
        executor = SubprocessExecutionProvider(provider="subprocess", profile=prefix, command=command)
        platform_host = "127.0.0.1"
    config = JobsServiceConfig(executors=[profile], enable_subprocess_executor=backend == "local")
    monkeypatch.setattr("nhx.core.jobs.api.v2.jobs.endpoints.profiles", config.executors)
    try:
        with create_test_client(
            JobsService,
            FilesService,
            client_type=ClientContext,
            service_configs={
                JobsService: config,
                FilesService: FilesConfig(default_storage_config=LocalStorageConfig(path=str(tmp_path / "files"))),
                HelixConfig: HelixConfig(
                    base_url="http://127.0.0.1", service_discovery={"platform": f"http://{platform_host}:{proxy.port}"}
                ),
            },
        ) as context:
            jobs_client = JobsClient.from_client(context.client)
            files_client = FilesClient.from_client(context.client)
            source = files_client.create_fileset(
                workspace="default",
                body=CreateFilesetRequest(name=prefix + "-source"),
            ).data()
            target = files_client.create_fileset(
                workspace="default",
                body=CreateFilesetRequest(name=prefix + "-target"),
            ).data()
            files_client.upload_file(
                workspace="default", name=source.name, path="input.txt", content=b"real transfer bytes\x00\xff"
            )
            before = target.commit_revision
            job = jobs_client.create_job(
                workspace="default",
                body=CreateHelixJobRequest(
                    name=prefix,
                    source="transfer-test",
                    spec={},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStepSpec(
                                name="work",
                                executor=executor,
                                filesets=StepFilesets(
                                    pull=[FilesetPull(fileset_id=source.id, directory=input_dir)],
                                    push=[FilesetPush(fileset_id=target.id, directory=output_dir)],
                                ),
                            )
                        ]
                    ),
                ),
            ).data()
            registry = BackendRegistry.from_config(nemo_client=context.client, profiles=config.executors)
            scheduler, reconciler = JobScheduler(registry, context.client), JobReconciler(registry, context.client)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    reconciler.step()
                    scheduler.step()
                    state = jobs_client.get_job_status(workspace="default", name=job.id).data()
                    if state.status.is_terminal():
                        break
                    time.sleep(0.05)
                assert state.status == (HelixJobStatus.COMPLETED if exit_code == 0 else HelixJobStatus.ERROR), (
                    state.model_dump_json(indent=2)
                )
                assert (
                    files_client.download_file(workspace="default", name=target.name, path="result.txt").read()
                    == b"real transfer bytes\x00\xff"
                )
                after = files_client.get_fileset_by_id(workspace="default", fileset_id=target.id).data()
                assert after.commit_revision != before
                commits = [path for method, path in requests if method == "POST" and path.endswith("/commit")]
                # The controller commits through its typed in-process Files client;
                # Task launchers only upload, so no Task commits appear at the proxy.
                assert commits == []
            finally:
                registry.shutdown_all_backends()
    finally:
        if backend == "docker":
            with closing(docker.from_env()) as client:
                for container in client.containers.list(all=True):
                    if prefix in container.name:
                        container.remove(force=True, v=True)
                for volume in client.volumes.list():
                    if prefix in volume.name:
                        volume.remove()
        proxy.stop()


def test_two_task_containers_upload_without_shared_output_storage(
    transfer_launchers: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import tarfile

    from nemo_helix_plugin.jobs.providers import ComputeResources, DistributedGPUExecutionProvider
    from nemo_helix_plugin.jobs.types import HelixJobStatusUpdateRequest
    from nhx.core.jobs.controllers.backends.kubernetes.volcano_job import (
        VolcanoJobExecutionProfile,
        VolcanoJobExecutionProfileConfig,
    )

    prefix = "transfer-multi-" + uuid4().hex[:10]
    proxy = HTTPServer(host="0.0.0.0", port=0)

    def forward(request: Request) -> Response:
        response = context.client.http_client.request(
            request.method,
            context.client.base_url + request.full_path.rstrip("?"),
            content=request.data,
            headers={"Content-Type": request.content_type or "application/octet-stream"},
        )
        return Response(
            response.content, status=response.status_code, content_type=response.headers.get("content-type")
        )

    proxy.expect_request(re.compile("/apis/.*")).respond_with_handler(forward)
    proxy.start()
    # The distributed profile supplies the real two-participant contract. This
    # test launches its Tasks in Docker, without a Kubernetes/GPU dependency.
    profile = VolcanoJobExecutionProfile(
        provider="gpu_distributed", profile=prefix, config=VolcanoJobExecutionProfileConfig()
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
                FilesService: FilesConfig(default_storage_config=LocalStorageConfig(path=str(tmp_path / "files"))),
            },
        ) as context:
            jobs_client = JobsClient.from_client(context.client)
            files_client = FilesClient.from_client(context.client)
            source = files_client.create_fileset(
                workspace="default", body=CreateFilesetRequest(name=prefix + "-source")
            ).data()
            target = files_client.create_fileset(
                workspace="default", body=CreateFilesetRequest(name=prefix + "-target")
            ).data()
            files_client.upload_file(
                workspace="default", name=source.name, path="input.txt", content=b"distributed Task input"
            )
            before = target.commit_revision
            transfers = StepFilesets(
                pull=[FilesetPull(fileset_id=source.id, directory="/input")],
                push=[FilesetPush(fileset_id=target.id, directory="/output")],
            )
            job = jobs_client.create_job(
                workspace="default",
                body=CreateHelixJobRequest(
                    name=prefix,
                    source="transfer-test",
                    spec={},
                    platform_spec=HelixJobSpec(
                        steps=[
                            HelixJobStepSpec(
                                name="work",
                                executor=DistributedGPUExecutionProvider(
                                    provider="gpu_distributed",
                                    profile=prefix,
                                    container=ContainerSpec(image="busybox:stable"),
                                    resources=ComputeResources(num_nodes=2, num_gpus=0),
                                ),
                                filesets=transfers,
                            )
                        ]
                    ),
                ),
            ).data()
            assert job.attempt_id is not None
            for status in (HelixJobStatus.PENDING, HelixJobStatus.ACTIVE):
                jobs_client.update_attempt_step_status(
                    workspace="default",
                    job=job.id,
                    attempt=job.attempt_id,
                    name="work",
                    body=HelixJobStatusUpdateRequest(status=status),
                )
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as tar:
                tar.add(transfer_launchers[1], arcname="jobs-launcher")
            with closing(docker.from_env()) as client:
                containers = []
                try:
                    for rank in range(2):
                        # The worker keeps running after writing its shard. The
                        # leader's script completion must stop that process while
                        # preserving the worker launcher and its private output.
                        delay = "0.4" if rank == 0 else "30"
                        command = (
                            f"mkdir -p /output; cat /input/input.txt > /output/shard-$NEMO_JOB_TASK.txt; sleep {delay}"
                        )
                        container = client.containers.create(
                            "busybox:stable",
                            name=f"{prefix}-{rank}",
                            command=["/jobs-launcher", "run", "--require-fileset-transfers", "--", "sh", "-c", command],
                            environment={
                                "NHX_BASE_URL": f"http://host.docker.internal:{proxy.port}",
                                "NEMO_JOB_WORKSPACE": "default",
                                "NEMO_JOB_UID": job.id,
                                "NEMO_JOB_ID": job.name,
                                "NEMO_JOB_ATTEMPT_ID": job.attempt_id,
                                "NEMO_JOB_STEP": "work",
                                "NEMO_JOB_TASK": f"task-{rank}",
                                "NEMO_JOB_FILESET_TRANSFERS": transfers.model_dump_json(),
                                "NEMO_JOB_TRANSFER_LEADER": "true" if rank == 0 else "false",
                            },
                        )
                        containers.append(container)
                        container.put_archive("/", archive.getvalue())
                        container.start()
                    for container in containers:
                        assert container.wait(timeout=30)["StatusCode"] == 0
                        container.reload()
                        assert container.attrs["Mounts"] == []
                    paths = {
                        entry.path
                        for entry in files_client.list_files(workspace="default", name=target.name).data().data
                    }
                    assert paths == {"shard-task-0.txt", "shard-task-1.txt"}
                    for path in paths:
                        assert (
                            files_client.download_file(workspace="default", name=target.name, path=path).read()
                            == b"distributed Task input"
                        )
                    published = files_client.get_fileset_by_id(workspace="default", fileset_id=target.id).data()
                    assert published.commit_revision != before
                    active = jobs_client.get_attempt_step(
                        workspace="default", job=job.id, attempt=job.attempt_id, name="work"
                    ).data()
                    assert active.status == HelixJobStatus.ACTIVE
                    jobs_client.update_attempt_step_status(
                        workspace="default",
                        job=job.id,
                        attempt=job.attempt_id,
                        name="work",
                        body=HelixJobStatusUpdateRequest(status=HelixJobStatus.COMPLETED),
                    )
                    assert (
                        jobs_client.get_job_status(workspace="default", name=job.id).data().status
                        == HelixJobStatus.COMPLETED
                    )
                    assert (
                        files_client.get_fileset_by_id(workspace="default", fileset_id=target.id).data().commit_revision
                        == published.commit_revision
                    )
                finally:
                    for container in containers:
                        container.remove(force=True, v=True)
    finally:
        proxy.stop()
