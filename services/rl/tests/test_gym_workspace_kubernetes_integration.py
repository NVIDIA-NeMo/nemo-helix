# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify GRPO workspace preparation on a disposable kind cluster.

Run with KUBECONFIG pointing at that cluster and NHX_GYM_WORKSPACE_KIND naming it
(without the ``kind-`` prefix). NHX_KIND_BINARY can select a kind executable.
This test builds a small CPU image and never submits a training job or needs
OpenSandbox. It exercises the Kubernetes PVC/subPath handoff OpenSandbox uses.
The unprepared case preserves the original PermissionError as a negative control.
The prepared cases run the production helper inside the training pod, then verify
wheel installation as both the training UID and a different sandbox UID.
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from kubernetes import client, config
from nhx.core.jobs.controllers.backends.kubernetes.common import build_permissions_init_container
from nhx.rl.tasks.training.backends.nemo_rl.sandbox_config import build_sandbox_mounts

pytestmark = pytest.mark.integration

_PACKAGE = "gym_workspace_probe"
_WHEEL = f"{_PACKAGE}-1.0-py3-none-any.whl"
_PROBE = """\
import importlib
import os
import shutil
from pathlib import Path
from sandboxed_gym.runtime import gym_host_runtime as runtime

work = Path('/job/work')
metadata = work.stat()
print(f'uid={os.getuid()} gid={os.getgid()} work_uid={metadata.st_uid} '
      f'work_gid={metadata.st_gid} work_mode={oct(metadata.st_mode & 0o777)}', flush=True)
shutil.copytree('/bundle', '/tmp/environment', ignore=shutil.ignore_patterns('..*'))
package = runtime._load_runtime_environment_package('/tmp/environment', required=True)
runtime._install_wheels_v1_dependencies(package, str(work))
assert importlib.import_module('gym_workspace_probe').VALUE == 42
print('wheel installed and imported', flush=True)
"""


def _run(*command: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stdout}\n{result.stderr}"


@pytest.fixture(scope="module")
def workspace_cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[client.CoreV1Api, str, str]]:
    cluster = os.environ.get("NHX_GYM_WORKSPACE_KIND")
    if not cluster:
        pytest.skip("set NHX_GYM_WORKSPACE_KIND to a disposable local kind cluster name")
    kind = os.environ.get("NHX_KIND_BINARY") or shutil.which("kind")
    assert kind, "kind is required when NHX_GYM_WORKSPACE_KIND is set"
    docker = shutil.which("docker")
    assert docker, "Docker is required to build the CPU probe image"

    repo = Path(__file__).resolve().parents[3]
    build_dir = tmp_path_factory.mktemp("gym-workspace-image")
    dockerfile = build_dir / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.13.15-slim-bookworm\n"
        "COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv\n"
        "RUN uv pip install --system pydantic==2.12.0 PyYAML==6.0.2\n"
        "COPY src/sandboxed_gym /opt/src/sandboxed_gym\n"
        "COPY --from=rl-source sandbox_config.py /opt/src/grpo_sandbox_config.py\n"
        "ENV PYTHONPATH=/opt/src HOME=/tmp UV_CACHE_DIR=/tmp/uv\n",
        encoding="utf-8",
    )
    image = f"gym-workspace-integration:{uuid4().hex[:12]}"
    namespace = f"gym-workspace-{uuid4().hex[:12]}"
    with config.new_client_from_config(context=f"kind-{cluster}") as api_client:
        core = client.CoreV1Api(api_client)
        namespace_created = False
        image_built = False
        try:
            _run(
                docker,
                "build",
                "--build-context",
                f"rl-source={repo / 'services/rl/src/nhx/rl/tasks/training/backends/nemo_rl'}",
                "-t",
                image,
                "-f",
                str(dockerfile),
                str(repo / "packages/sandboxed_gym"),
            )
            image_built = True
            _run(kind, "load", "docker-image", "--name", cluster, image)
            core.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace)))
            namespace_created = True
            core.create_namespaced_persistent_volume_claim(
                namespace,
                client.V1PersistentVolumeClaim(
                    metadata=client.V1ObjectMeta(name="storage"),
                    spec=client.V1PersistentVolumeClaimSpec(
                        access_modes=["ReadWriteOnce"],
                        resources=client.V1VolumeResourceRequirements(requests={"storage": "128Mi"}),
                    ),
                ),
            )
            core.create_namespaced_config_map(namespace, _bundle())
            yield core, namespace, image
        finally:
            if namespace_created:
                core.delete_namespace(namespace)
            if image_built:
                _run(docker, "image", "rm", image)


def _bundle() -> client.V1ConfigMap:
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{_PACKAGE}/__init__.py", "VALUE = 42\n")
        archive.writestr(
            f"{_PACKAGE}-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {_PACKAGE}\nVersion: 1.0\n",
        )
        archive.writestr(
            f"{_PACKAGE}-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{_PACKAGE}-1.0.dist-info/RECORD", "")
    return client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name="bundle"),
        data={
            "probe.py": _PROBE,
            "nemo-environment.yaml": (
                "format: wheels-v1\nconfig_paths:\n"
                "  - resources_servers/probe/configs/probe.yaml\n"
                "metadata:\n  name: workspace-probe\n"
            ),
            "config": "probe: {}\n",
            "requirements": f"{_PACKAGE}==1.0\n",
        },
        binary_data={"wheel": base64.b64encode(wheel.getvalue()).decode("ascii")},
    )


def _wait(core: client.CoreV1Api, namespace: str, name: str) -> tuple[str, str]:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        pod = core.read_namespaced_pod(name, namespace)
        if pod.status.phase in {"Succeeded", "Failed"}:
            with core.read_namespaced_pod_log(name, namespace, container="probe", _preload_content=False) as response:
                return pod.status.phase, response.data.decode("utf-8")
        time.sleep(1)
    events = core.list_namespaced_event(namespace, field_selector=f"involvedObject.name={name}")
    pytest.fail(f"pod {name} did not finish: {pod.status}\nevents: {events}")


def _pod(name: str, image: str, command: list[str]) -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1PodSpec(
            restart_policy="Never",
            automount_service_account_token=False,
            containers=[
                client.V1Container(
                    name="probe",
                    image=image,
                    image_pull_policy="Never",
                    command=command,
                    security_context=client.V1SecurityContext(run_as_user=1000, run_as_group=1000),
                    resources=client.V1ResourceRequirements(requests={"cpu": "100m", "memory": "128Mi"}),
                )
            ],
            volumes=[
                client.V1Volume(
                    name="storage",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name="storage"),
                )
            ],
        ),
    )


@pytest.mark.parametrize(
    ("prepare_workspace", "sandbox_uid"),
    [(False, 1000), (True, 1000), (True, 2000)],
    ids=["unprepared-control", "prepared-workspace", "different-sandbox-uid"],
)
def test_grpo_workspace_handoff(
    workspace_cluster: tuple[client.CoreV1Api, str, str], prepare_workspace: bool, sandbox_uid: int
) -> None:
    core, namespace, image = workspace_cluster
    job = f"job-{uuid4().hex[:8]}"
    mounts = build_sandbox_mounts(
        pvc_claim="storage",
        workspace="default",
        job_id=job,
        storage_root=Path("/training"),
        environment_path="/training/environment",
        dataset_path="/training/dataset",
    )
    preparation = (
        "from pathlib import Path; "
        "from grpo_sandbox_config import prepare_sandbox_workspace; "
        "prepare_sandbox_workspace(Path('/training')); "
        "prepare_sandbox_workspace(Path('/training'))"
        if prepare_workspace
        else "from pathlib import Path; assert not Path('/training/gym-work').exists()"
    )
    prepare = _pod(
        f"{job}-prepare",
        image,
        [
            "python",
            "-c",
            "from pathlib import Path; "
            "Path('/training/environment').mkdir(); "
            "Path('/training/dataset').mkdir(); " + preparation,
        ],
    )
    prepare.spec.init_containers = [
        client.V1Container(
            name="restrict-volume-root",
            image=image,
            image_pull_policy="Never",
            command=["sh", "-ec", "chown 0:0 /vol && chmod 0755 /vol"],
            volume_mounts=[client.V1VolumeMount(name="storage", mount_path="/vol")],
        ),
        build_permissions_init_container("storage", "default", job, image),
    ]
    prepare.spec.containers[0].volume_mounts = [
        client.V1VolumeMount(name="storage", mount_path="/training", sub_path=f"jobs/default/{job}")
    ]
    core.create_namespaced_pod(namespace, prepare)
    phase, logs = _wait(core, namespace, prepare.metadata.name)
    assert phase == "Succeeded", logs

    sandbox = _pod(f"{job}-sandbox", image, ["python", "/bundle/probe.py"])
    sandbox.spec.containers[0].security_context = client.V1SecurityContext(
        run_as_user=sandbox_uid, run_as_group=sandbox_uid
    )
    sandbox.spec.volumes.append(
        client.V1Volume(
            name="bundle",
            config_map=client.V1ConfigMapVolumeSource(
                name="bundle",
                items=[
                    client.V1KeyToPath(key="probe.py", path="probe.py"),
                    client.V1KeyToPath(key="nemo-environment.yaml", path="nemo-environment.yaml"),
                    client.V1KeyToPath(key="config", path="resources_servers/probe/configs/probe.yaml"),
                    client.V1KeyToPath(key="requirements", path="resources_servers/probe/requirements.txt"),
                    client.V1KeyToPath(key="wheel", path=f"wheels/{_WHEEL}"),
                ],
            ),
        )
    )
    sandbox.spec.containers[0].volume_mounts = [
        client.V1VolumeMount(name="storage", mount_path="/job/work", sub_path=mounts.workspace_sub_path),
        client.V1VolumeMount(name="bundle", mount_path="/bundle", read_only=True),
    ]
    core.create_namespaced_pod(namespace, sandbox)
    phase, logs = _wait(core, namespace, sandbox.metadata.name)
    print(logs)
    assert f"uid={sandbox_uid} gid={sandbox_uid}" in logs
    if prepare_workspace:
        assert phase == "Succeeded", logs
        assert "work_uid=1000 work_gid=1000 work_mode=0o777" in logs
        assert "wheel installed and imported" in logs
    else:
        assert phase == "Failed", logs
        assert "work_uid=0 work_gid=0 work_mode=0o755" in logs
        assert "PermissionError: [Errno 13] Permission denied: '/job/work/wheels-v1-site-packages'" in logs
