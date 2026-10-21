# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the worker zip and the job request that fetches it at startup."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from config import PodSize, Step, Sweep
from nemo_helix_plugin.jobs.providers import (
    ComputeResources,
    ComputeResourceSpec,
    ContainerSpec,
    CPUExecutionProvider,
)
from nemo_helix_plugin.jobs.spec import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

WORKER_PATH = "code/worker.zip"
RESULT_NAME = "service-stress"
BOOTSTRAP = """
import os
import sys
import time
import zipfile
from pathlib import Path

from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient

started = time.perf_counter()
workspace = os.environ["NEMO_JOB_WORKSPACE"]
fileset = os.environ["STRESS_FILESET"]
client = get_task_nemo_client("jobs", workspace=workspace)
payload = FilesClient.from_client(client).download_file(
    workspace=workspace, name=fileset, path="code/worker.zip"
).read()
destination = Path("/tmp/stress-worker")
destination.mkdir(parents=True, exist_ok=True)
archive = destination / "worker.zip"
archive.write_bytes(payload)
with zipfile.ZipFile(archive) as zipped:
    zipped.extractall(destination)
os.environ["STRESS_CODE_FETCH_SECONDS"] = str(time.perf_counter() - started)
sys.path.insert(0, str(destination))
import main
main.main()
"""


def build_zip(worker_dir: Path) -> bytes:
    """Zip the worker modules. The SDK stays in the task image."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(worker_dir.glob("*.py")):
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def job_environment(sweep: Sweep, step: Step, fileset: str) -> dict[str, str]:
    """Environment the bootstrap and worker read. Platform job variables stay unset."""
    values = {
        "STRESS_KIND": step.kind,
        "STRESS_STEP_ID": step.id,
        "STRESS_RUN_KEY": sweep.run_key,
        "STRESS_FILESET": fileset,
        "STRESS_CODE_PATH": WORKER_PATH,
        "STRESS_PAYLOAD_BYTES": str(sweep.payload_bytes),
    }
    if step.duration_seconds is not None:
        values["STRESS_DURATION_SECONDS"] = str(step.duration_seconds)
    if step.service is not None:
        values["STRESS_SERVICE"] = step.service
    if step.rate_per_second is not None:
        values["STRESS_RATE"] = str(step.rate_per_second)
    return values


def job_request(name: str, environment: dict[str, str], pod: PodSize, profile_name: str) -> CreateHelixJobRequest:
    """One CPU step on the selected profile. The image stays unset."""
    resources = ComputeResources(
        requests=ComputeResourceSpec(cpu=pod.cpu_request, memory=pod.memory_request),
        limits=ComputeResourceSpec(cpu=pod.cpu_limit, memory=pod.memory_limit),
    )
    container = ContainerSpec(entrypoint=["python", "-c"], command=[BOOTSTRAP])
    executor = CPUExecutionProvider(
        provider="cpu",
        profile=profile_name,
        container=container,
        resources=resources,
    )
    step = HelixJobStepSpec(
        name="run",
        environment=[HelixJobEnvironmentVariable(name=key, value=value) for key, value in sorted(environment.items())],
        executor=executor,
    )
    return CreateHelixJobRequest(
        name=name,
        spec={},
        source="stress-test",
        platform_spec=HelixJobSpec(steps=[step]),
    )
