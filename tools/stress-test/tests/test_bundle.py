# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import cast

import pytest
from bundle import BOOTSTRAP, WORKER_PATH, build_zip, digest, job_request
from config import PodSize
from helix import HelixCluster
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider, GPUExecutionProvider

WORKER = Path(__file__).resolve().parents[1] / "worker"
POD = PodSize("100m", "256Mi", "500m", "512Mi")


class _Stored:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content


class _Files:
    def __init__(self) -> None:
        self.stored = b""

    def download_file(self, *, workspace: str, name: str, path: str) -> _Stored:
        del workspace, name, path
        return _Stored(self.stored)

    def upload_file(self, *, workspace: str, name: str, path: str, content: bytes) -> object:
        del workspace, name, path
        self.stored = content

        class _Uploaded:
            def data(self) -> None:
                return None

        return _Uploaded()


def test_sync_worker_logs_the_bundle_and_the_send(caplog: pytest.LogCaptureFixture) -> None:
    cluster = HelixCluster.__new__(HelixCluster)
    cluster.workspace = "default"
    cluster.fileset = "stress-stress"
    files = _Files()
    cluster._files = cast(FilesClient, files)
    payload = b"zip-bytes"
    with caplog.at_level(logging.INFO, logger="stress"):
        assert cluster.sync_worker(payload) == digest(payload)
        cluster.sync_worker(payload)
    assert f"bundled {WORKER_PATH} ({len(payload)} bytes)" in caplog.text
    assert f"sent {WORKER_PATH} to the stress-stress fileset" in caplog.text
    assert f"{WORKER_PATH} already matches the stress-stress fileset" in caplog.text
    assert files.stored == payload


def test_zip_contains_only_worker_modules() -> None:
    payload = build_zip(WORKER)
    names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
    assert names == ["barrier.py", "hammer.py", "history.py", "main.py"]
    assert not any("nemo_helix_plugin" in name or name.endswith("Dockerfile") for name in names)


def test_exit_and_idle_are_inline_shell_and_hammers_fetch_the_worker() -> None:
    exit_job = job_request("stress-exit-01-1", {"STRESS_KIND": "exit"}, POD, "default")
    exit_executor = exit_job.platform_spec.steps[0].executor
    assert isinstance(exit_executor, CPUExecutionProvider)
    assert exit_executor.container.entrypoint == ["/bin/bash", "-c"]
    assert exit_executor.container.command == ["exit 0"]
    idle = job_request("stress-idle-01-1", {"STRESS_KIND": "idle"}, POD, "default")
    idle_executor = idle.platform_spec.steps[0].executor
    assert isinstance(idle_executor, CPUExecutionProvider)
    assert idle_executor.container.command == ['sleep "$STRESS_DURATION_SECONDS"']
    hammer = job_request("stress-files-01-1", {"STRESS_KIND": "hammer"}, POD, "default")
    hammer_executor = hammer.platform_spec.steps[0].executor
    assert isinstance(hammer_executor, CPUExecutionProvider)
    assert hammer_executor.container.entrypoint == ["python", "-c"]
    command = "\n".join(hammer_executor.container.command)
    barrier = job_request("stress-barrier-10-1", {"STRESS_KIND": "barrier"}, POD, "default")
    barrier_executor = barrier.platform_spec.steps[0].executor
    assert isinstance(barrier_executor, CPUExecutionProvider)
    assert barrier_executor.container.entrypoint == ["python", "-c"]
    assert "code/worker.zip" in "\n".join(barrier_executor.container.command)
    assert "code/worker.zip" in command
    assert "pip" not in command
    assert "uv " not in command
    assert "pip" not in BOOTSTRAP
    assert "uv " not in BOOTSTRAP
    override = job_request("stress-exit-01-1", {"STRESS_KIND": "exit"}, POD, "default", image="local/nhx-tasks:dev")
    override_executor = override.platform_spec.steps[0].executor
    assert isinstance(override_executor, CPUExecutionProvider)
    assert override_executor.container.image == "local/nhx-tasks:dev"


def test_job_request_keeps_the_provider_with_the_profile_name() -> None:
    request = job_request("stress-exit-01-1", {"STRESS_KIND": "exit"}, POD, "gpu", "cpu")
    executor = request.platform_spec.steps[0].executor
    assert isinstance(executor, CPUExecutionProvider)
    assert executor.profile == "gpu"
    gpu = job_request("stress-exit-01-1", {"STRESS_KIND": "exit"}, POD, "gpu", "gpu")
    gpu_executor = gpu.platform_spec.steps[0].executor
    assert isinstance(gpu_executor, GPUExecutionProvider)
    assert gpu_executor.profile == "gpu"


def test_gitignore_ignores_generated_runs_and_reports() -> None:
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "runs/" in text
    assert "output/" in text
