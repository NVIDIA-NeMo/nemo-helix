# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from bundle import BOOTSTRAP, build_zip, job_request
from config import PodSize
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider

WORKER = Path(__file__).resolve().parents[1] / "worker"
POD = PodSize("100m", "256Mi", "500m", "512Mi")


def test_zip_contains_only_worker_modules() -> None:
    payload = build_zip(WORKER)
    names = zipfile.ZipFile(io.BytesIO(payload)).namelist()
    assert names == ["hammer.py", "history.py", "main.py"]
    assert not any("nemo_helix_plugin" in name or name.endswith("Dockerfile") for name in names)


def test_job_request_uses_the_profile_image_and_does_not_install_packages() -> None:
    request = job_request("stress-exit-01-1", {"STRESS_KIND": "exit"}, POD, "default")
    executor = request.platform_spec.steps[0].executor
    assert isinstance(executor, CPUExecutionProvider)
    assert executor.container.image is None
    command = "\n".join(executor.container.command)
    assert "pip" not in command
    assert "uv " not in command
    assert "pip" not in BOOTSTRAP
    assert "uv " not in BOOTSTRAP


def test_gitignore_ignores_generated_runs_and_reports() -> None:
    text = (Path(__file__).resolve().parents[1] / ".gitignore").read_text()
    assert "runs/" in text
    assert "output/" in text
