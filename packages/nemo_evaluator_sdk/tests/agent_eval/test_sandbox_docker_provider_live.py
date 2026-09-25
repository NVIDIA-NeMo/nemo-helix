# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live Docker smoke test for the sandbox provider + facade.

Exercises the real ``docker run/exec/cp/rm`` plumbing (no fabric image needed — plain
busybox from NVIDIA's Docker Hub mirror), so it validates the boundary-crossing
inject/exec/retrieve loop end to end.
Skipped automatically where Docker is unavailable, mirroring how NeMo Gym guards its
real-Apptainer tests.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.api import AsyncSandbox
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxSpec
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.providers.docker import DockerSandboxProvider

# NVIDIA Docker Hub mirror (see Platform-Deploy setup_docker_e2e.sh). Unqualified
# ``busybox:latest`` resolves to docker.io and fails when Hub auth is unavailable.
_IMAGE = "dockerhub.nvidia.com/busybox:latest"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except subprocess.TimeoutExpired:
        return False  # an unresponsive daemon skips the live tests rather than hanging the run


pytestmark = pytest.mark.skipif(not _docker_ready(), reason="docker daemon not available")


async def test_inject_exec_retrieve_roundtrip(tmp_path: Path) -> None:
    provider = DockerSandboxProvider()
    spec = SandboxSpec(
        image=_IMAGE,
        workdir="/work",
        env={"GREETING": "hello"},
        files={"/work/seed.txt": "seed-content"},
    )
    async with AsyncSandbox(provider, spec) as sandbox:
        await sandbox.start()

        # Env + seeded file visible inside the sandbox.
        echo = await sandbox.exec("echo $GREETING")
        assert echo.ok and echo.stdout is not None and echo.stdout.strip() == "hello"
        seeded = await sandbox.exec("cat /work/seed.txt")
        assert seeded.stdout is not None and seeded.stdout.strip() == "seed-content"

        # The agent "does work": derive an output file under /out.
        work = await sandbox.exec("mkdir -p /out && tr a-z A-Z < /work/seed.txt > /out/result.txt")
        assert work.ok

        # Upload a dir, then retrieve the /out tree across the boundary.
        (tmp_path / "extra").mkdir()
        (tmp_path / "extra" / "note.md").write_text("uploaded", encoding="utf-8")
        await sandbox.upload_dir(tmp_path / "extra", "/work/extra")
        extra = await sandbox.exec("cat /work/extra/note.md")
        assert extra.stdout is not None and extra.stdout.strip() == "uploaded"

        out_dir = tmp_path / "out"
        await sandbox.download_dir("/out", out_dir)
        assert (out_dir / "result.txt").read_text(encoding="utf-8").strip() == "SEED-CONTENT"


async def test_exec_reports_command_failure(tmp_path: Path) -> None:
    provider = DockerSandboxProvider()
    async with AsyncSandbox(provider, SandboxSpec(image=_IMAGE, workdir="/work")) as sandbox:
        await sandbox.start()
        result = await sandbox.exec("exit 7")
        assert not result.ok and result.return_code == 7 and result.error_type is None
