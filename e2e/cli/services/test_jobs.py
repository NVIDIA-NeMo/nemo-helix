# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for jobs CLI commands."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable

import pytest
from nhx.testing import NemoRun, assert_exit_0

pytestmark = [pytest.mark.timeout(900)]


def _wait_for_job(
    nemo_run: NemoRun,
    job_name: str,
    workspace: str,
    timeout: float = 120.0,
    image_pull_timeout: float = 600.0,
) -> str:
    """Poll `nemo jobs get-status` until a terminal state is reached.

    Time spent in 'pending' (e.g. image pull) is not counted against timeout.
    A separate image_pull_timeout caps how long the job may stay pending.
    Returns final status string.
    """
    terminal = {"completed", "error", "cancelled"}
    start = time.monotonic()
    pending_deadline = start + image_pull_timeout
    deadline = start + timeout
    hard_deadline = start + image_pull_timeout + timeout
    while True:
        now = time.monotonic()
        if now > hard_deadline:
            raise TimeoutError(f"Job '{job_name}' did not reach a terminal state")
        result = nemo_run("jobs", "get-status", job_name, workspace=workspace)
        if result.returncode == 0:
            status = json.loads(result.stdout).get("status", "").lower()
            if status in terminal:
                return status
            if status == "pending":
                deadline = now + timeout
                if now > pending_deadline:
                    raise TimeoutError(f"Job '{job_name}' stuck in pending after {image_pull_timeout}s")
            elif now > deadline:
                raise TimeoutError(f"Job '{job_name}' did not reach a terminal state within {timeout}s")
        time.sleep(2)


def test_jobs_lifecycle(
    workspace: str,
    nemo_run: NemoRun,
    image: Callable[[str], str],
) -> None:
    """Full jobs lifecycle via CLI: create, list, get, get-status, get-logs, delete."""
    job_name = f"e2e-cli-job-{uuid.uuid4().hex[:8]}"
    platform_spec = {
        "steps": [
            {
                "name": "echo-step",
                "executor": {
                    "provider": "cpu",
                    "container": {
                        "image": image("nhx-tasks"),
                        "command": ["echo", "Hello from CLI e2e"],
                    },
                },
            }
        ]
    }

    result = nemo_run(
        "jobs",
        "create",
        job_name,
        "--source",
        "e2e-cli-test",
        "--spec",
        json.dumps({"test": "cli"}),
        "--platform-spec",
        json.dumps(platform_spec),
        workspace=workspace,
    )
    assert_exit_0(result, "jobs create failed")
    assert json.loads(result.stdout).get("name") == job_name

    status = _wait_for_job(nemo_run, job_name, workspace)
    assert status == "completed", f"job ended with status '{status}'"

    result = nemo_run("jobs", "list", workspace=workspace)
    assert_exit_0(result, "jobs list failed")
    assert "data" in json.loads(result.stdout)
    assert job_name in [i.get("name") for i in json.loads(result.stdout).get("data", []) if isinstance(i, dict)]

    result = nemo_run("jobs", "get", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get failed")
    assert json.loads(result.stdout).get("name") == job_name

    result = nemo_run("jobs", "get-status", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get-status failed")
    assert json.loads(result.stdout).get("status", "").lower() == "completed"

    result = nemo_run("jobs", "get-logs", job_name, workspace=workspace)
    assert_exit_0(result, "jobs get-logs failed")
    assert "data" in json.loads(result.stdout)

    nemo_run("jobs", "delete", job_name, workspace=workspace)
    result = nemo_run("jobs", "get", job_name, workspace=workspace)
    assert result.returncode != 0, "job should not exist after deletion"
