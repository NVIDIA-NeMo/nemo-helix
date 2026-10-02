# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent created and ran GPU jobs through the NeMo Helix jobs pipeline.

Tests job creation, GPU execution, failure handling, and agent trajectory.
"""

import base64
import json
import os
import time

import pytest
from nemo_helix_plugin.jobs.client import JobsClient

WORKSPACE = "gpu-job-workspace"


def _make_unsigned_jwt() -> str:
    """Create an unsigned JWT (alg=none) for local quickstart auth."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "verifier@harbor.local", "email": "verifier@harbor.local"}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}."


@pytest.fixture
def client() -> JobsClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return JobsClient(
        base_url=nhx_base_url,
        workspace=WORKSPACE,
        auth=_make_unsigned_jwt(),
    )


def _wait_for_terminal(client: JobsClient, job_name: str, max_wait: int = 60) -> str:
    """Wait for a job to reach a terminal status."""
    for _ in range(max_wait // 5):
        try:
            resp = client.get_job_status(name=job_name, workspace=WORKSPACE)
            status = resp.status if hasattr(resp, "status") else str(resp)
            if status in ("completed", "error", "cancelled"):
                return status
        except Exception:
            pass
        time.sleep(5)
    try:
        resp = client.get_job_status(name=job_name, workspace=WORKSPACE)
        return resp.status if hasattr(resp, "status") else str(resp)
    except Exception:
        return "unknown"


def _find_job_by_name(client: JobsClient, name: str):
    """Find a specific job by name."""
    jobs = client.list_jobs(workspace=WORKSPACE).items()
    for job in jobs:
        if job.name == name:
            return job
    return None


# --- Job existence and completion checks ---


def test_multiple_jobs_created(client: JobsClient) -> None:
    """Verify that at least 3 jobs were created."""
    jobs = list(client.list_jobs(workspace=WORKSPACE).items())
    assert len(jobs) >= 3, f"Expected at least 3 jobs, found {len(jobs)}: {[j.name for j in jobs]}"


def test_gpu_verify_job_completed(client: JobsClient) -> None:
    """Verify gpu-verify-job reached completed status (nvidia-smi ran on GPU)."""
    job = _find_job_by_name(client, "gpu-verify-job")
    assert job is not None, (
        f"Job 'gpu-verify-job' not found. Jobs: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "gpu-verify-job")
    assert status == "completed", f"Job 'gpu-verify-job' has status '{status}', expected 'completed'."


def test_gpu_compute_job_completed(client: JobsClient) -> None:
    """Verify gpu-compute-job reached completed status."""
    job = _find_job_by_name(client, "gpu-compute-job")
    assert job is not None, (
        f"Job 'gpu-compute-job' not found. Jobs: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "gpu-compute-job")
    assert status == "completed", f"Job 'gpu-compute-job' has status '{status}', expected 'completed'."


def test_gpu_fail_job_errored(client: JobsClient) -> None:
    """Verify gpu-fail-job reached error status (exit code 1)."""
    job = _find_job_by_name(client, "gpu-fail-job")
    assert job is not None, (
        f"Job 'gpu-fail-job' not found. Jobs: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "gpu-fail-job")
    assert status == "error", f"Job 'gpu-fail-job' has status '{status}', expected 'error'."


# --- Agent trajectory checks ---


def test_agent_polled_status() -> None:
    """Verify the agent polled for job status multiple times."""
    try:
        from trace_reader import get_session

        session = get_session()
        commands = session.get_bash_commands()
    except Exception:
        pytest.skip("trace_reader not available")
        return

    status_checks = [
        cmd for cmd in commands if "jobs" in cmd and ("get-status" in cmd or "get_status" in cmd or "status" in cmd)
    ]
    assert len(status_checks) >= 3, f"Agent checked job status {len(status_checks)} time(s), expected at least 3."


def test_agent_investigated_failure() -> None:
    """Verify the agent investigated the failing job."""
    try:
        from trace_reader import get_session

        session = get_session()
        commands = session.get_bash_commands()
    except Exception:
        pytest.skip("trace_reader not available")
        return

    fail_investigation = [cmd for cmd in commands if "gpu-fail-job" in cmd or "fail-job" in cmd or "fail_job" in cmd]
    assert len(fail_investigation) >= 2, (
        f"Agent only interacted with fail job {len(fail_investigation)} time(s). "
        f"Expected at least 2 (create + investigate)."
    )
