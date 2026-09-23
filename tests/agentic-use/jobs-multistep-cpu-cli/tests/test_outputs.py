# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the agent created multiple jobs, handled failure, and recovered.

Tests the full jobs lifecycle including success, failure diagnosis, and recovery.

Tests:
- Three jobs exist in the workspace
- success-job completed
- fail-job reached error status
- recovery-job completed
- Agent trajectory shows status polling
- Agent trajectory shows failure investigation
"""

import base64
import json
import os
import time

import pytest
from nemo_helix_plugin.jobs.client import JobsClient
from trace_reader import get_session

WORKSPACE = "job-test-workspace"


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


def _wait_for_terminal(client: JobsClient, job_name: str, max_wait: int = 30) -> str:
    """Wait for a job to reach a terminal status, with a safety timeout."""
    for _ in range(max_wait // 5):
        try:
            resp = client.get_job_status(name=job_name, workspace=WORKSPACE)
            status = resp.status if hasattr(resp, "status") else str(resp)
            if status in ("completed", "error", "cancelled"):
                return status
        except Exception:
            pass
        time.sleep(5)
    # Return whatever we got
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


# --- Job existence checks ---


def test_multiple_jobs_created(client: JobsClient) -> None:
    """Verify that at least 3 jobs were created."""
    jobs = list(client.list_jobs(workspace=WORKSPACE).items())
    assert len(jobs) >= 3, (
        f"Expected at least 3 jobs in workspace '{WORKSPACE}', found {len(jobs)}: {[j.name for j in jobs]}"
    )


def test_success_job_completed(client: JobsClient) -> None:
    """Verify success-job reached completed status."""
    job = _find_job_by_name(client, "success-job")
    assert job is not None, (
        "Job 'success-job' not found. "
        f"Jobs in workspace: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "success-job")
    assert status == "completed", f"Job 'success-job' has status '{status}', expected 'completed'."


def test_fail_job_errored(client: JobsClient) -> None:
    """Verify fail-job reached error status (exit code 1)."""
    job = _find_job_by_name(client, "fail-job")
    assert job is not None, (
        f"Job 'fail-job' not found. Jobs in workspace: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "fail-job")
    assert status == "error", (
        f"Job 'fail-job' has status '{status}', expected 'error'. The job was designed to fail with exit 1."
    )


def test_recovery_job_completed(client: JobsClient) -> None:
    """Verify recovery-job reached completed status after the failure."""
    job = _find_job_by_name(client, "recovery-job")
    assert job is not None, (
        "Job 'recovery-job' not found. "
        f"Jobs in workspace: {[j.name for j in client.list_jobs(workspace=WORKSPACE).items()]}"
    )
    status = _wait_for_terminal(client, "recovery-job")
    assert status == "completed", f"Job 'recovery-job' has status '{status}', expected 'completed'."


# --- Agent trajectory checks ---


def test_agent_created_multiple_jobs() -> None:
    """Verify the agent ran multiple job creation commands."""
    session = get_session()
    commands = session.get_bash_commands()

    create_commands = [cmd for cmd in commands if "jobs" in cmd and "create" in cmd]

    assert len(create_commands) >= 3, (
        f"Agent created {len(create_commands)} job(s), expected at least 3 (success-job, fail-job, recovery-job)."
    )


def test_agent_polled_status() -> None:
    """Verify the agent polled for job status multiple times."""
    session = get_session()
    commands = session.get_bash_commands()

    status_checks = [
        cmd
        for cmd in commands
        if "jobs" in cmd
        and ("get-status" in cmd or "get_status" in cmd or ("get " in cmd and "get-logs" not in cmd) or "status" in cmd)
    ]

    # Should poll multiple times across 3 jobs
    assert len(status_checks) >= 4, (
        f"Agent checked job status {len(status_checks)} time(s), expected at least 4 (polling across 3 jobs)."
    )


def test_agent_investigated_failure() -> None:
    """Verify the agent investigated the failing job."""
    session = get_session()
    commands = session.get_bash_commands()

    # Look for commands that inspect the failed job - status check, logs, or get
    fail_investigation = [cmd for cmd in commands if "fail-job" in cmd or "fail_job" in cmd]

    assert len(fail_investigation) >= 2, (
        f"Agent only interacted with fail-job {len(fail_investigation)} time(s). "
        f"Expected at least 2 interactions (create + investigate failure)."
    )
