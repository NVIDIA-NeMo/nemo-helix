# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the hello-world service jobs.

These tests verify the hello-world job API works correctly
when running against a fully deployed NHX platform.
"""

from nemo_helix_plugin.client.client import NemoClient
from nhx.testing.e2e import wait_for_platform_job


def test_job_lifecycle(client: NemoClient, workspace: str):
    """Test the complete job lifecycle: create, run, complete.

    This test verifies the job system works end-to-end:
    1. Create a job with a specific message
    2. Verify job is created with correct initial state
    3. Wait for the job to complete
    4. Verify job reaches completed status

    Note: This test uses the e2e backend which auto-completes jobs
    without running actual task code. Task output verification (filesets)
    requires a docker-compose based setup that runs real containers.
    """
    job_name = "lifecycle-test-job"
    test_message = "Hello from e2e lifecycle test!"

    # Create a job
    response = client._client.post(
        f"{client.base_url}/apis/hello-world/v2/workspaces/{workspace}/jobs",
        json={
            "name": job_name,
            "description": "Test job lifecycle",
            "spec": {"message": test_message},
        },
    )
    assert response.status_code == 201, f"Failed to create job: {response.text}"
    job = response.json()
    assert job["name"] == job_name
    assert job["spec"]["message"] == test_message

    # Wait for job completion
    completed = wait_for_platform_job(client, job_name, workspace)
    assert completed.status == "completed", f"Job did not complete successfully: {completed}"

    # Verify job can be retrieved after completion
    response = client._client.get(f"{client.base_url}/apis/hello-world/v2/workspaces/{workspace}/jobs/{job_name}")
    assert response.status_code == 200, f"Failed to get completed job: {response.text}"
    completed_job = response.json()
    assert completed_job["status"].upper() == "COMPLETED"
