# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import AsyncClient
from nhx.core.jobs.api.v2.jobs.schemas import CreatePlatformJobRequest
from nhx.core.jobs.app.schemas import PlatformJobSpec, PlatformJobStepSpec
from nhx.core.jobs.app.test_helpers import TestConstants


@pytest.mark.asyncio
async def test_job_pause_functionality(test_client: AsyncClient):
    """Test pausing a job that is currently active."""
    req = CreatePlatformJobRequest(
        name="test-job-pause",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID
    assert job_data["status"] == "created"

    # Move job to active state first
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Verify job is active
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "active"

    # Pause the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/pause")
    assert response.status_code == 200
    paused_job_data = response.json()

    # Verify the job was paused (status should be "pausing" initially)
    assert paused_job_data["status"] == "pausing"

    # Check that the step status was updated to pausing
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "pausing"


@pytest.mark.asyncio
async def test_job_resume_functionality(test_client: AsyncClient):
    """Test resuming a job that is currently paused."""
    req = CreatePlatformJobRequest(
        name="test-job-resume",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    # Move job to active, then paused state
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "paused"}
    )
    assert response.status_code == 200

    # Verify job is paused
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "paused"

    # Resume the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/resume")
    assert response.status_code == 200
    resumed_job_data = response.json()

    # Verify the job was resumed (status should be "resuming" initially)
    assert resumed_job_data["status"] == "resuming"

    # Check that the step status was updated to resuming
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "resuming"


@pytest.mark.asyncio
async def test_job_pause_resume_lifecycle(test_client: AsyncClient):
    """Test complete pause-resume lifecycle of a job."""
    req = CreatePlatformJobRequest(
        name="test-job-lifecycle",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
                PlatformJobStepSpec(name="step2", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    # Move first step to active
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Verify job is active
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

    # Pause the job while step1 is active
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "pausing"

    # Simulate step transitioning to paused
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "paused"}
    )
    assert response.status_code == 200

    # Verify job is now paused
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

    # Resume the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "resuming"

    # Simulate step transitioning from resuming back to active
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Verify job is active again
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

    # Complete the first step and move to second step
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "completed"}
    )
    assert response.status_code == 200

    # Verify second step is created and start it
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 200
    assert response.json()["status"] == "created"

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Test pause again on the second step
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "pausing"


@pytest.mark.asyncio
async def test_job_pause_nonexistent_job(test_client: AsyncClient):
    """Test pausing a job that doesn't exist returns 404."""
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs/nonexistent-job-id/pause")
    assert response.status_code == 404
    error_data = response.json()
    assert error_data["detail"] == "Job 'nonexistent-job-id' not found in workspace 'default'."


@pytest.mark.asyncio
async def test_job_resume_nonexistent_job(test_client: AsyncClient):
    """Test resuming a job that doesn't exist returns 404."""
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs/nonexistent-job-id/resume")
    assert response.status_code == 404
    error_data = response.json()
    assert error_data["detail"] == "Job 'nonexistent-job-id' not found in workspace 'default'."


@pytest.mark.asyncio
async def test_job_pause_pending_job(test_client: AsyncClient):
    """Test pausing a job that is pending, should just pause."""
    req = CreatePlatformJobRequest(
        name="test-job-no-active",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job (status: created)
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    # Try to pause a job that's still in created state (no active steps)
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/pause")
    assert response.status_code == 200

    # The job should remain in its current state since there's no active step to pause
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "pausing"


@pytest.mark.asyncio
async def test_job_resume_no_paused_steps(test_client: AsyncClient):
    """Test resuming a job with no paused steps (should handle gracefully)."""
    req = CreatePlatformJobRequest(
        name="test-job-no-paused",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job and move to active
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Try to resume a job that's not paused
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/resume")
    assert response.status_code == 200

    # The job should remain in its current state since there's no paused step to resume
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "active"
