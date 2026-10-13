# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from httpx import AsyncClient
from nhx.core.jobs.api.v2.jobs.schemas import CreateHelixJobRequest
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.app.test_helpers import TestConstants


@pytest.mark.asyncio
async def test_job_cancel_functionality(test_client: AsyncClient):
    """Test cancelling a job that is currently active."""
    req = CreateHelixJobRequest(
        name="test-job-cancel",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
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
    current_attempt = job_data["attempt_id"]

    # Cancel the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/cancel")
    assert response.status_code == 200
    cancelled_job_data = response.json()

    # Verify the job was cancelled (status should be "cancelling" initially)
    assert cancelled_job_data["status"] == "cancelling"

    # Check that the step status was updated to cancelling
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "cancelling"

    # verify the job is still on the same attempt
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "cancelling"
    assert current_attempt == job_data["attempt_id"]


@pytest.mark.asyncio
async def test_job_rerun_functionality(test_client: AsyncClient):
    """Test rerunning a job that has completed."""
    req = CreateHelixJobRequest(
        name="test-job-rerun",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID
    original_attempt_id = job_data["attempt_id"]

    # Move job through lifecycle to completed
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "completed"}
    )
    assert response.status_code == 200

    # Verify job is completed
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "completed"

    # Rerun the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/rerun")
    assert response.status_code == 200
    rerun_job_data = response.json()

    # Verify a new attempt was created
    assert rerun_job_data["attempt_id"] != original_attempt_id
    assert rerun_job_data["status"] == "created"

    # Verify that the new attempt has a fresh first step created
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()

    # The step should belong to the new attempt and be in created status
    assert step_data["attempt_id"] == rerun_job_data["attempt_id"]
    assert step_data["name"] == "step1"
    assert step_data["status"] == "created"


@pytest.mark.asyncio
async def test_job_cancel_rerun_lifecycle(test_client: AsyncClient):
    """Test complete cancel-rerun lifecycle of a job."""
    req = CreateHelixJobRequest(
        name="test-job-cancel-rerun",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
                HelixJobStepSpec(name="step2", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID
    original_attempt_id = job_data["attempt_id"]

    # Move first step to active
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Verify job is active
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

    # Cancel the job while step1 is active
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"

    # Simulate step transitioning to cancelled
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "cancelled"}
    )
    assert response.status_code == 200

    # Verify job is now cancelled
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # Rerun the job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/rerun")
    assert response.status_code == 200
    rerun_job_data = response.json()

    # Verify new attempt was created
    assert rerun_job_data["attempt_id"] != original_attempt_id
    assert rerun_job_data["status"] == "created"

    # Verify new attempt has fresh first step
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()

    # The first step should belong to the new attempt and be in created status
    assert step_data["attempt_id"] == rerun_job_data["attempt_id"]
    assert step_data["name"] == "step1"
    assert step_data["status"] == "created"

    # The second step should not exist yet (only created when first step completes)
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_job_cancel_while_resuming(test_client: AsyncClient):
    """Test cancelling a job immediately after resuming returns 200, not 500.

    Regression test for: cancel while in RESUMING state raises StateTransitionConflictError
    because RESUMING -> CANCELLING is not in the valid state machine transitions.
    """
    req = CreateHelixJobRequest(
        name="test-job-cancel-resuming",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job and advance to active
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_name = response.json()["name"]

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # Pause, then simulate step reaching paused
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/pause")
    assert response.status_code == 200

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "paused"}
    )
    assert response.status_code == 200

    # Resume (job is now in RESUMING state)
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "resuming"

    # Cancel immediately while still in RESUMING — should NOT return 500
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/cancel")
    assert response.status_code in (200, 202, 409), f"Expected 200/202/409, got {response.status_code}"


@pytest.mark.asyncio
async def test_job_cancel_nonexistent_job(test_client: AsyncClient):
    """Test cancelling a job that doesn't exist returns 404."""
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs/nonexistent-job-id/cancel")
    assert response.status_code == 404
    error_data = response.json()
    assert error_data["detail"] == "Job 'nonexistent-job-id' not found in workspace 'default'."


@pytest.mark.asyncio
async def test_job_rerun_nonexistent_job(test_client: AsyncClient):
    """Test rerunning a job that doesn't exist returns 404."""
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs/nonexistent-job-id/rerun")
    assert response.status_code == 404
    error_data = response.json()
    assert error_data["detail"] == "Job not found"


@pytest.mark.asyncio
async def test_job_cancel_no_active_steps(test_client: AsyncClient):
    """Test cancelling a job with no active steps (should handle gracefully)."""
    req = CreateHelixJobRequest(
        name="test-job-cancel-no-active",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job (status: created)
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    # Try to cancel a job that's still in created state (no active steps)
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/cancel")
    assert response.status_code == 200

    # The job should transition directly to cancelled if no active steps
    # This is different from pause which only affects active jobs
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    job_data = response.json()
    assert job_data["status"] == "cancelled"


@pytest.mark.asyncio
async def test_job_rerun_active_job(test_client: AsyncClient):
    """Test rerunning a job that is still active."""
    req = CreateHelixJobRequest(
        name="test-job-rerun-active",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    # Create job and move to active
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID
    original_attempt_id = job_data["attempt_id"]

    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200

    # must be cancelled before rerun
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "cancelled"}
    )
    assert response.status_code == 200

    # Rerun the cancelled job
    response = await test_client.post(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/rerun")
    assert response.status_code == 200
    rerun_job_data = response.json()

    # Verify new attempt was created
    assert rerun_job_data["attempt_id"] != original_attempt_id
    assert rerun_job_data["status"] == "created"
