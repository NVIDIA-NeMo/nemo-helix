# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for JobDispatcher cross-workspace operations.

These tests verify that the JobDispatcher correctly finds jobs and steps
across different workspaces, not just in the default workspace.

This captures a regression where dispatcher methods like list_steps,
get_current_job_step_by_name, and attempt_get_current were defaulting
to the "default" workspace instead of searching across all workspaces.
"""

import pytest
import pytest_asyncio
from nhx.common.entities import ALL_WORKSPACES, EntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateHelixJobRequest,
    HelixJobSortField,
    HelixJobStepsListFilter,
    HelixJobTaskUpdate,
)
from nhx.core.jobs.app.dispatcher import JobDispatcher
from nhx.core.jobs.app.test_helpers import TestConstants
from nhx.core.jobs.entities import HelixJobTask
from nhx.testing import create_test_client


@pytest.fixture(scope="function")
def multi_workspace_store():
    """Create an EntityClient for testing with multiple workspaces."""
    # Create store with both default and custom workspaces
    projects = ["default/test-project", "custom-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as client:
        yield client


@pytest_asyncio.fixture()
async def multi_workspace_dispatcher(multi_workspace_store, mock_files_client, mock_secrets_client) -> JobDispatcher:
    """Create a JobDispatcher with multi-workspace EntityStore."""
    return JobDispatcher(
        store=multi_workspace_store,
        files=mock_files_client,
        secrets=mock_secrets_client,
    )


def create_job_request(job_name: str) -> CreateHelixJobRequest:
    """Create a job request."""
    return CreateHelixJobRequest(
        name=job_name,
        description=f"Test job {job_name}",
        project="test-project",
        source="test-source",
        spec={"task": "test"},
        platform_spec=TestConstants.PLATFORM_SPEC,
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_list_steps_finds_jobs_in_non_default_workspace(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that list_steps finds job steps in non-default workspaces.

    This test verifies the fix for the bug where list_steps was only
    searching the default workspace instead of all workspaces when
    workspace=None was passed.
    """
    # Create a job in a custom workspace (not "default")
    custom_workspace = "custom-workspace"
    job_request = create_job_request("cross-workspace-job")
    job = await multi_workspace_dispatcher.create_job(job_request, custom_workspace)

    # Verify job was created in the custom workspace
    assert job.workspace == custom_workspace

    # Get the step entity to verify it exists
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", custom_workspace)
    assert step is not None
    assert step.workspace == custom_workspace

    # Now test list_steps with workspace=None (should search ALL workspaces)
    filter_obj = HelixJobStepsListFilter(status=[HelixJobStatus.CREATED])
    steps, total = await multi_workspace_dispatcher.list_steps(
        filter=filter_obj,
        sort=HelixJobSortField.CREATED_AT_ASC,
        limit=100,
        offset=0,
        workspace=ALL_WORKSPACES,  # This should search across all workspaces
    )

    # The step from the custom workspace should be found
    assert total >= 1, "Should find at least one step"
    step_ids = [s.id for s in steps]
    assert step.id in step_ids, f"Step {step.id} from {custom_workspace} not found in list_steps results"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_current_job_step_by_name_works_across_workspaces(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that get_current_job_step_by_name finds steps in non-default workspaces.

    This test verifies the fix for the bug where get_current_job_step_by_name
    was only searching the default workspace for attempts and steps.
    """
    # Create a job in a custom workspace
    custom_workspace = "custom-workspace"
    job_request = create_job_request("step-lookup-job")
    job = await multi_workspace_dispatcher.create_job(job_request, custom_workspace)

    # Verify we can find the step by name using the job ID
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", custom_workspace)

    assert step is not None, "Step should be found even though job is in non-default workspace"
    assert step.name == "basic"
    assert step.workspace == custom_workspace


@pytest.mark.asyncio
@pytest.mark.integration
async def test_attempt_get_current_works_across_workspaces(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that attempt_get_current finds attempts in non-default workspaces.

    This test verifies the fix for the bug where attempt_get_current
    was only searching the default workspace.
    """
    # Create a job in a custom workspace
    custom_workspace = "custom-workspace"
    job_request = create_job_request("attempt-lookup-job")
    job = await multi_workspace_dispatcher.create_job(job_request, custom_workspace)

    # Verify we can find the current attempt for this job
    attempt = await multi_workspace_dispatcher.get_current_attempt(job.name, custom_workspace)

    assert attempt is not None, "Attempt should be found even though job is in non-default workspace"
    assert attempt.workspace == custom_workspace
    assert attempt.job == job.id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_list_steps_with_specific_workspace_only_returns_that_workspace(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that list_steps with a specific workspace filters correctly.

    When a specific workspace is provided, only steps from that workspace
    should be returned.
    """
    # Create jobs in two different workspaces
    await multi_workspace_dispatcher.create_job(
        create_job_request("default-job"),
        "default",
    )
    await multi_workspace_dispatcher.create_job(
        create_job_request("custom-job"),
        "custom-workspace",
    )

    # List steps only from the custom workspace
    filter_obj = HelixJobStepsListFilter(status=[HelixJobStatus.CREATED])
    steps, total = await multi_workspace_dispatcher.list_steps(
        filter=filter_obj,
        sort=HelixJobSortField.CREATED_AT_ASC,
        limit=100,
        offset=0,
        workspace="custom-workspace",
    )

    # Should only find the step from custom-workspace
    step_workspaces = {s.workspace for s in steps}
    assert "custom-workspace" in step_workspaces
    # The default workspace step should not be in results
    for step in steps:
        assert step.workspace == "custom-workspace", f"Found step from wrong workspace: {step.workspace}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_update_job_step_status_works_across_workspaces(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that updating job step status works for jobs in non-default workspaces.

    This verifies the full workflow: create job -> find step -> update status
    all works correctly when the job is in a non-default workspace.
    """
    # Create a job in a custom workspace
    custom_workspace = "custom-workspace"
    job_request = create_job_request("status-update-job")
    job = await multi_workspace_dispatcher.create_job(job_request, custom_workspace)

    # Get the step
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", custom_workspace)
    assert step is not None

    # Update the step status to PENDING
    step.status = HelixJobStatus.PENDING
    await multi_workspace_store.update(step)

    # Verify we can still find the step after update
    updated_step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", custom_workspace)
    assert updated_step is not None
    assert updated_step.status == HelixJobStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.integration
async def test_create_or_update_task_finds_existing_task_by_name(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that create_or_update_task finds existing tasks by name, not entity ID.

    This reproduces a bug where get_task uses get_by_id with the task NAME
    instead of entity ID. Since tasks are parent-scoped, looking up by name
    via get_by_id fails, causing create_or_update_task to think the task
    doesn't exist and try to create a duplicate, resulting in:
        "unique constraint failed: entities.workspace, entities.entity_type, entities.name"

    The fix requires get_task to use a list query filtered by step_id and name.
    """
    # Create a job with a step
    job_request = create_job_request("task-lookup-job")
    job = await multi_workspace_dispatcher.create_job(job_request, "default")

    # Get the step
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", "default")
    assert step is not None

    task_name = "evaluation-task"

    # First call: create the task
    task_update = HelixJobTaskUpdate(
        status=HelixJobStatus.ACTIVE,
        status_details={"progress": 0},
    )
    task1 = await multi_workspace_dispatcher.create_or_update_task(
        job_name=job.name,
        task_name=task_name,  # This is actually the task NAME, not entity ID
        workspace="default",
        task_update=task_update,
        step=step,
    )
    assert task1 is not None
    assert task1.name == task_name
    assert task1.status == HelixJobStatus.ACTIVE

    # Second call: update the same task (should NOT create a duplicate)
    # This is where the bug manifests - if get_task doesn't find the existing
    # task, it tries to create another one with the same name, causing:
    # "unique constraint failed"
    task_update2 = HelixJobTaskUpdate(
        status=HelixJobStatus.COMPLETED,
        status_details={"progress": 100},
    )
    task2 = await multi_workspace_dispatcher.create_or_update_task(
        job_name=job.name,
        task_name=task_name,  # Same task name as before
        workspace="default",
        task_update=task_update2,
        step=step,
    )

    # Should be the same task, just updated
    assert task2 is not None
    assert task2.id == task1.id, "Should update existing task, not create a new one"
    assert task2.name == task_name
    assert task2.status == HelixJobStatus.COMPLETED

    # Verify only one task exists for this step
    tasks = await multi_workspace_dispatcher.list_tasks(step.id, workspace="default")
    task_names = [t.name for t in tasks]
    assert task_names.count(task_name) == 1, f"Should have exactly one task named '{task_name}', found {task_names}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_list_tasks_uses_step_workspace_for_non_default_jobs(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that list_tasks returns tasks for steps in non-default workspaces.

    This reproduces a bug where list_tasks omitted the workspace and fell back
    to the entity client's default workspace, so jobs.tasks.list(...) returned
    an empty task list for jobs outside ``default`` even though the task existed.
    """
    custom_workspace = "custom-workspace"
    job = await multi_workspace_dispatcher.create_job(create_job_request("custom-task-list-job"), custom_workspace)
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", custom_workspace)
    assert step is not None

    task = await multi_workspace_store.add(
        HelixJobTask(
            name="custom-task",
            workspace=custom_workspace,
            step_id=step.id,
            status=HelixJobStatus.ERROR,
        )
    )
    assert task.id is not None

    tasks = await multi_workspace_dispatcher.list_tasks(step.id, workspace=custom_workspace)

    assert [listed_task.id for listed_task in tasks] == [task.id]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_task_finds_task_by_name_not_entity_id(
    multi_workspace_dispatcher: JobDispatcher,
    multi_workspace_store: EntityClient,
):
    """Test that get_task correctly finds a task when given its name.

    The dispatcher's get_task method receives the task NAME from callers,
    not the entity ID. This test verifies it correctly finds tasks by name
    within a step's scope.
    """
    # Create a job with a step
    job_request = create_job_request("get-task-job")
    job = await multi_workspace_dispatcher.create_job(job_request, "default")

    # Get the step
    step = await multi_workspace_dispatcher.get_current_job_step_by_name(job.name, "basic", "default")
    assert step is not None

    task_name = "my-task"

    # Create a task directly in the store (simulating what create_or_update_task does)
    task = await multi_workspace_store.add(
        HelixJobTask(
            name=task_name,
            workspace=step.workspace,
            step_id=step.id,
            status=HelixJobStatus.ACTIVE,
        )
    )
    assert task.id is not None
    assert task.name == task_name

    # Now verify get_task can find it using the task NAME (not entity ID)
    found_task = await multi_workspace_dispatcher.get_task(step.id, task_name, "default")

    assert found_task is not None, f"get_task should find task by name '{task_name}'"
    assert found_task.id == task.id
    assert found_task.name == task_name
