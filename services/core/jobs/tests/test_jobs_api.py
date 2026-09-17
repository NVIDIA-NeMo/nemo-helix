# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import AsyncClient
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.schemas import FileStorageType, PlatformJobResultCreateRequest
from nemo_platform_plugin.jobs.types import PlatformJobTaskUpdate
from nmp.common.entities import ALL_WORKSPACES, DEFAULT_WORKSPACE
from nmp.common.entities.client import EntityValidationError
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.api.v2.jobs.endpoints import (
    _format_create_job_conflict,
    _format_entity_validation_error,
    get_platform_jobs_steps_list_filter,
)
from nmp.core.jobs.api.v2.jobs.schemas import (
    CreatePlatformJobRequest,
    PlatformJobResponse,
    PlatformJobSortField,
    PlatformJobStepsListFilter,
)
from nmp.core.jobs.app.dispatcher import (
    JobAlreadyExistsError,
    JobDispatcher,
    JobOutputLocationError,
    JobSecretValidationError,
    StateTransitionConflictError,
)
from nmp.core.jobs.app.providers import ContainerSpec, GPUExecutionProvider, SubprocessExecutionProvider
from nmp.core.jobs.app.schemas import (
    PlatformJobSecret,
    PlatformJobSpec,
    PlatformJobStepSpec,
)
from nmp.core.jobs.app.test_helpers import TestConstants
from pydantic import ValidationError
from starlette.datastructures import QueryParams


def expected_translated_executor_dump() -> Dict[str, Any]:
    """Return the expected persisted executor for ``TestConstants.TEST_EXECUTOR``.

    The Jobs API rewrites ``cpu/<profile>`` steps into ``subprocess/<profile>``
    steps before persistence (see
    ``translate_cpu_container_steps_to_subprocess`` in
    ``services/core/jobs/src/nmp/core/jobs/api/v2/jobs/endpoints.py``), so the
    round-trip representation of a step submitted with ``TestConstants.TEST_EXECUTOR``
    is the translated subprocess executor — with ``command`` set to
    ``container.entrypoint + container.command``.
    """
    container = TestConstants.TEST_EXECUTOR.container
    return SubprocessExecutionProvider(
        provider="subprocess",
        profile=TestConstants.TEST_EXECUTOR.profile,
        command=[*container.entrypoint, *container.command],
    ).model_dump()


@pytest.mark.asyncio
async def test_create_job_using_sdk(test_sdk: AsyncNeMoPlatform):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    job = (
        await jobs.create_job(
            workspace=DEFAULT_WORKSPACE,
            body=CreatePlatformJobRequest(
                name="test-job",
                source="testing",
                spec={},
                platform_spec=PlatformJobSpec.model_validate(
                    {
                        "steps": [
                            {
                                "name": "basic",
                                "executor": {
                                    "provider": "cpu",
                                    "profile": "default",
                                    # entrypoint+command are required so the cpu->subprocess
                                    # translation hop in the Jobs API (see
                                    # `translate_cpu_container_steps_to_subprocess`) can
                                    # produce a non-empty subprocess command. Real plugin
                                    # compilers always set both; mirroring that here keeps
                                    # the SDK round-trip path realistic.
                                    "container": {
                                        "image": "test-image",
                                        "entrypoint": ["python", "-m"],
                                        "command": ["nmp.testing.fake_task"],
                                    },
                                },
                            }
                        ]
                    },
                ),
            ),
        )
    ).data()
    assert job.id
    response = (await jobs.list_jobs(workspace=DEFAULT_WORKSPACE)).page()
    assert len(response.items) == 1
    job_item = response.items[0]
    assert job_item.name == "test-job"
    assert len(job_item.platform_spec.steps) == 1


@pytest.mark.asyncio
async def test_create_job_with_invalid_step_name(test_client: AsyncClient):
    """Test that jobs with invalid step names are rejected with 422 status."""
    response = await test_client.post(
        "/apis/jobs/v2/workspaces/default/jobs",
        json={
            "name": "test-job",
            "source": "test-source",
            "spec": {"param1": "value1"},
            "platform_spec": {
                "steps": [
                    {
                        "name": "invalidCamelCase",
                        "executor": TestConstants.TEST_EXECUTOR.model_dump(),
                        "config": {},
                    }
                ]
            },
        },
    )
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("invalidCamelCase" in str(err) for err in error_detail)


@pytest.mark.parametrize(
    "step_name,should_pass",
    [
        ("valid-step-name", True),
        ("step1", True),
        ("my-training-step", True),
        ("invalidCamelCase", False),
        ("UPPERCASE", False),
        # TODO(#3530): This should fail once we standardize names
        ("has_underscore", True),
        ("has spaces", False),
        ("-starts-with-dash", False),
        ("ends-with-dash-", False),
        ("a", False),
        ("123", False),
    ],
)
def test_step_name_validation(step_name: str, should_pass: bool):
    """Test PlatformJobStepSpec validates step names via Pydantic."""
    if should_pass:
        step = PlatformJobStepSpec(
            name=step_name,
            executor=TestConstants.TEST_EXECUTOR,
            config={},
        )
        assert step.name == step_name
    else:
        with pytest.raises(ValidationError) as exc_info:
            PlatformJobStepSpec(
                name=step_name,
                executor=TestConstants.TEST_EXECUTOR,
                config={},
            )
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("name",) for error in errors)


@pytest.mark.asyncio
@pytest.mark.skip("This is an integration test that requires secrets service.")
async def test_create_job_with_secrets(test_sdk: AsyncNeMoPlatform):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    request = CreatePlatformJobRequest(
        name="test-job",
        source="testing",
        spec={},
        platform_spec=PlatformJobSpec.model_validate(
            {
                "steps": [
                    {
                        "name": "basic",
                        "executor": {
                            "provider": "cpu",
                            "profile": "default",
                            "container": {
                                "image": "test-image",
                                "entrypoint": ["python", "-m"],
                                "command": ["nmp.testing.fake_task"],
                            },
                        },
                        "environment": [
                            {"name": "MY_SECRET_ENV", "from_secret": {"name": "secret_name_1"}},
                        ],
                    }
                ],
                "secrets": [PlatformJobSecret(name="secret_name_1", value="secret_value_1")],
            }
        ),
    )
    job = (await jobs.create_job(workspace=DEFAULT_WORKSPACE, body=request)).data()
    assert job.id
    response = await jobs.list_jobs(workspace=DEFAULT_WORKSPACE)
    page = response.page()
    assert len(page.items) == 1
    job_item = page.items[0]
    assert job_item.name == "test-job"
    assert len(job_item.platform_spec.steps) == 1

    # Assert that when created, we still got a secret reference back
    secrets = job_item.platform_spec.secrets
    assert secrets is not None
    assert len(secrets) == 1
    assert secrets[0].name == "secret_name_1"
    # We should not be returning the actual secret value in the job spec
    assert secrets[0].value is None
    # The secret should have a ref_id (used for in-memory secret storage)
    assert secrets[0].ref_id is not None
    assert secrets[0].ref_id.startswith("job-")


@pytest.mark.asyncio
async def test_create_job_with_invalid_project_name(test_client: AsyncClient):
    """Test that creating a job with an invalid project name (containing space) returns 422, not 500."""
    req = CreatePlatformJobRequest(
        name="test-job-invalid-project",
        project="A Project",  # Invalid: contains space
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )

    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())

    # Should return 422 Unprocessable Entity for validation error, not 500 Internal Server Error
    assert response.status_code == 422, (
        f"Expected 422 or 400 validation error, but got {response.status_code}. Response: {response.json()}"
    )


@pytest.mark.asyncio
async def test_create_job_validation_error_returns_actionable_message(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
):
    raw_error = "EntityValidationError: field 'name' is required; internal=db_version"
    with patch.object(mock_dispatcher, "create_job", new=AsyncMock(side_effect=EntityValidationError(raw_error))):
        response = await test_client.post(
            "/apis/jobs/v2/workspaces/default/jobs",
            json={
                "name": "bad-job",
                "source": "test-source",
                "spec": {},
                "platform_spec": {
                    "steps": [
                        {
                            "name": "valid-step",
                            "executor": TestConstants.TEST_EXECUTOR.model_dump(mode="json"),
                            "config": {},
                        }
                    ]
                },
            },
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Field 'name' is required" in detail
    assert "EntityValidationError" not in detail
    assert "db_version" not in detail


def test_format_entity_validation_error_unknown_type_message():
    raw_error = "field 'required_string_count' has an invalid value"
    detail = _format_entity_validation_error(EntityValidationError(raw_error), resource="job")

    assert detail == (
        "Invalid job request. Field 'required_string_count' has an invalid value. Update the request and try again."
    )


@pytest.mark.asyncio
async def test_create_job_conflict_hides_raw_exception_message(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
):
    raw_error = "ValueError: Job with name 'dup-job' already exists in workspace 'default'. internal=db_version"
    with patch.object(mock_dispatcher, "create_job", new=AsyncMock(side_effect=JobAlreadyExistsError(raw_error))):
        response = await test_client.post(
            "/apis/jobs/v2/workspaces/default/jobs",
            json={
                "name": "dup-job",
                "source": "test-source",
                "spec": {},
                "platform_spec": {
                    "steps": [
                        {
                            "name": "valid-step",
                            "executor": TestConstants.TEST_EXECUTOR.model_dump(mode="json"),
                            "config": {},
                        }
                    ]
                },
            },
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Job 'dup-job' already exists in workspace 'default'" in detail
    assert "ValueError" not in detail
    assert "db_version" not in detail


@pytest.mark.asyncio
async def test_create_job_missing_output_location_returns_422(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
):
    """A JobOutputLocationError maps to 422, not the 409 that other ValueErrors get.

    Locks the except-clause ordering: JobOutputLocationError subclasses ValueError, so if its
    handler stopped preceding ``except ValueError`` the status would silently regress to 409.
    """
    err = JobOutputLocationError("fileset 'ghost' not found in workspace 'default'")
    with patch.object(mock_dispatcher, "create_job", new=AsyncMock(side_effect=err)):
        response = await test_client.post(
            "/apis/jobs/v2/workspaces/default/jobs",
            json={
                "name": "eval-job",
                "source": "test-source",
                "spec": {},
                "output_location": "ghost",
                "platform_spec": {
                    "steps": [
                        {
                            "name": "valid-step",
                            "executor": TestConstants.TEST_EXECUTOR.model_dump(mode="json"),
                            "config": {},
                        }
                    ]
                },
            },
        )

    assert response.status_code == 422
    assert "ghost" in response.json()["detail"]


def test_create_job_conflict_does_not_classify_generic_message():
    raw_error = "unrelated secret already exists; internal=db_version"
    detail = _format_create_job_conflict(ValueError(raw_error), job_name="test-job", workspace="default")

    assert "conflicts with existing platform state" in detail
    assert "already exists" not in detail
    assert "referenced secrets" not in detail
    assert "db_version" not in detail


def test_create_job_secret_validation_error_returns_safe_guidance():
    raw_error = "Secret 'default/api-key' not found; internal=db_version"
    detail = _format_create_job_conflict(JobSecretValidationError(raw_error), job_name="test-job", workspace="default")

    assert "referenced secrets were not found or are not accessible" in detail
    assert "api-key" not in detail
    assert "db_version" not in detail


@pytest.mark.asyncio
async def test_create_job_conflict_sanitizes_log_fields(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
    caplog,
):
    with patch.object(mock_dispatcher, "create_job", new=AsyncMock(side_effect=ValueError("conflict"))):
        with caplog.at_level(logging.INFO, logger="nmp.core.jobs.api.v2.jobs.endpoints"):
            response = await test_client.post(
                "/apis/jobs/v2/workspaces/default%0Aforged/jobs",
                json={
                    "name": "dup-job\r\nforged",
                    "source": "test-source",
                    "spec": {},
                    "platform_spec": {
                        "steps": [
                            {
                                "name": "valid-step",
                                "executor": TestConstants.TEST_EXECUTOR.model_dump(mode="json"),
                                "config": {},
                            }
                        ]
                    },
                },
            )

    assert response.status_code == 409
    message = next(record.getMessage() for record in caplog.records if "Failed to create job" in record.msg)
    assert "\r" not in message
    assert "\n" not in message


@pytest.mark.asyncio
async def test_create_job_gpu_fail_fast_when_docker_no_gpus(test_client: AsyncClient):
    """Direct Jobs API create with GPU step fails fast with 422 when platform is Docker with no GPUs."""
    from nmp.common.config import Runtime

    gpu_executor = GPUExecutionProvider(
        provider="gpu",
        profile="default",
        container=ContainerSpec(image="gpu-image"),
    )
    req = CreatePlatformJobRequest(
        name="gpu-job",
        source="test-source",
        spec={},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="gpu_step", executor=gpu_executor, config={}),
            ]
        ),
    )
    mock_platform_config = MagicMock()
    mock_platform_config.runtime = Runtime.DOCKER
    mock_platform_config.docker.get_reserved_gpu_ids.return_value = []

    with patch("nemo_platform_plugin.jobs.docker.get_platform_config", return_value=mock_platform_config):
        response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())

    assert response.status_code == 422
    detail = response.json().get("detail", "")
    assert "GPU" in detail and "Docker" in detail


@pytest.mark.asyncio
async def test_hello_world_jobs_list(test_client: AsyncClient):
    fake_name = "my-fake-job"
    fake_description = "this is a fake job"
    fake_project = "proj-1234"
    # Use "default" workspace since jobs list endpoint defaults to filtering by "default"
    fake_workspace_id = "default"

    response = await test_client.post(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs",
        json={
            "name": fake_name,
            "description": fake_description,
            "project": fake_project,
            "workspace_id": fake_workspace_id,
            "spec": {"config": {"key": "Value"}, "target": "str"},
            "ownership": {"user": "fake-user", "service": "fake-ms"},
        },
    )
    assert response.status_code == 201, f"POST failed: {response.status_code} {response.text}"

    response = await test_client.get("/apis/jobs/v2/workspaces/default/hello-world/jobs")
    assert response.status_code == 200, f"GET failed: {response.status_code} {response.text}"

    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs")
    data = response.json()
    assert "data" in data, f"No 'data' key in response: {data}"
    assert len(data["data"]) > 0, f"No jobs returned: {data}"
    value = PlatformJobResponse.model_validate(data["data"][0])
    assert value.name == fake_name
    assert value.description == fake_description
    assert value.project == fake_project
    assert value.workspace == fake_workspace_id
    assert value.source == "hello-world"
    assert value.spec["target"] == "str"
    assert value.platform_spec.steps[0].name == "hello-world-step-1"

    # Post another job with a different name, don't provide a namespace - should default to "default"
    # Sleep to ensure different created_at timestamp for sort ordering
    await asyncio.sleep(1.1)
    fake_name_2 = "my-fake-job-2"
    response = await test_client.post(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs",
        json={
            "name": fake_name_2,
            "description": fake_description,
            "project": fake_project,
            "workspace_id": fake_workspace_id,
            "spec": {"config": {"key": "Value"}, "target": "str"},
            "ownership": {"user": "fake-user", "service": "fake-ms"},
        },
    )
    assert response.status_code == 201, f"POST failed: {response.status_code} {response.text}"
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs")
    data = response.json()
    assert len(data["data"]) == 2
    # Default listing is in descending order, so the second job we just created should be first now
    value = PlatformJobResponse.model_validate(data["data"][0])
    assert value.workspace == "default"
    assert value.project == fake_project
    assert value.source == "hello-world"
    assert value.spec["target"] == "str"
    assert value.platform_spec.steps[0].name == "hello-world-step-1"

    # Post another job, but this time don't provide a name. It should be "<source>-<job-id>"
    # Sleep to ensure different created_at timestamp for sort ordering
    await asyncio.sleep(1.1)
    response = await test_client.post(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs",
        json={
            "description": fake_description,
            "project": fake_project,
            "workspace_id": fake_workspace_id,
            "spec": {"config": {"key": "Value"}, "target": "str"},
            "ownership": {"user": "fake-user", "service": "fake-ms"},
        },
    )
    assert response.status_code == 201, f"POST failed: {response.status_code} {response.text}"
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs")
    data = response.json()
    assert len(data["data"]) == 3
    # Default listing is in descending order, so the second job we just created should be first now
    value = PlatformJobResponse.model_validate(data["data"][0])
    assert value.name.startswith("hello-world-")
    assert value.workspace == fake_workspace_id
    assert value.project == fake_project
    assert value.source == "hello-world"
    assert value.spec["target"] == "str"


@pytest.mark.asyncio
async def test_job_lifecycle_single_step(test_client: AsyncClient):
    req = CreatePlatformJobRequest(
        name="test-job",
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
    job_id = job_data["id"]
    job_name = job_data["name"]  # API URLs use job name, not ID
    assert job_data["name"] == "test-job"
    assert job_data["source"] == "test-source"
    assert job_data["status"] == "created"

    # List jobs
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["data"]) == 1
    assert list_data["data"][0]["id"] == job_id
    assert list_data["data"][0]["status"] == "created"

    # Get specific job
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    get_data = response.json()
    assert get_data["id"] == job_id
    assert get_data["status"] == "created"

    attempt_id = get_data["attempt_id"]

    # Assert that the platform_spec is created correctly
    assert len(get_data["platform_spec"]["steps"]) == 1
    assert get_data["platform_spec"]["steps"][0]["name"] == "step1"
    assert get_data["platform_spec"]["steps"][0]["executor"] == expected_translated_executor_dump()
    assert get_data["platform_spec"]["steps"][0]["config"] == {}

    # list all steps (scoped to this job name — list_steps injects filter.job = name)
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps")
    assert response.status_code == 200
    list_steps_body = response.json()
    steps_data = list_steps_body["data"]
    assert len(steps_data) == 1
    assert steps_data[0]["name"].startswith("step1")
    # The filter is echoed back as a dict (not a Pydantic instance or an empty {}).
    assert list_steps_body["filter"] == {"job": job_name}

    # Cross-job listing with no user filter should serialize as null (and
    # therefore be omitted under response_model_exclude_none=True), matching
    # list_jobs rather than leaking an empty dict.
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs/-/steps")
    assert response.status_code == 200
    assert "filter" not in response.json()

    # Assert from the api that the first step is created correctly
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["attempt_id"] == attempt_id
    assert step_data["name"].startswith("step1")
    assert step_data["status"] == "created"

    # Assert from the job status api that the status of the job is now created overall
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert response.status_code == 200
    status_data = response.json()
    # No tasks exist yet, but the job and first step status should be "created"
    assert status_data["status"] == "created"
    assert status_data["steps"][0]["name"].startswith("step1")
    assert status_data["steps"][0]["status"] == "created"

    # Update status of the first step to pending
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "pending"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "pending"

    # Ensure status update sticks on the step
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "pending"

    # Now also assert that the job status is now pending, so that job state is consistent
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "pending"

    # Assert from the job status api that the status of the job is now pending overall
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert response.status_code == 200
    status_data = response.json()
    # No tasks exist yet, but the job and first step status should be "pending"
    assert status_data["status"] == "pending"
    assert status_data["steps"][0]["name"].startswith("step1")
    assert status_data["steps"][0]["status"] == "pending"

    # Update the step status to active, and recheck the step and job status
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "active"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "active"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    # Assert from the job status api that the status of the job is now active overall
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert response.status_code == 200
    status_data = response.json()
    # No tasks exist yet, but the job and first step status should be "active"
    assert status_data["status"] == "active"
    assert status_data["steps"][0]["name"].startswith("step1")
    assert status_data["steps"][0]["status"] == "active"

    # Now update the step status to completed, and recheck the step and job status are all completed.
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "completed"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "completed"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "completed"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "completed"

    # Assert from the job status api that the status of the job is now completed overall
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert response.status_code == 200
    status_data = response.json()
    # No tasks exist yet, but the job and first step status should be "completed"
    assert status_data["status"] == "completed"
    assert status_data["steps"][0]["name"].startswith("step1")
    assert status_data["steps"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_job_lifecycle_multi_step(test_client: AsyncClient):
    req = CreatePlatformJobRequest(
        name="test-job",
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
    job_id = job_data["id"]
    job_name = job_data["name"]  # API URLs use job name, not ID
    assert job_data["name"] == "test-job"
    assert job_data["source"] == "test-source"
    assert job_data["status"] == "created"

    # List jobs
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs")
    assert response.status_code == 200
    list_data = response.json()
    assert len(list_data["data"]) == 1
    assert list_data["data"][0]["id"] == job_id
    assert list_data["data"][0]["status"] == "created"

    # Get specific job
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    get_data = response.json()
    assert get_data["id"] == job_id
    attempt_id = get_data["attempt_id"]

    # Assert that the platform_spec is created correctly
    assert len(get_data["platform_spec"]["steps"]) == 2
    assert get_data["platform_spec"]["steps"][0]["name"] == "step1"
    assert get_data["platform_spec"]["steps"][0]["executor"] == expected_translated_executor_dump()
    assert get_data["platform_spec"]["steps"][0]["config"] == {}
    assert get_data["platform_spec"]["steps"][1]["name"] == "step2"
    assert get_data["platform_spec"]["steps"][1]["executor"] == expected_translated_executor_dump()
    assert get_data["platform_spec"]["steps"][1]["config"] == {}

    # Assert from the api that the first step is created correctly
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["attempt_id"] == attempt_id
    assert step_data["name"].startswith("step1")
    assert step_data["status"] == "created"

    # Update status of the first step to pending
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "pending"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "pending"

    # Ensure status update sticks on the step
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "pending"

    # Now also assert that the job status is now pending, so that job state is consistent
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "pending"

    # Update the step status to active, and recheck the step and job status
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "active"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "active"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    # Now update the first step status to completed, recheck the step, and see that the job is still active.
    # Also check that the second step is now created.
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "completed"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "completed"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "completed"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "created"

    # Now move the second step to pending, and recheck the job status.
    # The job should still be active and the second step should be pending.
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2/status", json={"status": "pending"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "pending"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "pending"

    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    # Now update the second step status to active, and recheck the step and job status
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2/status", json={"status": "active"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "active"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "active"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    # Now update the second step status to completed, and recheck the step and job status are all completed.
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2/status", json={"status": "completed"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "completed"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step2")
    assert response.status_code == 200
    step_data = response.json()
    assert step_data["status"] == "completed"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "completed"


@pytest.mark.asyncio
async def test_job_lifecycle_errored_job(test_client: AsyncClient):
    """Test the lifecycle of a job that encounters an error."""
    # Create a job
    req = CreatePlatformJobRequest(
        name="test-job-errored",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[
                PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={}),
                PlatformJobStepSpec(name="step2", executor=TestConstants.TEST_EXECUTOR, config={}),
            ]
        ),
    )
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
    assert response.status_code == 201
    job_data = response.json()
    job_name = job_data["name"]  # API URLs use job name, not ID

    # Move the first step to active, verify the job is now active
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "active"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "active"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "active"

    # Simulate an error in the first step, and see that the job status is updated to errored
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/step1/status", json={"status": "error"}
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "error"
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert response.status_code == 200
    final_data = response.json()
    assert final_data["status"] == "error"


@pytest.mark.asyncio
async def test_job_paging(test_client: AsyncClient):
    """Test that paging works correctly with multiple jobs."""
    # Create multiple jobs for paging test
    job_ids = []
    for i in range(15):  # Create 15 jobs to test pagination
        req = CreatePlatformJobRequest(
            name=f"test-job-{i}",
            source="test-source",
            spec={"param1": f"value{i}"},
            platform_spec=PlatformJobSpec(
                steps=[PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={})]
            ),
        )
        response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req.model_dump())
        assert response.status_code == 201
        job_ids.append(response.json()["id"])

    # Test first page with default page size (10)
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()

    # Verify pagination metadata
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 10
    assert data["pagination"]["current_page_size"] == 10
    assert data["pagination"]["total_results"] == 15
    assert data["pagination"]["total_pages"] == 2
    assert len(data["data"]) == 10

    # Test second page
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=2&page_size=10")
    assert response.status_code == 200
    data = response.json()

    # Verify pagination metadata for second page
    assert data["pagination"]["page"] == 2
    assert data["pagination"]["page_size"] == 10
    assert data["pagination"]["current_page_size"] == 5  # Only 5 jobs left
    assert data["pagination"]["total_results"] == 15
    assert data["pagination"]["total_pages"] == 2
    assert len(data["data"]) == 5

    # Test with different page size
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=1&page_size=5")
    assert response.status_code == 200
    data = response.json()

    # Verify pagination metadata with smaller page size
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 5
    assert data["pagination"]["current_page_size"] == 5
    assert data["pagination"]["total_results"] == 15
    assert data["pagination"]["total_pages"] == 3
    assert len(data["data"]) == 5

    # Test last page with smaller page size
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=3&page_size=5")
    assert response.status_code == 200
    data = response.json()

    # Verify pagination metadata for last page
    assert data["pagination"]["page"] == 3
    assert data["pagination"]["page_size"] == 5
    assert data["pagination"]["current_page_size"] == 5
    assert data["pagination"]["total_results"] == 15
    assert data["pagination"]["total_pages"] == 3
    assert len(data["data"]) == 5

    # Test page beyond available data
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=10&page_size=10")
    assert response.status_code == 200
    data = response.json()

    # Verify pagination metadata for empty page
    # Note: When requesting a page beyond available data, the store may return 0 total_results
    # since it doesn't find any matching records within the offset range
    assert data["pagination"]["page"] == 10
    assert data["pagination"]["page_size"] == 10
    assert data["pagination"]["current_page_size"] == 0
    # The total_results may be 0 for out-of-range pages depending on store implementation
    assert data["pagination"]["total_results"] >= 0
    assert len(data["data"]) == 0

    # Test that all job IDs are returned across pages (no duplicates/missing)
    all_returned_ids = set()
    for page in range(1, 3):  # Pages 1 and 2 with page_size=10
        response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs?page={page}&page_size=10")
        data = response.json()
        for job in data["data"]:
            all_returned_ids.add(job["id"])

    # Verify all created jobs are returned
    assert len(all_returned_ids) == 15
    assert all_returned_ids == set(job_ids)

    # Invalid filter value passes through (entity store returns no results)
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=1&page_size=5&filter[status]=FOOBAR")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 0

    # Invalid filter field name is rejected
    response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs?page=1&page_size=5&filter[bogus]=test")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_job_result_crud(test_sdk: AsyncNeMoPlatform, sample_platform_job_request: CreatePlatformJobRequest):
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    sdk_job_resp = (await jobs.create_job(workspace=DEFAULT_WORKSPACE, body=sample_platform_job_request)).data()
    resp = (
        await jobs.create_job_result(
            name="result-name1",
            workspace=DEFAULT_WORKSPACE,
            job=sdk_job_resp.name,
            body=PlatformJobResultCreateRequest(
                artifact_url="default/test-fileset#myartifact",
                artifact_storage_type=FileStorageType.FILESET,
            ),
        )
    ).data()
    assert resp.name == "result-name1"
    assert resp.job == sdk_job_resp.id
    resp2 = (
        await jobs.create_job_result(
            name="result-name2",
            workspace=DEFAULT_WORKSPACE,
            job=sdk_job_resp.name,
            body=PlatformJobResultCreateRequest(
                artifact_url="default/test-fileset#myartifact",
                artifact_storage_type=FileStorageType.FILESET,
            ),
        )
    ).data()
    assert resp2.name == "result-name2"
    assert resp2.job == sdk_job_resp.id

    results = (await jobs.list_job_results(name=sdk_job_resp.name, workspace=DEFAULT_WORKSPACE)).data()
    assert len(results.data) == 2
    result2 = next(r for r in results.data if "result-name2" == r.name)

    another_result_2 = (
        await jobs.get_job_result(name=result2.name, workspace=DEFAULT_WORKSPACE, job=sdk_job_resp.name)
    ).data()
    assert result2 == another_result_2
    # The result's namespace is inherited from the parent job, not from the create request
    # Since the job uses the default namespace, the result should too
    assert result2.workspace == "default"


@pytest.mark.asyncio
async def test_job_result_download(
    test_sdk: AsyncNeMoPlatform,
    sample_platform_job_request: CreatePlatformJobRequest,
    mock_result_manager,
    tmp_path: Path,
):
    # set up some mock files that the result manager will return
    tmp_dir = tmp_path / "testdir"
    tmp_dir.mkdir()
    tmp_file = tmp_dir / "testfile.txt"
    testdata = "test data"
    tmp_file.write_text(testdata)
    mock_result_manager._tmp_dir = tmp_dir
    mock_result_manager._path = tmp_file

    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    sdk_job_resp = (await jobs.create_job(workspace=DEFAULT_WORKSPACE, body=sample_platform_job_request)).data()
    result = (
        await jobs.create_job_result(
            name="result-name1",
            workspace=DEFAULT_WORKSPACE,
            job=sdk_job_resp.name,
            body=PlatformJobResultCreateRequest(
                artifact_url="default/test-fileset#result_name1",
                artifact_storage_type=FileStorageType.FILESET,
            ),
        )
    ).data()

    with patch(
        "nmp.common.jobs.result_manager.async_result_manager_factory", return_value=mock_result_manager
    ) as factory:
        download = await jobs.download_job_result(name=result.name, workspace=DEFAULT_WORKSPACE, job=sdk_job_resp.name)
        download_bytes = await download.read()

    call_kwargs = factory.call_args.kwargs
    assert call_kwargs["job_name"] == sdk_job_resp.name
    assert call_kwargs["workspace"] == DEFAULT_WORKSPACE
    assert call_kwargs["sdk"]._client is test_sdk._client

    # make sure we deleted the temp files on the server
    assert not tmp_dir.exists()
    assert not tmp_file.exists()
    assert download_bytes.decode() == testdata

    # make the artifact a folder this time
    tmp_dir.mkdir()
    tmp_file.write_text(testdata)
    mock_result_manager._tmp_dir = tmp_dir
    mock_result_manager._path = tmp_dir

    with patch("nmp.common.jobs.result_manager.async_result_manager_factory", return_value=mock_result_manager):
        download = await jobs.download_job_result(name=result.name, workspace=DEFAULT_WORKSPACE, job=sdk_job_resp.name)
        tar_content = await download.read()

    # ensure we're returning the appropriate tar file
    assert "result-name1.tar.gz" in download.http_response.headers["content-disposition"]
    assert not tmp_dir.exists()
    assert not tmp_file.exists()

    # ensure the content of the tar includes the tmp_file
    tar_bytes = BytesIO(tar_content)
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        extracted_file = tar.extractfile(tar.getmember("testdir/testfile.txt"))
        assert extracted_file
        file_content = extracted_file.read()
        assert file_content.decode() == testdata


@pytest.mark.asyncio
async def test_job_status_details_crud(
    test_client: AsyncClient,
    test_sdk: AsyncNeMoPlatform,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    original_details = {"progress": 50, "metadata": {"key": "value"}}
    updated_details = {"progress": 75, "metadata": {"key": "value"}}
    patch = {"progress": 75}

    # Create a job
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    sdk_job_resp = (await jobs.create_job(workspace=DEFAULT_WORKSPACE, body=sample_platform_job_request)).data()
    job_name = sdk_job_resp.name  # API URLs use job name, not ID

    ### Test patching a job status details
    # Post a status update like we would from a job
    resp = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status-details", json=original_details
    )
    assert resp.status_code == 200

    # Update the job step from created to active
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/basic/status",
        json={"status": "active", "status_details": {"message": "Step is now active"}},
    )
    assert response.status_code == 200
    updated_data = response.json()
    assert updated_data["status"] == "active"
    assert updated_data["status_details"] == {"message": "Step is now active"}

    # Get the job and inspect it's status details
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert resp.status_code == 200
    data = resp.json()
    job_status_details = data.get("status_details")
    assert original_details == job_status_details

    # Patch the status details, and ensure they match the updated details
    resp = await test_client.patch(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status-details", json=patch)
    assert resp.status_code == 200
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}")
    assert resp.status_code == 200
    data = resp.json()
    job_status_details = data.get("status_details")
    assert updated_details == job_status_details

    # Get the job step and inspect it's status details
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/basic")
    assert resp.status_code == 200
    data = resp.json()
    step_status_details = data.get("status_details")
    assert step_status_details == {"message": "Step is now active"}

    # Now lets create a task associated with the step, and then update it's status
    task_resp = (
        await jobs.update_job_step_task(
            name="task-1",
            workspace=DEFAULT_WORKSPACE,
            job=job_name,
            step="basic",
            body=PlatformJobTaskUpdate(
                status=PlatformJobStatus.ACTIVE,
                status_details={"message": "Task is now active"},
            ),
        )
    ).data()
    assert task_resp is not None

    # Now get the task and inspect it's status details (URL uses task name, not ID)
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/basic/tasks/{task_resp.name}")
    assert resp.status_code == 200
    data = resp.json()
    task_status_details = data.get("status_details")
    assert task_status_details == {"message": "Task is now active"}

    # Finally, ensure that the job status endpoint returns the overall status details for the job, step, and task correctly
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert resp.status_code == 200
    status_data = resp.json()

    assert status_data["status"] == "active"
    # Task status_details are propagated to job level for progress tracking
    expected_merged_details = {**updated_details, "message": "Task is now active"}
    assert status_data["status_details"] == expected_merged_details
    # Verify step structure
    assert len(status_data["steps"]) == 1
    step_data = status_data["steps"][0]
    assert step_data["name"].startswith("basic")
    assert step_data["status"] == "active"
    assert step_data["error_details"] == {}
    assert step_data["status_details"] == {"message": "Step is now active"}
    # Verify task structure
    assert len(step_data["tasks"]) == 1
    task_data = step_data["tasks"][0]
    assert task_data["id"] == task_resp.id  # Use the actual generated task ID
    assert task_data["status"] == "active"
    assert task_data["error_details"] == {}
    assert task_data["error_stack"] is None
    assert task_data["status_details"] == {"message": "Task is now active"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_template,min_expected_results",
    [
        # Test filtering with created_at[gte] - should include both jobs
        ("filter[created_at][gte]={job1_created_at}", 2),
        # Test filtering with created_at[lte] - should include at least the first job
        ("filter[created_at][lte]={job1_created_at}", 1),
        # Test filtering with both gte and lte - exact match should get at least job1
        ("filter[created_at][gte]={job1_created_at}&filter[created_at][lte]={job1_created_at}", 1),
    ],
)
async def test_job_list_filter_with_datetime(test_client: AsyncClient, filter_template: str, min_expected_results: int):
    """Test that datetime filters with gte/lte work correctly."""
    # Create a few jobs at different times
    req1 = CreatePlatformJobRequest(
        name="test-job-1",
        source="test-source",
        spec={"param1": "value1"},
        platform_spec=PlatformJobSpec(
            steps=[PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={})]
        ),
    )
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req1.model_dump())
    assert response.status_code == 201
    job1 = response.json()
    job1_created_at = job1["created_at"]

    req2 = CreatePlatformJobRequest(
        name="test-job-2",
        source="test-source",
        spec={"param1": "value2"},
        platform_spec=PlatformJobSpec(
            steps=[PlatformJobStepSpec(name="step1", executor=TestConstants.TEST_EXECUTOR, config={})]
        ),
    )
    response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=req2.model_dump())
    assert response.status_code == 201

    # Build filter query from template
    filter_query = filter_template.format(job1_created_at=job1_created_at)

    # Test the filter
    response = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs?{filter_query}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) >= min_expected_results


class MockRequest:
    """Mock Request class for testing filter functions."""

    def __init__(self, query_string: str):
        self.query_params = QueryParams(query_string)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_string,expected_checks",
    [
        # Multiple status values
        (
            "filter[status]=active&filter[status]=pending",
            lambda result: (
                result.status is not None
                and PlatformJobStatus.ACTIVE in result.status
                and PlatformJobStatus.PENDING in result.status
            ),
        ),
        # Empty filter
        (
            "",
            lambda result: result is not None,
        ),
        # Filter by job
        (
            "filter[job]=test-job-123",
            lambda result: result.job == "test-job-123",
        ),
        # Filter by source
        (
            "filter[source]=test-source",
            lambda result: result.source == "test-source",
        ),
        # Multiple filters together
        (
            "filter[job]=test-job-123&filter[source]=my-source",
            lambda result: result.job == "test-job-123" and result.source == "my-source",
        ),
        # Multiple statuses
        (
            "filter[status]=active&filter[status]=pending",
            lambda result: (
                result.status is not None
                and PlatformJobStatus.ACTIVE in result.status
                and PlatformJobStatus.PENDING in result.status
            ),
        ),
    ],
)
async def test_get_platform_jobs_steps_list_filter(query_string, expected_checks):
    """Test the get_platform_jobs_steps_list_filter function with various filter parameters."""
    request = MockRequest(query_string)
    result = get_platform_jobs_steps_list_filter(request)  # ty: ignore[invalid-argument-type]
    assert expected_checks(result)


@pytest.mark.asyncio
async def test_get_platform_jobs_steps_list_filter_invalid():
    """Test that invalid status raises an error."""
    request = MockRequest("filter[status]=INVALID_STATUS")
    with pytest.raises(HTTPException) as exc_info:
        get_platform_jobs_steps_list_filter(request)  # ty: ignore[invalid-argument-type]
    assert exc_info.value.detail is not None
    assert "filter.status" in str(exc_info.value.detail)
    assert "Expected one or more of" in str(exc_info.value.detail)
    assert "ValidationError" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_cancel_job_conflict_hides_internal_transition(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
):
    raw_error = (
        "Invalid status transition from PlatformJobStatus.RESUMING to PlatformJobStatus.CANCELLING for step step-id-123"
    )
    with patch.object(
        mock_dispatcher, "cancel_job", new=AsyncMock(side_effect=StateTransitionConflictError(raw_error))
    ):
        response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs/conflicted-job/cancel")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Cannot cancel job 'conflicted-job'" in detail
    assert "PlatformJobStatus" not in detail
    assert "step-id-123" not in detail


@pytest.mark.asyncio
async def test_cancel_job_conflict_sanitizes_log_fields(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
    caplog,
):
    with patch.object(
        mock_dispatcher,
        "cancel_job",
        new=AsyncMock(side_effect=StateTransitionConflictError("invalid transition")),
    ):
        with caplog.at_level(logging.INFO, logger="nmp.core.jobs.api.v2.jobs.endpoints"):
            response = await test_client.post(
                "/apis/jobs/v2/workspaces/default%0Aforged/jobs/conflicted-job%0D%0Aforged/cancel"
            )

    assert response.status_code == 409
    message = next(record.getMessage() for record in caplog.records if "Cannot cancel job" in record.msg)
    assert "\r" not in message
    assert "\n" not in message


@pytest.mark.asyncio
async def test_delete_non_terminal_job_returns_409_and_keeps_job(
    test_client: AsyncClient,
    sample_platform_job_request: CreatePlatformJobRequest,
    caplog: pytest.LogCaptureFixture,
):
    request = sample_platform_job_request.model_copy(update={"name": "non-terminal-delete"})
    create_response = await test_client.post("/apis/jobs/v2/workspaces/default/jobs", json=request.model_dump())
    assert create_response.status_code == 201, create_response.text

    with caplog.at_level(logging.INFO, logger="nmp.core.jobs.api.v2.jobs.endpoints"):
        delete_response = await test_client.delete("/apis/jobs/v2/workspaces/default/jobs/non-terminal-delete")

    assert delete_response.status_code == 409
    assert "Cancel the job and wait for it to reach a terminal state" in delete_response.json()["detail"]
    log_record = next(record for record in caplog.records if "Cannot delete job" in record.getMessage())
    assert log_record.exc_info is None

    get_response = await test_client.get("/apis/jobs/v2/workspaces/default/jobs/non-terminal-delete")
    assert get_response.status_code == 200
    assert get_response.json()["status"] == PlatformJobStatus.CREATED.value


@pytest.mark.asyncio
async def test_factory_delete_non_terminal_job_propagates_409(test_client: AsyncClient):
    create_response = await test_client.post(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs",
        json={
            "name": "factory-non-terminal-delete",
            "description": "factory delete conflict",
            "spec": {"config": {"key": "value"}, "target": "str"},
            "ownership": {"user": "u", "service": "s"},
        },
    )
    assert create_response.status_code == 201, create_response.text

    delete_response = await test_client.delete(
        "/apis/jobs/v2/workspaces/default/hello-world/jobs/factory-non-terminal-delete"
    )

    assert delete_response.status_code == 409
    assert "Cancel the job and wait for it to reach a terminal state" in delete_response.json()["detail"]


@pytest.mark.asyncio
async def test_update_job_step_conflict_sanitizes_log_fields(
    test_client: AsyncClient,
    mock_dispatcher: JobDispatcher,
    caplog,
):
    with (
        patch.object(
            mock_dispatcher,
            "get_current_job_step_by_name",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch.object(
            mock_dispatcher,
            "update_job_status_from_step",
            new=AsyncMock(side_effect=StateTransitionConflictError("invalid transition")),
        ),
        caplog.at_level(logging.INFO, logger="nmp.core.jobs.api.v2.jobs.endpoints"),
    ):
        response = await test_client.patch(
            "/apis/jobs/v2/workspaces/default/jobs/job%0Aforged/steps/step%0D%0Aforged/status",
            json={"status": "pending"},
        )

    assert response.status_code == 409
    message = next(record.getMessage() for record in caplog.records if "Cannot update job step" in record.msg)
    assert "\r" not in message
    assert "\n" not in message


@pytest.mark.asyncio
async def test_job_steps_list_global_vs_workspaced(sample_platform_job_request: CreatePlatformJobRequest):
    """Test that global step listing returns steps from all workspaces while workspaced calls are filtered."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from nmp.common.entities.client import EntityClient
    from nmp.testing import create_test_client

    # Create entity store with multiple workspaces and projects
    projects = ["default/test-project", "other-workspace/test-project"]
    with create_test_client(client_type=EntityClient, projects=projects) as mock_store:
        # Create mock SDK with patched files client
        mock_nmp_client = MagicMock()
        mock_files = AsyncMock()
        mock_fileset_obj = MagicMock()
        mock_fileset_obj.name = "test-fileset-id"
        mock_resp = MagicMock()
        mock_resp.data.return_value = mock_fileset_obj
        mock_files.create_fileset.return_value = mock_resp

        with patch("nmp.core.jobs.app.dispatcher.client_from_platform", return_value=mock_files):
            # Create dispatcher with the multi-workspace store
            mock_dispatcher = JobDispatcher(store=mock_store, sdk=mock_nmp_client)

            # Create jobs in "default" workspace
            job1 = await mock_dispatcher.create_job(sample_platform_job_request, DEFAULT_WORKSPACE)

            job2_request = CreatePlatformJobRequest(
                name="test-job-2",
                description="Second test job",
                project="test-project",
                source=TestConstants.SOURCE,
                spec=TestConstants.SPEC_BASIC,
                platform_spec=TestConstants.PLATFORM_SPEC,
                ownership=TestConstants.OWNERSHIP_BASIC,
                custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
            )
            job2 = await mock_dispatcher.create_job(job2_request, DEFAULT_WORKSPACE)

            # Create jobs in "other-workspace"
            job3_request = CreatePlatformJobRequest(
                name="test-job-3",
                description="Third test job in other workspace",
                project="test-project",
                source=TestConstants.SOURCE,
                spec=TestConstants.SPEC_BASIC,
                platform_spec=TestConstants.PLATFORM_SPEC,
                ownership=TestConstants.OWNERSHIP_BASIC,
                custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
            )
            job3 = await mock_dispatcher.create_job(job3_request, "other-workspace")

            job4_request = CreatePlatformJobRequest(
                name="test-job-4",
                description="Fourth test job in other workspace",
                project="test-project",
                source=TestConstants.SOURCE,
                spec=TestConstants.SPEC_BASIC,
                platform_spec=TestConstants.PLATFORM_SPEC,
                ownership=TestConstants.OWNERSHIP_BASIC,
                custom_fields=TestConstants.CUSTOM_FIELDS_BASIC,
            )
            job4 = await mock_dispatcher.create_job(job4_request, "other-workspace")

        # Test global listing with wildcard - should return steps from all workspaces
        step_filter = PlatformJobStepsListFilter()
        steps_global, count_global = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace=ALL_WORKSPACES,
        )

        assert len(steps_global) == 4, f"Expected 4 steps globally, got {len(steps_global)}"
        assert count_global == 4, f"Expected count=4 globally, got {count_global}"

        # Verify steps from both workspaces are present
        global_job_names = {step.job for step in steps_global}
        assert job1.name in global_job_names, "job1 from default workspace should be in global results"
        assert job2.name in global_job_names, "job2 from default workspace should be in global results"
        assert job3.name in global_job_names, "job3 from other-workspace should be in global results"
        assert job4.name in global_job_names, "job4 from other-workspace should be in global results"

        # Test workspace-specific listing for "default" - should only return steps from default workspace
        steps_default, count_default = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace="default",
        )

        assert len(steps_default) == 2, f"Expected 2 steps in default workspace, got {len(steps_default)}"
        assert count_default == 2, f"Expected count=2 in default workspace, got {count_default}"

        default_job_names = {step.job for step in steps_default}
        assert job1.name in default_job_names, "job1 should be in default workspace results"
        assert job2.name in default_job_names, "job2 should be in default workspace results"
        assert job3.name not in default_job_names, "job3 from other-workspace should NOT be in default results"
        assert job4.name not in default_job_names, "job4 from other-workspace should NOT be in default results"

        # Test workspace-specific listing for "other-workspace" - should only return steps from other-workspace
        steps_other, count_other = await mock_dispatcher.list_steps(
            filter=step_filter,
            sort=PlatformJobSortField.CREATED_AT_ASC,
            limit=100,
            offset=0,
            workspace="other-workspace",
        )

        assert len(steps_other) == 2, f"Expected 2 steps in other-workspace, got {len(steps_other)}"
        assert count_other == 2, f"Expected count=2 in other-workspace, got {count_other}"

        other_job_names = {step.job for step in steps_other}
        assert job3.name in other_job_names, "job3 should be in other-workspace results"
        assert job4.name in other_job_names, "job4 should be in other-workspace results"
        assert job1.name not in other_job_names, "job1 from default should NOT be in other-workspace results"
        assert job2.name not in other_job_names, "job2 from default should NOT be in other-workspace results"

        # Verify workspace field in returned steps
        for step in steps_default:
            assert step.workspace == "default", "All steps in default results should have workspace='default'"

        for step in steps_other:
            assert step.workspace == "other-workspace", (
                "All steps in other results should have workspace='other-workspace'"
            )


@pytest.mark.asyncio
async def test_job_status_timestamps(
    test_client: AsyncClient,
    test_sdk: AsyncNeMoPlatform,
    sample_platform_job_request: CreatePlatformJobRequest,
):
    """Test that created_at and updated_at are present at job, step, and task levels in status response."""
    jobs = client_from_platform(test_sdk, AsyncJobsClient)
    sdk_job_resp = (await jobs.create_job(workspace=DEFAULT_WORKSPACE, body=sample_platform_job_request)).data()
    job_name = sdk_job_resp.name

    # Activate the step so the job is in a meaningful state
    response = await test_client.patch(
        f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/steps/basic/status",
        json={"status": "active"},
    )
    assert response.status_code == 200

    # Create a task for the step
    task_resp = (
        await jobs.update_job_step_task(
            name="task-1",
            workspace=DEFAULT_WORKSPACE,
            job=job_name,
            step="basic",
            body=PlatformJobTaskUpdate(status=PlatformJobStatus.ACTIVE),
        )
    ).data()
    assert task_resp is not None

    # Fetch the status and verify timestamps are present at all levels
    resp = await test_client.get(f"/apis/jobs/v2/workspaces/default/jobs/{job_name}/status")
    assert resp.status_code == 200
    status_data = resp.json()

    # Job-level timestamps
    assert status_data["created_at"] is not None
    assert status_data["updated_at"] is not None
    datetime.fromisoformat(status_data["created_at"])
    datetime.fromisoformat(status_data["updated_at"])

    # Step-level timestamps
    assert len(status_data["steps"]) == 1
    step_data = status_data["steps"][0]
    assert step_data["created_at"] is not None
    assert step_data["updated_at"] is not None
    datetime.fromisoformat(step_data["created_at"])
    datetime.fromisoformat(step_data["updated_at"])

    # Task-level timestamps
    assert len(step_data["tasks"]) == 1
    task_data = step_data["tasks"][0]
    assert task_data["created_at"] is not None
    assert task_data["updated_at"] is not None
    datetime.fromisoformat(task_data["created_at"])
    datetime.fromisoformat(task_data["updated_at"])
