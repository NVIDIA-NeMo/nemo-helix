# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Jobs service endpoint definitions."""

from __future__ import annotations

import json
from typing import get_args, get_origin

from nemo_helix_plugin.client.types import BinaryContent, CursorPagination, Paginated, PreparedRequest
from nemo_helix_plugin.jobs import endpoints
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import (
    FileStorageType,
    HelixJobLog,
    HelixJobResultCreateRequest,
    HelixJobStatus,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import (
    CreateHelixJobRequest,
    HelixJobAttemptResponse,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
    JobStatusDetailsUpdate,
    UpdateHelixJobRequest,
)


def _create_request() -> CreateHelixJobRequest:
    return CreateHelixJobRequest(
        spec={},
        source="test",
        platform_spec=HelixJobSpec(
            steps=[
                HelixJobStepSpec(
                    name="step-one",
                    executor=CPUExecutionProvider(container=ContainerSpec(image="img")),
                )
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Execution profiles
# ---------------------------------------------------------------------------


def test_get_execution_profiles() -> None:
    prepared = endpoints.get_execution_profiles()

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/jobs/v2/execution-profiles"
    assert prepared.path_params == {}
    assert prepared.content is None


# ---------------------------------------------------------------------------
# Job CRUD + lifecycle
# ---------------------------------------------------------------------------


def test_create_job() -> None:
    body = _create_request()
    prepared = endpoints.create_job(workspace="default", body=body)

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/jobs/v2/workspaces/{workspace}/jobs"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()
    assert prepared.content_type == "application/json"
    assert prepared.response_type is HelixJobResponse


def test_create_job_workspace_optional() -> None:
    prepared = endpoints.create_job(body=_create_request())
    assert prepared.path_params == {}


def test_list_jobs() -> None:
    prepared = endpoints.list_jobs(workspace="default")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.content is None
    assert get_origin(prepared.response_type) is Paginated


def test_list_jobs_with_query_params() -> None:
    prepared = endpoints.list_jobs(workspace="default", query_params={"page": 2, "page_size": 10})
    assert prepared.query_params == {"page": 2, "page_size": 10}


def test_get_job() -> None:
    prepared = endpoints.get_job(workspace="default", name="j-1")

    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "name": "j-1"}
    assert prepared.response_type is HelixJobResponse


def test_update_job_preserves_explicit_null_description() -> None:
    body = UpdateHelixJobRequest(description=None)
    prepared = endpoints.update_job(workspace="default", name="j-1", body=body)

    assert prepared.method == "PATCH"
    assert prepared.path_params == {"workspace": "default", "name": "j-1"}
    assert isinstance(prepared.content, bytes)
    assert json.loads(prepared.content) == {"description": None}


def test_delete_job() -> None:
    prepared = endpoints.delete_job(workspace="default", name="j-1")

    assert prepared.method == "DELETE"
    assert prepared.path_params == {"workspace": "default", "name": "j-1"}
    assert prepared.content is None
    assert prepared.response_type is None


def test_cancel_job() -> None:
    prepared = endpoints.cancel_job(workspace="default", name="j-1")
    assert prepared.method == "POST"
    assert prepared.path_template.endswith("/jobs/{name}/cancel")
    assert prepared.response_type is HelixJobResponse


def test_pause_job() -> None:
    prepared = endpoints.pause_job(workspace="default", name="j-1")
    assert prepared.method == "POST"
    assert prepared.path_template.endswith("/jobs/{name}/pause")


def test_resume_job() -> None:
    prepared = endpoints.resume_job(workspace="default", name="j-1")
    assert prepared.method == "POST"
    assert prepared.path_template.endswith("/jobs/{name}/resume")


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


def test_get_job_status() -> None:
    prepared = endpoints.get_job_status(workspace="default", name="j-1")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{name}/status")
    assert prepared.response_type is HelixJobStatusResponse


def test_list_and_get_job_attempts() -> None:
    listed = endpoints.list_job_attempts(
        workspace="default",
        job="j-1",
        query_params={"page": 2, "page_size": 5, "sort": "seq"},
    )
    fetched = endpoints.get_job_attempt(workspace="default", job="j-1", attempt="current")

    assert listed.path_template.endswith("/jobs/{job}/attempts")
    assert listed.query_params == {"page": 2, "page_size": 5, "sort": "seq"}
    assert get_args(listed.response_type)[0] is HelixJobAttemptResponse
    assert fetched.path_params == {"workspace": "default", "job": "j-1", "attempt": "current"}
    assert fetched.response_type is HelixJobAttemptResponse


def test_update_attempt_status_details() -> None:
    prepared = endpoints.update_job_attempt_status_details(
        workspace="default",
        job="j-1",
        attempt="att-1",
        body=JobStatusDetailsUpdate({"progress": 50}),
    )

    assert prepared.method == "PATCH"
    assert prepared.path_template.endswith("/attempts/{attempt}/status-details")
    assert json.loads(prepared.content) == {"progress": 50}


# ---------------------------------------------------------------------------
# Job logs
# ---------------------------------------------------------------------------


def test_list_job_logs() -> None:
    prepared = endpoints.list_job_logs(
        workspace="default", name="j-1", query_params={"limit": 50, "page_cursor": "abc", "tail": 500}
    )
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{name}/logs")
    assert prepared.query_params == {"limit": 50, "page_cursor": "abc", "tail": 500}
    assert get_origin(prepared.response_type) is Paginated
    assert get_args(prepared.response_type) == (HelixJobLog, CursorPagination)


# ---------------------------------------------------------------------------
# Job results
# ---------------------------------------------------------------------------


def test_create_job_result() -> None:
    body = HelixJobResultCreateRequest(artifact_url="s3://x", artifact_storage_type=FileStorageType.FILESET)
    prepared = endpoints.create_job_result(workspace="default", job="j-1", name="out", body=body)
    assert prepared.method == "POST"
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "out"}
    assert prepared.content == body.model_dump_json(exclude_unset=True).encode()


def test_list_job_results() -> None:
    prepared = endpoints.list_job_results(workspace="default", name="j-1")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{name}/results")


def test_download_job_result() -> None:
    prepared = endpoints.download_job_result(workspace="default", job="j-1", name="out")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/results/{name}/download")
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "out"}
    assert prepared.content is None
    assert prepared.response_type is BinaryContent


def test_attempt_result_endpoints_include_full_parent_path() -> None:
    body = HelixJobResultCreateRequest(artifact_url="s3://x", artifact_storage_type=FileStorageType.FILESET)
    created = endpoints.create_attempt_result(
        workspace="default",
        job="j-1",
        attempt="att-1",
        name="out",
        body=body,
    )
    listed = endpoints.list_attempt_results(workspace="default", job="j-1", attempt="att-1")
    fetched = endpoints.get_attempt_result(
        workspace="default",
        job="j-1",
        attempt="att-1",
        name="out",
    )

    expected = {"workspace": "default", "job": "j-1", "attempt": "att-1", "name": "out"}
    assert created.path_params == expected
    assert listed.path_params == {"workspace": "default", "job": "j-1", "attempt": "att-1"}
    assert fetched.path_params == expected


# ---------------------------------------------------------------------------
# Job steps
# ---------------------------------------------------------------------------


def test_list_steps() -> None:
    prepared = endpoints.list_steps(workspace="default", name="j-1")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/jobs/{name}/steps")
    assert get_origin(prepared.response_type) is Paginated


def test_list_steps_with_query_params() -> None:
    prepared = endpoints.list_steps(workspace="default", name="-", query_params={"page": 1, "sort": "created_at"})
    assert prepared.query_params == {"page": 1, "sort": "created_at"}


def test_get_job_step() -> None:
    prepared = endpoints.get_job_step(workspace="default", job="j-1", name="step-one")
    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "step-one"}
    assert prepared.response_type is HelixJobStepResponse


def test_update_job_step_status() -> None:
    body = HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE)
    prepared = endpoints.update_job_step_status(workspace="default", job="j-1", name="step-one", body=body)
    assert prepared.method == "PATCH"
    assert prepared.path_template.endswith("/steps/{name}/status")
    assert prepared.response_type is HelixJobStepResponse


def test_attempt_step_endpoints_include_full_parent_path() -> None:
    body = HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE)
    listed = endpoints.list_attempt_steps(workspace="default", job="j-1", attempt="att-1")
    fetched = endpoints.get_attempt_step(workspace="default", job="j-1", attempt="att-1", name="step-one")
    updated = endpoints.update_attempt_step_status(
        workspace="default",
        job="j-1",
        attempt="att-1",
        name="step-one",
        body=body,
    )

    assert listed.path_params == {"workspace": "default", "job": "j-1", "attempt": "att-1"}
    expected = {"workspace": "default", "job": "j-1", "attempt": "att-1", "name": "step-one"}
    assert fetched.path_params == expected
    assert updated.path_params == expected


# ---------------------------------------------------------------------------
# Job tasks
# ---------------------------------------------------------------------------


def test_list_job_step_tasks() -> None:
    prepared = endpoints.list_job_step_tasks(workspace="default", job="j-1", name="step-one")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/steps/{name}/tasks")
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "step-one"}


def test_update_job_step_task() -> None:
    body = HelixJobTaskUpdate(status=HelixJobStatus.COMPLETED)
    prepared = endpoints.update_job_step_task(workspace="default", job="j-1", step="step-one", name="task-1", body=body)
    assert prepared.method == "PUT"
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "step": "step-one", "name": "task-1"}
    assert prepared.response_type is HelixJobTaskResponse


def test_get_job_step_task() -> None:
    prepared = endpoints.get_job_step_task(workspace="default", job="j-1", step="step-one", name="task-1")
    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "step": "step-one", "name": "task-1"}
    assert prepared.response_type is HelixJobTaskResponse


def test_attempt_task_endpoints_include_full_parent_path() -> None:
    body = HelixJobTaskUpdate(status=HelixJobStatus.COMPLETED)
    listed = endpoints.list_attempt_step_tasks(workspace="default", job="j-1", attempt="att-1", step="step-one")
    updated = endpoints.update_attempt_step_task(
        workspace="default",
        job="j-1",
        attempt="att-1",
        step="step-one",
        name="task-1",
        body=body,
    )
    fetched = endpoints.get_attempt_step_task(
        workspace="default",
        job="j-1",
        attempt="att-1",
        step="step-one",
        name="task-1",
    )

    assert listed.path_params == {
        "workspace": "default",
        "job": "j-1",
        "attempt": "att-1",
        "step": "step-one",
    }
    assert updated.path_params["attempt"] == "att-1"
    assert fetched.path_params == updated.path_params


# ---------------------------------------------------------------------------
# Request-body serialisation
# ---------------------------------------------------------------------------


def test_create_job_body_roundtrip() -> None:
    body = _create_request()
    prepared = endpoints.create_job(workspace="default", body=body)
    assert isinstance(prepared.content, bytes)
    content = json.loads(prepared.content)
    assert content["source"] == "test"
    assert content["platform_spec"]["steps"][0]["name"] == "step-one"


def test_step_status_update_excludes_unset() -> None:
    body = HelixJobStatusUpdateRequest(status=HelixJobStatus.ACTIVE)
    prepared = endpoints.update_job_step_status(workspace="default", job="j-1", name="s", body=body)
    assert isinstance(prepared.content, bytes)
    content = json.loads(prepared.content)
    assert content["status"] == "active"


def test_step_with_context_response_type() -> None:
    prepared = endpoints.list_steps(workspace="default", name="j-1")
    # Paginated marker parametrised with the step-with-context model.
    assert get_args(prepared.response_type)[0] is HelixJobStepWithContext
