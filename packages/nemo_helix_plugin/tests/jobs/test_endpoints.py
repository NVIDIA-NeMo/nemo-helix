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
    JobStatusDetailsUpdate,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
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


def test_update_job_status_details() -> None:
    prepared = endpoints.update_job_status_details(
        workspace="default", name="j-1", body=JobStatusDetailsUpdate({"note": "x"})
    )
    assert prepared.method == "PATCH"
    assert prepared.path_template.endswith("/jobs/{name}/status-details")
    assert prepared.response_type is None
    # RootModel serialises to the bare JSON object
    assert isinstance(prepared.content, bytes)
    assert json.loads(prepared.content) == {"note": "x"}


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


def test_get_job_result() -> None:
    prepared = endpoints.get_job_result(workspace="default", job="j-1", name="out")
    assert prepared.method == "GET"
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "out"}


def test_download_job_result() -> None:
    prepared = endpoints.download_job_result(workspace="default", job="j-1", name="out")
    assert prepared.method == "GET"
    assert prepared.path_template.endswith("/results/{name}/download")
    assert prepared.path_params == {"workspace": "default", "job": "j-1", "name": "out"}
    assert prepared.content is None
    assert prepared.response_type is BinaryContent


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
