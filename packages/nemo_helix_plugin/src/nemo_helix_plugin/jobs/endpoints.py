# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Jobs service.

These are the single source of truth for the HTTP contract.  Each function
is decorated with an HTTP-method decorator and ``@abstractmethod``; the
decorator turns the signature into a :class:`PreparedRequest` builder.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any

from nemo_helix_plugin.client.endpoint import delete, get, patch, post, put
from nemo_helix_plugin.client.types import BinaryContent, CursorPagination, Paginated
from nemo_helix_plugin.jobs.backends import GenericExecutionProfile, discover_job_backends
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    E2EJobExecutionProfile,
    KubernetesJobExecutionProfile,
    OpenShellJobExecutionProfile,
    SubprocessJobExecutionProfile,
    VolcanoJobExecutionProfile,
)
from nemo_helix_plugin.jobs.schemas import (
    HelixJobLog,
    HelixJobResultCreateRequest,
    HelixJobResultResponse,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile
from nemo_helix_plugin.jobs.types import (
    CreateHelixJobRequest,
    HelixJobAttemptResponse,
    HelixJobListResultResponse,
    HelixJobListTaskResponse,
    HelixJobResponse,
    HelixJobStatusUpdateRequest,
    HelixJobStepResponse,
    HelixJobStepWithContext,
    HelixJobTaskResponse,
    HelixJobTaskUpdate,
    JobControl,
    JobLogsQueryParams,
    JobStatusDetailsQueryParams,
    JobStatusDetailsUpdate,
    ListAttemptsQueryParams,
    ListJobResultsQueryParams,
    ListJobsQueryParams,
    ListStepsQueryParams,
    UpdateHelixJobRequest,
)
from pydantic import SerializeAsAny, TypeAdapter, ValidatorFunctionWrapHandler, WrapValidator

# The execution-profiles endpoint returns a union over all configured backend
# profile types.
_BuiltinExecutionProfile = (
    DockerJobExecutionProfile
    | KubernetesJobExecutionProfile
    | VolcanoJobExecutionProfile
    | SubprocessJobExecutionProfile
    | OpenShellJobExecutionProfile
    | E2EJobExecutionProfile
)

_BUILTIN_PROFILE_MODELS = {
    model.model_fields["backend"].default: model
    for model in (
        DockerJobExecutionProfile,
        KubernetesJobExecutionProfile,
        VolcanoJobExecutionProfile,
        SubprocessJobExecutionProfile,
        OpenShellJobExecutionProfile,
        E2EJobExecutionProfile,
    )
}
_builtin_profile_adapter = TypeAdapter(_BuiltinExecutionProfile)


def _parse_execution_profile_response(value: Any, _handler: ValidatorFunctionWrapHandler) -> BaseExecutionProfile:
    if isinstance(value, BaseExecutionProfile):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise ValueError("Execution profile must be an object")
    if "backend" not in value:
        # Preserve the original built-in union's default selection behavior.
        return _builtin_profile_adapter.validate_python(value)
    backend = value["backend"]
    if not isinstance(backend, str):
        raise ValueError("Execution profile backend must be a string")
    model = _BUILTIN_PROFILE_MODELS.get(backend)
    if model is None:
        contribution = discover_job_backends().get(backend)
        model = contribution.profile_model if contribution else GenericExecutionProfile
    return model.model_validate(value)


# Explicit dispatch prevents invalid known configs from falling back to the
# generic model. WrapValidator accepts contributed subclasses while retaining
# stable built-in schemas; duck serialization preserves their config fields.
ExecutionProfile = Annotated[
    SerializeAsAny[_BuiltinExecutionProfile | GenericExecutionProfile],
    WrapValidator(_parse_execution_profile_response),
]

# ---------------------------------------------------------------------------
# Execution profiles
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/execution-profiles")
@abstractmethod
def get_execution_profiles() -> list[ExecutionProfile]: ...


# ---------------------------------------------------------------------------
# Job CRUD + lifecycle
# ---------------------------------------------------------------------------


@post("/apis/jobs/v2/workspaces/{workspace}/jobs")
@abstractmethod
def create_job(*, workspace: str | None = None, body: CreateHelixJobRequest) -> HelixJobResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs")
@abstractmethod
def list_jobs(
    *, workspace: str | None = None, query_params: ListJobsQueryParams | None = None
) -> Paginated[HelixJobResponse]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def get_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def update_job(*, workspace: str | None = None, name: str, body: UpdateHelixJobRequest) -> HelixJobResponse: ...


@delete("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}")
@abstractmethod
def delete_job(*, workspace: str | None = None, name: str) -> None: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/cancel")
@abstractmethod
def cancel_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/pause")
@abstractmethod
def pause_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/resume")
@abstractmethod
def resume_job(*, workspace: str | None = None, name: str) -> HelixJobResponse: ...


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/rerun")
@abstractmethod
def rerun_job(*, workspace: str | None = None, job: str) -> HelixJobResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/control")
@abstractmethod
def update_job_control(*, workspace: str | None = None, name: str, body: JobControl) -> HelixJobResponse: ...


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/status")
@abstractmethod
def get_job_status(*, workspace: str | None = None, name: str) -> HelixJobStatusResponse: ...


# ---------------------------------------------------------------------------
# Job attempts
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts")
@abstractmethod
def list_job_attempts(
    *, workspace: str | None = None, job: str, query_params: ListAttemptsQueryParams | None = None
) -> Paginated[HelixJobAttemptResponse]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}")
@abstractmethod
def get_job_attempt(*, workspace: str | None = None, job: str, attempt: str) -> HelixJobAttemptResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/status-details")
@abstractmethod
def update_job_attempt_status_details(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    body: JobStatusDetailsUpdate,
    query_params: JobStatusDetailsQueryParams | None = None,
) -> None: ...


# ---------------------------------------------------------------------------
# Job logs
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/logs")
@abstractmethod
def list_job_logs(
    *, workspace: str | None = None, name: str, query_params: JobLogsQueryParams | None = None
) -> Paginated[HelixJobLog, CursorPagination]: ...


# ---------------------------------------------------------------------------
# Job results
# ---------------------------------------------------------------------------


@post("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def create_job_result(
    *, workspace: str | None = None, job: str, name: str, body: HelixJobResultCreateRequest
) -> HelixJobResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/results")
@abstractmethod
def list_job_results(
    *, workspace: str | None = None, name: str, query_params: ListJobResultsQueryParams | None = None
) -> HelixJobListResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}")
@abstractmethod
def get_job_result(*, workspace: str | None = None, job: str, name: str) -> HelixJobResultResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/results/{name}/download")
@abstractmethod
def download_job_result(*, workspace: str | None = None, job: str, name: str) -> BinaryContent: ...


# ---------------------------------------------------------------------------
# Job steps
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/steps")
@abstractmethod
def list_steps(
    *, workspace: str | None = None, name: str, query_params: ListStepsQueryParams | None = None
) -> Paginated[HelixJobStepWithContext]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}")
@abstractmethod
def get_job_step(*, workspace: str | None = None, job: str, name: str) -> HelixJobStepResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/status")
@abstractmethod
def update_job_step_status(
    *, workspace: str | None = None, job: str, name: str, body: HelixJobStatusUpdateRequest
) -> HelixJobStepResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps")
@abstractmethod
def list_attempt_steps(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    query_params: ListStepsQueryParams | None = None,
) -> Paginated[HelixJobStepWithContext]: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{name}")
@abstractmethod
def get_attempt_step(*, workspace: str | None = None, job: str, attempt: str, name: str) -> HelixJobStepResponse: ...


@patch("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{name}/status")
@abstractmethod
def update_attempt_step_status(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    name: str,
    body: HelixJobStatusUpdateRequest,
) -> HelixJobStepResponse: ...


# ---------------------------------------------------------------------------
# Job tasks
# ---------------------------------------------------------------------------


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/tasks")
@abstractmethod
def list_job_step_tasks(*, workspace: str | None = None, job: str, name: str) -> HelixJobListTaskResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}")
@abstractmethod
def get_job_step_task(*, workspace: str | None = None, job: str, step: str, name: str) -> HelixJobTaskResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks")
@abstractmethod
def list_attempt_step_tasks(
    *, workspace: str | None = None, job: str, attempt: str, step: str
) -> HelixJobListTaskResponse: ...


@put("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks/{name}")
@abstractmethod
def update_attempt_step_task(
    *,
    workspace: str | None = None,
    job: str,
    attempt: str,
    step: str,
    name: str,
    body: HelixJobTaskUpdate,
) -> HelixJobTaskResponse: ...


@get("/apis/jobs/v2/workspaces/{workspace}/jobs/{job}/attempts/{attempt}/steps/{step}/tasks/{name}")
@abstractmethod
def get_attempt_step_task(
    *, workspace: str | None = None, job: str, attempt: str, step: str, name: str
) -> HelixJobTaskResponse: ...
