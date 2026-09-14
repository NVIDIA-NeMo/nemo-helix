# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job API endpoints."""

import logging
import math
import re
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.log_utils import sanitize_for_log
from nmp.common.api.common import Page, PaginationData
from nmp.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nmp.common.api.utils import generate_openapi_extra_params, parse_deep_object
from nmp.common.auth import AuthClient, AuthContext, get_auth_client
from nmp.common.config import get_platform_config
from nmp.common.entities.client import EntityConflictError, EntityNotFoundError, EntityValidationError
from nmp.common.jobs.docker import validate_gpu_available_for_docker
from nmp.common.jobs.exceptions import PlatformJobCompilationError
from nmp.common.jobs.log_client import JobLogsClient, dep_job_logs_client
from nmp.common.jobs.result_manager import download_from_result_info
from nmp.common.jobs.schemas import (
    InvalidPageCursorError,
    PlatformJobListResultResponse,
    PlatformJobLogPage,
    PlatformJobResultCreateRequest,
    PlatformJobResultResponse,
    PlatformJobStatus,
    PlatformJobStatusResponse,
)
from nmp.common.observability import scoped_app_ctx
from nmp.common.service.dependencies import get_sdk_client
from nmp.core.jobs.api.dependencies import dep_dispatcher
from nmp.core.jobs.api.v2.jobs.schemas import (
    CreatePlatformJobRequest,
    PlatformJobListSortField,
    PlatformJobListTaskResponse,
    PlatformJobResponse,
    PlatformJobsListFilter,
    PlatformJobSortField,
    PlatformJobStatusDetailsUpdateRequest,
    PlatformJobStatusUpdateRequest,
    PlatformJobStepsListFilter,
    PlatformJobStepWithContext,
    PlatformJobTaskUpdate,
    job_artifact_base_path,
)
from nmp.core.jobs.app.ctx import JobContext
from nmp.core.jobs.app.dispatcher import (
    JobAlreadyExistsError,
    JobDeletionConflictError,
    JobDispatcher,
    JobOutputLocationError,
    JobSecretValidationError,
    JobStatusUpdateSkippedError,
    StateTransitionConflictError,
)
from nmp.core.jobs.app.profiles import ExecutionProfileT
from nmp.core.jobs.app.providers import CPUExecutionProvider, SubprocessExecutionProvider
from nmp.core.jobs.app.schemas import (
    PlatformJobSpec,
)
from nmp.core.jobs.config import config, profiles
from nmp.core.jobs.entities import PlatformJobStep, PlatformJobTask
from pydantic import ValidationError
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()

platform_config = get_platform_config()
_JOB_STATUS_VALUES = ", ".join(job_status.value for job_status in PlatformJobStatus)
MAX_LOG_QUERY_LINES = 10_000
DEFAULT_LOG_QUERY_LIMIT = 100


def _format_validation_location(loc: Any, *, prefix: str | None = None) -> str:
    if isinstance(loc, (list, tuple)):
        parts = [str(part) for part in loc if part not in ("body", "query")]
    elif loc:
        parts = [str(loc)]
    else:
        parts = []
    if prefix:
        parts.insert(0, prefix)
    return ".".join(parts) if parts else (prefix or "request")


def _safe_validation_message(message: object) -> str:
    if not isinstance(message, str):
        return "has an invalid value"

    normalized = message.lower()
    if "required" in normalized or "missing" in normalized:
        return "is required"
    if "integer" in normalized:
        return "must be an integer"
    if "number" in normalized or "float" in normalized:
        return "must be a number"
    if "boolean" in normalized:
        return "must be a boolean"
    if "string" in normalized:
        return "must be a string"
    if "list" in normalized or "array" in normalized:
        return "must be a list"
    if "dictionary" in normalized or "object" in normalized:
        return "must be an object"
    return "has an invalid value"


def _format_pydantic_validation_error(exc: ValidationError, *, prefix: str | None = None) -> str:
    messages: list[str] = []
    for error in exc.errors():
        location = _format_validation_location(error.get("loc"), prefix=prefix)
        message = error.get("msg") or "Invalid value."
        if location.endswith("status") or ".status." in location:
            message = f"{message} Expected one or more of: {_JOB_STATUS_VALUES}."
        messages.append(f"{location}: {message}")
    if not messages:
        return "The request contains invalid values. Update the request and try again."
    return f"The request contains invalid values. {'; '.join(messages)}."


def _format_entity_validation_error(exc: EntityValidationError, *, resource: str) -> str:
    detail = str(exc.args[0]) if exc.args else ""
    field_match = re.search(r"field ['\"](?P<field>[A-Za-z_][A-Za-z0-9_.\[\]-]*)['\"](?P<message>.*)", detail)
    if field_match:
        field = field_match.group("field")
        message = _safe_validation_message(field_match.group("message"))
        return f"Invalid {resource} request. Field '{field}' {message}. Update the request and try again."

    return (
        f"Invalid {resource} request. Check that the name, project, platform_spec, and referenced resources are "
        "valid for this workspace, then try again."
    )


def _format_job_compilation_error(exc: PlatformJobCompilationError) -> str:
    detail = str(exc).lower()
    if "gpu" in detail and "docker" in detail:
        return (
            "Invalid job specification: this job requires GPU resources, but the platform is running on Docker with "
            "no GPUs configured. Configure GPUs in the platform config or choose a non-GPU execution profile."
        )
    return (
        "Invalid job specification. Check platform_spec.steps, executor profiles, and resource requirements, "
        "then try again."
    )


def _format_create_job_conflict(exc: ValueError, *, job_name: str | None, workspace: str) -> str:
    if isinstance(exc, JobAlreadyExistsError) and job_name:
        return (
            f"Job '{job_name}' already exists in workspace '{workspace}'. Choose a different job name or delete the "
            "existing job before creating a new one."
        )
    if isinstance(exc, JobSecretValidationError):
        return (
            "Unable to create job because one or more referenced secrets were not found or are not accessible. "
            "Verify each platform_spec secret reference uses the expected workspace/name and that you have access."
        )
    return (
        "Unable to create job because the request conflicts with existing platform state. Review the job name and "
        "referenced resources, then try again."
    )


def get_platform_jobs_steps_list_filter(request: Request) -> PlatformJobStepsListFilter:
    """Extract PlatformJobStepsListFilter from request query parameters."""
    try:
        filters = parse_deep_object(name="filter", params=request.query_params) or {}
        return PlatformJobStepsListFilter(**filters)
    except ValidationError as exc:
        logger.info("Invalid job steps filter parameters", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_format_pydantic_validation_error(exc, prefix="filter"),
        ) from exc


def validate_job_spec(
    job_spec: PlatformJobSpec,
    execution_profiles: list[ExecutionProfileT],
) -> None:
    """Validate that the job spec is compatible with the execution profile."""
    for step in job_spec.steps:
        profile_name = step.executor.profile
        execution_profile = next(
            (ep for ep in execution_profiles if ep.provider == step.executor.provider and ep.profile == profile_name),
            None,
        )
        if not execution_profile:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"The execution profile '{step.executor.provider}/{profile_name}' specified in step '{step.name}' does not exist.",
            )

        if step.requires_persistent_storage and not execution_profile.supports_persistent_storage:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"The selected execution profile '{execution_profile}' does not support persistent storage required by step '{step.name}'.",
            )
    try:
        validate_gpu_available_for_docker(job_spec.model_dump())
    except PlatformJobCompilationError as exc:
        logger.info("Invalid platform job specification", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_format_job_compilation_error(exc),
        ) from exc


def translate_cpu_container_steps_to_subprocess(
    job_spec: PlatformJobSpec,
    subprocess_profiles: set[str],
) -> PlatformJobSpec:
    """Translate CPU container steps when explicitly configured for subprocess compatibility."""
    if not subprocess_profiles:
        return job_spec

    translated_spec = job_spec.model_copy(deep=True)
    for step in translated_spec.steps:
        executor = step.executor
        if not isinstance(executor, CPUExecutionProvider) or executor.profile not in subprocess_profiles:
            continue
        command = [*executor.container.entrypoint, *executor.container.command]
        if not command:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Subprocess execution for step '{step.name}' requires container.entrypoint and/or container.command.",
            )
        step.executor = SubprocessExecutionProvider(provider="subprocess", profile=executor.profile, command=command)
    return translated_spec


def configured_subprocess_translation_profiles() -> set[str]:
    """Return explicitly configured subprocess profiles that should accept CPU container jobs."""
    return {profile.profile for profile in config.executors if profile.provider == "subprocess"}


# Execution Profiles Endpoint
@router.get("/v2/execution-profiles")
async def get_execution_profiles() -> list[ExecutionProfileT]:
    """Get all currently configured execution profiles.

    Returns the capability-filtered merge from jobs config. In local standalone
    the controller may prune the shared list further after registry boot; in
    split topologies the API advertises its own merge result (not controller
    process memory).
    """
    return profiles


@router.post(
    "/v2/workspaces/{workspace}/jobs",
    status_code=status.HTTP_201_CREATED,
)
async def create_job(
    workspace: str,
    request: CreatePlatformJobRequest,
    auth_client: AuthClient = Depends(get_auth_client),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> PlatformJobResponse:
    """Create a new platform job."""
    platform_spec = translate_cpu_container_steps_to_subprocess(
        request.platform_spec, configured_subprocess_translation_profiles()
    )
    request = request.model_copy(update={"platform_spec": platform_spec})
    validate_job_spec(request.platform_spec, profiles)

    try:
        return await dispatcher.create_job(
            request,
            workspace,
            auth_context=AuthContext.from_principal(auth_client.principal),
            sdk=sdk,
        )
    except JobOutputLocationError as exc:
        logger.info("Invalid output_location for workspace '%s'", sanitize_for_log(workspace), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.info(
            "Failed to create job '%s' in workspace '%s'",
            sanitize_for_log(request.name),
            sanitize_for_log(workspace),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_format_create_job_conflict(exc, job_name=request.name, workspace=workspace),
        ) from exc
    except EntityValidationError as exc:
        logger.info("Invalid job entity for workspace '%s'", sanitize_for_log(workspace), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_format_entity_validation_error(exc, resource="job"),
        ) from exc


@router.get(
    "/v2/workspaces/{workspace}/jobs",
    response_model=Page[PlatformJobResponse],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=PlatformJobsListFilter,
        filter_description="Filter jobs by workspace, project, name, status, source, created_at, and updated_at.",
    ),
    status_code=status.HTTP_200_OK,
)
async def list_jobs(
    workspace: str,
    page: int = Query(default=1, description="Page number.", gt=0),
    page_size: int = Query(default=10, description="Page size.", gt=0),
    sort: PlatformJobListSortField = Query(
        default=PlatformJobListSortField.CREATED_AT_DESC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(PlatformJobsListFilter)),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> Page[PlatformJobResponse]:
    """List platform jobs with filtering and pagination."""

    offset = (page - 1) * page_size
    jobs, total_count = await dispatcher.list_jobs(parsed, workspace, limit=page_size, offset=offset, sort=sort)

    return Page(
        data=jobs,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            current_page_size=len(jobs),
            total_pages=int(math.ceil(total_count / page_size)) if total_count > 0 else 1,
            total_results=total_count,
        ),
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/jobs/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def get_job(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResponse:
    """Get a platform job by name."""
    with scoped_app_ctx(JobContext(id=name)):
        job = await dispatcher.get_job(name, workspace)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )
        return job


@router.post(
    "/v2/workspaces/{workspace}/jobs/{name}/cancel",
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def cancel_job(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResponse:
    """Cancel a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        try:
            job = await dispatcher.cancel_job(name, workspace)
        except StateTransitionConflictError as exc:
            logger.info(
                "Cannot cancel job '%s' in workspace '%s'",
                sanitize_for_log(name),
                sanitize_for_log(workspace),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(f"Cannot cancel job '{name}' from its current state. Refresh the job status and try again."),
            ) from exc
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )
        return job


@router.post(
    "/v2/workspaces/{workspace}/jobs/{name}/pause",
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def pause_job(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResponse:
    """Pause a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        job = await dispatcher.pause_job(name, workspace)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )
        return job


@router.post(
    "/v2/workspaces/{workspace}/jobs/{name}/resume",
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def resume_job(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResponse:
    """Resume a paused platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        job = await dispatcher.resume_job(name, workspace)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )
        return job


@router.delete(
    "/v2/workspaces/{workspace}/jobs/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
        status.HTTP_409_CONFLICT: {
            "description": "Job, attempt, or step is not terminal; paused or cancelling jobs must reach a terminal state before deletion."
        },
    },
)
async def delete_job(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> None:
    """Delete a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        try:
            deleted = await dispatcher.delete_job(name, workspace)
        except JobDeletionConflictError as exc:
            logger.info(
                "Cannot delete job '%s' in workspace '%s'",
                sanitize_for_log(name),
                sanitize_for_log(workspace),
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )


@router.get(
    "/v2/workspaces/{workspace}/jobs/{name}/status",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def get_job_status(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobStatusResponse:
    """Get the status of a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        job_status = await dispatcher.get_job_status(name, workspace)
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )
        return job_status


@router.patch(
    "/v2/workspaces/{workspace}/jobs/{name}/status-details",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def update_job_status_details(
    name: str,
    request: PlatformJobStatusDetailsUpdateRequest,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> None:
    """Update the status details of a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        result = await dispatcher.update_job_status_details(name, workspace, request)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )


@router.get(
    "/v2/workspaces/{workspace}/jobs/{name}/logs",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def page_job_logs(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
    logs_client: JobLogsClient = Depends(dep_job_logs_client),
    limit: int | None = Query(
        default=None,
        description="Maximum number of logs to return",
        gt=0,
        le=MAX_LOG_QUERY_LINES,
        json_schema_extra={"x-schema-default": DEFAULT_LOG_QUERY_LIMIT},
    ),
    page_cursor: str | None = Query(default=None, description="Page cursor"),
    tail: int | None = Query(
        default=None, description="Number of newest log lines to return", gt=0, le=MAX_LOG_QUERY_LINES
    ),
    attempt_id: Optional[int] = Query(default=None, description="Filter logs by job attempt ID"),
    step_id: Optional[str] = Query(default=None, description="Filter logs by step name"),
    task_id: Optional[str] = Query(default=None, description="Filter logs by task ID"),
) -> PlatformJobLogPage:
    """Get paginated logs for a platform job."""
    with scoped_app_ctx(JobContext(id=name)):
        if tail is not None and limit is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tail cannot be combined with limit; tail controls the returned log window size.",
            )
        if tail is not None and page_cursor is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tail cannot be combined with page_cursor; pass the returned prev_page as page_cursor without tail.",
            )

        job = await dispatcher.get_job(name, workspace)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )

        try:
            filters: dict[str, str] = {
                "job": name,
                "job_attempt": str(attempt_id if attempt_id is not None else job.attempt_id),
            }
            if step_id:
                filters["job_step"] = step_id
            if task_id:
                filters["job_task"] = task_id
            return await logs_client.query_logs(
                job.fileset,
                workspace=workspace,
                filters=filters,
                page_size=limit or DEFAULT_LOG_QUERY_LIMIT,
                page_cursor=page_cursor,
                tail=tail,
                artifact_base_path=job_artifact_base_path(name, job.output_location),
            )
        except InvalidPageCursorError as e:
            logger.error(f"Invalid page cursor: {str(e)}")
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error when querying logs: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to query job logs")


# Job Results Endpoints
@router.post(
    "/v2/workspaces/{workspace}/jobs/{job}/results/{name}",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def create_job_result(
    job: str,
    name: str,
    workspace: str,
    request: PlatformJobResultCreateRequest,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResultResponse:
    """Create a new result for a job."""
    with scoped_app_ctx(JobContext(id=job, result_name=name)):
        job_entity = await dispatcher.get_job(job, workspace)
        if not job_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job}' not found in workspace '{workspace}'.",
            )
        try:
            result = await dispatcher.create_result(
                job_id=job_entity.id,
                job_name=job,
                result_name=name,
                workspace=workspace,
                artifact_url=request.artifact_url,
                artifact_storage_type=request.artifact_storage_type,
            )
        except EntityNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job}' not found in workspace '{workspace}'.",
            ) from exc
        return result.to_response()


@router.get(
    "/v2/workspaces/{workspace}/jobs/{name}/results",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def list_job_results(
    workspace: str,
    name: str,
    sort: PlatformJobSortField = Query(
        default=PlatformJobSortField.CREATED_AT_DESC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobListResultResponse:
    """List results for a job."""
    with scoped_app_ctx(JobContext(id=name)):
        job_entity = await dispatcher.get_job(name, workspace)
        if not job_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{name}' not found in workspace '{workspace}'.",
            )

        results, _ = await dispatcher.list_results(job_id=job_entity.id, workspace=workspace, sort=sort)
        return PlatformJobListResultResponse(data=[r.to_response() for r in results])


@router.get(
    "/v2/workspaces/{workspace}/jobs/{job}/results/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
)
async def get_job_result(
    job: str,
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobResultResponse:
    """Get a specific job result."""
    with scoped_app_ctx(JobContext(id=job, result_name=name)):
        result = await dispatcher.get_result(job, name, workspace)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Result '{name}' for job '{job}' not found in workspace '{workspace}'.",
            )
        return result.to_response()


@router.get(
    "/v2/workspaces/{workspace}/jobs/{job}/results/{name}/download",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Successful Response",
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
    response_class=FileResponse,
)
async def download_job_result(
    job: str,
    name: str,
    workspace: str,
    background_tasks: BackgroundTasks,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> FileResponse:
    """Download a job result file."""
    with scoped_app_ctx(JobContext(id=job, result_name=name)):
        result = await dispatcher.get_result(job, name, workspace)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Result '{name}' for job '{job}' not found in workspace '{workspace}'.",
            )

        filename, tmp_dir_path = await download_from_result_info(
            result_name=name,
            job_name=job,
            workspace=workspace,
            artifact_url=result.artifact_url,
            sdk=sdk,
        )
        background_tasks.add_task(lambda: tmp_dir_path.cleanup_tmp_dir())
        return FileResponse(path=tmp_dir_path.path, filename=filename, background=background_tasks)


# Job Steps Endpoints
@router.get(
    "/v2/workspaces/{workspace}/jobs/{name}/steps",
    response_model=Page[PlatformJobStepWithContext],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=PlatformJobStepsListFilter,
        filter_description="Filter steps by job, status, and source.",
    ),
)
async def list_steps(
    name: str,
    workspace: str,
    page: int = Query(default=1, description="Page number.", gt=0),
    page_size: int = Query(default=25, description="Page size.", gt=0),
    sort: PlatformJobSortField = Query(
        default=PlatformJobSortField.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    filter: PlatformJobStepsListFilter = Depends(get_platform_jobs_steps_list_filter),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> Page[PlatformJobStepWithContext]:
    """List job steps with pagination and filtering."""
    with scoped_app_ctx(JobContext(id=name)):
        if name != "-":
            filter.job = name

        offset = (page - 1) * page_size
        steps, total_count = await dispatcher.list_steps(
            filter=filter, sort=sort, limit=page_size, offset=offset, workspace=workspace
        )

        dumped = filter.model_dump(mode="json", exclude_none=True)
        return Page(
            data=steps,
            pagination=PaginationData(
                page=page,
                page_size=page_size,
                current_page_size=len(steps),
                total_pages=int(math.ceil(total_count / page_size)) if total_count > 0 else 1,
                total_results=total_count,
            ),
            sort=sort,
            filter=dumped or None,
        )


@router.get(
    "/v2/workspaces/{workspace}/jobs/{job}/steps/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
)
async def get_job_step(
    job: str,
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobStep:
    """Get a specific job step."""
    with scoped_app_ctx(JobContext(id=job, step_name=name)):
        step = await dispatcher.get_current_job_step_by_name(job, name, workspace)
        if not step:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{name}' for job '{job}' not found in workspace '{workspace}'.",
            )
        return step


@router.patch(
    "/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/status",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
        status.HTTP_409_CONFLICT: {"description": "Conflict"},
    },
)
async def update_job_step_status(
    job: str,
    name: str,
    request: PlatformJobStatusUpdateRequest,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobStep:
    """Update a job step status."""
    with scoped_app_ctx(JobContext(id=job, step_name=name)):
        step_entity = await dispatcher.get_current_job_step_by_name(job, name, workspace)
        if not step_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{name}' for job '{job}' not found in workspace '{workspace}'.",
            )

        try:
            step_entity, _ = await dispatcher.update_job_status_from_step(
                step_entity,
                request.status,
                status_details=request.status_details,
                error_details=request.error_details,
            )
        except StateTransitionConflictError as exc:
            logger.info(
                "Cannot update job step '%s' for job '%s' to status '%s'",
                sanitize_for_log(name),
                sanitize_for_log(job),
                sanitize_for_log(request.status),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot update job step '{name}' to status '{request.status}' from its current state. "
                    "Refresh the step status and try again."
                ),
            ) from exc
        except EntityConflictError as exc:
            logger.info(
                "Conflict updating job step '%s' for job '%s'",
                sanitize_for_log(name),
                sanitize_for_log(job),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Conflict updating job step: it was modified by another request. Refresh the step and retry.",
            ) from exc
        except JobStatusUpdateSkippedError:
            logger.info(
                "Skipping job step status update because the job was deleted",
                extra={
                    "job": sanitize_for_log(job),
                    "step": sanitize_for_log(name),
                    "workspace": sanitize_for_log(workspace),
                },
            )

        return step_entity


# Job Tasks Endpoints
@router.get(
    "/v2/workspaces/{workspace}/jobs/{job}/steps/{name}/tasks",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
)
async def list_job_step_tasks(
    job: str,
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobListTaskResponse:
    """List tasks for a job step."""
    with scoped_app_ctx(JobContext(id=job, step_name=name)):
        job_entity = await dispatcher.get_job(job, workspace)
        if not job_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job}' not found in workspace '{workspace}'.",
            )

        step_entity = await dispatcher.get_current_job_step_by_name(job, name, workspace)
        if not step_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{name}' for job '{job}' not found in workspace '{workspace}'.",
            )

        return PlatformJobListTaskResponse(data=await dispatcher.list_tasks(step_entity.id, workspace=workspace))


@router.put(
    "/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
)
async def update_job_step_task(
    job: str,
    step: str,
    name: str,
    update: PlatformJobTaskUpdate,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobTask:
    """Update a job step task."""
    with scoped_app_ctx(JobContext(id=job, step_name=step)):
        step_entity = await dispatcher.get_current_job_step_by_name(job, step, workspace)
        if not step_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{step}' for job '{job}' not found in workspace '{workspace}'.",
            )

        try:
            return await dispatcher.create_or_update_task(
                job,
                name,
                workspace,
                update,
                step_entity,
            )
        except EntityNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{step}' for job '{job}' not found in workspace '{workspace}'.",
            ) from exc


@router.get(
    "/v2/workspaces/{workspace}/jobs/{job}/steps/{step}/tasks/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Not Found"},
    },
)
async def get_job_step_task(
    job: str,
    step: str,
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> PlatformJobTask:
    """Get a specific job step task."""
    with scoped_app_ctx(JobContext(id=job, step_name=step, task_id=name)):
        step_entity = await dispatcher.get_current_job_step_by_name(job, step, workspace)
        if not step_entity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Step '{step}' for job '{job}' not found in workspace '{workspace}'.",
            )

        task = await dispatcher.get_task(step_entity.id, name, workspace)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(f"Task '{name}' for step '{step}' of job '{job}' not found in workspace '{workspace}'."),
            )
        return task
