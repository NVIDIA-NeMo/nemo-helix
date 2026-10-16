# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit Harbor tasksets and read stored runs/members under agent-evaluate."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.exceptions import RequestValidationError
from nemo_evals.api.dependencies import get_taskset_run_service
from nemo_evals.api.service.taskset_run_service import TasksetRunResolutionError, TasksetRunService
from nemo_evals.api.taskset_run_schemas import (
    TasksetRunInput,
    TasksetRunStatus,
    TasksetRunSubmitResponse,
    TasksetRunTask,
)
from nemo_evals.authz import scope
from nemo_evals.taskset_runs.entities import MemberPhase
from nemo_evals.taskset_runs.store import RunRequestConflict
from nemo_evals.taskset_runs.submission import TimeoutAboveMaximumError
from nemo_evals.taskset_runs.task_jobs import TaskJobsIncompleteError
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import Page

logger = logging.getLogger(__name__)


class TasksetRunPerms(PermissionSet, namespace="evaluator.taskset-runs"):
    """Permissions for run submission and the shared run lifecycle collection."""

    CREATE = perm("Submit a taskset run")
    LIST = perm("List taskset runs")
    READ = perm("Read a taskset run")
    CANCEL = perm("Request taskset run cancellation")


router = APIRouter()


@router.post(
    "/agent-evaluate/runs",
    summary="Submit Taskset Run",
    response_model=TasksetRunSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"model": TasksetRunSubmitResponse, "description": "Replay an identical existing run."},
        201: {"description": "Accept a new run with frozen taskset and agent inputs."},
        409: {"description": "The idempotency key already belongs to a different request."},
        422: {
            "description": "Invalid request, excessive timeout, missing or incompatible source, "
            "unconfigured execution, or a task without images."
        },
        503: {"description": "The run was accepted but some task Jobs weren't created; retry the same request."},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetRunPerms.CREATE])
async def submit_taskset_run(
    workspace: str,
    req: TasksetRunInput,
    response: Response,
    caller: str = Depends(get_effective_principal_id),
    service: TasksetRunService = Depends(get_taskset_run_service),
) -> TasksetRunSubmitResponse:
    """Freeze and accept a Harbor run, returning its original status on a keyed retry.

    The caller must be able to read every source: the taskset, its member tasks, their metrics and
    metric models. Submission copies those definitions into a plan stored in this workspace, so
    anyone who can read this workspace's files or jobs can read the copy, even without access to
    the source workspaces.

    Args:
        workspace: Workspace that owns the run.
        req: Validated stored Harbor taskset and execution inputs.
        response: HTTP response whose status distinguishes acceptance from replay.
        caller: Effective authenticated principal, supplied by the platform dependency.
        service: Caller-delegated request-scoped acceptance service.

    Returns:
        Public run status with advisory overlap warnings and one notice per task Job;
        201 for acceptance or 200 for replay. Submission creates one task Job per task
        right after acceptance; a keyed replay creates any that are still missing.

    Raises:
        RequestValidationError: The execution budget exceeds the operator maximum.
        HTTPException: Source denial (403), conflicting key (409), resolution rejection
            (422), missing task Jobs after acceptance (503), or a sanitized unexpected failure (500).
    """
    try:
        outcome = await service.submit(req, workspace=workspace, caller=caller)
    except TimeoutAboveMaximumError as exc:
        raise RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", "timeout_seconds"),
                    "msg": f"Value error, {exc}",
                    "input": req.timeout_seconds,
                    "ctx": {"error": str(exc)},
                },
            ]
        ) from exc
    except RunRequestConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=f"run {exc.run_id} exists for this idempotency_key with a different request",
            headers={"x-should-retry": "false"},
        ) from exc
    except TasksetRunResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TaskJobsIncompleteError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"x-should-retry": "true"}) from exc
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to submit taskset run in workspace %s", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Internal server error") from None
    response.status_code = status.HTTP_201_CREATED if outcome.created else status.HTTP_200_OK
    return outcome.status


@router.get(
    "/agent-evaluate/runs",
    summary="List Taskset Runs",
    response_model=Page[TasksetRunStatus],
    response_model_exclude_none=True,
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetRunPerms.LIST])
async def list_taskset_runs(
    workspace: str,
    active: bool = Query(default=False, description="Include only accepted and running runs."),
    mine: bool = Query(default=False, description="Include only runs created by the effective caller."),
    page: int = Query(default=1, ge=1, description="One-based page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Maximum runs on the requested page."),
    caller: str = Depends(get_effective_principal_id),
    service: TasksetRunService = Depends(get_taskset_run_service),
) -> Page[TasksetRunStatus]:
    """Read a newest-first workspace run page with independent active/mine filters.

    Args:
        workspace: Workspace containing the records.
        active: Restrict to accepted and running phases when True.
        mine: Restrict to the effective caller's records when True.
        page: One-based page number, validated by FastAPI.
        page_size: Maximum page size, from 1 to 1000.
        caller: Effective principal supplied by the platform dependency.
        service: Shared request-scoped run service.

    Returns:
        Stored statuses and filtered pagination counts, excluding null optionals.

    Raises:
        HTTPException: Unexpected storage or projection failure, sanitized to 500.
    """
    try:
        return await service.list_runs(
            workspace,
            caller=caller,
            active=active,
            mine=mine,
            page=page,
            page_size=page_size,
        )
    except Exception:
        logger.exception("Failed to list taskset runs in workspace %s", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get(
    "/agent-evaluate/runs/{run_id}",
    summary="Get Taskset Run",
    response_model=TasksetRunStatus,
    response_model_exclude_none=True,
    responses={404: {"description": "Run not found in this workspace."}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetRunPerms.READ])
async def get_taskset_run(
    workspace: str,
    run_id: str,
    service: TasksetRunService = Depends(get_taskset_run_service),
) -> TasksetRunStatus:
    """Read one stored run without loading its plan or joining its tasks and Jobs.

    Args:
        workspace: Workspace that scopes the record lookup.
        run_id: Public run name in that workspace.
        service: Shared request-scoped run service.

    Returns:
        Current stored status, excluding null optionals and internal metadata.

    Raises:
        HTTPException: The record is absent (404), or an unexpected storage or
            projection failure occurs (sanitized 500).
    """
    try:
        result = await service.get(workspace, run_id)
    except Exception:
        logger.exception("Failed to get taskset run %s/%s", sanitize_for_log(workspace), sanitize_for_log(run_id))
        raise HTTPException(status_code=500, detail="Internal server error") from None
    if result is None:
        raise HTTPException(status_code=404, detail="Taskset run not found")
    return result


@router.get(
    "/agent-evaluate/runs/{run_id}/tasks",
    summary="List Taskset Run Tasks",
    response_model=Page[TasksetRunTask],
    response_model_exclude_none=True,
    responses={404: {"description": "Run not found in this workspace."}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetRunPerms.READ])
async def list_taskset_run_tasks(
    workspace: str,
    run_id: str,
    phase: MemberPhase | None = Query(default=None, description="Include only members in this stored phase."),
    page: int = Query(default=1, ge=1, description="One-based page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Maximum tasks on the requested page."),
    service: TasksetRunService = Depends(get_taskset_run_service),
) -> Page[TasksetRunTask]:
    """Read stored run members in numeric task-index order without inspecting Jobs.

    Args:
        workspace: Workspace containing the run and its children.
        run_id: Public run name whose persisted entity ID scopes the members.
        phase: Optional stored member phase filter.
        page: One-based page number, validated by FastAPI.
        page_size: Maximum page size, from 1 to 1000.
        service: Shared request-scoped run service.

    Returns:
        Stored task page and filtered counts; existing runs with no members return
        an empty page. Null Job/error details are omitted.

    Raises:
        HTTPException: The parent run is absent (404), or unexpected storage or
            projection failure occurs (sanitized 500).
    """
    try:
        result = await service.list_tasks(workspace, run_id, phase=phase, page=page, page_size=page_size)
    except Exception:
        logger.exception(
            "Failed to list taskset run tasks %s/%s", sanitize_for_log(workspace), sanitize_for_log(run_id)
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
    if result is None:
        raise HTTPException(status_code=404, detail="Taskset run not found")
    return result
