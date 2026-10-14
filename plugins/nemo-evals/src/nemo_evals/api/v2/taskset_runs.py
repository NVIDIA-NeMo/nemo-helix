# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit stored Harbor tasksets as durable runs under the agent-evaluate collection."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.exceptions import RequestValidationError
from nemo_evals.api.dependencies import get_taskset_run_service
from nemo_evals.api.service.taskset_run_service import TasksetRunResolutionError, TasksetRunService
from nemo_evals.api.taskset_run_schemas import TasksetRunInput, TasksetRunSubmitResponse
from nemo_evals.authz import scope
from nemo_evals.taskset_runs.store import RunRequestConflict
from nemo_evals.taskset_runs.submission import TimeoutAboveMaximumError
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.dependencies import get_effective_principal_id
from nemo_helix_plugin.log_utils import sanitize_for_log

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
        422: {"description": "Invalid request, excessive timeout, or missing or incompatible source."},
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

    Args:
        workspace: Workspace that owns the run.
        req: Validated stored Harbor taskset and execution inputs.
        response: HTTP response whose status distinguishes acceptance from replay.
        caller: Effective authenticated principal, supplied by the platform dependency.
        service: Caller-delegated request-scoped acceptance service.

    Returns:
        Public run status with advisory overlap warnings; 201 for acceptance or 200
        for replay. Submission creates no Jobs or member records.

    Raises:
        RequestValidationError: The execution budget exceeds the operator maximum.
        HTTPException: Source denial (403), conflicting key (409), resolution rejection
            (422), or a sanitized unexpected failure (500).
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to submit taskset run in workspace %s", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Internal server error") from None
    response.status_code = status.HTTP_201_CREATED if outcome.created else status.HTTP_200_OK
    return outcome.status
