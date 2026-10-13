# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, HTTPException, status
from nhx.common.auth import AuthClient, AuthContext, get_auth_client
from nhx.core.jobs.api.dependencies import dep_dispatcher
from nhx.core.jobs.api.v2.jobs.endpoints import job_operation_errors
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobResponse
from nhx.core.jobs.app.dispatcher import JobDispatcher

router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/jobs/{job}/rerun",
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
        status.HTTP_409_CONFLICT: {"description": "The job cannot be rerun, or its status changed during the request"},
    },
)
async def rerun_job(
    job: str,
    workspace: str,
    auth_client: AuthClient = Depends(get_auth_client),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> HelixJobResponse:
    """Rerun a failed job from its surviving storage as a new attempt."""
    with job_operation_errors("rerun", job, workspace):
        job_response = await dispatcher.rerun_job(
            job,
            workspace=workspace,
            auth_context=AuthContext.from_principal(auth_client.principal),
        )
    if not job_response:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_response
