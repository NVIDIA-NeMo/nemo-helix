# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, HTTPException, status
from nhx.core.jobs.api.dependencies import dep_dispatcher
from nhx.core.jobs.api.v2.jobs.schemas import PlatformJobResponse
from nhx.core.jobs.app.dispatcher import JobDispatcher

# Creating a separate router for Rerun, so it can be included in tests, but not the actual release
router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/jobs/{job}/rerun",
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job not Found"},
    },
)
async def rerun_job(
    job: str, workspace: str, dispatcher: JobDispatcher = Depends(dep_dispatcher)
) -> PlatformJobResponse:
    job_response = await dispatcher.rerun_job(job, workspace=workspace)
    if not job_response:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_response
