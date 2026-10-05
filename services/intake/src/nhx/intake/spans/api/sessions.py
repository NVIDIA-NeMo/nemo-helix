# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read API for ClickHouse-backed Intake session details."""

from fastapi import APIRouter, Depends, HTTPException, status
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access
from nhx.intake.spans.api.sessions_schemas import Session
from nhx.intake.spans.service import SessionNotFoundError

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Sessions"


@router.get(
    "/v2/workspaces/{workspace}/sessions/{id}",
    response_model=Session,
    response_model_exclude_none=True,
    tags=[API_TAG],
    responses={404: {"description": "Session not found"}},
)
async def get_session(workspace: str, id: str, service: SpansServiceDep) -> Session:
    try:
        session = await service.get_session(workspace=workspace, session_id=id)
    except SessionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {workspace}/{id} not found")
    return Session.from_domain(session)
