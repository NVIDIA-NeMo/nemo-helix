# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and delete APIs for ClickHouse-backed Intake sessions."""

from fastapi import APIRouter, Depends, HTTPException, status
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import get_auth_client
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.repository.clickhouse.executor import ClickHouseQueryError
from nhx.intake.repository.session import TraceIndexCollisionError
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access
from nhx.intake.spans.api.sessions_schemas import Session
from nhx.intake.spans.service import SessionInEvaluationError, SessionNotFoundError
from pydantic import BaseModel

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Sessions"


class SessionDeletionPermission(BaseModel):
    """Permission hint for session deletion; write scopes and protection are checked on DELETE."""

    allowed: bool


@router.get(
    "/v2/workspaces/{workspace}/sessions/{id}/deletion-permission",
    response_model=SessionDeletionPermission,
    tags=[API_TAG],
)
async def get_session_deletion_permission(
    workspace: str, id: str, auth_client: AuthClient = Depends(get_auth_client)
) -> SessionDeletionPermission:
    """Check deletion permission without requiring existing telemetry.

    This is a UI hint; DELETE still enforces permissions, write scopes, and
    evaluation protection. It remains available during partial-cleanup retries.
    """
    return SessionDeletionPermission(allowed=await auth_client.has_permissions(workspace, ["intake.sessions.delete"]))


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


@router.delete(
    "/v2/workspaces/{workspace}/sessions/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=[API_TAG],
    responses={
        403: {"description": "Insufficient permission or write scope"},
        404: {"description": "Session not found"},
        409: {"description": "Session has a protected evaluation or conflicting trace-index ownership"},
        503: {"description": "Telemetry storage unavailable; retry deletion to finish cleanup"},
    },
)
async def delete_session(workspace: str, id: str, service: SpansServiceDep) -> None:
    """Soft-delete a session's current spans and traces, preserving evaluator results and annotations.

    Evaluation-linked sessions can be deleted only after every associated evaluation
    is confirmed soft-deleted. Deletion does not block subsequent ingestion into the same session ID.
    Conflicting ownership of a root span's trace-index key prevents deletion of the entire session.
    """
    try:
        await service.delete_session(workspace=workspace, session_id=id)
    except (SessionInEvaluationError, TraceIndexCollisionError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except SessionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {workspace}/{id} not found")
    except ClickHouseQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=CLICKHOUSE_UNAVAILABLE_MESSAGE
        ) from exc
