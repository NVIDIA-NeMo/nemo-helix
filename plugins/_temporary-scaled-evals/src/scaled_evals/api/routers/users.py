# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.schemas.common import (
    ActivityRecord,
    ListEnvelope,
    UserSummaryResponse,
    page_from_rows,
)
from scaled_evals.api.schemas.evaluations import Evaluation
from scaled_evals.api.schemas.tasks import Task
from scaled_evals.api.schemas.users import CurrentUserResponse
from scaled_evals.api.settings import settings
from scaled_evals.api.tenancy import record_principal

router = APIRouter(prefix="/users", tags=["users"])

Principal = Annotated[CurrentPrincipal, Depends(current_principal)]
Db = Annotated[Database, Depends(get_db)]


@router.get("/me", response_model=CurrentUserResponse)
def users_me(current: Principal, db: Db) -> CurrentUserResponse:
    record_principal(db, current)
    usage = db.users.quota_usage(current.owner_id)
    return CurrentUserResponse(
        **{
            "id": current.owner_id,
            "name": current.display_name or current.username or current.owner_id,
            "email": current.email,
            "teams": [],
            "quotas": {
                "evaluations_active_max": settings.control_plane_per_user_run_limit,
                "evaluations_active": usage["evaluations_active"],
                "tasks_owned": usage["tasks_owned"],
                "sandbox_slots_max": settings.control_plane_per_user_run_limit,
                "sandbox_slots_active": usage["sandbox_slots_active"],
            },
            "principal": {
                "source": current.source,
                "owner_type": current.owner_type,
                "owner_id": current.owner_id,
                "username": current.username,
                "display_name": current.display_name,
                "groups": current.groups,
                "roles": current.roles,
            },
            "stub": current.source in {"disabled", "anonymous"},
        }
    )


@router.get("/me/summary", response_model=UserSummaryResponse)
def users_me_summary(current: Principal, db: Db) -> UserSummaryResponse:
    record_principal(db, current)
    summary = db.users.summary(current.owner_id)
    recent = db.users.recent_activity(current.owner_id, limit=10)
    return UserSummaryResponse(**summary, recent=recent)


@router.get("/me/evaluations", response_model=ListEnvelope[Evaluation])
def users_me_evaluations(
    current: Principal,
    db: Db,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> ListEnvelope[Evaluation]:
    rows = db.evaluations.list(
        limit=limit,
        cursor=cursor,
        order="desc",
        status=None,
        task_id=None,
        shared=False,
        owner_id=current.owner_id,
    )
    return page_from_rows(rows, limit, Evaluation)


@router.get("/me/tasks", response_model=ListEnvelope[Task])
def users_me_tasks(
    current: Principal,
    db: Db,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> ListEnvelope[Task]:
    rows = db.tasks.list(limit=limit, cursor=cursor, order="desc", owner_id=current.owner_id)
    return page_from_rows(rows, limit, Task)


@router.get("/me/activity", response_model=ListEnvelope[ActivityRecord])
def users_me_activity(
    current: Principal,
    db: Db,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> ListEnvelope[ActivityRecord]:
    # Activity is a bounded cross-resource feed. Cursor support will be added
    # with durable audit events; reject accidental silent misuse for now.
    if cursor is not None:
        from scaled_evals.api.schemas.common import decode_cursor

        decode_cursor(cursor)
    rows = db.users.recent_activity(current.owner_id, limit=limit)
    return ListEnvelope(data=[ActivityRecord(**row) for row in rows])
