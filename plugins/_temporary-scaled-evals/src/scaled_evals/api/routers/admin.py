# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, time, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from scaled_evals.api.auth import CurrentPrincipal
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.schemas.common import (
    AdminCapacityResponse,
    AdminComputeSummaryResponse,
    AdminFailureSummaryResponse,
    AdminUsageSummaryResponse,
    AdminUserRecord,
    AdminUserSummaryResponse,
    ListEnvelope,
    page_from_rows,
)
from scaled_evals.api.schemas.evaluations import Evaluation
from scaled_evals.api.settings import settings
from scaled_evals.api.tenancy import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])
Db = Annotated[Database, Depends(get_db)]
Admin = Annotated[CurrentPrincipal, Depends(require_admin)]


@router.get("/users", response_model=ListEnvelope[AdminUserRecord])
def admin_users(
    db: Db,
    _admin: Admin,
    q: str = Query(default="", min_length=0, max_length=200),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> ListEnvelope[AdminUserRecord]:
    rows = db.users.list(q=q.strip(), limit=limit, cursor=cursor)
    return page_from_rows(rows, limit, AdminUserRecord)


@router.get("/users/{user_id}/summary", response_model=AdminUserSummaryResponse)
def admin_user_summary(user_id: str, db: Db, _admin: Admin) -> AdminUserSummaryResponse:
    summary = db.users.summary(user_id)
    return AdminUserSummaryResponse(user_id=user_id, **summary)


@router.get("/users/{user_id}/evaluations", response_model=ListEnvelope[Evaluation])
def admin_user_evaluations(
    user_id: str,
    db: Db,
    _admin: Admin,
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
        owner_id=user_id,
    )
    return page_from_rows(rows, limit, Evaluation)


@router.get("/capacity", response_model=AdminCapacityResponse)
def admin_capacity(db: Db, _admin: Admin) -> AdminCapacityResponse:
    return AdminCapacityResponse(
        **db.users.capacity(),
        cluster_limit=settings.control_plane_cluster_run_limit,
        per_user_limit=settings.control_plane_per_user_run_limit,
    )


@router.get("/usage", response_model=AdminUsageSummaryResponse)
def admin_usage(
    db: Db,
    _admin: Admin,
    limit: int = Query(default=20, ge=1, le=100),
) -> AdminUsageSummaryResponse:
    return AdminUsageSummaryResponse(**db.users.usage_by_actor(limit=limit))


_FAILURE_CATEGORY_COPY = {
    "inference_http_504": ("Inference HTTP 504", "Gateway failures returned by model inference."),
    "inference_rate_limit": ("Inference rate limit", "Provider throttling and HTTP 429s."),
    "inference_timeout": ("Inference timeout", "Model requests that exceeded provider timeouts."),
    "sandbox_startup": ("Sandbox startup", "Creation, image pull, admission, or startup failures."),
    "runtime_cleanup": ("Runtime cleanup", "Sandbox or runtime resources left after execution."),
    "evaluation_timeout": ("Evaluation timeout", "Runs that exceeded their execution deadline."),
    "trial_cancelled": ("Trial cancelled", "Trials interrupted before producing a result."),
    "object_storage": ("Object storage", "Task, result, or artifact storage failures."),
    "runtime_infrastructure": (
        "Runtime infrastructure",
        "Runner, Kubernetes, connection, or sandbox execution failures.",
    ),
    "agent_exit": ("Agent process exit", "Agent processes that terminated unsuccessfully."),
    "task_configuration": ("Task or configuration", "Invalid or unavailable task inputs."),
    "control_plane_state": (
        "Control-plane state",
        "Conflicting or already-terminal lifecycle transitions.",
    ),
    "other": ("Other", "Failures that do not yet match a known operational category."),
}


def _admin_window(
    *, days: int, from_time: datetime | None, to_time: datetime | None, noun: str
) -> tuple[datetime, datetime, int]:
    if (from_time is None) != (to_time is None):
        raise HTTPException(status_code=422, detail="from and to must be provided together")
    if from_time is not None and to_time is not None:
        if from_time.tzinfo is None or to_time.tzinfo is None:
            raise HTTPException(status_code=422, detail="from and to must include timezone offsets")
        if from_time >= to_time:
            raise HTTPException(status_code=422, detail="from must be earlier than to")
        if to_time - from_time > timedelta(days=90):
            raise HTTPException(status_code=422, detail=f"custom {noun} windows may not exceed 90 days")
        window_start = from_time.astimezone(UTC)
        window_end = to_time.astimezone(UTC)
        window_days = min(90, max(1, (window_end.date() - window_start.date()).days + 1))
        return window_start, window_end, window_days
    window_end = datetime.now(UTC)
    window_start = datetime.combine(window_end.date(), time.min, tzinfo=UTC) - timedelta(days=days - 1)
    return window_start, window_end, days


@router.get("/compute", response_model=AdminComputeSummaryResponse)
def admin_compute(
    db: Db,
    _admin: Admin,
    days: int = Query(default=7, ge=1, le=90),
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
) -> AdminComputeSummaryResponse:
    window_start, window_end, window_days = _admin_window(
        days=days, from_time=from_time, to_time=to_time, noun="compute"
    )
    return AdminComputeSummaryResponse(
        **db.users.compute_summary(
            window_days=window_days,
            window_start=window_start,
            window_end=window_end,
        )
    )


@router.get("/failures", response_model=AdminFailureSummaryResponse)
def admin_failures(
    db: Db,
    _admin: Admin,
    days: int = Query(default=7, ge=1, le=90),
    examples: int = Query(default=3, ge=1, le=10),
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
) -> AdminFailureSummaryResponse:
    window_start, window_end, window_days = _admin_window(
        days=days, from_time=from_time, to_time=to_time, noun="failure"
    )
    summary = db.users.failure_summary(
        window_days=window_days,
        window_start=window_start,
        window_end=window_end,
        examples_per_category=examples,
    )
    for category in summary["categories"]:
        label, description = _FAILURE_CATEGORY_COPY.get(category["key"], _FAILURE_CATEGORY_COPY["other"])
        category["label"] = label
        category["description"] = description
    return AdminFailureSummaryResponse(**summary)
