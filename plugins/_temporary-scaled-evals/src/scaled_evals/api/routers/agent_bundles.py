# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_scaled_evals_plugin.authz import admin_only

from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.schemas.agent_bundles import (
    AgentBundle,
    AgentBundleCreate,
    AgentBundleQualificationUpdate,
    AgentBundleVisibility,
)
from scaled_evals.api.schemas.common import DeleteResponse, ListEnvelope, page_from_rows
from scaled_evals.api.tenancy import IsAdmin
from scaled_evals.api.utils import make_id

router = APIRouter(prefix="/agent-bundles", tags=["agent-bundles"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


@router.post("", status_code=201, response_model=AgentBundle)
def create_agent_bundle(body: AgentBundleCreate, db: Db, current: Principal) -> AgentBundle:
    """Register an immutable private bundle owned by the authenticated caller."""
    try:
        row = db.agent_bundles.create(
            make_id("ab"),
            owner_id=current.owner_id,
            **body.model_dump(),
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    return AgentBundle(**row)


@router.get("", response_model=ListEnvelope[AgentBundle])
def list_agent_bundles(
    db: Db,
    current: Principal,
    admin: IsAdmin,
    mine: bool = False,
    visibility: AgentBundleVisibility | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> ListEnvelope[AgentBundle]:
    rows = db.agent_bundles.list_accessible(
        owner_id=current.owner_id,
        include_all=admin,
        mine=mine,
        visibility=visibility,
        limit=limit,
        cursor=cursor,
        order=order,
    )
    return page_from_rows(rows, limit, AgentBundle)


@router.get("/{bundle_id}", response_model=AgentBundle)
def get_agent_bundle(bundle_id: str, db: Db, current: Principal, admin: IsAdmin) -> AgentBundle:
    row = db.agent_bundles.get_accessible(
        bundle_id,
        owner_id=current.owner_id,
        include_all=admin,
    )
    if row is None:
        raise _http_error(404, "not_found", "agent bundle not found")
    return AgentBundle(**row)


@router.post("/{bundle_id}/qualification", response_model=AgentBundle)
@admin_only
def set_agent_bundle_qualification(
    bundle_id: str,
    body: AgentBundleQualificationUpdate,
    db: Db,
    current: Principal,
) -> AgentBundle:
    try:
        row = db.agent_bundles.set_qualification(
            bundle_id,
            status=body.status,
            evidence=body.evidence,
            qualified_by=current.owner_id,
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if row is None:
        raise _http_error(404, "not_found", "agent bundle not found")
    return AgentBundle(**row)


@router.post("/{bundle_id}/promote", response_model=AgentBundle)
@admin_only
def promote_agent_bundle(bundle_id: str, db: Db, current: Principal) -> AgentBundle:
    try:
        row = db.agent_bundles.promote(bundle_id, qualified_by=current.owner_id)
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if row is None:
        existing = db.agent_bundles.get_accessible(bundle_id, owner_id=current.owner_id, include_all=True)
        if existing is None:
            raise _http_error(404, "not_found", "agent bundle not found")
        raise _http_error(
            409,
            "agent_bundle_not_qualified",
            "agent bundle must be qualified before it can be promoted",
        )
    return AgentBundle(**row)


@router.delete("/{bundle_id}", response_model=DeleteResponse)
def delete_agent_bundle(bundle_id: str, db: Db, current: Principal, admin: IsAdmin) -> DeleteResponse:
    try:
        deleted = db.agent_bundles.delete(bundle_id, owner_id=current.owner_id, include_all=admin)
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if not deleted:
        raise _http_error(404, "not_found", "agent bundle not found")
    return DeleteResponse(id=bundle_id)
