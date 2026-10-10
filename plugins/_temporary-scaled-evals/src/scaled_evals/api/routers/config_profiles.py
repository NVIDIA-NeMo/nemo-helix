# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.schemas.common import DeleteResponse, ListEnvelope, page_from_rows
from scaled_evals.api.schemas.config_profiles import (
    ConfigProfile,
    ConfigProfileCreate,
    ConfigProfileType,
    ConfigProfileUpdate,
)
from scaled_evals.api.tenancy import is_admin, record_principal
from scaled_evals.api.utils import make_id
from scaled_evals.intake.config import validate_intake_profile_config
from scaled_evals.models.gym_profile import validate_gym_profile_config
from scaled_evals.models.harbor_profile import validate_harbor_profile_config

router = APIRouter(prefix="/config-profiles", tags=["config-profiles"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]


def _write_owner_scope(current: CurrentPrincipal) -> str | None:
    # None disables owner filtering: local auth-disabled development and
    # configured admins keep unrestricted writes, including pre-ownership
    # rows whose owner_id is NULL.
    if current.source == "disabled" or is_admin(current):
        return None
    return current.owner_id


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _validate_profile_config(profile_type: ConfigProfileType, config: dict[str, object]) -> None:
    validators = {
        "harbor": validate_harbor_profile_config,
        "gym": validate_gym_profile_config,
        "intake": validate_intake_profile_config,
    }
    try:
        validators[profile_type](config)
    except ValueError as exc:
        raise _http_error(422, "invalid_config", str(exc)) from exc


@router.post("", status_code=201, response_model=ConfigProfile)
def create_profile(body: ConfigProfileCreate, db: Db, current: Principal) -> ConfigProfile:
    """Create a reusable config profile owned by the caller.

    Input: `ConfigProfileCreate` (`name`, `type`, `config`). Config is validated
    against the selected profile type before persistence.
    Output: the persisted `ConfigProfile`.

    Errors: 422 on schema validation (bad `name` or unknown `type`).
    """
    _validate_profile_config(body.type, body.config)
    record_principal(db, current)
    row = db.config_profiles.create(
        make_id("cfg"),
        name=body.name,
        type=body.type,
        config=body.config,
        owner_id=current.owner_id,
    )
    return ConfigProfile(**row)


@router.get("", response_model=ListEnvelope[ConfigProfile])
def list_profiles(
    db: Db,
    current: Principal,
    type: ConfigProfileType | None = None,
    mine: bool = False,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> ListEnvelope[ConfigProfile]:
    """Paginated list of live (not soft-deleted) config profiles, newest first.

    Profiles are org-readable shared resources; `?mine=true` narrows the list
    to rows the caller owns.

    Input: optional `?type=` and `?mine=` filters, `?limit=N` (1..100,
    default 20), `?cursor=`.
    Output: `ListEnvelope[ConfigProfile]` with `next_cursor` when more rows exist.
    """
    rows = db.config_profiles.list(
        type=type,
        limit=limit,
        cursor=cursor,
        order=order,
        owner_id=current.owner_id if mine else None,
        q=q,
    )
    return page_from_rows(rows, limit, ConfigProfile)


@router.get("/{profile_id}", response_model=ConfigProfile)
def get_profile(profile_id: str, db: Db) -> ConfigProfile:
    """Fetch a config profile by id. 404 if not found or soft-deleted."""
    row = db.config_profiles.get(profile_id)
    if row is None:
        raise _http_error(404, "not_found", "config profile not found")
    return ConfigProfile(**row)


@router.patch("/{profile_id}", response_model=ConfigProfile)
def patch_profile(profile_id: str, body: ConfigProfileUpdate, db: Db, current: Principal) -> ConfigProfile:
    """Update a config profile's `name` and/or `config`. Type is immutable;
    Gym config is validated before persistence.

    Writes are owner-scoped: 404 if not found, or if the caller neither owns
    the profile nor is an admin.
    """
    existing = db.config_profiles.get(profile_id)
    if existing is None:
        raise _http_error(404, "not_found", "config profile not found")
    if body.config is not None:
        _validate_profile_config(existing["type"], body.config)
    try:
        row = db.config_profiles.update(
            profile_id,
            name=body.name,
            config=body.config,
            owner_id=_write_owner_scope(current),
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if row is None:
        raise _http_error(404, "not_found", "config profile not found")
    return ConfigProfile(**row)


@router.delete("/{profile_id}", status_code=200, response_model=DeleteResponse)
def delete_profile(profile_id: str, db: Db, current: Principal) -> DeleteResponse:
    """Soft-delete a config profile (sets `deleted_at`).

    Writes are owner-scoped: 404 if not found, already deleted, or the caller
    neither owns the profile nor is an admin; 409 if an active evaluation
    still references it.
    """
    try:
        deleted = db.config_profiles.soft_delete(profile_id, owner_id=_write_owner_scope(current))
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if not deleted:
        raise _http_error(404, "not_found", "config profile not found")
    return DeleteResponse(id=profile_id)
