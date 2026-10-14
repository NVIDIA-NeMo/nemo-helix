# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_scaled_evals_plugin.authz import admin_only

from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.benchmark_store import TaskRef
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import Conflict, InvalidReference
from scaled_evals.api.schemas.benchmarks import (
    SLUG_MAX_LEN,
    Benchmark,
    BenchmarkCreate,
    BenchmarkCreateResponse,
    BenchmarkDetail,
    BenchmarkQualificationUpdate,
    BenchmarkRevisionCreate,
    BenchmarkRevisionCreateResponse,
    BenchmarkTaskMember,
    BenchmarkUpdate,
    BenchmarkVariantCreate,
    BenchmarkVariantCreateResponse,
)
from scaled_evals.api.schemas.common import (
    DeleteResponse,
    ListEnvelope,
    page_from_rows,
)
from scaled_evals.api.utils import make_id

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]

_SLUG_NORMALIZE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_NORMALIZE.sub("-", name.lower()).strip("-")
    return slug[:SLUG_MAX_LEN] or "benchmark"


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _member_cursor(cursor: str | None) -> int | None:
    if cursor is None:
        return None
    try:
        value = int(cursor)
    except ValueError as exc:
        raise _http_error(422, "invalid_cursor", "invalid cursor") from exc
    if value < 0:
        raise _http_error(422, "invalid_cursor", "invalid cursor")
    return value


def _member_page(rows: list[dict], limit: int) -> ListEnvelope[BenchmarkTaskMember]:
    page_rows = rows[:limit]
    next_cursor = str(rows[limit]["position"]) if len(rows) > limit else None
    return ListEnvelope(
        data=[BenchmarkTaskMember(**row) for row in page_rows],
        next_cursor=next_cursor,
    )


def _task_refs(tasks: list) -> list[TaskRef]:
    return [TaskRef(task_id=t.task_id, task_revision=t.task_revision) for t in tasks]


@router.post("", status_code=201, response_model=BenchmarkCreateResponse)
def create_benchmark(body: BenchmarkCreate, db: Db) -> BenchmarkCreateResponse:
    """Register a benchmark and its first revision.

    A benchmark is a revisioned *collection of tasks*. Input: `BenchmarkCreate`
    with the member `tasks` (each an existing task id, optionally pinned to a
    `task_revision`; null = the task's latest at eval time). Slug is derived
    from `name` if omitted. Revision 1 is created with that member set and is
    immutable — change the set by snapshotting a new revision.

    Errors: 409 `slug_conflict` if the slug is in use on a live benchmark;
    422 if a member task / pinned task revision does not exist or is duplicated.
    """
    benchmark_id = make_id("bm")
    slug = body.slug or _slugify(body.name)
    try:
        row = db.benchmarks.create_with_initial_revision(
            benchmark_id,
            name=body.name,
            slug=slug,
            description=body.description,
            visibility=body.visibility,
            tasks=_task_refs(body.tasks),
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    except InvalidReference as exc:
        raise _http_error(422, "invalid_task_reference", exc.message) from exc

    return BenchmarkCreateResponse(
        **row,
        revision=1,
        links={
            "self": f"/benchmarks/{benchmark_id}",
            "revisions": f"/benchmarks/{benchmark_id}/revisions",
            "tasks": f"/benchmarks/{benchmark_id}/tasks",
        },
    )


@router.post(
    "/{benchmark_id}/variants",
    status_code=201,
    response_model=BenchmarkVariantCreateResponse,
)
def create_benchmark_variant(benchmark_id: str, body: BenchmarkVariantCreate, db: Db) -> BenchmarkVariantCreateResponse:
    """Derive a metadata-only variant of an existing benchmark revision.

    Copies the base member task pins unchanged and stores an allowlisted
    ``operational_policy`` (today: ``agent_timeout_floor_sec``). Does not accept
    a new task set — membership changes still go through ``/revisions``.

    Errors: 404 if the base benchmark/revision is missing; 409 ``slug_conflict``;
    422 on invalid policy or unresolvable base revision.
    """
    variant_id = make_id("bm")
    slug = body.slug or _slugify(body.name)
    try:
        row = db.benchmarks.create_variant(
            variant_id,
            base_benchmark_id=benchmark_id,
            from_revision=body.from_revision,
            name=body.name,
            slug=slug,
            description=body.description,
            visibility=body.visibility,
            operational_policy=body.operational_policy.model_dump(),
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    except InvalidReference as exc:
        raise _http_error(422, "invalid_variant", exc.message) from exc
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")

    return BenchmarkVariantCreateResponse(
        id=row["id"],
        name=row["name"],
        slug=row["slug"],
        description=row["description"],
        visibility=row["visibility"],
        qualification_status=row["qualification_status"],
        qualification_evidence=row["qualification_evidence"],
        qualified_at=row["qualified_at"],
        qualified_by=row["qualified_by"],
        current_revision=row["current_revision"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        revision=row["revision"],
        derived_from=row["derived_from"],
        operational_policy=row["operational_policy"],
        links={
            "self": f"/benchmarks/{variant_id}",
            "base": f"/benchmarks/{benchmark_id}",
            "tasks": f"/benchmarks/{variant_id}/tasks",
        },
    )


@router.post(
    "/{benchmark_id}/revisions",
    status_code=201,
    response_model=BenchmarkRevisionCreateResponse,
)
def revise_benchmark(benchmark_id: str, body: BenchmarkRevisionCreate, db: Db) -> BenchmarkRevisionCreateResponse:
    """Snapshot a new immutable revision with a new member set.

    Allocates `revision = max(revision) + 1`, records the given `tasks` as that
    revision's membership, and bumps `current_revision`. The previous revision
    is left untouched (reproducible history).

    Errors: 404 if the benchmark does not exist or is soft-deleted; 422 if a
    member task / pinned task revision is missing or duplicated.
    """
    try:
        new_revision = db.benchmarks.create_next_revision(
            benchmark_id,
            description=body.description,
            tasks=_task_refs(body.tasks),
        )
    except InvalidReference as exc:
        raise _http_error(422, "invalid_task_reference", exc.message) from exc
    if new_revision is None:
        raise _http_error(404, "not_found", "benchmark not found")

    return BenchmarkRevisionCreateResponse(
        id=benchmark_id,
        revision=new_revision,
        links={
            "self": f"/benchmarks/{benchmark_id}",
            "tasks": f"/benchmarks/{benchmark_id}/tasks",
        },
    )


@router.get("", response_model=ListEnvelope[Benchmark])
def list_benchmarks(
    db: Db,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> ListEnvelope[Benchmark]:
    """Paginated list of live (not soft-deleted) benchmarks, newest first."""
    rows = db.benchmarks.list(limit=limit, cursor=cursor, order=order, q=q)
    return page_from_rows(rows, limit, Benchmark)


@router.get("/by-slug/{slug}", response_model=Benchmark)
def get_benchmark_by_slug(slug: str, db: Db) -> Benchmark:
    """Resolve a benchmark by its slug. 404 if not found or soft-deleted."""
    row = db.benchmarks.get_by_slug(slug)
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")
    return Benchmark(**row)


@router.get("/{benchmark_id}/tasks", response_model=ListEnvelope[BenchmarkTaskMember])
def list_benchmark_tasks(
    benchmark_id: str,
    db: Db,
    revision: int | None = Query(default=None, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
) -> ListEnvelope[BenchmarkTaskMember]:
    """List the member tasks of a benchmark revision (latest if `?revision` omitted).

    Output: ordered `ListEnvelope[BenchmarkTaskMember]`. 404 if the benchmark is
    missing. An out-of-range or empty revision simply returns an empty page.
    """
    row = db.benchmarks.get(benchmark_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")
    rev = revision if revision is not None else row["current_revision"]
    if rev is None:
        return ListEnvelope(data=[], next_cursor=None)
    rows = db.benchmarks.list_tasks(
        benchmark_id,
        rev,
        limit=limit,
        cursor=_member_cursor(cursor),
    )
    return _member_page(rows, limit)


@router.get("/{benchmark_id}", response_model=BenchmarkDetail)
def get_benchmark(
    benchmark_id: str,
    db: Db,
    include_tasks: bool = Query(default=True),
) -> BenchmarkDetail:
    """Fetch a benchmark by id, optionally including latest revision member tasks.

    404 if not found or soft-deleted.
    """
    row = db.benchmarks.get_detail(benchmark_id) if include_tasks else db.benchmarks.get(benchmark_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")
    payload = dict(row)
    payload["revision"] = payload.get("revision", payload.get("current_revision"))
    payload["tasks"] = payload.get("tasks", [])
    payload.setdefault("derived_from", None)
    payload.setdefault("operational_policy", {})
    return BenchmarkDetail(**payload)


@router.patch("/{benchmark_id}", response_model=Benchmark)
def patch_benchmark(benchmark_id: str, body: BenchmarkUpdate, db: Db) -> Benchmark:
    """Update mutable metadata (`name`, `slug`, `visibility`, `description`).

    Revision membership is immutable and never touched here (snapshot a new
    revision to change it).

    Errors: 404 if not found; 409 `slug_conflict` on a colliding live slug.
    """
    try:
        row = db.benchmarks.update(
            benchmark_id,
            name=body.name,
            slug=body.slug,
            description=body.description,
            visibility=body.visibility,
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")
    return Benchmark(**row)


@router.post("/{benchmark_id}/qualification", response_model=Benchmark)
@admin_only
def set_benchmark_qualification(
    benchmark_id: str,
    body: BenchmarkQualificationUpdate,
    db: Db,
    current: Principal,
) -> Benchmark:
    row = db.benchmarks.set_qualification(
        benchmark_id,
        status=body.status,
        evidence=body.evidence,
        qualified_by=current.owner_id,
    )
    if row is None:
        raise _http_error(404, "not_found", "benchmark not found")
    return Benchmark(**row)


@router.post("/{benchmark_id}/promote", response_model=Benchmark)
@admin_only
def promote_benchmark(benchmark_id: str, db: Db, current: Principal) -> Benchmark:
    row = db.benchmarks.promote(benchmark_id, qualified_by=current.owner_id)
    if row is None:
        existing = db.benchmarks.get(benchmark_id)
        if existing is None:
            raise _http_error(404, "not_found", "benchmark not found")
        raise _http_error(
            409,
            "benchmark_not_qualified",
            "benchmark must be qualified before it can be promoted",
        )
    return Benchmark(**row)


@router.delete("/{benchmark_id}", status_code=200, response_model=DeleteResponse)
def delete_benchmark(benchmark_id: str, db: Db) -> DeleteResponse:
    """Soft-delete a benchmark (sets `deleted_at`). 404 if not found/already deleted."""
    deleted = db.benchmarks.soft_delete(benchmark_id)
    if not deleted:
        raise _http_error(404, "not_found", "benchmark not found")
    return DeleteResponse(id=benchmark_id)
