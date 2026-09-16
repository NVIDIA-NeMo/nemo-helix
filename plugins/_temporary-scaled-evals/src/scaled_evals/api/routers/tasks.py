# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from scaled_evals.api import artifacts
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.build.task_image_identity import (
    TaskImageIdentityError,
    validate_task_image_request,
)
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import Conflict
from scaled_evals.api.schemas.common import (
    DeleteResponse,
    ListEnvelope,
    page_from_rows,
)
from scaled_evals.api.schemas.tasks import (
    SLUG_MAX_LEN,
    Task,
    TaskCreate,
    TaskCreateResponse,
    TaskDetail,
    TaskFinalizeRequest,
    TaskFinalizeResponse,
    TaskPackReconciliationItem,
    TaskPackReconciliationResponse,
    TaskRevisionCreateResponse,
    TaskUpdate,
    TaskUpload,
)
from scaled_evals.api.settings import settings
from scaled_evals.api.utils import make_id

router = APIRouter(prefix="/tasks", tags=["tasks"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]

_SLUG_NORMALIZE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class UploadedTaskPack:
    revision: int
    object_key: str
    size_bytes: int


def _slugify(name: str) -> str:
    slug = _SLUG_NORMALIZE.sub("-", name.lower()).strip("-")
    return slug[:SLUG_MAX_LEN] or "task"


def _http_error(status: int, code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": details or {}}},
    )


def _upload_for(object_key: str) -> TaskUpload:
    # Broker-upload model: hand the client the fileset coordinates it uploads the tarball to
    # directly via the Files SDK. The fileset is pre-created so the client's upload can't 404.
    target = artifacts.upload_target(object_key)
    return TaskUpload(
        workspace=target.workspace,
        fileset=target.fileset,
        path=target.path,
        remote_path=target.remote_path,
    )


def _format_bytes(size_bytes: int) -> str:
    gib = size_bytes / (1024 * 1024 * 1024)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = size_bytes / (1024 * 1024)
    return f"{mib:.2f} MiB"


def _validate_uploaded_task_pack(task_id: str, db: Db, *, expected_revision: int | None = None) -> UploadedTaskPack:
    revision = db.tasks.revision_for_finalize(task_id, expected_revision=expected_revision)
    if revision is None:
        raise _http_error(404, "not_found", "task not found")
    if revision.status != "uploading":
        raise _http_error(
            409,
            "not_finalizable",
            f"revision {revision.revision} is {revision.status}, not uploading",
        )

    try:
        size_bytes = artifacts.object_size(revision.object_key)
    except Exception as exc:  # noqa: BLE001 - any Files transport failure is a retryable 503
        raise _http_error(
            503,
            "object_store_unavailable",
            "could not verify task pack size against Files; retry finalize later",
        ) from exc

    if size_bytes is None:
        # Missing: the client has not uploaded the tarball to the fileset yet. (On Files a file
        # always reports a size, so there is no separate "unknown content-length" case.)
        raise _http_error(
            409,
            "task_pack_missing",
            "task pack upload is missing; upload the tarball to the fileset returned by "
            "POST /tasks before finalizing this revision",
            {"object_key": revision.object_key},
        )

    max_size = settings.task_pack_max_size_bytes
    if size_bytes > max_size:
        artifacts.delete_object(revision.object_key)
        raise _http_error(
            413,
            "task_pack_too_large",
            "task pack is too large; upload a smaller gzip tarball before finalizing "
            f"(limit {_format_bytes(max_size)}, uploaded {_format_bytes(size_bytes)})",
            {
                "object_key": revision.object_key,
                "limit_bytes": max_size,
                "uploaded_bytes": size_bytes,
            },
        )

    return UploadedTaskPack(
        revision=revision.revision,
        object_key=revision.object_key,
        size_bytes=size_bytes,
    )


def _raise_quota_exceeded(object_key: str, used_bytes: int, uploaded_bytes: int) -> None:
    quota = settings.task_pack_tenant_storage_quota_bytes
    artifacts.delete_object(object_key)
    raise _http_error(
        413,
        "tenant_storage_quota_exceeded",
        "task pack would exceed tenant storage quota; delete old tasks/revisions or "
        "upload a smaller pack before finalizing "
        f"(quota {_format_bytes(quota)}, current {_format_bytes(used_bytes)}, "
        f"upload {_format_bytes(uploaded_bytes)})",
        {
            "object_key": object_key,
            "quota_bytes": quota,
            "used_bytes": used_bytes,
            "uploaded_bytes": uploaded_bytes,
        },
    )


def _reconcile_task_packs(
    *,
    db: Db,
    owner_id: str,
    repair: bool,
    limit: int,
) -> TaskPackReconciliationResponse:
    revisions = db.tasks.task_pack_revisions_for_owner(owner_id=owner_id, limit=limit)
    items: list[TaskPackReconciliationItem] = []
    repaired = 0
    for revision in revisions:
        try:
            missing = not artifacts.object_exists(revision.object_key)
        except Exception as exc:  # noqa: BLE001 - any Files transport failure is a retryable 503
            raise _http_error(
                503,
                "object_store_unavailable",
                "could not verify task pack objects against Files; retry later",
            ) from exc
        item_repaired = False
        if missing and repair:
            item_repaired = db.tasks.mark_task_pack_missing(
                owner_id=owner_id,
                task_id=revision.task_id,
                revision=revision.revision,
                object_key=revision.object_key,
                previous_status=revision.status,
            )
            repaired += int(item_repaired)
        items.append(
            TaskPackReconciliationItem(
                task_id=revision.task_id,
                revision=revision.revision,
                status=revision.status,
                object_key=revision.object_key,
                missing=missing,
                repaired=item_repaired,
            )
        )
    return TaskPackReconciliationResponse(
        owner_id=owner_id,
        checked=len(revisions),
        missing=sum(1 for item in items if item.missing),
        repaired=repaired,
        items=items,
    )


def _raise_if_revision_changed(expected: UploadedTaskPack, actual_revision: int) -> None:
    if actual_revision != expected.revision:
        raise _http_error(
            409,
            "not_finalizable",
            f"latest task revision changed before finalize completed; retry finalize for revision {actual_revision}",
            {
                "expected_revision": expected.revision,
                "actual_revision": actual_revision,
            },
        )


@router.post("", status_code=201, response_model=TaskCreateResponse)
def create_task(body: TaskCreate, db: Db, current: Principal) -> TaskCreateResponse:
    """Register a task and its first revision.

    Input: `TaskCreate` body. Slug is derived from `name` if omitted.
    Output: `TaskCreateResponse` — the new row plus `revision=1`,
    `status="uploading"`, an `upload` block with the PUT URL for the
    tarball, and `links` for follow-up calls (self, finalize).

    Errors: 409 `slug_conflict` if the slug is already in use on a live
    task; 422 on schema validation (see `SLUG_PATTERN`).
    """
    task_id = make_id("task")
    slug = body.slug or _slugify(body.name)
    object_key = f"{task_id}/rev/1/tarball.tar.gz"

    try:
        row = db.tasks.create_with_initial_revision(
            task_id,
            name=body.name,
            slug=slug,
            description=body.description,
            visibility=body.visibility,
            object_key=object_key,
            owner_id=current.owner_id,
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc

    db.commit()
    return TaskCreateResponse(
        **row,
        revision=1,
        status="uploading",
        upload=_upload_for(object_key),
        links={
            "self": f"/tasks/{task_id}",
            "finalize": f"/tasks/{task_id}/finalize",
        },
    )


@router.post("/reconcile-packs", response_model=TaskPackReconciliationResponse)
def reconcile_task_packs(
    db: Db,
    current: Principal,
    repair: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=5000),
) -> TaskPackReconciliationResponse:
    """Verify this owner's runnable task-pack objects and repair missing rows.

    Missing `ready`/`building` packs are marked `failed` with stable
    `task_object_missing` build errors when `repair=true`. Revision numbers are
    preserved; the task's `current_revision` is moved only to an existing ready
    revision for the same task.
    """
    return _reconcile_task_packs(db=db, owner_id=current.owner_id, repair=repair, limit=limit)


@router.post("/{task_id}/finalize", status_code=202, response_model=TaskFinalizeResponse)
def finalize_task(
    task_id: str,
    db: Db,
    response: Response,
    _current: Principal,
    body: TaskFinalizeRequest | None = None,
) -> TaskFinalizeResponse:
    """Finalize the latest revision so it can be evaluated.

    Locks the task's latest revision, which must be in `uploading` (its
    tarball has been PUT), and flips it to `building` (stamping
    `build_started_at`). Hosted deployments queue an image-builder-service
    build/sign job from that same stored tarball. Local deployments without the
    image-builder service keep the in-cluster BuildKit fallback. On success the
    revision becomes `ready` with `image_ref`/`image_digest` and the task's
    `current_revision`; on failure `failed` with `build_error`. Poll
    `GET /tasks/{id}` for the outcome.

    Reuse image: if the request body sets `image_ref`, the build is skipped and
    the revision is pointed at that already signed image after registry identity
    verification. This is the path for externally produced images or prior
    builder-service outputs that should not be rebuilt.

    Errors: 404 if the task/revision is missing or soft-deleted;
    409 `not_finalizable` if the latest revision is not in `uploading`;
    503 `build_disabled` if no managed builder or local BuildKit backend is
    available.
    """
    prebuilt_image = body.image_ref if body else None
    if body and body.image_digest and prebuilt_image is None:
        raise _http_error(
            422,
            "invalid_request",
            "image_digest can only be supplied with image_ref when reusing a signed image",
        )
    if (
        prebuilt_image is None
        and not settings.image_builder_service_url
        and not settings.cloud_build_enabled
        and not settings.buildkit_enabled
    ):
        raise _http_error(
            503,
            "build_disabled",
            "task finalize builds are disabled because no managed builder "
            "(the image-builder service or Google Cloud Build) and no local BuildKit backend "
            "is configured; pass image_ref and image_digest to reuse an already signed image",
        )
    if prebuilt_image is None and settings.image_builder_service_url and not settings.image_builder_service_token:
        raise _http_error(
            422,
            "builder_auth_required",
            "image-builder-service finalize needs IMAGE_BUILDER_SERVICE_TOKEN configured",
        )
    if (
        prebuilt_image is None
        and not settings.image_builder_service_url
        and settings.cloud_build_enabled
    ):
        # Cloud Build reads the task pack from a GCS storageSource, but task packs now live
        # in the Files service, not GCS, so a Cloud Build finalize would build against an
        # object that was never written. Refuse it: use the image-builder service or local
        # BuildKit, or pass a prebuilt image_ref. (The Cloud Build path is slated for removal.)
        raise _http_error(
            503,
            "cloud_build_unsupported",
            "Google Cloud Build finalize is no longer supported: task packs are stored in the "
            "Files service, which Cloud Build cannot read as a build source. Configure the "
            "image-builder service or local BuildKit, or pass image_ref + image_digest to "
            "reuse an already-signed image.",
        )

    build_backend: str | None = None
    build_payload: dict[str, str] = {}
    build_credentials: dict[str, str] = {}
    if prebuilt_image is not None and settings.task_image_validation_mode == "resolve":
        assert body is not None
        try:
            normalized_ref = validate_task_image_request(prebuilt_image, body.image_digest)
        except TaskImageIdentityError as exc:
            raise _http_error(422, "invalid_task_image", str(exc)) from exc
        build_backend = "prebuilt"
        build_payload = {"image_ref": normalized_ref}
        if body.image_digest:
            build_payload["expected_digest"] = body.image_digest
    elif prebuilt_image is None and settings.image_builder_service_url:
        build_backend = "image_builder_service"
        build_payload = {"context_path": "."}
        builder_source_commit = settings.image_builder_source_commit.strip()
        if builder_source_commit:
            build_payload["builder_source_commit"] = builder_source_commit
    elif prebuilt_image is None:
        # Cloud Build is refused above (cloud_build_unsupported) since packs live in Files,
        # so the only remaining unmanaged path is local BuildKit.
        build_backend = "buildkit"

    uploaded_pack = _validate_uploaded_task_pack(task_id, db, expected_revision=(body.revision if body else None))

    if prebuilt_image is not None and settings.task_image_validation_mode == "disabled":
        assert body is not None
        assert uploaded_pack is not None
        revision = db.tasks.finalize_latest_revision_prebuilt(
            task_id,
            image_ref=prebuilt_image,
            image_digest=body.image_digest or "",
            tarball_sha256=body.tarball_sha256,
            expected_revision=uploaded_pack.revision,
            exact_revision=body.revision is not None,
            tarball_size_bytes=uploaded_pack.size_bytes,
            tenant_storage_quota_bytes=settings.task_pack_tenant_storage_quota_bytes,
        )
        if revision is None:
            raise _http_error(404, "not_found", "task not found")
        if revision.quota_exceeded:
            _raise_quota_exceeded(
                revision.object_key,
                revision.previous_storage_bytes,
                revision.uploaded_bytes,
            )
        _raise_if_revision_changed(uploaded_pack, revision.revision)
        if not revision.finalized:
            raise _http_error(
                409,
                "not_finalizable",
                f"revision {revision.revision} is {revision.status}, not uploading",
            )
        response.status_code = 200
        return TaskFinalizeResponse(id=task_id, revision=revision.revision, status="ready")

    revision = db.tasks.mark_latest_revision_building(
        task_id,
        build_backend=build_backend,
        build_payload=build_payload,
        build_credentials=build_credentials,
        tarball_sha256=(body.tarball_sha256 if body else None),
        expected_revision=uploaded_pack.revision if uploaded_pack is not None else None,
        exact_revision=bool(body and body.revision is not None),
        tarball_size_bytes=uploaded_pack.size_bytes if uploaded_pack is not None else None,
        tenant_storage_quota_bytes=(
            settings.task_pack_tenant_storage_quota_bytes if uploaded_pack is not None else None
        ),
    )
    if revision is None:
        raise _http_error(404, "not_found", "task not found")
    if uploaded_pack is not None:
        if revision.quota_exceeded:
            _raise_quota_exceeded(
                revision.object_key,
                revision.previous_storage_bytes,
                revision.uploaded_bytes,
            )
        _raise_if_revision_changed(uploaded_pack, revision.revision)
    if revision.status != "uploading":
        raise _http_error(
            409,
            "not_finalizable",
            f"revision {revision.revision} is {revision.status}, not uploading",
        )

    # The request commits only durable job parameters. Hosted builder secrets
    # come from deployment configuration, never the task finalize request.
    return TaskFinalizeResponse(id=task_id, revision=revision.revision, status="building")


@router.post(
    "/{task_id}/revisions",
    status_code=201,
    response_model=TaskRevisionCreateResponse,
)
def revise_task(task_id: str, db: Db) -> TaskRevisionCreateResponse:
    """Create a new immutable revision on an existing task.

    Allocates `revision = max(revision) + 1`, inserts a fresh
    `task_revisions` row in `uploading` status with its own tarball
    object key, and bumps `tasks.current_revision`. Mirrors the create
    flow: returns the new `revision`, an `upload` block (Files upload target) for
    the new tarball, and `links` (self, finalize).

    The build is not triggered here — the new revision sits in `uploading`
    until `POST /tasks/{id}/finalize` runs the build.

    Errors: 404 if the task does not exist or is soft-deleted.
    """
    new_revision = db.tasks.create_next_revision(task_id)
    if new_revision is None:
        raise _http_error(404, "not_found", "task not found")

    return TaskRevisionCreateResponse(
        id=task_id,
        revision=new_revision.revision,
        status="uploading",
        upload=_upload_for(new_revision.object_key),
        links={
            "self": f"/tasks/{task_id}",
            "finalize": f"/tasks/{task_id}/finalize",
        },
    )


@router.get("", response_model=ListEnvelope[Task])
def list_tasks(
    db: Db,
    current: Principal,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    order: str = Query(default="desc", pattern="^(asc|desc)$"),
    mine: bool = False,
    q: str | None = Query(default=None, min_length=1, max_length=200),
) -> ListEnvelope[Task]:
    """Paginated list of live (not soft-deleted) tasks, newest first.

    Input: optional `?limit=N` (1..100, default 20), `?cursor=`.
    Output: `ListEnvelope[Task]` with `next_cursor` when more rows exist.
    """
    rows = db.tasks.list(
        limit=limit,
        cursor=cursor,
        order=order,
        owner_id=current.owner_id if mine else None,
        q=q,
    )
    return page_from_rows(rows, limit, Task)


@router.get("/by-slug/{slug}", response_model=Task)
def get_task_by_slug(slug: str, db: Db) -> Task:
    """Resolve a task by its slug. 404 if not found or soft-deleted.

    Output: `Task`.
    """
    row = db.tasks.get_by_slug(slug)
    if row is None:
        raise _http_error(404, "not_found", "task not found")
    return Task(**row)


@router.get("/{task_id}", response_model=TaskDetail)
def get_task(task_id: str, db: Db) -> TaskDetail:
    """Fetch a task by id, including its latest revision's build status.

    404 if not found or soft-deleted.

    Output: `TaskDetail` — the task plus the latest revision's
    `revision`, `status`, `image_ref`, `image_digest`, and `build_error`
    (all null until a revision exists). This is how clients observe a
    finalize build progress `building → ready | failed`.
    """
    row = db.tasks.get_detail(task_id)
    if row is None:
        raise _http_error(404, "not_found", "task not found")
    return TaskDetail(**row)


@router.patch("/{task_id}", response_model=Task)
def patch_task(task_id: str, body: TaskUpdate, db: Db) -> Task:
    """Update mutable metadata (`name`, `slug`, `visibility`, `description`).

    Only fields present in the body are changed; revision contents (tasks,
    build artifacts) are immutable and never touched here.

    Input: `TaskUpdate` (any subset of the mutable fields).
    Output: the updated `Task`.

    Errors: 404 if not found or soft-deleted; 409 `slug_conflict` if the new
    slug collides with another live task.
    """
    try:
        row = db.tasks.update(
            task_id,
            name=body.name,
            slug=body.slug,
            description=body.description,
            visibility=body.visibility,
        )
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if row is None:
        raise _http_error(404, "not_found", "task not found")
    return Task(**row)


@router.delete("/{task_id}", status_code=200, response_model=DeleteResponse)
def delete_task(task_id: str, db: Db) -> DeleteResponse:
    """Soft-delete a task (sets `deleted_at`).

    Output: `{"id", "deleted": true}`.
    Errors: 404 if not found or already deleted; 409 if an active evaluation
    still references it.
    """
    try:
        deleted = db.tasks.soft_delete(task_id)
    except Conflict as exc:
        raise _http_error(409, exc.code, exc.message) from exc
    if not deleted:
        raise _http_error(404, "not_found", "task not found")
    return DeleteResponse(id=task_id)
