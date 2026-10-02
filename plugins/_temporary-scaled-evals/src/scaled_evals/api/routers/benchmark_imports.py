# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response

from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.repositories.base_repository import InvalidReference
from scaled_evals.api.repositories.benchmark_repository import TaskRef
from scaled_evals.api.routers.tasks import _upload_for, finalize_task
from scaled_evals.api.schemas.benchmark_imports import (
    BenchmarkImport,
    BenchmarkImportBenchmark,
    BenchmarkImportCreate,
    BenchmarkImportImages,
    BenchmarkImportPrepareResponse,
    BenchmarkImportPublishRequest,
    BenchmarkImportTask,
)
from scaled_evals.api.schemas.tasks import TaskFinalizeRequest
from scaled_evals.api.utils import make_id
from scaled_evals.benchmark_import import canonical_manifest_sha256

router = APIRouter(prefix="/benchmark-imports", tags=["benchmark-imports"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _ensure_upload_fileset(object_key: str) -> None:
    """Pre-create the Files fileset for a newly-allocated uploading revision.

    Benchmark-import creates task revisions in ``uploading`` status; the CLI then uploads the
    pack directly to Files. Ensure the destination fileset exists at revision-allocation time
    (mirroring the create/revise routes) so that direct upload can't 404 — the ``_detail`` read
    path deliberately does NOT create it. A Files failure maps to the retryable
    ``503 object_store_unavailable`` (same error model as task create/revise + finalize) rather
    than an unhandled 500.
    """
    from scaled_evals.api import s3

    try:
        s3.upload_target(object_key, ensure_fileset=True)
    except Exception as exc:  # noqa: BLE001 - any Files transport failure is a retryable 503
        raise _http_error(
            503,
            "object_store_unavailable",
            f"could not provision the task-pack upload target {object_key!r} in object storage; retry shortly",
        ) from exc


def _import_status(tasks: list[dict[str, Any]], benchmarks: list[dict[str, Any]]) -> str:
    statuses = {row["status"] for row in tasks}
    if "failed" in statuses:
        return "failed"
    if statuses & {"pending", "uploading"}:
        return "uploading"
    if "building" in statuses:
        return "preparing"
    if all(row.get("benchmark_revision") is not None for row in benchmarks):
        return "ready"
    return "prepared"


def _detail(db: Database, row: dict[str, Any]) -> BenchmarkImport:
    tasks = db.benchmark_imports.tasks(row["id"])
    benchmarks = db.benchmark_imports.benchmarks(row["id"])
    task_models = []
    for task in tasks:
        # Detail is a READ path: compute the upload coordinates without a live Files write.
        # The fileset is created when the task is created / revised (ensure_fileset=True there),
        # not on every detail render — so a GET never depends on Files availability.
        upload = (
            _upload_for(task["tarball_object_key"], ensure_fileset=False).model_dump()
            if task["status"] == "uploading"
            else None
        )
        task_models.append(
            BenchmarkImportTask(
                slug=task["slug"],
                task_id=task["task_id"],
                task_revision=task["task_revision"],
                pack_path=task["pack_path"],
                pack_sha256=task["pack_sha256"],
                status=task["status"],
                image_ref=task.get("image_ref") or task.get("requested_image_ref"),
                image_digest=task.get("image_digest") or task.get("requested_image_digest"),
                image_metadata=task.get("requested_image_metadata") or {},
                build_error=task.get("build_error"),
                upload=upload,
            )
        )
    benchmark_models = [
        BenchmarkImportBenchmark(
            slug=item["slug"],
            name=item["name"],
            task_slugs=list(item["task_slugs"]),
            benchmark_id=item.get("benchmark_id"),
            benchmark_revision=item.get("benchmark_revision"),
        )
        for item in benchmarks
    ]
    return BenchmarkImport(
        id=row["id"],
        manifest_sha256=row["manifest_sha256"],
        visibility=row["visibility"],
        description=row.get("description"),
        status=_import_status(tasks, benchmarks),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        tasks=task_models,
        benchmarks=benchmark_models,
    )


def _owned_import(import_id: str, db: Database, current: CurrentPrincipal) -> dict[str, Any]:
    row = db.benchmark_imports.get(import_id, owner_id=current.owner_id)
    if row is None:
        raise _http_error(404, "not_found", "benchmark import not found")
    return row


def _attach_images(
    import_id: str,
    images: dict[str, Any],
    db: Database,
) -> None:
    if not images:
        return
    tasks = {task["slug"]: task for task in db.benchmark_imports.tasks(import_id)}
    unknown = sorted(set(images) - set(tasks))
    if unknown:
        raise _http_error(422, "unknown_image_slug", f"images reference unknown tasks: {unknown}")
    for slug, image in images.items():
        metadata = image.model_dump(mode="json")
        attached = db.benchmark_imports.attach_task_image(
            import_id,
            slug,
            image_ref=image.image_ref,
            image_digest=image.image_digest,
            image_metadata=metadata,
        )
        if attached:
            continue
        task = tasks[slug]
        if (
            task.get("requested_image_ref") == image.image_ref
            and task.get("requested_image_digest") == image.image_digest
        ):
            continue
        raise _http_error(
            409,
            "image_identity_conflict",
            f"task {slug!r} is already preparing or has a different image identity",
        )


@router.post("", status_code=201, response_model=BenchmarkImport)
def create_benchmark_import(
    body: BenchmarkImportCreate, db: Db, current: Principal, response: Response
) -> BenchmarkImport:
    manifest = body.manifest.model_dump(mode="json")
    observed_sha = canonical_manifest_sha256(manifest)
    if observed_sha != body.manifest_sha256:
        raise _http_error(
            422,
            "manifest_digest_mismatch",
            f"manifest digest is {observed_sha}, not {body.manifest_sha256}",
        )
    unknown_images = sorted(set(body.images) - {task.slug for task in body.manifest.tasks})
    if unknown_images:
        raise _http_error(422, "unknown_image_slug", f"images reference unknown tasks: {unknown_images}")

    db.benchmark_imports.lock_identity(current.owner_id, body.manifest_sha256, body.visibility)
    existing_import = db.benchmark_imports.get_by_identity(
        owner_id=current.owner_id,
        manifest_sha256=body.manifest_sha256,
        visibility=body.visibility,
    )
    if existing_import is not None:
        _attach_images(existing_import["id"], body.images, db)
        response.status_code = 200
        return _detail(db, existing_import)

    import_id = make_id("bmi")
    row = db.benchmark_imports.create(
        import_id,
        owner_id=current.owner_id,
        manifest_sha256=body.manifest_sha256,
        manifest=manifest,
        visibility=body.visibility,
        description=body.description,
    )
    for position, task in enumerate(body.manifest.tasks):
        existing_task = db.tasks.get_by_slug(task.slug)
        if existing_task is not None and existing_task.get("owner_id") != current.owner_id:
            raise _http_error(
                409,
                "task_slug_owned_by_another_user",
                f"task slug {task.slug!r} is owned by another user",
            )
        if existing_task is None:
            task_id = make_id("task")
            revision = 1
            object_key = f"{task_id}/rev/1/tarball.tar.gz"
            db.tasks.create_with_initial_revision(
                task_id,
                name=task.name,
                slug=task.slug,
                description=body.description,
                visibility=body.visibility,
                object_key=object_key,
                owner_id=current.owner_id,
            )
            _ensure_upload_fileset(object_key)
        else:
            task_id = existing_task["id"]
            if body.visibility == "public" and existing_task.get("visibility") != "public":
                db.tasks.update(task_id, visibility="public")
            detail = db.tasks.get_detail(task_id)
            reusable = bool(
                detail
                and detail.get("tarball_sha256") == task.pack_sha256
                and detail.get("status") in {"building", "ready"}
            )
            if reusable:
                revision = int(detail["revision"])
            else:
                new_revision = db.tasks.create_next_revision(task_id)
                assert new_revision is not None
                revision = new_revision.revision
                _ensure_upload_fileset(new_revision.object_key)
        image = body.images.get(task.slug)
        db.benchmark_imports.add_task(
            import_id,
            position=position,
            slug=task.slug,
            task_id=task_id,
            task_revision=revision,
            pack_path=task.pack,
            pack_sha256=task.pack_sha256,
            image_ref=image.image_ref if image else None,
            image_digest=image.image_digest if image else None,
            image_metadata=image.model_dump(mode="json") if image else None,
        )
    for position, benchmark in enumerate(body.manifest.benchmarks):
        db.benchmark_imports.add_benchmark(
            import_id,
            position=position,
            slug=benchmark.slug,
            name=benchmark.name,
            task_slugs=benchmark.tasks,
        )
    return _detail(db, row)


@router.get("/{import_id}", response_model=BenchmarkImport)
def get_benchmark_import(import_id: str, db: Db, current: Principal) -> BenchmarkImport:
    return _detail(db, _owned_import(import_id, db, current))


@router.post("/{import_id}/images", response_model=BenchmarkImport)
def attach_benchmark_import_images(
    import_id: str, body: BenchmarkImportImages, db: Db, current: Principal
) -> BenchmarkImport:
    row = _owned_import(import_id, db, current)
    _attach_images(import_id, body.images, db)
    return _detail(db, row)


@router.post("/{import_id}/prepare", response_model=BenchmarkImportPrepareResponse)
def prepare_benchmark_import(import_id: str, db: Db, current: Principal) -> BenchmarkImportPrepareResponse:
    row = _owned_import(import_id, db, current)
    accepted = 0
    skipped = 0
    errors: dict[str, str] = {}
    for task in db.benchmark_imports.tasks(import_id):
        if task["status"] != "uploading":
            skipped += 1
            continue
        request = TaskFinalizeRequest(
            revision=task["task_revision"],
            image_ref=task.get("requested_image_ref"),
            image_digest=task.get("requested_image_digest"),
            tarball_sha256=task["pack_sha256"],
        )
        try:
            finalize_task(task["task_id"], db, Response(), current, request)
            accepted += 1
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            error = detail.get("error", detail)
            errors[task["slug"]] = str(error.get("message") or error)
    detail = _detail(db, row)
    return BenchmarkImportPrepareResponse(
        import_id=import_id,
        accepted=accepted,
        skipped=skipped,
        errors=errors,
        status=detail.status,
    )


@router.post("/{import_id}/retry", response_model=BenchmarkImport)
def retry_benchmark_import(import_id: str, db: Db, current: Principal) -> BenchmarkImport:
    row = _owned_import(import_id, db, current)
    for task in db.benchmark_imports.tasks(import_id):
        if task["status"] != "failed":
            continue
        revision = db.tasks.create_next_revision(task["task_id"])
        assert revision is not None
        db.benchmark_imports.set_task_revision(import_id, task["slug"], task_revision=revision.revision)
    return _detail(db, row)


@router.post("/{import_id}/publish", response_model=BenchmarkImport)
def publish_benchmark_import(
    import_id: str,
    db: Db,
    current: Principal,
    body: BenchmarkImportPublishRequest | None = None,
) -> BenchmarkImport:
    row = _owned_import(import_id, db, current)
    task_rows = db.benchmark_imports.tasks(import_id)
    if any(task["status"] != "ready" for task in task_rows):
        raise _http_error(409, "import_not_prepared", "all imported task revisions must be ready")
    tasks_by_slug = {task["slug"]: task for task in task_rows}
    if body and body.smoke_size is not None:
        existing_benchmarks = db.benchmark_imports.benchmarks(import_id)
        existing_slugs = {benchmark["slug"] for benchmark in existing_benchmarks}
        position = len(existing_benchmarks)
        base_slugs = {item["slug"] for item in row["manifest"].get("benchmarks", []) if item.get("slug")}
        for benchmark in existing_benchmarks:
            if benchmark["slug"] not in base_slugs:
                continue
            smoke_slug = f"{benchmark['slug']}-smoke-{body.smoke_size}"
            if smoke_slug in existing_slugs:
                continue
            if len(benchmark["task_slugs"]) < body.smoke_size:
                raise _http_error(
                    422,
                    "smoke_size_too_large",
                    f"benchmark {benchmark['slug']!r} has fewer than {body.smoke_size} tasks",
                )
            selected = sorted(
                benchmark["task_slugs"],
                key=lambda slug: (hashlib.sha256(slug.encode()).hexdigest(), slug),
            )[: body.smoke_size]
            db.benchmark_imports.add_benchmark(
                import_id,
                position=position,
                slug=smoke_slug,
                name=f"{benchmark['name']} ({body.smoke_size}-task smoke)",
                task_slugs=selected,
            )
            existing_slugs.add(smoke_slug)
            position += 1
    for benchmark in db.benchmark_imports.benchmarks(import_id):
        db.benchmark_imports.lock_benchmark_slug(benchmark["slug"])
        refs = [
            TaskRef(
                task_id=tasks_by_slug[slug]["task_id"],
                task_revision=tasks_by_slug[slug]["task_revision"],
            )
            for slug in benchmark["task_slugs"]
        ]
        existing = db.benchmarks.get_by_slug(benchmark["slug"])
        if (
            existing is not None
            and existing.get("owner_id") != current.owner_id
            and not (existing.get("owner_id") is None and current.source == "disabled")
        ):
            raise _http_error(
                409,
                "benchmark_slug_owned_by_another_user",
                f"benchmark slug {benchmark['slug']!r} is owned by another user",
            )
        if existing is None:
            benchmark_id = make_id("bm")
            created = db.benchmarks.create_with_initial_revision(
                benchmark_id,
                name=benchmark["name"],
                slug=benchmark["slug"],
                description=row.get("description"),
                visibility=row["visibility"],
                tasks=refs,
            )
            revision = int(created["current_revision"])
        else:
            benchmark_id = existing["id"]
            if row["visibility"] == "public" and existing.get("visibility") != "public":
                db.benchmarks.update(benchmark_id, visibility="public")
            current_revision = int(existing["current_revision"])
            members = db.benchmarks.list_tasks(benchmark_id, current_revision)
            current_refs = [(member["task_id"], member["task_revision"]) for member in members]
            expected_refs = [(ref.task_id, ref.task_revision) for ref in refs]
            if current_refs == expected_refs:
                revision = current_revision
            else:
                try:
                    new_revision = db.benchmarks.create_next_revision(
                        benchmark_id, description=row.get("description"), tasks=refs
                    )
                except InvalidReference as exc:
                    raise _http_error(422, "invalid_task_reference", exc.message) from exc
                assert new_revision is not None
                revision = new_revision
        db.benchmark_imports.set_benchmark_revision(
            import_id,
            benchmark["slug"],
            benchmark_id=benchmark_id,
            benchmark_revision=revision,
        )
    return _detail(db, row)
