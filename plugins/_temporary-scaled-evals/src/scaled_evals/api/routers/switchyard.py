# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from scaled_evals.api import artifacts
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.build import cloud_build
from scaled_evals.api.build.buildkit import BuildError
from scaled_evals.api.build.image_builder_service import (
    UploadedArchiveMetadata,
    inspect_uploaded_archive_file,
    resolve_uploaded_archive_file_details,
)
from scaled_evals.api.build.task_image_identity import (
    TaskImageIdentityError,
    normalize_digest,
    parse_task_image_ref,
    require_allowed_task_image,
    resolve_task_image,
)
from scaled_evals.api.db import Database, get_db
from scaled_evals.api.routers.config_profiles import _write_owner_scope
from scaled_evals.api.settings import settings
from scaled_evals.api.tenancy import record_principal
from scaled_evals.api.utils import make_id

router = APIRouter(prefix="/switchyard", tags=["switchyard"])

Db = Annotated[Database, Depends(get_db)]
Principal = Annotated[CurrentPrincipal, Depends(current_principal)]

_DEFAULT_SOURCE_PROJECT = "NVIDIA-NeMo/Switchyard"
_DEFAULT_CONTEXT_PATH = "."
_DEFAULT_DOCKERFILE_PATH = "Dockerfile"
_CONTEXT_HASH_RE = re.compile(r"[0-9a-fA-F]{12,64}")
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
_GITHUB_PROJECT_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class SwitchyardPublishRequest(BaseModel):
    source_project: str = Field(default=_DEFAULT_SOURCE_PROJECT, min_length=1)
    source_ref: str = Field(min_length=40, max_length=40)
    context_path: str = Field(default=_DEFAULT_CONTEXT_PATH, min_length=1)
    dockerfile_path: str | None = Field(default=None, min_length=1)
    context_hash: str | None = Field(
        default=None,
        description="Optional build-context hash when the builder requires caller verification.",
    )
    profile_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _COMMIT_RE.fullmatch(normalized):
            raise ValueError("source_ref must be a full 40-character commit SHA")
        return normalized

    @field_validator("source_project")
    @classmethod
    def validate_source_project(cls, value: str) -> str:
        normalized = value.strip().strip("/")
        if not _GITHUB_PROJECT_RE.fullmatch(normalized):
            raise ValueError("source_project must be a GitHub owner/repo path")
        return normalized

    @field_validator("context_path")
    @classmethod
    def validate_context_path(cls, value: str) -> str:
        normalized = value.strip() or "."
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("context_path must stay within the uploaded archive")
        return path.as_posix()

    @field_validator("dockerfile_path")
    @classmethod
    def validate_dockerfile_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts:
            raise ValueError("dockerfile_path must stay within the uploaded archive")
        return path.as_posix()

    @field_validator("context_hash")
    @classmethod
    def validate_context_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not _CONTEXT_HASH_RE.fullmatch(normalized):
            raise ValueError("context_hash must be 12-64 hex characters")
        return normalized


class SwitchyardPublishResponse(BaseModel):
    profile_id: str
    profile_name: str
    reused_profile: bool
    source_project: str
    source_ref: str
    context_path: str
    dockerfile_path: str
    dockerfile_sha256: str | None = None
    context_hash: str | None = None
    context_archive_sha256: str | None = None
    image_ref: str
    image_digest: str
    config: dict[str, object]


class SwitchyardPublishJobResponse(BaseModel):
    build_id: str
    status: Literal["queued", "building", "succeeded", "failed"]
    result: SwitchyardPublishResponse | None = None
    build_error: str | None = None


SwitchyardPublishResult = SwitchyardPublishResponse | SwitchyardPublishJobResponse

_CLOUD_BUILD_TERMINAL = frozenset({"SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED"})
_SWITCHYARD_SUBSTITUTION_PREFIX = "_SCALED_EVALS_SWITCHYARD_"


def _http_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"error": {"code": code, "message": message, "details": {}}},
    )


def _profile_name(source_project: str, source_ref: str, requested: str | None) -> str:
    if requested:
        return requested.strip()
    return f"Switchyard {source_project}@{source_ref[:12]}"


def _publish_config(
    *,
    image_ref: str,
    image_digest: str,
    source_project: str,
    source_ref: str,
    context_path: str,
    dockerfile_path: str,
    dockerfile_sha256: str,
    context_hash: str,
    context_archive_sha256: str,
    builder_source_commit: str | None,
    builder: str,
) -> dict[str, object]:
    config: dict[str, object] = {
        "mode": "managed",
        "image": image_ref,
        "image_digest": image_digest,
        "source_project": source_project,
        "source_ref": source_ref,
        "source_commit": source_ref,
        "context_path": context_path,
        "dockerfile_path": dockerfile_path,
        "dockerfile_sha256": dockerfile_sha256,
        "context_hash": context_hash,
        "context_archive_sha256": context_archive_sha256,
        "builder": builder,
    }
    if builder_source_commit:
        config["builder_source_commit"] = builder_source_commit
    return config


def _find_existing_profile(
    db: Database,
    *,
    source_project: str,
    source_ref: str,
    context_path: str,
    context_hash: str,
    builder_source_commit: str | None,
    owner_id: str | None,
) -> dict | None:
    return db.config_profiles.find_switchyard_publish(
        source_project=source_project,
        source_ref=source_ref,
        context_path=context_path,
        context_hash=context_hash,
        builder_source_commit=builder_source_commit,
        owner_id=owner_id,
    )


def _resolved_builder_source_commit() -> str | None:
    resolved = settings.image_builder_source_commit.strip().lower()
    return resolved or None


def _dockerfile_path_for_context(context_path: str, dockerfile_path: str | None) -> str:
    if dockerfile_path:
        return dockerfile_path
    return _DEFAULT_DOCKERFILE_PATH if context_path == "." else f"{context_path}/Dockerfile"


def _approved_switchyard_image_repository() -> str:
    configured = (settings.switchyard_image or "").strip()
    if not configured:
        # No default repository: publishing to a registry the operator did not
        # name is worse than refusing to publish.
        raise BuildError("Switchyard publish requires SWITCHYARD_IMAGE to be configured")
    try:
        return parse_task_image_ref(configured).repository_ref
    except TaskImageIdentityError as exc:
        raise BuildError(f"configured SWITCHYARD_IMAGE is invalid: {exc}") from exc


def _switchyard_cloud_build_image_ref(source_ref: str, context_hash: str) -> str:
    """Derive the GAR tag-form ref for a Cloud Build Switchyard image.

    Uses a dedicated ``scaled-evals-switchyard`` image name alongside the
    configured task-image registry (a new image name in the existing GAR
    repository, so no new registry needs provisioning), tagged by source commit
    and verified build-context hash. Including both identities avoids immutable
    tag collisions when one commit exposes multiple supported build contexts.
    """
    registry = (settings.image_registry or "").strip().rstrip("/")
    if not registry:
        raise BuildError("Cloud Build Switchyard publish requires IMAGE_REGISTRY")
    base = registry.rsplit("/", 1)[0] if "/" in registry else registry
    return f"{base}/scaled-evals-switchyard:git-{source_ref[:12]}-ctx-{context_hash[:12]}"


def _submit_cloud_build_switchyard(
    archive_path: Path,
    body: SwitchyardPublishRequest,
    metadata: UploadedArchiveMetadata,
    *,
    owner_id: str,
) -> dict[str, object]:
    """Submit the Switchyard image to Cloud Build without waiting for completion."""

    object_key = f"switchyard-contexts/{body.source_ref}/{metadata.context_hash}.tar.gz"
    artifacts.upload_context_archive(archive_path, object_key)
    target_ref = _switchyard_cloud_build_image_ref(body.source_ref, metadata.context_hash)
    substitutions = {
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}PURPOSE": "publish",
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}OWNER_ID": owner_id,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}SOURCE_PROJECT": body.source_project,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}SOURCE_REF": body.source_ref,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}CONTEXT_PATH": body.context_path,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}DOCKERFILE_PATH": metadata.dockerfile_path,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}DOCKERFILE_SHA256": metadata.dockerfile_sha256,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}CONTEXT_HASH": metadata.context_hash,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}ARCHIVE_SHA256": metadata.context_archive_sha256,
        f"{_SWITCHYARD_SUBSTITUTION_PREFIX}PROFILE_NAME": _profile_name(
            body.source_project, body.source_ref, body.profile_name
        ),
    }
    if builder_commit := _resolved_builder_source_commit():
        substitutions[f"{_SWITCHYARD_SUBSTITUTION_PREFIX}BUILDER_SOURCE_COMMIT"] = builder_commit
    build = cloud_build.submit_image_build_from_gcs(
        object_key,
        target_ref,
        substitutions=substitutions,
    )
    build_id = str(build.get("id") or "").strip()
    if not build_id:
        raise BuildError(f"Cloud Build response did not include build id: {build}")
    return build


def _normalize_cloud_build_switchyard_identity(image_ref: str, image_digest: str) -> tuple[str, str]:
    """Validate a GAR-built Switchyard image against the task-image allowlist.

    Cloud Build images are not cosign-signed into the signature-approved
    repository, so they are checked against the approved task-image registries
    instead of the hosted repository, keeping the tag-form ref and digest ref.
    """
    try:
        ref = parse_task_image_ref(image_ref)
        if ref.digest is not None:
            raise TaskImageIdentityError("cloud build Switchyard image_ref must be tag-form")
        require_allowed_task_image(ref)
        digest_value = image_digest.strip().lower()
        if "@" in digest_value:
            digest_ref = parse_task_image_ref(digest_value)
            if digest_ref.digest is None or digest_ref.repository_ref != ref.repository_ref:
                raise TaskImageIdentityError("cloud build Switchyard image_digest repository does not match image_ref")
            digest = digest_ref.digest
        else:
            digest = normalize_digest(digest_value)
    except TaskImageIdentityError as exc:
        raise BuildError(f"invalid Switchyard image identity: {exc}") from exc
    return ref.normalized_ref, ref.digest_ref(digest)


def _normalize_switchyard_image_identity(image_ref: str, image_digest: str) -> tuple[str, str]:
    try:
        ref = parse_task_image_ref(image_ref)
        if ref.digest is not None:
            raise TaskImageIdentityError(
                "Switchyard publish requires the signed tag-form image_ref admitted by the cluster"
            )
        expected_repository = _approved_switchyard_image_repository()
        if ref.repository_ref != expected_repository:
            raise TaskImageIdentityError(
                "Switchyard publish returned image_ref outside the approved repository "
                f"{expected_repository!r}: {ref.repository_ref!r}"
            )
        digest_value = image_digest.strip().lower()
        if "@" in digest_value:
            digest_ref = parse_task_image_ref(digest_value)
            if digest_ref.digest is None or digest_ref.repository_ref != ref.repository_ref:
                raise TaskImageIdentityError("Switchyard publish image_digest repository does not match image_ref")
            digest = digest_ref.digest
        else:
            digest = normalize_digest(digest_value)
    except TaskImageIdentityError as exc:
        raise BuildError(f"invalid Switchyard image identity: {exc}") from exc
    return ref.normalized_ref, ref.digest_ref(digest)


def _profile_response(
    existing: dict,
    *,
    requested: SwitchyardPublishRequest,
    reused_profile: bool,
) -> SwitchyardPublishResponse:
    config = dict(existing["config"])
    normalize = (
        _normalize_cloud_build_switchyard_identity
        # Only 'cloudbuild' selects the other normalizer, so a legacy profile
        # storing the builder service's former name still lands here.
        if str(config.get("builder") or "image_builder_service") == "cloudbuild"
        else _normalize_switchyard_image_identity
    )
    try:
        image_ref, image_digest = normalize(
            str(config["image"]),
            str(config["image_digest"]),
        )
    except BuildError as exc:
        raise _http_error(502, "invalid_switchyard_image_identity", str(exc)) from exc
    config["image"] = image_ref
    config["image_digest"] = image_digest
    return SwitchyardPublishResponse(
        profile_id=str(existing["id"]),
        profile_name=str(existing["name"]),
        reused_profile=reused_profile,
        source_project=str(config.get("source_project") or requested.source_project),
        source_ref=str(config.get("source_ref") or requested.source_ref),
        context_path=str(config.get("context_path") or requested.context_path),
        dockerfile_path=str(
            config.get("dockerfile_path")
            or _dockerfile_path_for_context(requested.context_path, requested.dockerfile_path)
        ),
        dockerfile_sha256=config.get("dockerfile_sha256") if isinstance(config.get("dockerfile_sha256"), str) else None,
        context_hash=config.get("context_hash") if isinstance(config.get("context_hash"), str) else None,
        context_archive_sha256=config.get("context_archive_sha256")
        if isinstance(config.get("context_archive_sha256"), str)
        else None,
        image_ref=image_ref,
        image_digest=image_digest,
        config=config,
    )


def _cloud_build_job_status(build: dict[str, object]) -> str:
    status = str(build.get("status") or "QUEUED").upper()
    if status == "SUCCESS":
        return "succeeded"
    if status in _CLOUD_BUILD_TERMINAL:
        return "failed"
    if status in {"QUEUED", "PENDING"}:
        return "queued"
    return "building"


def _switchyard_build_metadata(build: dict[str, object]) -> dict[str, str]:
    substitutions = build.get("substitutions")
    if not isinstance(substitutions, dict):
        raise BuildError("Cloud Build is not a scaled-evals Switchyard publication")
    values = {str(key): str(value) for key, value in substitutions.items()}
    purpose = values.get(f"{_SWITCHYARD_SUBSTITUTION_PREFIX}PURPOSE")
    if purpose != "publish":
        raise BuildError("Cloud Build is not a scaled-evals Switchyard publication")
    return values


def _metadata_value(metadata: dict[str, str], name: str) -> str:
    value = metadata.get(f"{_SWITCHYARD_SUBSTITUTION_PREFIX}{name}", "").strip()
    if not value:
        raise BuildError(f"Cloud Build Switchyard metadata omitted {name.lower()}")
    return value


def _optional_metadata_value(metadata: dict[str, str], name: str) -> str | None:
    return metadata.get(f"{_SWITCHYARD_SUBSTITUTION_PREFIX}{name}", "").strip() or None


def _request_from_cloud_build(metadata: dict[str, str]) -> SwitchyardPublishRequest:
    return SwitchyardPublishRequest(
        source_project=_metadata_value(metadata, "SOURCE_PROJECT"),
        source_ref=_metadata_value(metadata, "SOURCE_REF"),
        context_path=_metadata_value(metadata, "CONTEXT_PATH"),
        dockerfile_path=_metadata_value(metadata, "DOCKERFILE_PATH"),
        context_hash=_metadata_value(metadata, "CONTEXT_HASH"),
        profile_name=_metadata_value(metadata, "PROFILE_NAME"),
    )


def _persist_cloud_build_profile(
    db: Database,
    current: CurrentPrincipal,
    *,
    build: dict[str, object],
    metadata: dict[str, str],
    requested: SwitchyardPublishRequest,
) -> SwitchyardPublishResponse:
    owner_scope = _write_owner_scope(current)
    existing = _find_existing_profile(
        db,
        source_project=requested.source_project,
        source_ref=requested.source_ref,
        context_path=requested.context_path,
        context_hash=_metadata_value(metadata, "CONTEXT_HASH"),
        builder_source_commit=_optional_metadata_value(metadata, "BUILDER_SOURCE_COMMIT"),
        owner_id=owner_scope,
    )
    if existing is not None:
        return _profile_response(existing, requested=requested, reused_profile=True)

    images = build.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise BuildError("Cloud Build Switchyard publication did not include one target image")
    image_ref = str(images[0])
    expected_ref = _switchyard_cloud_build_image_ref(
        requested.source_ref,
        _metadata_value(metadata, "CONTEXT_HASH"),
    )
    if image_ref != expected_ref:
        raise BuildError("Cloud Build Switchyard target image did not match its source revision")
    builder_digest = cloud_build.image_digest(build, image_ref)
    try:
        resolved = resolve_task_image(image_ref, expected_digest=builder_digest)
    except TaskImageIdentityError as exc:
        raise BuildError(f"cloud build Switchyard image verification failed: {exc}") from exc
    normalized_ref, normalized_digest = _normalize_cloud_build_switchyard_identity(
        resolved.runtime_ref,
        resolved.immutable_ref,
    )
    config = _publish_config(
        image_ref=normalized_ref,
        image_digest=normalized_digest,
        source_project=requested.source_project,
        source_ref=requested.source_ref,
        context_path=requested.context_path,
        dockerfile_path=_metadata_value(metadata, "DOCKERFILE_PATH"),
        dockerfile_sha256=_metadata_value(metadata, "DOCKERFILE_SHA256"),
        context_hash=_metadata_value(metadata, "CONTEXT_HASH"),
        context_archive_sha256=_metadata_value(metadata, "ARCHIVE_SHA256"),
        builder_source_commit=_optional_metadata_value(metadata, "BUILDER_SOURCE_COMMIT"),
        builder="cloudbuild",
    )
    record_principal(db, current)
    row = db.config_profiles.create(
        make_id("cfg"),
        name=_metadata_value(metadata, "PROFILE_NAME"),
        type="switchyard",
        config=config,
        owner_id=current.owner_id,
    )
    return _profile_response(row, requested=requested, reused_profile=False)


def _spool_context_archive(context_file: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(context_file, output)
    if destination.stat().st_size <= 0:
        raise BuildError("uploaded Switchyard context archive is empty")


@router.post("/publish", status_code=202, response_model=SwitchyardPublishResult)
def publish_switchyard(
    db: Db,
    current: Principal,
    source_ref: Annotated[str, Form(min_length=40, max_length=40)],
    source_project: Annotated[str, Form(min_length=1)] = _DEFAULT_SOURCE_PROJECT,
    context_path: Annotated[str, Form(min_length=1)] = _DEFAULT_CONTEXT_PATH,
    dockerfile_path: Annotated[str | None, Form()] = None,
    context_hash: Annotated[str | None, Form()] = None,
    profile_name: Annotated[str | None, Form(min_length=1, max_length=200)] = None,
    context: Annotated[
        UploadFile | None,
        File(description="Deterministic gzip archive of the Switchyard build context."),
    ] = None,
) -> SwitchyardPublishResult:
    """Submit a Switchyard publication and return without waiting on Cloud Build."""
    body = SwitchyardPublishRequest(
        source_project=source_project,
        source_ref=source_ref,
        context_path=context_path,
        dockerfile_path=dockerfile_path,
        context_hash=context_hash,
        profile_name=profile_name,
    )
    context_file = context.file if context is not None else None
    return _publish_switchyard(body, db, current, context_file=context_file)


@router.get("/publishes/{build_id}", response_model=SwitchyardPublishJobResponse)
def get_switchyard_publish(
    build_id: str,
    db: Db,
    current: Principal,
) -> SwitchyardPublishJobResponse:
    """Read and, on success, finalize one asynchronous Cloud Build publication."""
    try:
        build = cloud_build.get_build(build_id)
        metadata = _switchyard_build_metadata(build)
        owner_scope = _write_owner_scope(current)
        if owner_scope is not None and _metadata_value(metadata, "OWNER_ID") != current.owner_id:
            raise _http_error(404, "not_found", "Switchyard publication not found")
        status = _cloud_build_job_status(build)
        if status == "failed":
            detail = build.get("failureInfo") or build.get("statusDetail")
            return SwitchyardPublishJobResponse(
                build_id=build_id,
                status=status,
                build_error=str(detail or f"Cloud Build ended with {build.get('status')}"),
            )
        if status != "succeeded":
            return SwitchyardPublishJobResponse(build_id=build_id, status=status)
        requested = _request_from_cloud_build(metadata)
        result = _persist_cloud_build_profile(
            db,
            current,
            build=build,
            metadata=metadata,
            requested=requested,
        )
        return SwitchyardPublishJobResponse(
            build_id=build_id,
            status="succeeded",
            result=result,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - expose concise Cloud Build failure
        raise _http_error(502, "switchyard_publish_status_failed", str(exc)) from exc


def _publish_switchyard(
    body: SwitchyardPublishRequest,
    db: Database,
    current: CurrentPrincipal,
    *,
    context_file: BinaryIO | None,
) -> SwitchyardPublishResult:
    owner_id = _write_owner_scope(current)
    resolved_builder_source_commit = _resolved_builder_source_commit()
    dockerfile_path = _dockerfile_path_for_context(body.context_path, body.dockerfile_path)

    if body.context_hash is not None:
        existing = _find_existing_profile(
            db,
            source_project=body.source_project,
            source_ref=body.source_ref,
            context_path=body.context_path,
            context_hash=body.context_hash,
            builder_source_commit=resolved_builder_source_commit,
            owner_id=owner_id,
        )
        if existing is not None:
            return _profile_response(existing, requested=body, reused_profile=True)

    if context_file is None:
        raise _http_error(
            400,
            "switchyard_context_required",
            "upload a Switchyard build context archive",
        )

    with tempfile.TemporaryDirectory(prefix="se-switchyard-publish-") as tmp:
        archive_path = Path(tmp) / "context.tar.gz"
        try:
            _spool_context_archive(context_file, archive_path)
            metadata = inspect_uploaded_archive_file(
                archive_path,
                context_path=body.context_path,
                dockerfile_path=dockerfile_path,
            )
        except BuildError as exc:
            raise _http_error(400, "invalid_switchyard_context", str(exc)) from exc
        if body.context_hash is not None and metadata.context_hash != body.context_hash:
            raise _http_error(
                400,
                "switchyard_context_hash_mismatch",
                "provided context_hash does not match the uploaded Switchyard context",
            )

        existing = _find_existing_profile(
            db,
            source_project=body.source_project,
            source_ref=body.source_ref,
            context_path=body.context_path,
            context_hash=metadata.context_hash,
            builder_source_commit=resolved_builder_source_commit,
            owner_id=owner_id,
        )
        if existing is not None:
            return _profile_response(existing, requested=body, reused_profile=True)

        if settings.image_builder_service_url:
            builder = "image_builder_service"
            if not settings.image_builder_service_token:
                raise _http_error(
                    422,
                    "builder_auth_required",
                    "Switchyard publish needs IMAGE_BUILDER_SERVICE_TOKEN configured",
                )
            try:
                data = resolve_uploaded_archive_file_details(
                    archive_path,
                    context_path=body.context_path,
                )
            except Exception as exc:  # noqa: BLE001 - expose concise builder failure as API error
                raise _http_error(502, "switchyard_publish_failed", str(exc)) from exc
            try:
                image_ref, image_digest = _normalize_switchyard_image_identity(
                    str(data["image_ref"]),
                    str(data["image_digest"]),
                )
            except BuildError as exc:
                raise _http_error(502, "invalid_switchyard_image_identity", str(exc)) from exc
        elif settings.cloud_build_enabled:
            builder = "cloudbuild"
            try:
                record_principal(db, current)
                build = _submit_cloud_build_switchyard(
                    archive_path,
                    body,
                    metadata,
                    owner_id=current.owner_id,
                )
            except Exception as exc:  # noqa: BLE001 - expose concise builder failure as API error
                raise _http_error(502, "switchyard_publish_failed", str(exc)) from exc
            return SwitchyardPublishJobResponse(
                build_id=str(build["id"]),
                status=_cloud_build_job_status(build),
            )
        else:
            raise _http_error(
                503,
                "build_disabled",
                "Switchyard publish requires a managed builder: set "
                "IMAGE_BUILDER_SERVICE_URL or enable CLOUD_BUILD_ENABLED (GKE)",
            )

    context_archive_sha256 = str(data.get("context_archive_sha256") or metadata.context_archive_sha256)
    context_hash = str(data.get("context_hash") or metadata.context_hash)
    builder_commit = data.get("builder_source_commit")
    if not isinstance(builder_commit, str):
        builder_commit = resolved_builder_source_commit

    config = _publish_config(
        image_ref=image_ref,
        image_digest=image_digest,
        source_project=body.source_project,
        source_ref=body.source_ref,
        context_path=body.context_path,
        dockerfile_path=dockerfile_path,
        dockerfile_sha256=metadata.dockerfile_sha256,
        context_hash=context_hash,
        context_archive_sha256=context_archive_sha256,
        builder_source_commit=builder_commit,
        builder=builder,
    )
    record_principal(db, current)
    row = db.config_profiles.create(
        make_id("cfg"),
        name=_profile_name(body.source_project, body.source_ref, body.profile_name),
        type="switchyard",
        config=config,
        owner_id=current.owner_id,
    )
    return SwitchyardPublishResponse(
        profile_id=str(row["id"]),
        profile_name=str(row["name"]),
        reused_profile=False,
        source_project=body.source_project,
        source_ref=body.source_ref,
        context_path=body.context_path,
        dockerfile_path=dockerfile_path,
        dockerfile_sha256=metadata.dockerfile_sha256,
        context_hash=context_hash,
        context_archive_sha256=context_archive_sha256,
        image_ref=image_ref,
        image_digest=image_digest,
        config=config,
    )
