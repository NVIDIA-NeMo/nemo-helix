# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve uploaded task packs via an external image-builder service.

The service is named only by `IMAGE_BUILDER_SERVICE_URL`, so any builder exposing
this contract can back it. The task source of truth is the tarball already
uploaded to scaled-evals object storage. Finalize sends that stored archive to the
service, which builds, scans, signs, and pushes the task image. No task
finalization path asks the service to clone user Git source.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

import httpx

from scaled_evals.api import artifacts
from scaled_evals.api.build.errors import BuildError
from scaled_evals.api.build.uploaded_context import (
    UploadedArchiveMetadata,
    archive_context_directory,
    compute_context_hash,
    inspect_uploaded_archive_file,
)
from scaled_evals.api.settings import settings

_RESOLVE_PATH = "/v1/eval-images/resolve"
__all__ = [
    "BuildError",
    "UploadedArchiveMetadata",
    "archive_context_directory",
    "compute_context_hash",
    "inspect_uploaded_archive_file",
    "resolve_uploaded_archive",
    "resolve_uploaded_archive_details",
    "resolve_uploaded_archive_file",
    "resolve_uploaded_archive_file_details",
    "resolve_uploaded_revision_image",
]


def resolve_uploaded_revision_image(
    *,
    tarball_object_key: str,
    context_path: str = ".",
    builder_source_commit: str | None = None,
    oc_token: str | None = None,
    force_rebuild: bool = False,
    service_url: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """Resolve a stored task-pack archive into a signed image via the builder service."""

    with tempfile.TemporaryDirectory(prefix="se-upload-") as tmp:
        archive_path = Path(tmp) / "context.tar.gz"
        artifacts.download_object(tarball_object_key, str(archive_path))
        data = resolve_uploaded_archive_file_details(
            archive_path,
            context_path=context_path,
            builder_source_commit=builder_source_commit,
            oc_token=oc_token,
            force_rebuild=force_rebuild,
            service_url=service_url,
            timeout_s=timeout_s,
        )
    return _image_identity(data)


def resolve_uploaded_archive_file(
    archive_path: Path,
    *,
    context_path: str = ".",
    builder_source_commit: str | None = None,
    oc_token: str | None = None,
    force_rebuild: bool = False,
    service_url: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """Post a gzip build-context archive file to the image-builder service."""

    data = resolve_uploaded_archive_file_details(
        archive_path,
        context_path=context_path,
        builder_source_commit=builder_source_commit,
        oc_token=oc_token,
        force_rebuild=force_rebuild,
        service_url=service_url,
        timeout_s=timeout_s,
    )
    return _image_identity(data)


def resolve_uploaded_archive(
    archive: bytes,
    *,
    context_path: str = ".",
    builder_source_commit: str | None = None,
    oc_token: str | None = None,
    force_rebuild: bool = False,
    service_url: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str, str]:
    """Post a gzip build-context archive to the image-builder service."""

    data = resolve_uploaded_archive_details(
        archive,
        context_path=context_path,
        builder_source_commit=builder_source_commit,
        oc_token=oc_token,
        force_rebuild=force_rebuild,
        service_url=service_url,
        timeout_s=timeout_s,
    )
    return _image_identity(data)


def resolve_uploaded_archive_details(
    archive: bytes,
    *,
    context_path: str = ".",
    builder_source_commit: str | None = None,
    oc_token: str | None = None,
    force_rebuild: bool = False,
    service_url: str | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Post a gzip build-context archive to the image-builder service."""

    if not archive:
        raise BuildError("uploaded build context archive is empty")
    with tempfile.TemporaryDirectory(prefix="se-archive-") as tmp:
        archive_path = Path(tmp) / "context.tar.gz"
        archive_path.write_bytes(archive)
        return resolve_uploaded_archive_file_details(
            archive_path,
            context_path=context_path,
            builder_source_commit=builder_source_commit,
            oc_token=oc_token,
            force_rebuild=force_rebuild,
            service_url=service_url,
            timeout_s=timeout_s,
        )


def resolve_uploaded_archive_file_details(
    archive_path: Path,
    *,
    context_path: str = ".",
    builder_source_commit: str | None = None,
    oc_token: str | None = None,
    force_rebuild: bool = False,
    service_url: str | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Post a gzip build-context archive file to the image-builder service."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise BuildError(f"uploaded build context archive does not exist: {archive_path}")
    if archive_path.stat().st_size <= 0:
        raise BuildError("uploaded build context archive is empty")
    context_path = context_path.strip() or "."
    metadata = inspect_uploaded_archive_file(archive_path, context_path=context_path)
    payload = {
        "context_path": context_path,
        "context_hash": metadata.context_hash,
        "context_archive_sha256": metadata.context_archive_sha256,
    }
    resolved_builder_source_commit = _resolved_builder_source_commit(builder_source_commit)
    if resolved_builder_source_commit:
        payload["builder_source_commit"] = resolved_builder_source_commit
    if force_rebuild:
        payload["force_rebuild"] = "true"

    with archive_path.open("rb") as archive_file:
        data = _post_resolve(
            data=payload,
            files={"context": ("context.tar.gz", archive_file, "application/gzip")},
            oc_token=oc_token,
            service_url=service_url,
            timeout_s=timeout_s,
        )
    _verify_builder_source_commit(data, resolved_builder_source_commit)
    _image_identity(data)
    data.setdefault("context_hash", metadata.context_hash)
    data.setdefault("context_archive_sha256", metadata.context_archive_sha256)
    if resolved_builder_source_commit:
        data.setdefault("builder_source_commit", resolved_builder_source_commit)
    return data


def _post_resolve(
    *,
    data: dict[str, str],
    files: dict[str, tuple[str, bytes | BinaryIO, str]],
    oc_token: str | None,
    service_url: str | None,
    timeout_s: float | None = None,
) -> dict:
    resolved_service_url = (service_url or settings.image_builder_service_url).strip()
    if not resolved_service_url:
        raise BuildError(
            "image builder service is not configured (IMAGE_BUILDER_SERVICE_URL unset)"
        )

    bearer = oc_token or settings.image_builder_service_token
    if not bearer:
        raise BuildError("no image builder service auth: configure IMAGE_BUILDER_SERVICE_TOKEN")

    parsed_service_url = urlsplit(resolved_service_url)
    if (
        parsed_service_url.scheme.lower() != "https"
        or not parsed_service_url.hostname
        or parsed_service_url.username is not None
        or parsed_service_url.password is not None
    ):
        raise BuildError("image builder service URL must use HTTPS and contain no userinfo")

    url = resolved_service_url.rstrip("/") + _RESOLVE_PATH
    headers = {"Authorization": f"Bearer {bearer}"}
    try:
        timeout = timeout_s if timeout_s is not None else settings.image_builder_service_timeout_s
        response = httpx.post(
            url,
            data=data,
            files=files,
            headers=headers,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise BuildError(f"image builder service request failed: {exc}") from exc

    if response.status_code >= 400:
        raise BuildError(
            f"image builder service returned {response.status_code}: {response.text[:1000]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise BuildError("image builder service returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise BuildError("image builder service returned malformed JSON: expected an object")
    return payload


def _resolved_builder_source_commit(builder_source_commit: str | None) -> str:
    resolved = (builder_source_commit or settings.image_builder_source_commit).strip().lower()
    if resolved and not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise BuildError("builder_source_commit must be a full 40-character hex commit")
    return resolved


def _verify_builder_source_commit(data: dict, requested: str) -> None:
    if not requested:
        return
    returned = (data.get("builder_source_commit") or "").strip().lower()
    if returned != requested:
        raise BuildError(
            "image builder service returned builder_source_commit that did not match "
            "the requested builder_source_commit"
        )


def _image_identity(data: dict) -> tuple[str, str]:
    image_ref = (data.get("image_ref") or "").strip()
    image_digest = (data.get("image_digest") or "").strip()
    if not image_ref:
        raise BuildError(f"image builder service returned no image_ref: {data}")
    if not image_digest:
        raise BuildError(f"image builder service returned no image_digest: {data}")
    return image_ref, image_digest
