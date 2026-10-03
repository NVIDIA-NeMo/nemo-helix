# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor Viewer-compatible archive and best-effort upload helpers."""

from __future__ import annotations

import copy
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from scaled_evals.api import artifacts
from scaled_evals.api.settings import settings

RESULT_NAMESPACE = "scaled_evals"
RESULT_HARBOR_VIEWER_KEY = "harbor_viewer"


@dataclass(frozen=True)
class HarborViewerPublication:
    """Metadata for a durable Viewer archive and its optional online upload."""

    job_name: str
    archive_size_bytes: int
    upload_url: str
    viewer_url: str | None = None
    viewer_path: str | None = None
    api_path: str | None = None
    bytes_uploaded: int | None = None
    upload_error: str | None = None


def _harbor_viewer_metadata(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    metadata = result.get(RESULT_NAMESPACE)
    if not isinstance(metadata, Mapping):
        return None
    viewer = metadata.get(RESULT_HARBOR_VIEWER_KEY)
    return viewer if isinstance(viewer, Mapping) else None


def harbor_viewer_url_from_result(result: Mapping[str, Any] | None) -> str | None:
    """Return a persisted Harbor Viewer URL from an evaluation result envelope."""
    viewer = _harbor_viewer_metadata(result)
    if viewer is None:
        return None
    url = viewer.get("url")
    return str(url) if isinstance(url, str) and url.strip() else None


def harbor_viewer_archive_available_from_result(result: Mapping[str, Any] | None) -> bool:
    """Return whether a Viewer-compatible archive was durably stored."""
    viewer = _harbor_viewer_metadata(result)
    return isinstance(viewer, Mapping) and isinstance(viewer.get("archive"), Mapping)


def harbor_viewer_upload_url_from_result(result: Mapping[str, Any] | None) -> str | None:
    """Return the browser-reachable manual upload endpoint."""
    viewer = _harbor_viewer_metadata(result)
    if viewer is None:
        return None
    url = viewer.get("upload_url")
    return str(url) if isinstance(url, str) and url.strip() else None


def result_with_harbor_viewer_publication(
    result: Mapping[str, Any],
    publication: HarborViewerPublication | None,
) -> dict[str, Any]:
    """Copy a result envelope and attach durable Harbor Viewer metadata."""
    copied = copy.deepcopy(dict(result))
    if publication is None:
        return copied
    namespace = copied.setdefault(RESULT_NAMESPACE, {})
    if not isinstance(namespace, dict):
        namespace = {}
        copied[RESULT_NAMESPACE] = namespace
    viewer = {
        "job_name": publication.job_name,
        "archive": {
            "file_name": artifacts.HARBOR_VIEWER_ARCHIVE_FILE_NAME,
            "format": "tar.gz",
            "size_bytes": publication.archive_size_bytes,
        },
        "upload_url": publication.upload_url,
    }
    if publication.viewer_url:
        viewer["url"] = publication.viewer_url
    if publication.viewer_path:
        viewer["viewer_path"] = publication.viewer_path
    if publication.api_path:
        viewer["api_path"] = publication.api_path
    namespace[RESULT_HARBOR_VIEWER_KEY] = viewer
    return copied


def publish_harbor_job_archive(
    *,
    job_name: str,
    job_dir: Path | str,
) -> HarborViewerPublication | None:
    """Persist one Viewer-compatible archive and optionally upload it.

    Empty ``HARBOR_VIEWER_BASE_URL`` disables the integration. Object-store
    persistence errors are raised to the caller; online upload errors are
    captured so the manual download-and-upload path remains available.
    """
    base_url = settings.harbor_viewer_base_url.strip().rstrip("/")
    if not base_url:
        return None

    root = Path(job_dir)
    _validate_harbor_job_dir(root)
    with tempfile.NamedTemporaryFile(prefix=f"{job_name}-", suffix=".tar.gz") as archive:
        _write_harbor_job_archive(job_name=job_name, job_dir=root, archive_path=archive.name)
        archive_path = Path(archive.name)
        archive_size_bytes = artifacts.upload_file(
            archive_path,
            artifacts.evaluation_harbor_viewer_archive_key(job_name),
            content_type="application/gzip",
        )
        overwrite = str(settings.harbor_viewer_upload_overwrite).lower()
        upload_url = f"{base_url}/api/uploads/jobs?overwrite={overwrite}"
        if not settings.harbor_viewer_auto_upload:
            return HarborViewerPublication(
                job_name=job_name,
                archive_size_bytes=archive_size_bytes,
                upload_url=upload_url,
            )

        archive.seek(0)
        headers = _upload_headers()
        params = {"overwrite": overwrite}
        files = {
            "archive": (
                f"{job_name}.tar.gz",
                archive,
                "application/gzip",
            )
        }
        try:
            with httpx.Client(timeout=settings.harbor_viewer_upload_timeout_seconds) as client:
                response = client.post(
                    f"{base_url}/api/uploads/jobs",
                    params=params,
                    files=files,
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001 — preserve the manual fallback
            return HarborViewerPublication(
                job_name=job_name,
                archive_size_bytes=archive_size_bytes,
                upload_url=upload_url,
                upload_error=str(exc),
            )

    uploaded_job_name = str(payload.get("job_name") or job_name)
    viewer_path = str(payload.get("viewer_path") or f"/jobs/{quote(uploaded_job_name, safe='')}")
    return HarborViewerPublication(
        job_name=uploaded_job_name,
        archive_size_bytes=archive_size_bytes,
        upload_url=upload_url,
        viewer_url=urljoin(f"{base_url}/", viewer_path.lstrip("/")),
        viewer_path=viewer_path,
        api_path=str(payload["api_path"]) if payload.get("api_path") else None,
        bytes_uploaded=int(payload["bytes"]) if isinstance(payload.get("bytes"), int) else None,
    )


def _validate_harbor_job_dir(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in ("config.json", "result.json"):
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(source)


def _upload_headers() -> dict[str, str]:
    token = settings.harbor_viewer_upload_token.strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _write_harbor_job_archive(*, job_name: str, job_dir: Path, archive_path: str) -> None:
    """Write a Harbor Viewer-compatible job archive.

    Reuses the scaled-evals artifact staging path so text secrets are redacted
    exactly like object-store artifacts before leaving the worker.
    """
    with (
        artifacts._staged_artifacts(job_dir) as (staged_root, files),  # noqa: SLF001
        tarfile.open(archive_path, mode="w:gz") as archive,
    ):
        for source in files:
            relative = source.relative_to(staged_root).as_posix()
            if artifacts._is_unsafe_artifact_path(relative):  # noqa: SLF001
                raise ValueError(f"unsafe Harbor Viewer artifact path: {relative!r}")
            archive.add(
                source,
                arcname=f"{job_name}/{relative}",
                recursive=False,
                filter=lambda info, path=source: artifacts._normalize_archive_info(  # noqa: SLF001
                    info,
                    path.stat().st_mtime,
                ),
            )
