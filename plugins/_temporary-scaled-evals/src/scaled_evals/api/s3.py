# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact + task-pack storage for scaled-evals, backed by the NeMo Helix Files service.

This module keeps its historical name and public function signatures so callers and the test
suite are unchanged, but the transport now delegates to :mod:`scaled_evals.api._files_backend`
(filesets) instead of talking to raw object storage (S3/GCS) directly. The scaled-evals-owned
write-path concerns — secret/.env redaction, sha256 manifests, and server-side ``results.tar.gz``
bundling — stay here, in front of Files.

Object keys are the same flat strings as before (``evaluations/{id}/artifacts/...``,
``{task_id}/rev/{n}/tarball.tar.gz``, ...); the backend maps each onto a fileset + in-fileset
path. See ``_files_backend.split_key``.
"""

import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scaled_evals.api import _files_backend
from scaled_evals.api.redaction import redact_json_text, redact_secret_text
from scaled_evals.api.settings import settings

ARTIFACT_MANIFEST_PATH = "scaled-evals-manifest.json"
ARCHIVE_FILE_NAME = "results.tar.gz"
HARBOR_VIEWER_ARCHIVE_FILE_NAME = "harbor-viewer.tar.gz"
_SECRET_FILE_NAMES = frozenset({".env", "target.env", "daytona.env"})
_TEXT_ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".toml", ".md"})

# `settings` is imported for backward-compatible references (e.g. archive size limits); the
# object-store endpoint/credential settings it used to read are no longer touched here.
_ = settings

# Re-export the broker-upload target type so callers keep going through the `s3` facade.
UploadTarget = _files_backend.UploadTarget


def upload_target(object_key: str) -> UploadTarget:
    """Resolve the direct-to-Files upload coordinates for a task pack (broker-upload model)."""
    return _files_backend.upload_target(object_key)


class PresignUnsupportedError(RuntimeError):
    """Presigned direct-to-storage URLs are not available once artifacts live on Files.

    The Files service brokers uploads/downloads through the SDK (see the task-pack broker
    upload flow and the server-side ``stream_object`` download path), so scaled-evals no longer
    hands callers a presigned URL. Kept as an explicit error rather than silently returning a
    dead URL so any missed caller fails loudly.
    """


def can_presign_get() -> bool:
    """Presigned GET is gone post-Files-migration; callers stream via the API instead."""
    return False


def presign_put(object_key: str, expires_in: int = 900) -> dict[str, str]:
    """Removed: task packs are now uploaded directly to Files by the client (broker model)."""
    raise PresignUnsupportedError(
        "presigned task-pack upload is no longer supported; the client uploads directly to the "
        "Files fileset returned by POST /tasks"
    )


def presign_get(object_key: str, expires_in: int = 900) -> str:
    """Removed: artifact/archive downloads stream through the API (see stream_object)."""
    raise PresignUnsupportedError(
        "presigned download is no longer supported; stream via the evaluations download endpoint"
    )


def object_size(object_key: str) -> int | None:
    """Return object size in bytes, or ``None`` when the file is not present.

    ``None`` (missing) is unsafe for task-pack quota enforcement, matching the prior contract
    where a missing Content-Length blocked finalize.
    """
    return _files_backend.object_size(object_key)


def object_exists(object_key: str) -> bool:
    """Return whether a file exists in the fileset backing this key."""
    return _files_backend.object_exists(object_key)


def delete_object(object_key: str) -> None:
    """Delete one file from its fileset (a no-op if already absent)."""
    _files_backend.delete_object(object_key)


def stream_object(object_key: str) -> Iterator[bytes]:
    """Stream a file's bytes for server-side proxying (FastAPI StreamingResponse).

    Chunked via the fileset filesystem so a large archive never fully materializes in a request
    thread beyond one chunk at a time.
    """
    return _files_backend.stream_object(object_key)


def read_json_object(object_key: str) -> dict[str, Any]:
    """Read one JSON object for worker-side evidence composition."""
    value = json.loads(_files_backend.get_bytes(object_key))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {object_key}")
    return value


def put_json_object(object_key: str, value: dict[str, Any]) -> None:
    """Write one JSON object to its fileset."""
    body = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    _files_backend.put_bytes(object_key, body, content_type="application/json")


def put_text_object(object_key: str, value: str) -> None:
    """Write UTF-8 text to its fileset."""
    _files_backend.put_bytes(object_key, value.encode(), content_type="text/plain; charset=utf-8")


def read_text_object_if_exists(object_key: str) -> str | None:
    """Read UTF-8 text, returning ``None`` when the file is not present."""
    try:
        return _files_backend.get_bytes(object_key).decode("utf-8", errors="replace")
    except _files_backend.MissingObjectError:
        return None


def evaluation_live_log_key(evaluation_id: str, execution_number: int) -> str:
    """Stable live-log snapshot key for one evaluation execution."""
    return f"evaluations/{evaluation_id}/live/{execution_number}/runner.log"


def evaluation_artifact_prefix(evaluation_id: str) -> str:
    """Stable object prefix for all artifacts produced by one evaluation."""
    return f"evaluations/{evaluation_id}/artifacts/"


def _is_unsafe_artifact_path(relative_path: str) -> bool:
    # Raw split, not PurePosixPath.parts — the latter normalizes away "." and
    # "//" segments, which would let them slip through undetected. Empty parts
    # also reject leading/trailing/doubled slashes.
    if not relative_path or "\\" in relative_path:
        return True
    return any(part in {"", ".", ".."} for part in relative_path.split("/"))


def evaluation_artifact_key(evaluation_id: str, path: str) -> str:
    """Object key for an artifact path relative to an evaluation.

    Rejects traversal/absolute paths: S3-compatible gateways may normalize
    `..` segments in request URLs, so an unchecked caller-supplied path could
    resolve outside the evaluation's artifact prefix.
    """
    relative = path.lstrip("/")
    if _is_unsafe_artifact_path(relative):
        raise ValueError(f"unsafe artifact path: {path!r}")
    return f"{evaluation_artifact_prefix(evaluation_id)}{relative}"


def evaluation_archive_key(evaluation_id: str) -> str:
    """Stable object key for one evaluation's downloadable artifact bundle."""
    return f"evaluations/{evaluation_id}/{ARCHIVE_FILE_NAME}"


def evaluation_harbor_viewer_archive_key(evaluation_id: str) -> str:
    """Stable object key for one Harbor Viewer-compatible job archive."""
    return f"evaluations/{evaluation_id}/{HARBOR_VIEWER_ARCHIVE_FILE_NAME}"


def upload_file(path: Path, object_key: str, *, content_type: str) -> int:
    """Upload one local file to a stable object key and return its size."""
    return _files_backend.upload_file(path, object_key, content_type=content_type)


def _is_secret_file(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    return any(part in _SECRET_FILE_NAMES or part.endswith(".env") for part in rel.parts)


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.relative_to(root).as_posix() != ARTIFACT_MANIFEST_PATH
            and not _is_secret_file(root, path)
        ):
            yield path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _bytes_sha256(body: bytes) -> str:
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _artifact_manifest(root: Path, files: list[Path]) -> bytes:
    entries = [
        {
            "path": source.relative_to(root).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
        }
        for source in files
    ]
    return _artifact_manifest_entries(entries)


def _artifact_manifest_entries(entries: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "schema_version": "scaled-evals-artifacts-v1",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "files": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _redacted_text_body(path: Path) -> bytes | None:
    if path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
        return None
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    redacted = (
        redact_json_text(text, lines=path.suffix.lower() == ".jsonl")
        if path.suffix.lower() in {".json", ".jsonl"}
        else redact_secret_text(text)
    )
    if redacted == text:
        return None
    return redacted.encode("utf-8")


@contextmanager
def _staged_artifacts(root: Path) -> Iterator[tuple[Path, list[Path]]]:
    """Materialize immutable, already-redacted bytes for hashing and persistence."""
    with tempfile.TemporaryDirectory(prefix="scaled-evals-artifacts-") as tmp:
        staged_root = Path(tmp)
        for source in _iter_files(root):
            relative = source.relative_to(root)
            destination = staged_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            redacted = _redacted_text_body(source)
            if redacted is None:
                shutil.copy2(source, destination)
            else:
                destination.write_bytes(redacted)
                shutil.copystat(source, destination)
        yield staged_root, list(_iter_files(staged_root))


def sync_directory_to_prefix(root: Path | str, prefix: str) -> int:
    """Upload all regular files under ``root`` to ``prefix``.

    Returns the number of uploaded files. Missing roots are a no-op so failed
    evaluations with no materialized job dir can still be marked terminal.
    """
    root_path = Path(root)
    if not root_path.exists():
        return 0
    with _staged_artifacts(root_path) as (staged_root, files):
        _files_backend.put_bytes(
            f"{prefix.rstrip('/')}/{ARTIFACT_MANIFEST_PATH}",
            _artifact_manifest(staged_root, files),
            content_type="application/json",
        )
        for source in files:
            object_key = f"{prefix.rstrip('/')}/{source.relative_to(staged_root).as_posix()}"
            _files_backend.upload_file(source, object_key, content_type="application/octet-stream")
    return len(files) + 1


def replace_directory_at_prefix(root: Path | str, prefix: str) -> int:
    """Replace one stable object prefix with the files currently under ``root``.

    Deleting first means a partial upload can leave an incomplete newest result,
    but it cannot mix files from different evaluation executions.
    """
    normalized = f"{prefix.rstrip('/')}/"
    for item in list_objects(normalized):
        object_key = str(item.get("key") or "")
        if object_key.startswith(normalized):
            delete_object(object_key)
    return sync_directory_to_prefix(root, normalized)


def sync_evidence_files(root: Path | str, prefix: str) -> int:
    """Upload terminal provenance/SBOM bytes and rebuild the remote file manifest."""
    from scaled_evals.models.provenance import MANIFEST_FILE_NAME
    from scaled_evals.models.sbom import SBOM_FILE_NAME

    root_path = Path(root)
    for name in (SBOM_FILE_NAME, MANIFEST_FILE_NAME):
        source = root_path / name
        if not source.is_file():
            raise FileNotFoundError(source)
        _files_backend.put_bytes(
            f"{prefix.rstrip('/')}/{name}",
            source.read_bytes(),
            content_type="application/json",
        )
    _rebuild_remote_artifact_manifest(prefix)
    return 3


def _rebuild_remote_artifact_manifest(prefix: str) -> None:
    normalized = f"{prefix.rstrip('/')}/"
    entries: list[dict[str, Any]] = []
    for item in sorted(list_objects(normalized), key=lambda value: value["key"]):
        key = str(item["key"])
        relative = key[len(normalized) :]
        if not relative or relative == ARTIFACT_MANIFEST_PATH:
            continue
        body = _files_backend.get_bytes(key)
        entries.append({"path": relative, "size_bytes": len(body), "sha256": _bytes_sha256(body)})
    _files_backend.put_bytes(
        f"{normalized}{ARTIFACT_MANIFEST_PATH}",
        _artifact_manifest_entries(entries),
        content_type="application/json",
    )


def list_objects(prefix: str) -> list[dict[str, Any]]:
    """List object metadata below ``prefix`` from the backing filesets."""
    return _files_backend.list_objects(prefix)


class ArchiveBuildError(RuntimeError):
    """Raised when a results archive cannot be built from synced artifacts."""


def _safe_archive_member_name(relative_path: str) -> str:
    if _is_unsafe_artifact_path(relative_path):
        raise ArchiveBuildError(f"unsafe artifact path: {relative_path!r}")
    return f"artifacts/{relative_path}"


def _add_local_file_to_archive(tar: tarfile.TarFile, root: Path, source: Path) -> int:
    relative_path = source.relative_to(root).as_posix()
    body = _redacted_text_body(source)
    if body is not None:
        info = tarfile.TarInfo(_safe_archive_member_name(relative_path))
        info.size = len(body)
        info.mode = 0o644
        info.mtime = int(source.stat().st_mtime)
        tar.addfile(info, io.BytesIO(body))
        return len(body)

    tar.add(
        source,
        arcname=_safe_archive_member_name(relative_path),
        recursive=False,
        filter=lambda info: _normalize_archive_info(info, source.stat().st_mtime),
    )
    return source.stat().st_size


def _normalize_archive_info(info: tarfile.TarInfo, mtime: float) -> tarfile.TarInfo:
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = int(mtime)
    return info


def _upload_archive(evaluation_id: str, tmp: Any, size_bytes: int) -> dict[str, Any]:
    dest_key = evaluation_archive_key(evaluation_id)
    tmp.seek(0)
    _files_backend.put_bytes(dest_key, tmp.read(), content_type="application/gzip")
    return {"object_key": dest_key, "size_bytes": size_bytes}


def build_evaluation_archive_from_directory(evaluation_id: str, root: Path | str) -> dict[str, Any]:
    """Create ``results.tar.gz`` directly from local artifacts, then upload it.

    This is the dispatch-worker happy path after an evaluation finishes: artifacts
    are still on the local worker filesystem, so we avoid listing and downloading
    the just-synced S3 objects only to re-compress them.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise ArchiveBuildError("no local artifacts found")
    with _staged_artifacts(root_path) as (staged_root, files):
        manifest = _artifact_manifest(staged_root, files)
        archive_file_count = len(files) + 1
        if archive_file_count > settings.evaluation_archive_max_files:
            raise ArchiveBuildError(
                f"archive would include {archive_file_count} files; limit is {settings.evaluation_archive_max_files}"
            )

        with tempfile.TemporaryFile() as tmp:
            source_bytes = len(manifest)
            with tarfile.open(fileobj=tmp, mode="w:gz") as tar:
                info = tarfile.TarInfo(_safe_archive_member_name(ARTIFACT_MANIFEST_PATH))
                info.size = len(manifest)
                info.mode = 0o644
                info.mtime = int(datetime.now(tz=UTC).timestamp())
                tar.addfile(info, io.BytesIO(manifest))
                for source in files:
                    source_bytes += _add_local_file_to_archive(tar, staged_root, source)
                    if source_bytes > settings.evaluation_archive_max_source_bytes:
                        raise ArchiveBuildError(
                            f"archive source bytes {source_bytes} exceed "
                            f"limit {settings.evaluation_archive_max_source_bytes}"
                        )
            size_bytes = tmp.tell()
            archive = _upload_archive(evaluation_id, tmp, size_bytes)

    return {
        **archive,
        "file_count": archive_file_count,
        "source_bytes": source_bytes,
    }


def build_evaluation_archive(evaluation_id: str) -> dict[str, Any]:
    """Create ``results.tar.gz`` from an evaluation's synced artifact objects.

    The API request path never calls this. It is intended for the dispatch /
    archive worker so compression and object-store reads do not occupy a FastAPI
    request thread.
    """
    source_prefix = evaluation_artifact_prefix(evaluation_id)
    objects = [item for item in list_objects(source_prefix) if item.get("key", "").startswith(source_prefix)]
    if not objects:
        raise ArchiveBuildError("no synced artifacts found")
    if len(objects) > settings.evaluation_archive_max_files:
        raise ArchiveBuildError(
            f"archive would include {len(objects)} files; limit is {settings.evaluation_archive_max_files}"
        )
    total_source_bytes = sum(int(item.get("size_bytes") or 0) for item in objects)
    if total_source_bytes > settings.evaluation_archive_max_source_bytes:
        raise ArchiveBuildError(
            f"archive source bytes {total_source_bytes} exceed limit {settings.evaluation_archive_max_source_bytes}"
        )

    with tempfile.TemporaryFile() as tmp:
        with tarfile.open(fileobj=tmp, mode="w:gz") as tar:
            for item in sorted(objects, key=lambda obj: obj["key"]):
                key = item["key"]
                relative_path = key[len(source_prefix) :]
                info = tarfile.TarInfo(_safe_archive_member_name(relative_path))
                info.size = int(item.get("size_bytes") or 0)
                info.mode = 0o644
                updated_at = item.get("updated_at")
                if isinstance(updated_at, str):
                    try:
                        info.mtime = int(datetime.fromisoformat(updated_at).timestamp())
                    except ValueError:
                        info.mtime = int(datetime.now(tz=UTC).timestamp())
                else:
                    info.mtime = int(datetime.now(tz=UTC).timestamp())
                tar.addfile(info, io.BytesIO(_files_backend.get_bytes(key)))
        size_bytes = tmp.tell()
        archive = _upload_archive(evaluation_id, tmp, size_bytes)
    return {
        **archive,
        "file_count": len(objects),
        "source_bytes": total_source_bytes,
    }


def upload_context_archive(archive_path: Path, object_key: str) -> None:
    """Upload a local build-context archive to its fileset (server-side put)."""
    _files_backend.upload_file(archive_path, object_key, content_type="application/gzip")


def download_object(object_key: str, dest_path: str) -> None:
    """Download a file to a local path (server-side fetch, e.g. the task build reading a pack)."""
    _files_backend.download_object(object_key, dest_path)


def ensure_bucket() -> str:
    """No-op retained for the startup call site: Files owns storage provisioning.

    Historically this created the object-store bucket on a fresh RustFS/MinIO volume. With
    artifacts on the Files service there is no scaled-evals-owned bucket to create; filesets are
    created on demand at write time. ``check_bucket`` remains the single source of truth for
    whether storage is actually usable.
    """
    return settings.files_workspace


def check_bucket() -> None:
    """Confirm the Files service is reachable. Raises on failure so ``/v1/readyz`` can report it."""
    _files_backend.readiness_probe()
