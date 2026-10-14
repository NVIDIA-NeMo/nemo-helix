# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic build-context archive helpers for uploaded-context builds.

These helpers are intentionally dependency-free so the CLI can package a local
build context without importing server settings.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scaled_evals.api.build.errors import BuildError


@dataclass(frozen=True)
class UploadedArchiveMetadata:
    context_hash: str
    context_archive_sha256: str
    dockerfile_path: str
    dockerfile_sha256: str


def inspect_uploaded_archive_file(
    archive_path: Path,
    *,
    context_path: str = ".",
    dockerfile_path: str | None = None,
) -> UploadedArchiveMetadata:
    """Validate an uploaded archive and return its stable build identity."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise BuildError(f"uploaded build context archive does not exist: {archive_path}")
    if archive_path.stat().st_size <= 0:
        raise BuildError("uploaded build context archive is empty")
    context_path = context_path.strip() or "."
    dockerfile_path = _resolved_dockerfile_path(context_path, dockerfile_path)
    with tempfile.TemporaryDirectory(prefix="se-context-") as tmp:
        context_dir = Path(tmp) / "context"
        _extract_uploaded_archive_file(archive_path, context_dir)
        build_root = context_dir if context_path == "." else context_dir / context_path
        build_dockerfile = build_root / "Dockerfile"
        if not build_dockerfile.is_file():
            raise BuildError(f"uploaded build context has no Dockerfile at {context_path}")
        source_dockerfile = context_dir / dockerfile_path
        if not source_dockerfile.is_file():
            raise BuildError(f"uploaded build context has no source Dockerfile at {dockerfile_path}")
        if source_dockerfile.read_bytes() != build_dockerfile.read_bytes():
            raise BuildError(
                "uploaded build context root Dockerfile does not match "
                f"source Dockerfile {dockerfile_path}"
            )
        context_hash = compute_context_hash(build_root)
        dockerfile_sha256 = _sha256_file(source_dockerfile)
    return UploadedArchiveMetadata(
        context_hash=context_hash,
        context_archive_sha256=_sha256_file(archive_path),
        dockerfile_path=dockerfile_path,
        dockerfile_sha256=dockerfile_sha256,
    )


def archive_context_directory(context_dir: Path, *, dockerfile_path: str | None = None) -> bytes:
    """Create a deterministic gzip archive from a local build context."""

    context_dir = context_dir.resolve()
    if not context_dir.is_dir():
        raise BuildError(f"build context directory does not exist: {context_dir}")
    dockerfile_path = _resolved_dockerfile_path(".", dockerfile_path)
    source_dockerfile = context_dir / dockerfile_path
    if not source_dockerfile.is_file():
        raise BuildError(f"build context has no source Dockerfile at {dockerfile_path}")
    if source_dockerfile.is_symlink():
        raise BuildError(f"refusing symlink in build context: {dockerfile_path}")
    root_dockerfile = context_dir / "Dockerfile"
    synthetic_root_dockerfile = dockerfile_path != "Dockerfile"
    if synthetic_root_dockerfile and root_dockerfile.exists():
        raise BuildError(
            "build context already has a root Dockerfile; omit dockerfile_path or use Dockerfile"
        )
    with io.BytesIO() as raw:
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            for path in _context_paths(context_dir):
                rel = path.relative_to(context_dir)
                _reject_context_path(rel, path)
                arcname = "." if rel == Path(".") else rel.as_posix()
                info = archive.gettarinfo(str(path), arcname=arcname)
                _normalize_tar_info(info)
                if info.isfile():
                    with path.open("rb") as fileobj:
                        archive.addfile(info, fileobj)
                elif info.isdir():
                    archive.addfile(info)
                else:
                    raise BuildError(f"refusing non-regular build context entry: {rel.as_posix()}")
            if synthetic_root_dockerfile:
                info = archive.gettarinfo(str(source_dockerfile), arcname="Dockerfile")
                _normalize_tar_info(info)
                with source_dockerfile.open("rb") as fileobj:
                    archive.addfile(info, fileobj)
        return raw.getvalue()


def compute_context_hash(context_dir: Path) -> str:
    """Compute the file-tree hash expected by the image-builder service."""

    digest = hashlib.sha256()
    for path in sorted(context_dir.rglob("*")):
        rel = path.relative_to(context_dir)
        if _is_ignored_context_path(rel):
            continue
        _reject_context_path(rel, path)
        if not path.is_file():
            continue
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"755" if path.stat().st_mode & 0o111 else b"644")
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _extract_uploaded_archive_file(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                _safe_member_path(member)
            for member in members:
                rel = _safe_member_path(member)
                target = destination / rel
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise BuildError(f"could not read archive member: {member.name}")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    except (OSError, tarfile.TarError) as exc:
        raise BuildError(f"could not extract uploaded build context archive: {exc}") from exc


def _safe_member_path(member: tarfile.TarInfo) -> Path:
    path = Path(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise BuildError(f"unsafe build context archive path: {member.name}")
    if member.issym() or member.islnk():
        raise BuildError(f"refusing link in build context archive: {member.name}")
    if not (member.isdir() or member.isfile()):
        raise BuildError(f"refusing non-regular build context archive member: {member.name}")
    if any(part == ".DS_Store" or part.startswith("._") for part in path.parts):
        raise BuildError(f"refusing macOS metadata in build context archive: {member.name}")
    return path


def _is_ignored_context_path(relative: Path) -> bool:
    """Return True for entries pruned from build context packaging.

    Prunes version-control and virtualenv artifacts (`.git`, `.venv*`) so a
    local checkout with a `.venv` can be packaged without hand-cleaning it
    first. Symlinks outside these directories are still
    rejected by `_reject_context_path` rather than silently dropped.
    """
    return any(part == ".git" or part.startswith(".venv") for part in relative.parts)


def _reject_context_path(relative: Path, path: Path) -> None:
    if any(part == ".DS_Store" or part.startswith("._") for part in relative.parts):
        raise BuildError(f"refusing macOS metadata in build context: {relative.as_posix()}")
    if path.is_symlink():
        raise BuildError(f"refusing symlink in build context: {relative.as_posix()}")


def _normalize_tar_info(info: tarfile.TarInfo) -> None:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isfile():
        info.mode = 0o755 if info.mode & 0o111 else 0o644
    elif info.isdir():
        info.mode = 0o755


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_paths(context_dir: Path) -> Iterable[Path]:
    yield context_dir
    for path in sorted(context_dir.rglob("*"), key=lambda p: p.as_posix()):
        rel = path.relative_to(context_dir)
        if _is_ignored_context_path(rel):
            continue
        yield path


def _resolved_dockerfile_path(context_path: str, dockerfile_path: str | None) -> str:
    if dockerfile_path is None or not dockerfile_path.strip():
        return "Dockerfile" if context_path == "." else f"{context_path.rstrip('/')}/Dockerfile"
    normalized = dockerfile_path.strip()
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise BuildError("dockerfile_path must stay within the uploaded archive")
    if any(part == ".DS_Store" or part.startswith("._") for part in path.parts):
        raise BuildError(f"refusing macOS metadata in build context: {normalized}")
    return path.as_posix()
