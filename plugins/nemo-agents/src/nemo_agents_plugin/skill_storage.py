# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-owned storage for skill archives.

Every published skill revision points at an archive in one service-owned fileset per workspace,
:data:`SKILLS_FILESET_NAME`, at ``<skill-name>/<files_hash>/skill_archive``. The key is the archive
digest, so identical bytes always land at the same key and an object found there is reused only
after it has been read back and verified.

Publishing never trusts the caller's fileset beyond one bounded download: the source directory is
copied into private staging, archived with :func:`~nemo_helix_plugin.skill_archive.build_skill_archive`
(which applies the format contract, including the ``evals/`` exclusion), uploaded, and read back
through :func:`~nemo_helix_plugin.skill_archive.verify_skill_archive` before any revision row exists.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from nemo_helix_plugin.agents.skills import SkillArchiveDescriptor
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.storage_config import StorageConfigType
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.refs import parse_entity_ref
from nemo_helix_plugin.skill_archive import (
    DEFAULT_LIMITS,
    SkillArchiveError,
    SkillArchiveInfo,
    SkillArchiveLimits,
    verify_skill_archive,
)

logger = logging.getLogger(__name__)

#: The service-owned fileset (one per workspace) that holds every published skill archive.
SKILLS_FILESET_NAME = "nemo-agents-skills"

#: Object name of the archive under ``<skill-name>/<files_hash>/``.
SKILL_ARCHIVE_OBJECT = "skill_archive"

_CHUNK_BYTES = 1024 * 1024


class SkillSourceError(ValueError):
    """The source fileset or directory is missing, unreadable, unsafe, or over a limit."""


class SkillStorageError(RuntimeError):
    """The stored archive could not be written or did not verify after writing."""


def archive_object_path(skill_name: str, files_hash: str) -> str:
    """Path of an archive inside :data:`SKILLS_FILESET_NAME`."""
    return f"{skill_name}/{files_hash}/{SKILL_ARCHIVE_OBJECT}"


def archive_ref(workspace: str, skill_name: str, files_hash: str) -> str:
    """Fully qualified fileset reference of an archive (``workspace/fileset#path``)."""
    return f"{workspace}/{SKILLS_FILESET_NAME}#{archive_object_path(skill_name, files_hash)}"


def parse_fileset_dir_ref(ref: str) -> tuple[str, str, str]:
    """Split ``workspace/fileset#dir`` into ``(workspace, fileset, dir)``; ``dir`` has no outer slashes.

    ``dir`` is ``""`` for the fileset root (``#.`` or ``#/``).
    """
    base, separator, path = ref.partition("#")
    if not separator:
        raise SkillSourceError(f"Fileset reference {ref!r} must name a directory with '#path'")
    parsed = parse_entity_ref(base)
    if not parsed.workspace:
        raise SkillSourceError(f"Fileset reference {ref!r} must name its workspace")
    directory = path.strip("/")
    if directory == ".":
        directory = ""
    parts = PurePosixPath(directory).parts if directory else ()
    if any(part in {"", ".", ".."} for part in parts):
        raise SkillSourceError(f"Fileset reference {ref!r} has an unsafe path")
    return parsed.workspace, parsed.name, directory


class _BoundedWriter:
    """Write downloaded chunks while enforcing a byte limit and hashing what was written."""

    def __init__(self, output: BinaryIO, limit: int, what: str) -> None:
        self.output = output
        self.limit = limit
        self.what = what
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > self.limit:
            raise SkillSourceError(f"{self.what} exceeds the {self.limit}-byte limit")
        self.digest.update(chunk)
        self.output.write(chunk)


class SkillArchiveStore:
    """Stage skill sources and keep verified archives in the service-owned skills fileset.

    Two clients, so service authority never reads caller-supplied content:

    - ``files_client`` writes and reads the service-owned skills fileset.
    - ``source_files_client`` lists and downloads the caller's source. It must carry the *caller's*
      authority, so the Files service applies the caller's own access (including any storage secret
      behind the fileset). Defaults to ``files_client`` for callers that have only one client.
    """

    def __init__(
        self,
        files_client: AsyncFilesClient,
        *,
        source_files_client: AsyncFilesClient | None = None,
        limits: SkillArchiveLimits = DEFAULT_LIMITS,
    ) -> None:
        self.files_client = files_client
        self.source_files_client = source_files_client or files_client
        self.limits = limits

    async def stage_source(self, source_ref: str, dest: Path) -> None:
        """Download the directory ``source_ref`` names into the empty private directory ``dest``.

        Every check on the listing — paths, path length, entry count including directories, per-file
        and total size — runs before anything is created, and sizes are enforced again while
        streaming, since a listing is only a claim. Files are created exclusively, so a listing with
        colliding paths fails rather than overwriting. The Files listing is not paginated, so its
        response is held in memory once; nothing is allocated per entry until it passes the checks.

        Raises:
            SkillSourceError: Missing fileset, empty directory, unsafe path, or a limit exceeded.
        """
        workspace, fileset, directory = parse_fileset_dir_ref(source_ref)
        prefix = f"{directory}/" if directory else ""
        try:
            listing = (
                await self.source_files_client.list_files(
                    workspace=workspace, name=fileset, query_params={"path": directory} if directory else None
                )
            ).data()
        except NotFoundError as exc:
            raise SkillSourceError(f"Source fileset {workspace}/{fileset} was not found") from exc

        members: list[tuple[str, PurePosixPath, int]] = []
        directories: set[PurePosixPath] = set()
        total = 0
        for item in listing.data:
            path = item.path.lstrip("/")
            if prefix and not path.startswith(prefix):
                continue
            relative = PurePosixPath(path[len(prefix) :])
            if not relative.parts or relative.is_absolute() or any(p in {"", ".", ".."} for p in relative.parts):
                raise SkillSourceError(f"Source file path {path!r} is unsafe")
            if len(str(relative).encode("utf-8")) > self.limits.max_path_length:
                raise SkillSourceError(
                    f"Source file path {path!r} exceeds the {self.limits.max_path_length}-byte limit"
                )
            if item.size > self.limits.max_file_bytes:
                raise SkillSourceError(f"Source file {path!r} exceeds the {self.limits.max_file_bytes}-byte limit")
            total += item.size
            if total > self.limits.max_total_payload_bytes:
                raise SkillSourceError(
                    f"Source {source_ref!r} exceeds the {self.limits.max_total_payload_bytes}-byte limit"
                )
            directories.update(relative.parents)
            # Files and directories both become archive entries.
            if len(members) + 1 + len(directories) > self.limits.max_entries:
                raise SkillSourceError(f"Source {source_ref!r} holds more than {self.limits.max_entries} entries")
            members.append((path, relative, item.size))
        if not members:
            raise SkillSourceError(f"Source {source_ref!r} holds no files")

        remaining = self.limits.max_total_payload_bytes
        for path, relative, _ in members:
            target = dest.joinpath(*relative.parts)
            await asyncio.to_thread(target.parent.mkdir, mode=0o700, parents=True, exist_ok=True)
            limit = min(self.limits.max_file_bytes, remaining)
            written = await self._download(
                self.source_files_client, workspace, fileset, path, target, limit=limit, what=f"Source file {path!r}"
            )
            remaining -= written

    async def store(self, workspace: str, archive_path: Path, info: SkillArchiveInfo) -> SkillArchiveDescriptor:
        """Put a built archive at its digest key and verify the stored bytes before returning.

        An object already at the key is reused only if it verifies; otherwise it is overwritten (the
        key names exact bytes, so the correct content is never in doubt) and verified again.

        Raises:
            SkillStorageError: The stored object does not verify after writing.
        """
        descriptor = SkillArchiveDescriptor(
            archive_ref=archive_ref(workspace, info.skill_name, info.files_hash),
            files_hash=info.files_hash,
            skill_name=info.skill_name,
        )
        await self._ensure_fileset(workspace)
        try:
            await self.verify(descriptor, scratch=archive_path.parent)
        except (NotFoundError, SkillArchiveError, SkillSourceError) as exc:
            if not isinstance(exc, NotFoundError):
                logger.warning(
                    "Existing skill archive failed verification; rewriting it",
                    extra={"archive_ref": descriptor.archive_ref, "error": str(exc)},
                )
        else:
            return descriptor

        path = archive_object_path(info.skill_name, info.files_hash)
        content = await asyncio.to_thread(archive_path.read_bytes)
        try:
            (
                await self.files_client.upload_file(
                    workspace=workspace, name=SKILLS_FILESET_NAME, path=path, content=content
                )
            ).data()
        except NemoTransportError:
            # A lost response may still have written the object; the readback below decides.
            logger.warning("Skill archive upload response was lost; verifying by readback")
        try:
            await self.verify(descriptor, scratch=archive_path.parent)
        except (NotFoundError, SkillArchiveError, SkillSourceError) as exc:
            raise SkillStorageError(
                f"Stored skill archive {descriptor.archive_ref} failed verification: {exc}"
            ) from exc
        return descriptor

    async def verify(self, descriptor: SkillArchiveDescriptor, *, scratch: Path) -> SkillArchiveInfo:
        """Download the stored archive into ``scratch`` and verify it against its descriptor.

        Raises:
            NotFoundError: No object at the descriptor's key.
            SkillArchiveError: Hash mismatch or any format violation.
            SkillSourceError: The object is larger than the archive limit.
        """
        workspace, fileset, path = parse_fileset_dir_ref(descriptor.archive_ref)
        fd, name = await asyncio.to_thread(_private_temp_file, scratch)
        os.close(fd)
        local = Path(name)
        try:
            await self._download(
                self.files_client,
                workspace,
                fileset,
                path,
                local,
                limit=self.limits.max_archive_bytes,
                what=f"Stored archive {descriptor.archive_ref}",
                replace=True,
            )
            info = await asyncio.to_thread(
                verify_skill_archive, local, expected_files_hash=descriptor.files_hash, limits=self.limits
            )
        finally:
            local.unlink(missing_ok=True)
        if info.skill_name != descriptor.skill_name:
            raise SkillArchiveError(
                f"Stored archive {descriptor.archive_ref} holds skill {info.skill_name!r}, not {descriptor.skill_name!r}"
            )
        return info

    async def _ensure_fileset(self, workspace: str) -> None:
        """Create the skills fileset if absent, and refuse one that is not plain platform storage.

        Known gap: the Files service has no way to reserve a fileset for a service, so a workspace
        editor can still pre-create it or overwrite its objects. Every read verifies the archive
        against its pinned hash, so tampering is detected, not silently served. Refusing non-local
        storage at least stops a pre-created fileset from pointing at storage the caller controls.
        """
        fileset = (
            await self.files_client.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=SKILLS_FILESET_NAME, description="Published skill archives (managed by nemo-agents)."
                ),
                exist_ok=True,
            )
        ).data()
        if fileset.storage.type != StorageConfigType.LOCAL:
            raise SkillStorageError(
                f"Fileset {workspace}/{SKILLS_FILESET_NAME} is reserved for skill archives and must use platform "
                f"storage, but it uses {fileset.storage.type!r} storage"
            )

    async def _download(
        self,
        client: AsyncFilesClient,
        workspace: str,
        fileset: str,
        path: str,
        target: Path,
        *,
        limit: int,
        what: str,
        replace: bool = False,
    ) -> int:
        response = await client.download_file(workspace=workspace, name=fileset, path=path)
        mode = "wb" if replace else "xb"
        with target.open(mode) as output:
            writer = _BoundedWriter(output, limit, what)
            async with response.stream(chunk_size=_CHUNK_BYTES) as chunks:
                async for chunk in chunks:
                    writer.write(chunk)
        return writer.size


def _private_temp_file(directory: Path) -> tuple[int, str]:
    return tempfile.mkstemp(prefix=".readback-", dir=directory)
