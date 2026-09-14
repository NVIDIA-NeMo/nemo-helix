# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared taskset source contracts and strict materialized-tree identity."""

import hashlib
import os
import stat
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> object: ...


@dataclass(frozen=True, slots=True)
class _SymlinkExpansionEnd:
    path: Path


class TasksetSourceMaterialization(BaseModel):
    """Receipt returned after a taskset source has been materialized."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_uri: str
    materialized_root: Path
    revision_digest: Digest | None
    member_digests: dict[str, Digest]
    materialization_digest: Digest

    @field_validator("source_uri")
    @classmethod
    def _validate_source_uri(cls, value: str) -> str:
        try:
            scheme = urlsplit(value).scheme
        except ValueError as exc:
            raise ValueError("source_uri must be a nonempty absolute URI") from exc
        if not value or not scheme or any(character.isspace() for character in value):
            raise ValueError("source_uri must be a nonempty absolute URI")
        return value

    @field_validator("materialized_root")
    @classmethod
    def _validate_materialized_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("materialized_root must be an absolute path")
        return value

    @field_validator("member_digests", mode="before")
    @classmethod
    def _copy_member_digests(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return dict(value)
        return value

    @field_validator("member_digests")
    @classmethod
    def _validate_member_ids(cls, value: dict[str, Digest]) -> dict[str, Digest]:
        if "" in value:
            raise ValueError("member digest keys must be nonempty member IDs")
        return value


class TasksetSourceAdapter(Protocol):
    schemes: frozenset[str]

    async def materialize(self, source_uri: str, *, destination_root: Path) -> TasksetSourceMaterialization: ...


def digest_harbor_tree(root: Path) -> str:
    """Return a deterministic SHA-256 hex digest of a readable Harbor tree."""
    try:
        root_stat = root.stat()
    except FileNotFoundError:
        raise
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"Harbor tree root is not a directory: {root}")

    resolved_root = root.resolve(strict=True)
    digest = hashlib.sha256()
    root_identity = _directory_identity(resolved_root.stat())
    _hash_directory_contents(
        digest,
        physical_directory=resolved_root,
        logical_directory=Path(),
        root=resolved_root,
        directory_ancestors=frozenset({root_identity}),
    )
    return digest.hexdigest()


def _update_frame(digest: _HashWriter, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big"))
    digest.update(value)


def _hash_entry_header(digest: _HashWriter, entry_type: bytes, logical_path: Path) -> None:
    _update_frame(digest, entry_type)
    _update_frame(digest, os.fsencode(logical_path.as_posix()))


def _hash_directory_contents(
    digest: _HashWriter,
    *,
    physical_directory: Path,
    logical_directory: Path,
    root: Path,
    directory_ancestors: frozenset[tuple[int, int]],
) -> None:
    with os.scandir(physical_directory) as entries:
        sorted_entries = sorted(entries, key=lambda entry: entry.name)
    for entry in sorted_entries:
        physical_path = physical_directory / entry.name
        logical_path = logical_directory / entry.name
        _hash_node(
            digest,
            physical_path=physical_path,
            logical_path=logical_path,
            root=root,
            directory_ancestors=directory_ancestors,
        )


def _hash_node(
    digest: _HashWriter,
    *,
    physical_path: Path,
    logical_path: Path,
    root: Path,
    directory_ancestors: frozenset[tuple[int, int]],
) -> None:
    entry_stat = physical_path.lstat()
    mode = entry_stat.st_mode
    if stat.S_ISREG(mode):
        _hash_regular_file(digest, physical_path, logical_path, entry_stat)
        return
    if stat.S_ISDIR(mode):
        _hash_entry_header(digest, b"directory", logical_path)
        identity = _directory_identity(entry_stat)
        if identity in directory_ancestors:
            raise ValueError(f"cyclic symlink at {logical_path.as_posix()!r}")
        _hash_directory_contents(
            digest,
            physical_directory=physical_path,
            logical_directory=logical_path,
            root=root,
            directory_ancestors=directory_ancestors | {identity},
        )
        return
    if stat.S_ISLNK(mode):
        _hash_symlink(
            digest,
            physical_path=physical_path,
            logical_path=logical_path,
            root=root,
            directory_ancestors=directory_ancestors,
        )
        return
    raise ValueError(f"unsupported filesystem entry at {logical_path.as_posix()!r}")


def _hash_regular_file(
    digest: _HashWriter, physical_path: Path, logical_path: Path, entry_stat: os.stat_result
) -> None:
    _hash_entry_header(digest, b"file", logical_path)
    _update_frame(digest, (entry_stat.st_mode & 0o111).to_bytes(2, byteorder="big"))
    _hash_file_contents(digest, physical_path)


def _hash_file_contents(digest: _HashWriter, physical_path: Path) -> None:
    with physical_path.open("rb") as stream:
        expected_size = os.fstat(stream.fileno()).st_size
        digest.update(expected_size.to_bytes(8, byteorder="big"))
        bytes_read = 0
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            bytes_read += len(chunk)
    if bytes_read != expected_size:
        raise OSError(f"file changed while hashing: {physical_path}")


def _hash_symlink(
    digest: _HashWriter,
    *,
    physical_path: Path,
    logical_path: Path,
    root: Path,
    directory_ancestors: frozenset[tuple[int, int]],
) -> None:
    target = os.readlink(physical_path)
    if os.path.isabs(target):
        raise ValueError(f"absolute symlink at {logical_path.as_posix()!r} is not allowed")

    _hash_entry_header(digest, b"symlink", logical_path)
    _update_frame(digest, os.fsencode(target))
    target_path = _resolve_contained_path(
        physical_path.parent,
        target=target,
        root=root,
        followed_links=frozenset({physical_path}),
    )
    target_stat = target_path.lstat()
    if stat.S_ISREG(target_stat.st_mode):
        _update_frame(digest, b"file-target")
        _update_frame(digest, (target_stat.st_mode & 0o111).to_bytes(2, byteorder="big"))
        _hash_file_contents(digest, target_path)
        return
    if stat.S_ISDIR(target_stat.st_mode):
        _update_frame(digest, b"directory-target")
        identity = _directory_identity(target_stat)
        if identity in directory_ancestors:
            raise ValueError(f"cyclic symlink at {logical_path.as_posix()!r}")
        _hash_directory_contents(
            digest,
            physical_directory=target_path,
            logical_directory=logical_path,
            root=root,
            directory_ancestors=directory_ancestors | {identity},
        )
        return
    raise ValueError(f"unsupported filesystem entry targeted by symlink at {logical_path.as_posix()!r}")


def _resolve_contained_path(
    start: Path,
    *,
    target: str,
    root: Path,
    followed_links: frozenset[Path],
) -> Path:
    if start != root and root not in start.parents:
        raise ValueError(f"symlink target escapes root: {start}")

    current = start
    pending: deque[str | _SymlinkExpansionEnd] = deque(Path(target).parts)
    active_links = set(followed_links)
    while pending:
        part = pending.popleft()
        if isinstance(part, _SymlinkExpansionEnd):
            active_links.remove(part.path)
            continue
        if part == ".":
            continue
        if part == "..":
            if current == root:
                raise ValueError(f"symlink target escapes root: {target}")
            current = current.parent
            continue

        candidate = current / part
        entry_stat = candidate.lstat()
        if stat.S_ISLNK(entry_stat.st_mode):
            if candidate in active_links:
                raise ValueError(f"cyclic symlink at {candidate.relative_to(root).as_posix()!r}")
            nested_target = os.readlink(candidate)
            if os.path.isabs(nested_target):
                raise ValueError(f"absolute symlink at {candidate.relative_to(root).as_posix()!r} is not allowed")
            active_links.add(candidate)
            pending.appendleft(_SymlinkExpansionEnd(candidate))
            pending.extendleft(reversed(Path(nested_target).parts))
            continue

        current = candidate
        while pending and isinstance(pending[0], _SymlinkExpansionEnd):
            expansion_end = pending.popleft()
            assert isinstance(expansion_end, _SymlinkExpansionEnd)
            active_links.remove(expansion_end.path)
        if pending and not stat.S_ISDIR(entry_stat.st_mode):
            raise NotADirectoryError(f"symlink target component is not a directory: {candidate}")
    return current


def _directory_identity(entry_stat: os.stat_result) -> tuple[int, int]:
    return entry_stat.st_dev, entry_stat.st_ino
