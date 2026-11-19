# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, bounded archives of agent skill bundles.

One module owns the archive format so every boundary applies the same rules: the agents plugin
builds an archive when it publishes a skill revision, and the evals worker verifies and extracts
it before injecting the skill into a Harbor trial.

Format contract (``files_hash`` covers exactly these bytes, so changing any rule changes digests):

- A gzip-free POSIX tar (``tarfile.PAX_FORMAT``) with a single root directory named after the
  skill (the frontmatter ``name``).
- Members are sorted by normalised relative path; directories precede their contents.
- Only regular files and directories. Symlinks, hard links, devices and FIFOs are rejected.
- Paths are relative, NFC-normalised, contain no ``..`` or empty segments, and are unique after
  normalisation (case-sensitive).
- Modes are normalised: directories ``0o755``; files ``0o755`` if any execute bit was set in the
  source, otherwise ``0o644``. The executable bit is therefore part of the digest.
- uid/gid ``0``, uname/gname empty, mtime ``0``; no PAX extended headers beyond what long paths need.
- A top-level ``evals/`` directory in the source is excluded: it is test material (rubrics,
  expected answers) and must never reach the agent.
- ``SKILL.md`` must exist at the root, be a regular file, and parse as frontmatter with a valid
  ``name`` and ``description``.

``files_hash`` = lowercase hex SHA-256 of the complete archive bytes.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import shutil
import stat
import tarfile
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import yaml
from nemo_helix_plugin.agents.skills import SKILL_NAME_MAX_LENGTH, SKILL_NAME_PATTERN, SkillFrontmatter
from pydantic import JsonValue, ValidationError

#: Top-level source directories never captured into a skill archive.
EXCLUDED_TOP_LEVEL_DIRS: frozenset[str] = frozenset({"evals"})

SKILL_FILE_NAME = "SKILL.md"

#: Upper bound on the YAML frontmatter block (between the ``---`` lines) of a ``SKILL.md``.
MAX_FRONTMATTER_BYTES = 64 * 1024

_DIR_MODE = 0o755
_EXEC_FILE_MODE = 0o755
_FILE_MODE = 0o644
_CHUNK_BYTES = 1024 * 1024
#: Largest single PAX extended header accepted; canonical archives only carry a ``path`` record.
_MAX_PAX_HEADER_BYTES = 64 * 1024
_FRONTMATTER_DELIMITER = "---"
_SKILL_NAME_RE = re.compile(SKILL_NAME_PATTERN)


@dataclass(frozen=True)
class SkillArchiveLimits:
    """Bounds applied when building, downloading and extracting an archive."""

    max_archive_bytes: int = 32 * 1024 * 1024
    max_total_payload_bytes: int = 128 * 1024 * 1024
    max_file_bytes: int = 32 * 1024 * 1024
    max_entries: int = 4096
    max_path_length: int = 512


DEFAULT_LIMITS = SkillArchiveLimits()


@dataclass(frozen=True)
class SkillArchiveInfo:
    """What building (or verifying) an archive established about it."""

    files_hash: str
    skill_name: str
    frontmatter: SkillFrontmatter
    archive_bytes: int
    payload_bytes: int
    entry_count: int


class SkillArchiveError(ValueError):
    """Base class: the bundle or archive violates the format contract."""


class SkillArchiveLimitError(SkillArchiveError):
    """A size, entry-count or path-length limit was exceeded."""


class SkillArchivePathError(SkillArchiveError):
    """A member is not a regular file or directory, escapes the root, or collides after normalisation."""


class SkillFrontmatterError(SkillArchiveError):
    """``SKILL.md`` is missing, not a regular file, or its frontmatter is invalid."""


class SkillArchiveHashMismatchError(SkillArchiveError):
    """The archive bytes do not hash to the expected ``files_hash``."""


def parse_skill_frontmatter(skill_md: str) -> SkillFrontmatter:
    """Parse the YAML frontmatter at the top of a ``SKILL.md``.

    ``metadata.tags`` becomes :attr:`SkillFrontmatter.tags`; every other key is kept in ``extra``.

    Raises:
        SkillFrontmatterError: No frontmatter block, invalid YAML, or a missing/invalid ``name`` or
            ``description``.
    """
    block = _frontmatter_block(skill_md)
    try:
        loaded = yaml.load(block, Loader=_FrontmatterLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter is not valid YAML: {exc}") from None
    if not isinstance(loaded, dict):
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter must be a YAML mapping")
    try:
        fields = _json_mapping(loaded)
    except RecursionError:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter is nested too deeply") from None

    name = fields.pop("name", None)
    description = fields.pop("description", None)
    if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name) or len(name) > SKILL_NAME_MAX_LENGTH:
        raise SkillFrontmatterError(
            f"{SKILL_FILE_NAME} frontmatter 'name' must start with a letter and use lowercase alphanumerics separated by single hyphens "
            f"(2-{SKILL_NAME_MAX_LENGTH} characters)"
        )
    if not isinstance(description, str) or not description.strip():
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter 'description' must be a non-empty string")

    tags: list[str] = []
    metadata = fields.get("metadata")
    if isinstance(metadata, dict) and "tags" in metadata:
        raw_tags = metadata["tags"]
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter 'metadata.tags' must be a list of strings")
        tags = [tag for tag in raw_tags if isinstance(tag, str)]
        rest = {key: value for key, value in metadata.items() if key != "tags"}
        if rest:
            fields["metadata"] = rest
        else:
            del fields["metadata"]
    try:
        return SkillFrontmatter(name=name, description=description, tags=tags, extra=fields)
    except ValidationError as exc:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter is invalid: {exc}") from None


def build_skill_archive(
    source_dir: Path, archive_path: Path, *, limits: SkillArchiveLimits = DEFAULT_LIMITS
) -> SkillArchiveInfo:
    """Capture ``source_dir`` (the directory holding ``SKILL.md``) into a deterministic archive.

    Walks without following symlinks, applies the format contract, and writes ``archive_path``
    exclusively (it must not exist). The root directory in the archive is the frontmatter name,
    whatever the source directory is called.

    Raises:
        SkillArchiveError: Any contract violation; nothing is left at ``archive_path``.
    """
    try:
        root_stat = os.lstat(source_dir)
    except FileNotFoundError:
        raise SkillArchivePathError(f"Skill source directory {source_dir} does not exist") from None
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SkillArchivePathError(f"Skill source {source_dir} is not a directory")

    frontmatter = parse_skill_frontmatter(_read_source_skill_md(source_dir, limits))
    entries = _inventory_source(source_dir, frontmatter.name, limits)

    # Opened exclusively: an existing path is never ours to delete, so FileExistsError propagates as-is.
    with archive_path.open("xb") as raw:
        try:
            _write_archive(raw, entries, limits)
        except BaseException:
            raw.close()
            archive_path.unlink(missing_ok=True)
            raise
    try:
        # Re-read what was written through the same checks the worker applies, so a build can never
        # produce an archive that verification would refuse.
        info = _inspect_archive(_read_archive_bytes(archive_path, limits), limits)
        if info.skill_name != frontmatter.name:
            raise SkillFrontmatterError(f"{SKILL_FILE_NAME} changed while the archive was being built")
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    return info


def verify_skill_archive(
    archive_path: Path, *, expected_files_hash: str, limits: SkillArchiveLimits = DEFAULT_LIMITS
) -> SkillArchiveInfo:
    """Hash the archive and validate every member against the contract without extracting.

    Raises:
        SkillArchiveHashMismatchError: The bytes do not hash to ``expected_files_hash``.
        SkillArchiveError: Any other contract violation.
    """
    data = _read_archive_bytes(archive_path, limits)
    _check_hash(data, expected_files_hash)
    return _inspect_archive(data, limits)


def extract_skill_archive(
    archive_path: Path,
    dest_parent: Path,
    *,
    expected_files_hash: str,
    limits: SkillArchiveLimits = DEFAULT_LIMITS,
) -> Path:
    """Verify, then extract into a fresh ``dest_parent / <skill_name>`` and return that path.

    Verification happens before any byte is written. ``dest_parent`` must exist; the skill
    directory must not. Writes are exclusive (``O_EXCL``), modes come from the archive's normalised
    values, and the running payload total is enforced while streaming.

    Raises:
        SkillArchiveHashMismatchError: The bytes do not hash to ``expected_files_hash``.
        SkillArchiveError: Any other contract violation; a partial extraction is removed.
    """
    # Both passes read the same in-memory bytes, so what is extracted is exactly what was hashed.
    data = _read_archive_bytes(archive_path, limits)
    _check_hash(data, expected_files_hash)
    info = _inspect_archive(data, limits)

    if not dest_parent.is_dir():
        raise SkillArchivePathError(f"Extraction parent {dest_parent} is not an existing directory")
    skill_dir = dest_parent / info.skill_name
    try:
        # Private until complete; the final directory modes are applied last.
        skill_dir.mkdir(mode=0o700)
    except FileExistsError:
        raise SkillArchivePathError(f"Extraction target {skill_dir} already exists") from None
    try:
        _extract_members(data, skill_dir, limits)
    except BaseException:
        shutil.rmtree(skill_dir, ignore_errors=True)
        raise
    return skill_dir


# --------------------------------------------------------------------------------------------------
# Frontmatter
# --------------------------------------------------------------------------------------------------


class _FrontmatterLoader(yaml.SafeLoader):
    """Safe loader that refuses aliases (no expansion bombs) and keeps timestamps as strings."""

    def compose_node(self, parent: yaml.Node | None, index: object) -> yaml.Node | None:
        if self.check_event(yaml.AliasEvent):
            raise yaml.YAMLError("YAML aliases are not allowed in skill frontmatter")
        return super().compose_node(parent, index)


_FrontmatterLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _frontmatter_block(skill_md: str) -> str:
    """Return the YAML text between the leading ``---`` line and the next ``---`` line."""
    # Only the bounded prefix can hold a valid block (opening line, at most MAX_FRONTMATTER_BYTES of
    # YAML, closing line), so a huge body is never split into lines. Characters are at least one
    # UTF-8 byte, so this character cut never drops bytes the byte limit below would admit.
    text = skill_md.removeprefix("﻿")[: MAX_FRONTMATTER_BYTES + 2 * (len(_FRONTMATTER_DELIMITER) + 2)]
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != _FRONTMATTER_DELIMITER:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} must start with a '---' frontmatter block")
    size = 0
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip() == _FRONTMATTER_DELIMITER:
            return "".join(lines[1:index])
        size += len(line.encode("utf-8", errors="surrogatepass"))
        if size > MAX_FRONTMATTER_BYTES:
            raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
    raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter block is not closed with '---'")


def _json_value(value: object) -> JsonValue:
    """Convert a YAML-loaded value to JSON, refusing anything JSON cannot represent."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return _json_mapping(value)
    raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter contains a non-JSON value ({type(value).__name__})")


def _json_mapping(value: dict[object, object]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SkillFrontmatterError(f"{SKILL_FILE_NAME} frontmatter keys must be strings")
        result[key] = _json_value(item)
    return result


def _decode_skill_md(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} is not valid UTF-8") from None


# --------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------


def _utf8_length(path: str) -> int:
    try:
        return len(path.encode("utf-8"))
    except UnicodeEncodeError:
        raise SkillArchivePathError(f"Path {path!r} is not valid UTF-8") from None


def _check_segment(segment: str) -> None:
    if segment in {"", ".", ".."} or "/" in segment or "\x00" in segment:
        raise SkillArchivePathError(f"Invalid path segment {segment!r}")


def _check_path_length(path: str, limits: SkillArchiveLimits) -> None:
    if _utf8_length(path) > limits.max_path_length:
        raise SkillArchiveLimitError(f"Path {path[:64]!r}... exceeds {limits.max_path_length} bytes")


def _normalise_segment(raw: str) -> str:
    """NFC-normalise one source name and reject names that cannot be a safe path segment."""
    _utf8_length(raw)
    segment = unicodedata.normalize("NFC", raw)
    _check_segment(segment)
    return segment


def _validate_member_path(name: str, limits: SkillArchiveLimits) -> list[str]:
    """Validate an archive member path and return its segments."""
    _check_path_length(name, limits)
    if name.startswith("/"):
        raise SkillArchivePathError(f"Archive member {name!r} is absolute")
    if unicodedata.normalize("NFC", name) != name:
        raise SkillArchivePathError(f"Archive member {name!r} is not NFC-normalised")
    segments = name.split("/")
    for segment in segments:
        _check_segment(segment)
    return segments


# --------------------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _SourceEntry:
    archive_name: str
    source: Path
    is_dir: bool
    mode: int
    size: int


def _open_regular_file(path: Path, expected: os.stat_result) -> IO[bytes]:
    """Open ``path`` without following a link and check it is still the file that was inventoried."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise SkillArchivePathError(f"{path} changed while the skill was being captured")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _read_source_skill_md(source_dir: Path, limits: SkillArchiveLimits) -> str:
    path = source_dir / SKILL_FILE_NAME
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} is missing from {source_dir}") from None
    if not stat.S_ISREG(info.st_mode):
        raise SkillFrontmatterError(f"{SKILL_FILE_NAME} in {source_dir} is not a regular file")
    with _open_regular_file(path, info) as source:
        raw = source.read(limits.max_file_bytes + 1)
    if len(raw) > limits.max_file_bytes:
        raise SkillArchiveLimitError(f"{SKILL_FILE_NAME} exceeds {limits.max_file_bytes} bytes")
    return _decode_skill_md(raw)


def _inventory_source(source_dir: Path, skill_name: str, limits: SkillArchiveLimits) -> list[_SourceEntry]:
    """Walk ``source_dir`` without following links and return the archive entries in archive order."""
    entries = [_SourceEntry(skill_name, source_dir, is_dir=True, mode=_DIR_MODE, size=0)]
    seen: set[str] = {skill_name}
    payload = 0
    pending: list[tuple[Path, str]] = [(source_dir, skill_name)]
    while pending:
        directory, archive_dir = pending.pop()
        # Iterate rather than materialize, so an enormous directory hits the entry limit as it
        # streams in instead of being listed into memory first.
        with os.scandir(directory) as iterator:
            for child in iterator:
                if (
                    directory == source_dir
                    and child.name in EXCLUDED_TOP_LEVEL_DIRS
                    and child.is_dir(follow_symlinks=False)
                ):
                    continue
                name = f"{archive_dir}/{_normalise_segment(child.name)}"
                _check_path_length(name, limits)
                if name in seen:
                    raise SkillArchivePathError(f"Path {name!r} collides with another entry after normalisation")
                seen.add(name)
                if len(entries) >= limits.max_entries:
                    raise SkillArchiveLimitError(f"Skill has more than {limits.max_entries} entries")
                info = child.stat(follow_symlinks=False)
                path = Path(child.path)
                if stat.S_ISDIR(info.st_mode):
                    entries.append(_SourceEntry(name, path, is_dir=True, mode=_DIR_MODE, size=0))
                    pending.append((path, name))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_size > limits.max_file_bytes:
                        raise SkillArchiveLimitError(f"File {name!r} exceeds {limits.max_file_bytes} bytes")
                    payload += info.st_size
                    if payload > limits.max_total_payload_bytes:
                        raise SkillArchiveLimitError(f"Skill payload exceeds {limits.max_total_payload_bytes} bytes")
                    mode = _EXEC_FILE_MODE if info.st_mode & 0o111 else _FILE_MODE
                    entries.append(_SourceEntry(name, path, is_dir=False, mode=mode, size=info.st_size))
                elif stat.S_ISLNK(info.st_mode):
                    raise SkillArchivePathError(f"Symlinks are not allowed in a skill: {name!r}")
                else:
                    raise SkillArchivePathError(f"Only regular files and directories are allowed in a skill: {name!r}")
    # A path sorts before every path it prefixes, so each directory precedes its contents.
    entries.sort(key=lambda entry: entry.archive_name)
    return entries


class _BoundedWriter:
    """File-like sink that refuses to grow past the archive limit."""

    def __init__(self, raw: IO[bytes], limit: int) -> None:
        self._raw = raw
        self._limit = limit
        self._size = 0

    def write(self, data: bytes) -> int:
        self._size += len(data)
        if self._size > self._limit:
            raise SkillArchiveLimitError(f"Archive exceeds {self._limit} bytes")
        return self._raw.write(data)

    def tell(self) -> int:
        return self._size


def _write_archive(raw: IO[bytes], entries: list[_SourceEntry], limits: SkillArchiveLimits) -> None:
    sink = _BoundedWriter(raw, limits.max_archive_bytes)
    with tarfile.TarFile(fileobj=sink, mode="w", format=tarfile.PAX_FORMAT) as archive:  # ty: ignore[invalid-argument-type] -- duck-typed sink
        for entry in entries:
            header = tarfile.TarInfo(entry.archive_name)
            header.mode = entry.mode
            header.uid = header.gid = 0
            header.uname = header.gname = ""
            header.mtime = 0
            if entry.is_dir:
                header.type = tarfile.DIRTYPE
                archive.addfile(header)
                continue
            header.type = tarfile.REGTYPE
            header.size = entry.size
            info = os.lstat(entry.source)
            if not stat.S_ISREG(info.st_mode) or info.st_size != entry.size:
                raise SkillArchivePathError(f"{entry.source} changed while the skill was being captured")
            with _open_regular_file(entry.source, info) as source:
                archive.addfile(header, source)
                if source.read(1):
                    raise SkillArchivePathError(f"{entry.source} changed while the skill was being captured")


# --------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------


def _read_archive_bytes(archive_path: Path, limits: SkillArchiveLimits) -> bytes:
    """Read the whole archive, refusing anything larger than ``max_archive_bytes``."""
    try:
        fd = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        raise SkillArchiveError(f"Archive {archive_path} does not exist") from None
    except OSError as exc:
        raise SkillArchivePathError(f"Archive {archive_path} cannot be opened as a regular file: {exc}") from None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise SkillArchivePathError(f"Archive {archive_path} is not a regular file")
        if info.st_size > limits.max_archive_bytes:
            raise SkillArchiveLimitError(f"Archive exceeds {limits.max_archive_bytes} bytes")
        data = stream.read(limits.max_archive_bytes + 1)
    if len(data) > limits.max_archive_bytes:
        raise SkillArchiveLimitError(f"Archive exceeds {limits.max_archive_bytes} bytes")
    return data


def _check_hash(data: bytes, expected_files_hash: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_files_hash:
        raise SkillArchiveHashMismatchError(f"Archive hashes to {actual}, expected {expected_files_hash}")


class _SafeTarInfo(tarfile.TarInfo):
    """Refuse non-canonical record types before tarfile allocates or recurses on them.

    Only regular files, directories and a single bounded PAX extended header per member (for long
    or non-ASCII paths) are accepted; GNU long names, sparse maps, global headers, links and
    devices are refused while parsing.
    """

    def _proc_gnusparse_00(self, next, raw_headers):
        raise SkillArchivePathError("Sparse tar members are not allowed")

    def _proc_gnusparse_01(self, next, pax_headers):
        raise SkillArchivePathError("Sparse tar members are not allowed")

    def _proc_gnusparse_10(self, next, pax_headers, tarfile):
        raise SkillArchivePathError("Sparse tar members are not allowed")

    def _proc_member(self, tarfile_: tarfile.TarFile) -> tarfile.TarInfo:
        in_pax = getattr(tarfile_, "_skill_in_pax", False)
        if self.type == tarfile.XHDTYPE:
            if in_pax:
                raise SkillArchivePathError("Chained tar extended headers are not allowed")
            if self.size > _MAX_PAX_HEADER_BYTES:
                raise SkillArchiveLimitError("Tar extended header is too large")
            setattr(tarfile_, "_skill_in_pax", True)
            try:
                return super()._proc_member(tarfile_)  # ty: ignore[unresolved-attribute] -- stdlib private hook
            finally:
                setattr(tarfile_, "_skill_in_pax", False)
        if self.type not in (tarfile.REGTYPE, tarfile.DIRTYPE):
            raise SkillArchivePathError(
                f"Only regular files and directories are allowed in a skill archive (tar type {self.type!r})"
            )
        return super()._proc_member(tarfile_)  # ty: ignore[unresolved-attribute] -- stdlib private hook


def _iter_members(data: bytes) -> Iterator[tuple[tarfile.TarFile, tarfile.TarInfo]]:
    """Yield members of an in-memory tar, then require only zero padding after the last one."""
    try:
        with tarfile.TarFile(fileobj=io.BytesIO(data), mode="r", tarinfo=_SafeTarInfo) as archive:
            for member in archive:
                yield archive, member
            end = archive.offset
    except tarfile.TarError as exc:
        raise SkillArchiveError(f"Archive is not a valid tar: {exc}") from None
    trailer = data[end:]
    if len(trailer) < 2 * tarfile.BLOCKSIZE or len(data) % tarfile.RECORDSIZE or trailer.count(0) != len(trailer):
        raise SkillArchiveError("Archive is truncated or has data after the end-of-archive marker")


def _member_name(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return member.name.removesuffix("/")
    if member.name.endswith("/"):
        raise SkillArchivePathError(f"File member {member.name!r} has a trailing slash")
    return member.name


def _check_header(member: tarfile.TarInfo, name: str) -> None:
    """Require the normalised header fields the format contract fixes."""
    if member.isdir():
        expected_modes: tuple[int, ...] = (_DIR_MODE,)
        if member.size:
            raise SkillArchivePathError(f"Directory {name!r} has a payload")
    else:
        expected_modes = (_FILE_MODE, _EXEC_FILE_MODE)
    if member.mode not in expected_modes:
        raise SkillArchivePathError(f"Member {name!r} has a non-normalised mode {oct(member.mode)}")
    if (member.uid, member.gid, member.uname, member.gname, member.mtime) != (0, 0, "", "", 0):
        raise SkillArchivePathError(f"Member {name!r} has non-normalised ownership or mtime")
    if member.linkname or member.sparse is not None:
        raise SkillArchivePathError(f"Member {name!r} carries link or sparse metadata")
    if member.pax_headers.keys() - {"path"}:
        raise SkillArchivePathError(f"Member {name!r} carries unsupported PAX headers")


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise SkillArchivePathError(f"Member {member.name!r} has no readable payload")
    with source:
        content = source.read(member.size + 1)
    if len(content) != member.size:
        raise SkillArchiveError(f"Member {member.name!r} is truncated")
    return content


def _inspect_archive(data: bytes, limits: SkillArchiveLimits) -> SkillArchiveInfo:
    """Validate every member of ``data`` against the format contract without writing anything."""
    root: str | None = None
    previous: str | None = None
    directories: set[str] = set()
    entry_count = 0
    payload = 0
    frontmatter: SkillFrontmatter | None = None
    for archive, member in _iter_members(data):
        entry_count += 1
        if entry_count > limits.max_entries:
            raise SkillArchiveLimitError(f"Archive has more than {limits.max_entries} entries")
        name = _member_name(member)
        segments = _validate_member_path(name, limits)
        _check_header(member, name)
        # Strictly increasing names give both the canonical order and uniqueness.
        if previous is not None and name <= previous:
            raise SkillArchivePathError(f"Member {name!r} is duplicated or out of order")
        previous = name

        if root is None:
            if len(segments) != 1 or not member.isdir():
                raise SkillArchivePathError("Archive must start with a single root directory")
            if not _SKILL_NAME_RE.fullmatch(name) or len(name) > SKILL_NAME_MAX_LENGTH:
                raise SkillArchivePathError(f"Archive root {name!r} is not a valid skill name")
            root = name
        elif segments[0] != root or len(segments) < 2:
            raise SkillArchivePathError(f"Member {name!r} is outside the single root {root!r}")
        elif "/".join(segments[:-1]) not in directories:
            raise SkillArchivePathError(f"Member {name!r} has no parent directory entry")
        elif len(segments) == 2 and segments[1] in EXCLUDED_TOP_LEVEL_DIRS and member.isdir():
            raise SkillArchivePathError(f"Archive contains the excluded top-level directory {segments[1]!r}")

        if member.isdir():
            directories.add(name)
            continue
        if member.size > limits.max_file_bytes:
            raise SkillArchiveLimitError(f"File {name!r} exceeds {limits.max_file_bytes} bytes")
        payload += member.size
        if payload > limits.max_total_payload_bytes:
            raise SkillArchiveLimitError(f"Archive payload exceeds {limits.max_total_payload_bytes} bytes")
        if len(segments) == 2 and segments[1] == SKILL_FILE_NAME:
            frontmatter = parse_skill_frontmatter(_decode_skill_md(_read_member(archive, member)))

    if root is None:
        raise SkillArchivePathError("Archive is empty")
    if frontmatter is None:
        raise SkillFrontmatterError(f"Archive has no regular {SKILL_FILE_NAME} at its root")
    if frontmatter.name != root:
        raise SkillFrontmatterError(
            f"Archive root {root!r} does not match the {SKILL_FILE_NAME} name {frontmatter.name!r}"
        )
    return SkillArchiveInfo(
        files_hash=hashlib.sha256(data).hexdigest(),
        skill_name=root,
        frontmatter=frontmatter,
        archive_bytes=len(data),
        payload_bytes=payload,
        entry_count=entry_count,
    )


# --------------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------------


def _write_member(fd: int, source: IO[bytes], size: int) -> int:
    """Copy exactly ``size`` bytes from ``source`` to ``fd`` and return how many were written."""
    written = 0
    while written < size:
        chunk = source.read(min(_CHUNK_BYTES, size - written))
        if not chunk:
            raise SkillArchiveError("Archive member is truncated")
        view = memoryview(chunk)
        while view:
            view = view[os.write(fd, view) :]
        written += len(chunk)
    return written


def _extract_members(data: bytes, skill_dir: Path, limits: SkillArchiveLimits) -> None:
    """Write the already-verified members of ``data`` under ``skill_dir``, which the caller created."""
    parent = skill_dir.parent
    directories: list[Path] = [skill_dir]
    total = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    for archive, member in _iter_members(data):
        name = _member_name(member)
        if name == skill_dir.name:
            continue
        path = parent / name
        try:
            if member.isdir():
                path.mkdir(mode=0o700)
                directories.append(path)
                continue
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            # For example two names differing only in case on a case-insensitive filesystem.
            raise SkillArchivePathError(f"Extraction path {name!r} already exists") from None
        try:
            source = archive.extractfile(member)
            if source is None:
                raise SkillArchivePathError(f"Member {name!r} has no readable payload")
            with source:
                total += _write_member(fd, source, member.size)
            if total > limits.max_total_payload_bytes:
                raise SkillArchiveLimitError(f"Archive payload exceeds {limits.max_total_payload_bytes} bytes")
            os.fchmod(fd, member.mode)
        finally:
            os.close(fd)
    for directory in directories:
        directory.chmod(_DIR_MODE)
