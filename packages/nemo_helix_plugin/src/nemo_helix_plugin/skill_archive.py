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

from dataclasses import dataclass
from pathlib import Path

from nemo_helix_plugin.agents.skills import SkillFrontmatter

#: Top-level source directories never captured into a skill archive.
EXCLUDED_TOP_LEVEL_DIRS: frozenset[str] = frozenset({"evals"})

SKILL_FILE_NAME = "SKILL.md"


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
    raise NotImplementedError


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
    raise NotImplementedError


def verify_skill_archive(
    archive_path: Path, *, expected_files_hash: str, limits: SkillArchiveLimits = DEFAULT_LIMITS
) -> SkillArchiveInfo:
    """Hash the archive and validate every member against the contract without extracting.

    Raises:
        SkillArchiveHashMismatchError: The bytes do not hash to ``expected_files_hash``.
        SkillArchiveError: Any other contract violation.
    """
    raise NotImplementedError


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
    raise NotImplementedError
