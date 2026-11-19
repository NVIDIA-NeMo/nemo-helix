# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills a Harbor run injects into every trial: validation and a content digest.

Harbor stages each injected skill directory under ``<skills_dir>/<basename>`` and uploads the
whole tree. So the basename is the skill's name, and the digest covers everything Harbor would
upload: every relative path, the owner-execute bit of each file, and each file's bytes.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from pathlib import Path

#: The file Harbor requires at the root of a skill directory.
SKILL_FILE_NAME = "SKILL.md"
_CHUNK_BYTES = 1 << 20


def validate_skill_dirs(skills: Sequence[Path]) -> None:
    """Reject skill paths Harbor would misread, or that would collide once staged.

    Each path must be absolute (Harbor reads a relative ``org/name`` as a git source), must be a
    real directory rather than a symlink to one (Harbor names a skill by its resolved basename),
    and must hold a regular ``SKILL.md`` at its root (otherwise Harbor treats it as a collection
    and expands its children). Two skills with the same basename would be staged to the same
    directory, and Harbor keeps only the last.
    """
    seen: dict[str, Path] = {}
    for path in skills:
        if not path.is_absolute():
            raise ValueError(f"skill path must be absolute: {path}")
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"skill path must be an existing directory, not a symlink: {path}")
        skill_file = path / SKILL_FILE_NAME
        if skill_file.is_symlink() or not skill_file.is_file():
            raise ValueError(f"skill directory {path} has no regular {SKILL_FILE_NAME} at its root")
        previous = seen.get(path.name)
        if previous is not None:
            raise ValueError(f"duplicate skill name {path.name!r}: {previous} and {path}")
        seen[path.name] = path


def _feed(digest: hashlib._Hash, label: bytes, payload: bytes) -> None:
    """Append a length-framed field, so two different trees never produce the same byte stream."""
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def skill_tree_digest(root: Path) -> str:
    """Lowercase hex SHA-256 over a skill tree: sorted relative paths, executable bit, file bytes.

    Only the owner-execute bit of files counts: read/write bits follow the umask, while ``+x`` on a
    script changes what the agent can run. Symlinks and special files are rejected rather than
    followed, so the digest always describes the bytes Harbor uploads.
    """
    digest = hashlib.sha256()
    entries: list[Path] = []
    # os.walk rather than rglob: it never follows a directory symlink, on every supported Python.
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        entries.extend(Path(directory, name) for name in (*dirnames, *filenames))
    for path in sorted(entries, key=lambda entry: entry.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"skill {root.name!r} contains a symlink: {relative}")
        _feed(digest, b"path", relative.encode("utf-8"))
        if stat.S_ISDIR(mode):
            _feed(digest, b"dir", b"")
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"skill {root.name!r} contains a special file: {relative}")
        _feed(digest, b"mode", b"x" if mode & stat.S_IXUSR else b"-")
        content = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                content.update(chunk)
        _feed(digest, b"file", content.digest())
    return digest.hexdigest()


def skill_digests(skills: Sequence[Path]) -> list[dict[str, str]]:
    """Validate ``skills`` and describe each as ``{"name", "digest"}``, in the order given."""
    validate_skill_dirs(skills)
    return [{"name": path.name, "digest": skill_tree_digest(path)} for path in skills]
