# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Harbor package validation and stable capture, independent of platform clients."""

import os
import re
import shutil
import stat
import tomllib
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Markdown instruction files may carry repository license comments. Those are file metadata, not
# agent-facing task instructions.
_SPDX_HTML_COMMENT_RE = re.compile(r"<!--\s*SPDX-(?:FileCopyrightText|License-Identifier):[^>]*-->\s*")

# Filename that marks a directory as a Harbor task, and the template dir to skip.
TASK_CONFIG_FILENAME = "task.toml"
TASK_TEMPLATE_DIRNAME = "task_template"

CHUNK_BYTES = 1024 * 1024
MAX_ENTRIES = 100_000
MAX_FILE_BYTES = 4 * 1024**3
MAX_PAYLOAD_BYTES = 16 * 1024**3
MAX_CONFIG_BYTES = 1024**2
MAX_IGNORE_BYTES = 256 * 1024
MAX_INSTRUCTION_BYTES = 1024**2
MAX_METADATA_BYTES = 16 * 1024**2
MAX_STEPS = 128


def validate_archive_path(value: str) -> str:
    """Require an unambiguous relative POSIX path before normalization."""
    if (
        not value
        or len(value.encode("utf-8")) > 4096
        or any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in value.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 or char in "\\%?#:" for char in value)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError(f"Unsafe or ambiguous archive path: {value!r}")
    return value


def _strip_leading_spdx_html_comments(text: str) -> str:
    """Remove leading SPDX HTML comments from Markdown prompt content."""
    position = 0
    while match := _SPDX_HTML_COMMENT_RE.match(text, position):
        position = match.end()
    return text[position:]


def normalize_harbor_instruction(instruction: str | None, *, task_id: str) -> str:
    """Use identical scoring inputs for archived and stored Harbor definitions.

    Args:
        instruction: Original root instruction, or None when absent.
        task_id: Task identity used for absent instructions and validation errors.

    Returns:
        The stripped instruction, or task_id when no instruction was supplied.

    Raises:
        ValueError: If an explicit instruction contains only SPDX comments or whitespace.
    """
    if instruction is None:
        return task_id
    normalized = _strip_leading_spdx_html_comments(instruction).strip()
    if not normalized:
        raise ValueError(f"Harbor task {task_id!r} instruction is empty after removing SPDX comments and whitespace")
    return normalized


@dataclass(frozen=True)
class NativeTask:
    """Validated metadata extracted from a native Harbor task package.

    ``task_id`` is the native execution identity; ``instruction`` may be absent
    for a multistep package. ``config`` contains the parsed package configuration.
    This value carries neither Entity Store identity nor a runtime task instance.
    """

    task_id: str
    instruction: str | None
    config: dict[str, Any]


def _read_metadata(path: Path, limit: int) -> bytes:
    """Read a regular metadata file without following symlinks or exceeding the byte limit."""
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Metadata must be a regular file: {path.name}")
        with path.open("rb") as stream:
            value = stream.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"Missing or unreadable metadata: {path.name}") from exc
    if len(value) > limit:
        raise ValueError(f"Metadata exceeds byte limit: {path.name}")
    return value


def validate_native_task_inputs(root: Path) -> NativeTask:
    """Validate bounded task metadata before loading a native Harbor task.

    Algorithm:
        - Read ``task.toml`` and instruction files with per-file and aggregate limits.
        - Validate step names and directories without following paths outside the task root.
        - Let Harbor validate the staged task, then normalize its root instruction.

    Args:
        root: Captured task directory containing ``task.toml`` and its environment.

    Returns:
        Native task identity, optional root instruction, and parsed configuration.

    Raises:
        ValueError: Required metadata is unsafe, malformed, missing, or exceeds a limit.
    """
    config_bytes = _read_metadata(root / TASK_CONFIG_FILENAME, MAX_CONFIG_BYTES)
    config = tomllib.loads(config_bytes.decode("utf-8"))
    steps = config.get("steps", [])
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise ValueError("Invalid or excessive Harbor steps")
    names: set[str] = set()
    instructions = [root / "instruction.md"] if os.path.lexists(root / "instruction.md") else []
    for step in steps:
        name = step.get("name") if isinstance(step, dict) else None
        if not isinstance(name, str) or "/" in name:
            raise ValueError("Step name must be a single directory component")
        validate_archive_path(name)
        if name.casefold() in names:
            raise ValueError("Duplicate step name")
        names.add(name.casefold())
        directory = root / "steps" / name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Missing packaged step directory")
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("Step escapes task root")
        instructions.append(directory / "instruction.md")
    total = len(config_bytes)
    ignore = root / ".gitignore"
    if os.path.lexists(ignore):
        total += len(_read_metadata(ignore, MAX_IGNORE_BYTES))
    instruction = None
    for path in instructions:
        value = _read_metadata(path, MAX_INSTRUCTION_BYTES)
        total += len(value)
        if total > MAX_METADATA_BYTES:
            raise ValueError("Task metadata exceeds aggregate byte limit")
        if path == root / "instruction.md":
            instruction = value.decode("utf-8")
    if total > MAX_METADATA_BYTES:
        raise ValueError("Task metadata exceeds aggregate byte limit")
    from harbor.models.task.task import Task

    try:
        task = Task(root)
        if not (root / "environment").is_dir() or not Task.is_valid_dir(root):
            raise ValueError("Invalid native Harbor task")
    except OSError as exc:
        raise ValueError("Missing or unreadable native Harbor artifact") from exc
    normalize_harbor_instruction(instruction, task_id=task.name)
    return NativeTask(task.name, instruction, config)


def remove_owned_tree(root: Path) -> None:
    """Remove only invocation-owned staging, including restrictive directory modes."""
    if not root.exists():
        return
    root.chmod(stat.S_IMODE(root.stat().st_mode) | 0o700)
    for directory, _, _ in os.walk(root, topdown=True, followlinks=False):
        os.chmod(directory, stat.S_IMODE(os.stat(directory).st_mode) | 0o700)
        for child in Path(directory).iterdir():
            if not child.is_symlink() and child.is_dir():
                child.chmod(stat.S_IMODE(child.stat().st_mode) | 0o700)
    shutil.rmtree(root)


@contextmanager
def private_directory(parent: Path | None = None) -> Iterator[Path]:
    """Yield a private temporary directory and remove its contents on exit, including on failure."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="harbor-", dir=parent))
    try:
        yield root
    finally:
        remove_owned_tree(root)


def _raise_walk_error(error: OSError) -> None:
    raise error


def check_internal_symlink(path: Path, root: Path) -> str:
    """Return a link's target after proving it stays inside ``root`` lexically and on disk.

    Call only on a complete tree: a link checked before a later link it traverses exists
    (``b -> a/..`` before ``a -> .``) passes here yet escapes once the tree is finished.
    Dangling links and cycles are rejected.
    """
    target = os.readlink(path)
    lexical = os.path.normpath(os.path.join(path.parent.relative_to(root), target))
    try:
        resolved = path.resolve(strict=True)
        # A link to its own ancestor directory (``a -> .``, ``up -> ..``) is a cycle for link-following copies.
        inside = resolved.is_relative_to(root.resolve(strict=True)) and not (
            resolved.is_dir() and path.parent.resolve(strict=True).is_relative_to(resolved)
        )
    except (OSError, RuntimeError):
        inside = False
    if (
        not target
        or len(os.fsencode(target)) > 4096
        or os.path.isabs(target)
        or lexical == ".."
        or lexical.startswith("../")
        or not inside
    ):
        raise ValueError(f"Symlink escapes task root: {path.relative_to(root.parent).as_posix()}")
    return target


def inventory_task_entries(root: Path) -> list[tuple[Path, os.stat_result]]:
    """List task entries in stable path order, rejecting unsafe paths, types, collisions, and excess size."""
    validate_archive_path(root.name)
    if root.name.casefold() == TASK_TEMPLATE_DIRNAME or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("Task root must be a real, non-reserved directory")
    entries = [(root, root.lstat())]
    seen = {root.name.casefold()}
    total = 0
    # Walk without following links; only task-internal links are kept, never descended into.
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root.parent).as_posix()
            validate_archive_path(relative)
            if relative.casefold() in seen:
                raise ValueError("Duplicate case-folded archive path")
            seen.add(relative.casefold())
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                check_internal_symlink(path, root)
            elif not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError(f"Unsupported task entry: {relative}")
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if info.st_size > MAX_FILE_BYTES or total > MAX_PAYLOAD_BYTES:
                    raise ValueError("Task payload exceeds limits")
            entries.append((path, info))
            if len(entries) > MAX_ENTRIES:
                raise ValueError("Too many task entries")
    return sorted(entries, key=lambda pair: pair[0].relative_to(root.parent).as_posix())


def capture_validated_task(root: Path, parent: Path) -> tuple[Path, NativeTask]:
    """Capture a stable, validated copy of a Harbor task in private staging.

    Algorithm:
        - Inventory regular files, directories, and task-internal symlinks, rejecting unsafe entries.
        - Copy files without following symlinks and detect identity, size, or timestamp changes.
        - Recreate symlinks, then re-check them on the completed private copy.
        - Restore source modes and validate the copy as a native Harbor task.

    Args:
        root: Source Harbor task directory.
        parent: Existing private directory that will own the captured task.

    Returns:
        Captured task path beneath ``parent`` and its validated native metadata.

    Raises:
        ValueError: The source is unsafe, changes during capture, or is not a valid Harbor task.
    """
    entries = inventory_task_entries(root)
    target = parent / root.name
    links: list[Path] = []
    for path, info in entries:
        destination = target / path.relative_to(root)
        if stat.S_ISDIR(info.st_mode):
            destination.mkdir(mode=0o700)
        elif stat.S_ISLNK(info.st_mode):
            # The source link may change after inventory; the copy is re-checked below.
            os.symlink(os.readlink(path), destination)
            links.append(destination)
        else:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as source, destination.open("xb") as output:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("Task changed during capture")
                size = 0
                while chunk := source.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > info.st_size:
                        raise ValueError("Task grew during capture")
                    output.write(chunk)
                after = os.fstat(source.fileno())
                if size != info.st_size or (before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError("Task changed during capture")
            destination.chmod(stat.S_IMODE(info.st_mode))
    # Check links on the complete copy while its directories are still owner-traversable.
    for link in links:
        check_internal_symlink(link, target)
    for path, info in reversed(entries):
        if stat.S_ISDIR(info.st_mode):
            (target / path.relative_to(root)).chmod(stat.S_IMODE(info.st_mode))
    native = validate_native_task_inputs(target)
    return target, native


def capture_task(root: Path, parent: Path) -> Path:
    """Capture and validate a package, returning its private staged path."""
    return capture_validated_task(root, parent)[0]


def local_task_identity(root: Path) -> str:
    """Read bounded task identity for cache lookup without loading/copying payloads."""
    config = tomllib.loads(_read_metadata(root / TASK_CONFIG_FILENAME, MAX_CONFIG_BYTES).decode("utf-8"))
    section = config.get("task", {})
    if not isinstance(section, dict):
        raise ValueError("Invalid native task identity")
    name = section.get("name", root.name)
    if not isinstance(name, str) or not name:
        raise ValueError("Invalid native task identity")
    return name
