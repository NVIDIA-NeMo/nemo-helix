# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point at evaluation code that is neither a Harbor nor a Gym task.

Standard library only. These are leads for the agent to raise with the user, who
decides whether to convert them. They never count as discovered evals, and a
heuristic hit proves nothing about what the code does.

A directory is a lead when it holds an eval-named file, or a file carrying one of
the trace formats the rest of Eval Author reads: OpenTelemetry, MLflow tracing,
ATIF trajectories, or NeMo Intake.

The scan stays bounded on large monorepos. In a git repository it reads only the
files git tracks or would track, so ``.gitignore`` keeps out build output, data
dumps, and vendored trees. It reads only the head of each file, where imports and
an ATIF ``schema_version`` sit, and stops after ``_MAX_FILES`` files, saying so.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from providers.harbor._explore import _PRUNE_DIR_NAMES, walk_dirs

_CODE_SUFFIXES = frozenset({".py", ".ipynb", ".js", ".ts", ".sh"})
# Data files count only for ATIF, so a package.json that merely lists a tracing SDK is not a lead.
_DATA_SUFFIXES = frozenset({".json", ".jsonl"})
_MAX_READ_BYTES = 64 * 1024
_MAX_FILES = 10_000
_MAX_CANDIDATES = 50
_GIT_TIMEOUT_SEC = 60
_EVAL_NAME = re.compile(r"(?:^|[_.-])(?:evals?|evaluat(?:e|ion|or)|bench(?:mark)?s?)(?:[_.-]|$)", re.IGNORECASE)
_NAME_SIGNAL = "eval-named file"
_ATIF_SIGNAL = "holds ATIF trajectories"
# Each trace format and the bytes that mark it.
_TRACE_SIGNALS = {
    # An import, not the bare word: services that merely mention OpenTelemetry are not evals.
    "imports OpenTelemetry": re.compile(rb"(?m)^\s*(?:from|import)\s+opentelemetry\b|[\"']@opentelemetry/"),
    "uses MLflow tracing": re.compile(rb"mlflow\.(?:trace|start_span|get_trace|search_traces|genai)\b"),
    _ATIF_SIGNAL: re.compile(rb"ATIF-v\d"),
    "sends to NeMo Intake": re.compile(rb"apis/intake/|nemo intake "),
}


def find(repo_root: Path, owned: list[Path]) -> tuple[list[dict], dict]:
    """Return directories holding eval-like code outside ``owned`` paths, and how much of the repo was scanned."""
    owned_roots = {path.resolve() for path in owned}
    by_directory: dict[Path, tuple[set[str], list[str]]] = {}
    scanned = 0
    complete = True
    for path in _candidate_files(repo_root):
        if scanned == _MAX_FILES:
            complete = False
            break
        directory = path.parent
        if _is_owned(directory.resolve(), owned_roots, repo_root.resolve()):
            continue
        scanned += 1
        found = _signals(path)
        if found:
            signals, files = by_directory.setdefault(directory, (set(), []))
            signals |= found
            files.append(path.name)
    candidates = [
        {"path": directory.relative_to(repo_root).as_posix() or ".", "signals": sorted(signals), "files": sorted(files)}
        for directory, (signals, files) in sorted(by_directory.items())
    ][:_MAX_CANDIDATES]
    return candidates, {"files_scanned": scanned, "complete": complete}


def _candidate_files(repo_root: Path) -> Iterator[Path]:
    """Yield the files worth reading, from git's view of the repository when there is one."""
    listed = _git_files(repo_root)
    if listed is None:
        listed = (path for directory in walk_dirs(repo_root) for path in sorted(_entries(directory)))
    for path in listed:
        if path.suffix.lower() in _CODE_SUFFIXES | _DATA_SUFFIXES and not path.is_symlink() and path.is_file():
            yield path


def _git_files(repo_root: Path) -> list[Path] | None:
    """Return tracked and untracked-but-not-ignored files under ``repo_root``, or None outside a git repository."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    names = sorted(name for name in completed.stdout.decode("utf-8", "replace").split("\0") if name)
    # Committed generated trees, such as a vendored node_modules, stay out as they do in the directory walk.
    return [repo_root / name for name in names if not _PRUNE_DIR_NAMES.intersection(Path(name).parts[:-1])]


def _entries(directory: Path) -> list[Path]:
    try:
        return list(directory.iterdir())
    except OSError:
        return []


def _is_owned(directory: Path, owned_roots: set[Path], repo_root: Path) -> bool:
    """Return whether a directory sits inside a Harbor or Gym path, checking its ancestors against a set."""
    for candidate in (directory, *directory.parents):
        if candidate in owned_roots:
            return True
        if candidate == repo_root:
            return False
    return False


def _signals(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    if suffix in _CODE_SUFFIXES:
        names = set(_TRACE_SIGNALS)
        signals = {_NAME_SIGNAL} if _EVAL_NAME.search(path.stem) else set()
    elif suffix in _DATA_SUFFIXES:
        names, signals = {_ATIF_SIGNAL}, set()
    else:
        return set()
    try:
        with path.open("rb") as source:
            head = source.read(_MAX_READ_BYTES)
    except OSError:
        return signals
    return signals | {name for name in names if _TRACE_SIGNALS[name].search(head)}
