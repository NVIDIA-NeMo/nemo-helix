# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split resolution from file paths.

Given the files in one partition, group them into splits by inferring each file's split from its
path — a split-named directory first, then the shard-stripped filename. A declared split map from a
dataset card would take precedence over this inference, but card parsing is not wired up yet, so
path inference is the only source today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from nemo_profiler_plugin.source import FileEntry

# Strips a shard suffix like "-00000" or "-00000-of-00003" from a file stem. A bare trailing number
# must be zero-padded to count: `-\d{2,}` alone also matched years and versions, turning
# covid-19.jsonl into a "covid" split and data-2024.jsonl into "data". The leading zero is what
# carries that distinction, so the padded form only needs to be three characters -- requiring four
# excluded `train-001.jsonl`, an ordinary numbering, and gave every such shard a split of its own.
#
# `N-of-M` needs no padding rule at all: nothing but a shard count is spelled that way, so any
# number of digits is safe there.
#
# An unpadded trailing number (`train-1.jsonl`) is deliberately still not a shard suffix. On a
# single path it cannot be told from a version, and this function sees one path at a time.
_SHARD_SUFFIX = re.compile(r"-(?:\d+-of-\d+|0\d{2,})$")

# Common on-disk split words -> the canonical concept they normalize to.
_CANONICAL_ALIASES = {
    "train": "train",
    "test": "test",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
}


@dataclass(frozen=True)
class ResolvedSplit:
    """A split and the files that belong to it."""

    name: str  # the on-disk split name, e.g. "train" or "train_prefs"
    canonical: str | None  # normalized concept (train | validation | test), or None
    entries: list[FileEntry]


def _canonical_for(split_name: str) -> str | None:
    """Map a split name to its canonical concept, tolerating variant suffixes (train_prefs -> train)."""
    lowered = split_name.lower()
    for alias, canonical in _CANONICAL_ALIASES.items():
        if lowered == alias or lowered.startswith(f"{alias}_") or lowered.startswith(f"{alias}-"):
            return canonical
    return None


def is_split_directory(name: str) -> bool:
    """Whether a directory name denotes a split rather than a partition.

    Partition grouping uses it to keep ``train/`` and ``test/`` in one partition rather than two,
    since they are two splits of one dataset.
    """
    return _canonical_for(name) is not None


def _split_name(path: str) -> str:
    """The split a file belongs to.

    A split-named directory on the path wins over the filename, nearest one first: the
    ``data/train/`` layout names its shards ``0000.parquet``, so reading the stem alone would file
    every split under the same name. Failing that, the shard-stripped stem
    (``train-00000-of-00003.parquet`` -> ``train``).
    """
    parts = PurePosixPath(path).parts
    for directory in reversed(parts[:-1]):
        if is_split_directory(directory):
            return directory
    # Leading dots are stripped before the extension is: `.split(".")[0]` reads everything before
    # the first dot, which for a dot-prefixed name (`._train.jsonl`, the AppleDouble files a
    # macOS-made tarball is full of) is the empty string -- a split whose on-disk name was "".
    return _SHARD_SUFFIX.sub("", parts[-1].lstrip(".").split(".")[0])


def _glob_matches(pattern: str, path: str) -> bool:
    """Match ``path`` against ``pattern`` where ``*`` spans any run of characters except ``/``.

    The narrowest dialect on purpose: this reading of ``*`` is shared by shell globs, :mod:`glob`,
    fsspec and HF ``data_files``, so a pattern means the same thing wherever it is pasted. ``**`` is
    never produced, since its meaning differs between those tools.
    """
    return re.fullmatch("[^/]*".join(re.escape(part) for part in pattern.split("*")), path) is not None


# Glob metacharacters other than the `*` this module produces on purpose. `_glob_matches` escapes
# them, so verification reads them as literals -- but the pattern is handed to a consumer, and every
# tool the dialect note names reads them as syntax. `data[a]/train*.jsonl` then selects nothing, and
# `what?/train*.jsonl` selects a neighbouring partition's shards as well, which is exactly the near
# miss the contract promises cannot happen. Escaping is no answer either: the escape syntax is the
# part those tools do not share. A path holding one gets no pattern, which is what None already says.
_GLOB_METACHARACTERS = frozenset("*?[]")


def infer_data_files(split_name: str, entries: list[FileEntry], all_paths: list[str]) -> str | None:
    """A glob selecting exactly ``entries`` out of ``all_paths``, or None if no single one does.

    Rebuilt from the same evidence the split was read from: the directory the files share, and the
    split name their filenames start with.

    Candidates are tried most specific first, because the general ones over-match: in ``helpsteer2/``
    a ``train`` split wants ``helpsteer2/train*.parquet``, where ``helpsteer2/*.parquet`` would take
    the validation shards too. Each is matched against every path the source listed, and the first to
    reproduce the split exactly wins.

    A near miss returns None. A glob is an instruction to go read files, so an approximate one is not
    a smaller version of the right answer -- it pulls in a README, or another split's shards.
    """
    paths = [entry.path for entry in entries]
    directories = {PurePosixPath(path).parent.as_posix() for path in paths}
    if len(directories) != 1:
        # Shards spread across subdirectories need `**`, whose meaning is not shared across glob
        # implementations, so report no pattern rather than an ambiguous one.
        return None
    directory = directories.pop()
    prefix = "" if directory == "." else f"{directory}/"
    names = [PurePosixPath(path).name for path in paths]
    suffixes = {PurePosixPath(path).suffix for path in paths}
    # A partition may hold more than one format; only a single shared suffix can go in the pattern.
    suffix = suffixes.pop() if len(suffixes) == 1 else None

    # Stems to anchor on: the bare split name first, then the same name with each separator. Plain
    # `train*` is what a person would write, so it is tried first; the separator variants cover the
    # case where `train` sits beside `train_prefs` and `train*` would take both. A stem is offered
    # only when every filename in the split carries it, so it can never exclude a file it should
    # match.
    stems = [split_name] + [f"{split_name}{sep}" for sep in ("-", ".", "_")] if split_name else []
    # Checked over every literal the pattern is built from, since one is enough to change what the
    # whole pattern selects.
    if any(_GLOB_METACHARACTERS & set(text) for text in [prefix, suffix or "", *stems]):
        return None
    candidates: list[str] = []
    for stem in stems:
        if not all(name.startswith(stem) for name in names):
            continue
        if suffix:
            candidates.append(f"{prefix}{stem}*{suffix}")
        candidates.append(f"{prefix}{stem}*")
    if suffix:
        candidates.append(f"{prefix}*{suffix}")
    candidates.append(f"{prefix}*")

    target = set(paths)
    for candidate in candidates:
        if {path for path in all_paths if _glob_matches(candidate, path)} == target:
            return candidate
    return None


def resolve_splits(entries: list[FileEntry]) -> list[ResolvedSplit]:
    """Group files into splits by path inference.

    Each file's split name comes from a split-named directory on its path, else its shard-stripped
    stem; the canonical concept is matched against common aliases (val/valid/dev -> validation). When
    no file carries a recognizable split, every file lands in one "default" split.
    """
    grouped: dict[str, list[FileEntry]] = {}
    for entry in entries:
        grouped.setdefault(_split_name(entry.path), []).append(entry)

    canonicals = {name: _canonical_for(name) for name in grouped}
    if not any(canonicals.values()):
        return [ResolvedSplit(name="default", canonical=None, entries=list(entries))]

    return [ResolvedSplit(name=name, canonical=canonicals[name], entries=grouped[name]) for name in sorted(grouped)]
