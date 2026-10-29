# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which files of a dataset fileset hold training data and which hold validation data.

Every Customizer backend picks its files by the same rule:

- A top-level file matching :data:`TRAIN_PATTERNS` (``train*.jsonl``, ``training*.jsonl``)
  holds training data, as do the files directly in a :data:`TRAIN_DIRS` directory
  (``train/``, ``training/``).
- A top-level file matching :data:`VAL_PATTERNS` (``val*.jsonl``, ``validation*.jsonl``,
  ``dev*.jsonl``) holds validation data, as do the files directly in a :data:`VAL_DIRS`
  directory (``val/``, ``validation/``, ``dev/``).
- When no file matches either way, every top-level ``.jsonl`` file holds training data.

Names match case-sensitively. A ``.json`` file is left out of a split that has ``.jsonl``
files, so a dataset can carry wrapped JSON beside its rows. Paths are relative to the
fileset root, as it lists them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Literal

logger = logging.getLogger(__name__)

Split = Literal["train", "validation"]

DATA_SUFFIXES = (".jsonl", ".json")

# Top-level file names that hold training data
TRAIN_PATTERNS = [
    "train*.jsonl",
    "training*.jsonl",
    "train*.json",
    "training*.json",
]
# Directories whose files hold training data
TRAIN_DIRS = ["train", "training"]

# Top-level file names that hold validation data
VAL_PATTERNS = [
    "val*.jsonl",
    "validation*.jsonl",
    "val*.json",
    "validation*.json",
    "dev*.jsonl",
    "dev*.json",
]
# Directories whose files hold validation data
VAL_DIRS = ["val", "validation", "dev"]


def split_of(path: str) -> Split | None:
    """The split *path* holds by its name or its directory, or None when it says neither."""
    parts = PurePosixPath(path.lstrip("/")).parts
    if len(parts) == 1:
        if any(fnmatchcase(parts[0], pattern) for pattern in TRAIN_PATTERNS):
            return "train"
        if any(fnmatchcase(parts[0], pattern) for pattern in VAL_PATTERNS):
            return "validation"
    elif len(parts) == 2:
        if parts[0] in TRAIN_DIRS:
            return "train"
        if parts[0] in VAL_DIRS:
            return "validation"
    return None


def select_split_files(paths: Iterable[str], suffixes: Sequence[str] = DATA_SUFFIXES) -> tuple[list[str], list[str]]:
    """Pick the training and validation files among *paths*, by the rule in this module's docstring.

    Args:
        paths: File paths relative to the dataset root.
        suffixes: The extensions of data files.

    Returns:
        The training files and the validation files, sorted. Either may be empty.
    """
    files = _data_files(paths, suffixes)
    train = [path for path in files if split_of(path) == "train"]
    validation = [path for path in files if split_of(path) == "validation"]
    if not train and not validation:
        train = [path for path in files if "/" not in path and path.endswith(".jsonl")]
        if len(train) > 1:
            logger.warning(
                f"Found {len(train)} JSONL files without train/val naming. Treating all as training data: {train}"
            )
    return _prefer_jsonl(train), _prefer_jsonl(validation)


def _data_files(paths: Iterable[str], suffixes: Sequence[str]) -> list[str]:
    return sorted(path.lstrip("/") for path in paths if PurePosixPath(path).suffix.lower() in suffixes)


def _prefer_jsonl(files: list[str]) -> list[str]:
    """Drop ``.json`` files when ``.jsonl`` files are present.

    Retrieval Stage 1 publishes a wrapped ``train.json`` beside the inline
    ``training.jsonl``; both are named for training, and merging them would emit
    invalid JSONL.
    """
    jsonl_files = [path for path in files if path.lower().endswith(".jsonl")]
    if jsonl_files and len(jsonl_files) < len(files):
        skipped = [path for path in files if not path.lower().endswith(".jsonl")]
        logger.info(f"Ignoring non-JSONL dataset file(s) found alongside JSONL: {skipped}")
    return jsonl_files or files
