# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which files of a dataset fileset hold training data and which hold validation data.

Every Customizer backend picks its files by the same rule:

- A top-level file whose name contains ``train`` holds training data: ``train.jsonl``,
  ``training.jsonl``, ``my_train_rows.jsonl``.
- Otherwise, a top-level file whose name contains ``val`` or starts with ``dev`` holds
  validation data: ``validation.jsonl``, ``val.jsonl``, ``eval.jsonl``, ``dev.jsonl``.
- So do the files directly in a ``train`` or ``training`` directory, and in a ``val``,
  ``validation`` or ``dev`` directory.
- When no file is named either way, every top-level data file holds training data, or,
  with none at the top level, every data file at any depth.

A ``.json`` file is left out of a split that has ``.jsonl`` files, so a dataset can carry
wrapped JSON beside its rows. Paths are relative to the fileset root, as it lists them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath
from typing import Literal

logger = logging.getLogger(__name__)

Split = Literal["train", "validation"]

DATA_SUFFIXES = (".jsonl", ".json")
TRAIN_DIRS = frozenset({"train", "training"})
VAL_DIRS = frozenset({"val", "validation", "dev"})


def split_of(path: str) -> Split | None:
    """The split *path* holds by its name or its directory, or None when it says neither."""
    parts = PurePosixPath(path.lstrip("/")).parts
    if len(parts) == 1:
        name = PurePosixPath(parts[0]).stem.lower()
        if "train" in name:
            return "train"
        if "val" in name or name.startswith("dev"):
            return "validation"
    elif len(parts) == 2:
        directory = parts[0].lower()
        if directory in TRAIN_DIRS:
            return "train"
        if directory in VAL_DIRS:
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
        train = [path for path in files if "/" not in path] or files
        if len(train) > 1:
            logger.warning(f"Found {len(train)} data files without train/val naming. Treating all as training: {train}")
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
