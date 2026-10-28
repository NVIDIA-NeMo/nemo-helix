# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Row checks for Unsloth training data, matching how ``unsloth_sft`` loads it.

With ``apply_chat_template``, ``unsloth_sft`` renders each row's ``messages`` itself.
Otherwise TRL's ``SFTTrainer`` reads the ``text_field`` column, or a conversational
``messages`` row, or a ``prompt`` / ``completion`` pair.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any, Literal

TRAIN_FILE = "train.jsonl"
VAL_FILE = "validation.jsonl"


class DatasetFormatError(ValueError):
    """A dataset row Unsloth training cannot read."""


def select_data_files(paths: Iterable[str], split: Literal["train", "validation"] | None = None) -> list[str]:
    """Pick the files training loads from a dataset's file paths.

    ``train.jsonl`` or ``validation.jsonl`` at the top level wins for its split;
    otherwise every ``.jsonl`` file, at any depth, falling back to ``.json``.

    Args:
        paths: File paths relative to the dataset root, as a fileset lists them.
        split: The split being loaded, when the fileset may hold both.

    Returns:
        The relative paths training loads, sorted.

    Raises:
        DatasetFormatError: If there is no JSON or JSONL file.
    """
    paths = sorted(path.lstrip("/") for path in paths)
    preferred = {"train": TRAIN_FILE, "validation": VAL_FILE}.get(split or "")
    if preferred in paths:
        return [preferred]
    for pattern in ("*.jsonl", "*.json"):
        files = [path for path in paths if fnmatchcase(PurePosixPath(path).name, pattern)]
        if files:
            return files
    raise DatasetFormatError("No .jsonl/.json files found. The dataset fileset must hold a JSON or JSONL file.")


def validate_lines(lines: Iterable[str], source: str, *, apply_chat_template: bool, text_field: str) -> int:
    """Check that each JSONL line is a row training can read.

    Args:
        lines: JSONL lines, numbered from 1 in error messages.
        source: Where the lines came from, for error messages.
        apply_chat_template: ``dataset.apply_chat_template`` from the job.
        text_field: ``dataset.text_field`` from the job.

    Returns:
        The number of rows checked.

    Raises:
        DatasetFormatError: If a line is not a JSON object, or has none of the fields training reads.
    """
    rows = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row: Any = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetFormatError(f"{source} line {line_number} is not valid JSON: {e}") from e
        if not isinstance(row, dict):
            raise DatasetFormatError(f"{source} line {line_number} is not a JSON object")
        if apply_chat_template:
            if not _is_messages(row.get("messages")):
                raise DatasetFormatError(
                    f"{source} line {line_number} needs a 'messages' list of {{'role', 'content'}} objects, "
                    "because dataset.apply_chat_template is true"
                )
        elif not (
            isinstance(row.get(text_field), str)
            or _is_messages(row.get("messages"))
            or (isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str))
        ):
            raise DatasetFormatError(
                f"{source} line {line_number} needs a string '{text_field}' field (dataset.text_field), "
                "a 'messages' list, or 'prompt' and 'completion' strings"
            )
        rows += 1
    if rows == 0:
        raise DatasetFormatError(f"{source} has no rows")
    return rows


def _is_messages(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(message, dict) and isinstance(message.get("role"), str) and "content" in message
            for message in value
        )
    )
