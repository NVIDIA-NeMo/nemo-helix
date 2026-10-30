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
from typing import Any

from nhx.customization_common.dataset_files import select_split_files


class DatasetFormatError(ValueError):
    """A dataset row Unsloth training cannot read."""


def select_data_files(paths: Iterable[str]) -> list[str]:
    """Pick the files training loads from a dataset's file paths: its training and validation files.

    Uses the rule every Customizer backend shares (:mod:`nhx.customization_common.dataset_files`),
    as ``unsloth_sft`` does.

    Args:
        paths: File paths relative to the dataset root, as a fileset lists them.

    Returns:
        The training files, then any validation files, as relative paths.

    Raises:
        DatasetFormatError: If the dataset has no training file.
    """
    train, validation = select_split_files(paths)
    if not train:
        raise DatasetFormatError(
            "No training .jsonl/.json files found. "
            "Expected files matching patterns like train*.jsonl or a train/ directory."
        )
    return train + validation


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
            isinstance(row.get(text_field), str) or _is_messages(row.get("messages")) or _is_prompt_completion(row)
        ):
            raise DatasetFormatError(
                f"{source} line {line_number} needs a string '{text_field}' field (dataset.text_field), "
                "a 'messages' list, or 'prompt' and 'completion' as strings or 'messages'-style lists"
            )
        rows += 1
    if rows == 0:
        raise DatasetFormatError(f"{source} has no rows")
    return rows


def _is_prompt_completion(row: dict[str, Any]) -> bool:
    """A ``prompt`` / ``completion`` pair, both strings or both lists of messages, as TRL reads them."""
    prompt, completion = row.get("prompt"), row.get("completion")
    return (isinstance(prompt, str) and isinstance(completion, str)) or (
        _is_messages(prompt) and _is_messages(completion)
    )


def _is_messages(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(message, dict) and isinstance(message.get("role"), str) and "content" in message
            for message in value
        )
    )
