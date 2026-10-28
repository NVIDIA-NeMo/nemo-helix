# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Literal

import pytest
from nhx.unsloth.tasks.training.datasets.validation import DatasetFormatError, select_data_files, validate_lines

_MESSAGES = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


@pytest.mark.parametrize(
    ("paths", "split", "expected"),
    [
        (["train.jsonl", "validation.jsonl"], "train", ["train.jsonl"]),
        (["train.jsonl", "validation.jsonl"], "validation", ["validation.jsonl"]),
        (["/b.jsonl", "a/c.jsonl", "notes.json"], None, ["a/c.jsonl", "b.jsonl"]),
        (["data.json", "README.md"], "train", ["data.json"]),
    ],
)
def test_select_data_files_picks_what_training_loads(
    paths: list[str], split: Literal["train", "validation"] | None, expected: list[str]
) -> None:
    assert select_data_files(paths, split) == expected


def test_select_data_files_requires_a_json_file() -> None:
    with pytest.raises(DatasetFormatError, match="No .jsonl/.json files"):
        select_data_files(["README.md"])


@pytest.mark.parametrize(
    "row",
    [{"text": "a rendered example"}, _MESSAGES, {"prompt": "q", "completion": "a"}],
)
def test_rows_trl_reads_pass_without_the_chat_template(row: dict) -> None:
    assert validate_lines([json.dumps(row)], "rows", apply_chat_template=False, text_field="text") == 1


def test_chat_template_requires_messages() -> None:
    with pytest.raises(DatasetFormatError, match="line 1 needs a 'messages' list"):
        validate_lines([json.dumps({"text": "x"})], "rows", apply_chat_template=True, text_field="text")
    assert validate_lines([json.dumps(_MESSAGES)], "rows", apply_chat_template=True, text_field="text") == 1


def test_errors_name_the_source_and_line() -> None:
    lines = [json.dumps({"text": "ok"}), "", json.dumps({"question": "q"})]
    with pytest.raises(DatasetFormatError, match=r"^ws/data#train.jsonl line 3 needs a string 'text' field"):
        validate_lines(lines, "ws/data#train.jsonl", apply_chat_template=False, text_field="text")


def test_rejects_non_json_and_empty_input() -> None:
    with pytest.raises(DatasetFormatError, match="line 1 is not valid JSON"):
        validate_lines(["{not json"], "rows", apply_chat_template=False, text_field="text")
    with pytest.raises(DatasetFormatError, match="has no rows"):
        validate_lines(["", "  "], "rows", apply_chat_template=False, text_field="text")
