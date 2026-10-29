# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.rl.tasks.training.datasets.preparation import DatasetFormatError, select_dataset_files


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["train.jsonl", "validation.jsonl"], (["train.jsonl"], ["validation.jsonl"])),
        (["training.jsonl", "eval.jsonl"], (["training.jsonl"], ["eval.jsonl"])),
        (["my_train_rows.jsonl", "dev.jsonl"], (["my_train_rows.jsonl"], ["dev.jsonl"])),
        (["data.jsonl"], (["data.jsonl"], [])),
    ],
)
def test_select_dataset_files_reads_any_train_and_val_name(
    paths: list[str], expected: tuple[list[str], list[str]]
) -> None:
    assert select_dataset_files(paths) == expected


def test_select_dataset_files_reads_jsonl_only() -> None:
    with pytest.raises(DatasetFormatError, match="No training files found"):
        select_dataset_files(["train.json", "validation.jsonl"])
