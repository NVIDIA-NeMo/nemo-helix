# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.customization_common.dataset_files import select_split_files, split_of


@pytest.mark.parametrize(
    ("path", "split"),
    [
        ("train.jsonl", "train"),
        ("training.jsonl", "train"),
        ("my_train_rows.jsonl", "train"),
        ("Pretrain-Corpus.jsonl", "train"),
        ("trainval.jsonl", "train"),
        ("validation.jsonl", "validation"),
        ("val.jsonl", "validation"),
        ("eval.jsonl", "validation"),
        ("dev.jsonl", "validation"),
        ("train/part-0.jsonl", "train"),
        ("Training/rows.jsonl", "train"),
        ("dev/rows.jsonl", "validation"),
        ("/validation.jsonl", "validation"),
        ("rows.jsonl", None),
        ("test.jsonl", None),
        ("eval_beir/corpus.jsonl", None),
        ("additional/train.json", None),
        ("data/train/part-0.jsonl", None),
    ],
)
def test_split_of_reads_the_name_or_the_directory(path: str, split: str | None) -> None:
    assert split_of(path) == split


def test_named_files_pick_both_splits_and_leave_others_out() -> None:
    paths = ["training.jsonl", "eval.jsonl", "test.jsonl", "README.md", "eval_beir/corpus.jsonl"]

    assert select_split_files(paths) == (["training.jsonl"], ["eval.jsonl"])


def test_train_and_training_are_the_same_split() -> None:
    assert select_split_files(["train.jsonl", "training.jsonl"]) == (["train.jsonl", "training.jsonl"], [])


def test_without_names_every_top_level_file_is_training_data() -> None:
    assert select_split_files(["b.jsonl", "a.jsonl", "nested/c.jsonl"]) == (["a.jsonl", "b.jsonl"], [])


def test_without_top_level_files_nested_files_are_training_data() -> None:
    assert select_split_files(["data/c.jsonl", "data/a.jsonl"]) == (["data/a.jsonl", "data/c.jsonl"], [])


def test_only_validation_files_leave_training_empty() -> None:
    assert select_split_files(["validation.jsonl", "rows.jsonl"]) == ([], ["validation.jsonl"])


def test_json_is_dropped_beside_jsonl_and_used_alone() -> None:
    assert select_split_files(["train.json", "training.jsonl"]) == (["training.jsonl"], [])
    assert select_split_files(["train.json"]) == (["train.json"], [])


def test_suffixes_limit_the_data_files() -> None:
    assert select_split_files(["train.json", "rows.parquet"], (".jsonl",)) == ([], [])
