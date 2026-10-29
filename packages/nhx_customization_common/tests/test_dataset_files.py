# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.customization_common.dataset_files import select_split_files, split_of


@pytest.mark.parametrize(
    ("path", "split"),
    [
        ("train.jsonl", "train"),
        ("training.jsonl", "train"),
        ("train_part-0.jsonl", "train"),
        ("train.json", "train"),
        ("validation.jsonl", "validation"),
        ("val.jsonl", "validation"),
        ("val_part-0.json", "validation"),
        ("dev.jsonl", "validation"),
        ("train/part-0.jsonl", "train"),
        ("training/rows.json", "train"),
        ("dev/rows.jsonl", "validation"),
        ("/validation.jsonl", "validation"),
        # A pattern matches from the start of the name, case-sensitively.
        ("my_train_rows.jsonl", None),
        ("pretrain.jsonl", None),
        ("constraints.jsonl", None),
        ("eval.jsonl", None),
        ("retrieval.jsonl", None),
        ("Train.jsonl", None),
        ("Training/rows.jsonl", None),
        ("rows.jsonl", None),
        ("test.jsonl", None),
        ("eval_beir/corpus.jsonl", None),
        ("additional/train.json", None),
        ("train/nested/part-0.jsonl", None),
        ("data/train/part-0.jsonl", None),
    ],
)
def test_split_of_reads_the_name_or_the_directory(path: str, split: str | None) -> None:
    assert split_of(path) == split


def test_named_files_pick_both_splits_and_leave_others_out() -> None:
    paths = ["training.jsonl", "validation.jsonl", "eval.jsonl", "test.jsonl", "README.md", "eval_beir/corpus.jsonl"]

    assert select_split_files(paths) == (["training.jsonl"], ["validation.jsonl"])


def test_train_and_training_are_the_same_split() -> None:
    assert select_split_files(["train.jsonl", "training.jsonl"]) == (["train.jsonl", "training.jsonl"], [])


def test_without_names_every_top_level_jsonl_file_is_training_data() -> None:
    assert select_split_files(["b.jsonl", "a.jsonl", "c.json", "nested/d.jsonl"]) == (["a.jsonl", "b.jsonl"], [])


def test_without_names_json_and_nested_files_are_not_training_data() -> None:
    assert select_split_files(["rows.json", "data/c.jsonl", "data/a.jsonl"]) == ([], [])


def test_only_validation_files_leave_training_empty() -> None:
    assert select_split_files(["validation.jsonl", "rows.jsonl"]) == ([], ["validation.jsonl"])


def test_json_is_dropped_beside_jsonl_and_used_alone() -> None:
    assert select_split_files(["train.json", "training.jsonl"]) == (["training.jsonl"], [])
    assert select_split_files(["train.json"]) == (["train.json"], [])


def test_suffixes_limit_the_data_files() -> None:
    assert select_split_files(["train.json", "rows.parquet"], (".jsonl",)) == ([], [])
