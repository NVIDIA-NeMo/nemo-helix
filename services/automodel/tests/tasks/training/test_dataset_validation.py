# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nhx.automodel.entities.values import TrainingType
from nhx.automodel.tasks.training.datasets import validation as validation_mod
from nhx.automodel.tasks.training.datasets.preparation import DatasetFormatError
from nhx.automodel.tasks.training.datasets.validation import DatasetValidator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_validate_dataset_compiles_schema_once(tmp_path: Path, mocker) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"query": "q", "pos_doc": "p", "neg_doc": []} for _ in range(50)])
    compile_spy = mocker.spy(validation_mod, "_compile_validator")

    DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path))

    assert compile_spy.call_count == 1


def test_validate_dataset_accepts_embedding_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"query": "q", "pos_doc": "p", "neg_doc": ["n"]}])
    DatasetValidator(training_type=TrainingType.SFT).validate_dataset(str(path))


def test_validate_lines_names_the_source_and_line() -> None:
    lines = [
        json.dumps({"prompt": "p", "completion": "c"}),
        "",
        json.dumps({"question": "no prompt or messages"}),
    ]
    with pytest.raises(DatasetFormatError, match=r"^default/data#train.jsonl line 3 does not conform"):
        DatasetValidator(training_type=TrainingType.SFT).validate_lines(lines, "default/data#train.jsonl")


def test_validate_lines_counts_validated_rows() -> None:
    lines = [json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})] * 2
    assert DatasetValidator(training_type=TrainingType.SFT).validate_lines(lines, "rows") == 2
