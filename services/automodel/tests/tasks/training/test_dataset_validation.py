# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nmp.automodel.entities.values import TrainingType
from nmp.automodel.tasks.training.datasets import validation as validation_mod
from nmp.automodel.tasks.training.datasets.validation import DatasetValidator


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
