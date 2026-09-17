# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nmp.automodel.tasks.training.datasets.preparation import (
    DatasetFormatError,
    discover_dataset_files,
    prepare_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _retrieval_rows(count: int) -> list[dict]:
    return [{"query": f"q{i}", "pos_doc": f"p{i}", "neg_doc": [f"n{i}"]} for i in range(count)]


def _stage1_artifacts(root: Path) -> Path:
    """Mimic the retrieval Stage 1 artifacts layout."""
    _write_jsonl(root / "training.jsonl", _retrieval_rows(20))
    (root / "train.json").write_text(
        json.dumps({"corpus": {"d0": "text"}, "data": _retrieval_rows(20)}, indent=2),
        encoding="utf-8",
    )
    beir = root / "eval_beir"
    (beir / "qrels").mkdir(parents=True)
    _write_jsonl(beir / "corpus.jsonl", [{"_id": "d0", "text": "text"}])
    _write_jsonl(beir / "queries.jsonl", [{"_id": "q0", "text": "q"}])
    (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq0\td0\t1\n", encoding="utf-8")
    return root


def test_discover_prefers_inline_jsonl_over_wrapped_json(tmp_path: Path) -> None:
    train_files, val_files = discover_dataset_files(_stage1_artifacts(tmp_path))

    assert [f.name for f in train_files] == ["training.jsonl"]
    assert val_files == []


def test_prepare_dataset_accepts_stage1_artifacts(tmp_path: Path) -> None:
    prepared = prepare_dataset(_stage1_artifacts(tmp_path), output_dir=tmp_path / "merged")

    rows = [json.loads(line) for line in prepared.train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert all(row.keys() == {"query", "pos_doc", "neg_doc"} for row in rows)
    assert prepared.train_samples + prepared.validation_samples == 20


def test_discover_keeps_json_when_no_jsonl_present(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.json", _retrieval_rows(3))

    train_files, _ = discover_dataset_files(tmp_path)

    assert [f.name for f in train_files] == ["train.json"]


def test_discover_requires_a_training_file(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "validation.jsonl", _retrieval_rows(3))

    with pytest.raises(DatasetFormatError, match="No training files found"):
        discover_dataset_files(tmp_path)
