# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nemo_evaluator_sdk.retrieval.beir import BeirDataset, BeirDatasetError


def _write_beir_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Alpha", "text": "First document."},
        {"_id": "d2", "title": "", "text": "Second document."},
        {"_id": "d3", "text": "Third document."},
    ]
    queries = [
        {"_id": "q1", "text": "first"},
        {"_id": "q2", "text": "third"},
    ]
    (root / "corpus.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in corpus),
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in queries),
        encoding="utf-8",
    )
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t2\n",
        encoding="utf-8",
    )
    return root


def test_loads_beir_test_split(tmp_path: Path) -> None:
    dataset = BeirDataset.from_path(_write_beir_dataset(tmp_path / "beir"))

    assert dataset.corpus["d1"].content == "Alpha\nFirst document."
    assert dataset.corpus["d3"].title == ""
    assert dataset.queries["q2"].text == "third"
    assert dataset.qrels == {"q1": {"d1": 1}, "q2": {"d3": 2}}
    assert dataset.dropped_qrel_rows == 0
    assert dataset.query_rows() == [
        {"query_id": "q1", "query": "first", "qrels": {"d1": 1}},
        {"query_id": "q2", "query": "third", "qrels": {"d3": 2}},
    ]


def test_discovers_eval_beir_below_fileset_root(tmp_path: Path) -> None:
    _write_beir_dataset(tmp_path / "eval_beir")

    assert BeirDataset.from_path(tmp_path).root == tmp_path / "eval_beir"


def test_rejects_missing_layout(tmp_path: Path) -> None:
    with pytest.raises(BeirDatasetError, match=r"corpus\.jsonl.*queries\.jsonl.*qrels/test\.tsv"):
        BeirDataset.from_path(tmp_path)


def test_keeps_first_duplicate_qrel_score(tmp_path: Path) -> None:
    root = _write_beir_dataset(tmp_path / "beir")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t2\nq2\td3\t2\n",
        encoding="utf-8",
    )

    dataset = BeirDataset.from_path(root)

    assert dataset.qrels == {"q1": {"d1": 1}, "q2": {"d3": 2}}
    assert dataset.dropped_qrel_rows == 1


def test_rejects_unknown_qrel_document(tmp_path: Path) -> None:
    root = _write_beir_dataset(tmp_path / "beir")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tmissing\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(BeirDatasetError, match="unknown corpus ids: missing"):
        BeirDataset.from_path(root)
