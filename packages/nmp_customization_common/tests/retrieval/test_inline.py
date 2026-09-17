# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd
import pytest
from nmp.customization_common.retrieval.inline import move_aux_files_to_additional, wrapped_to_inline_jsonl
from nmp.customization_common.retrieval.unroll import unroll_training_data


def test_unroll_and_inline_jsonl(tmp_path: Path) -> None:
    records = [
        {
            "question_id": "q1",
            "question": "what?",
            "corpus_id": "c",
            "pos_doc": ["alpha", "beta"],
            "neg_doc": ["gamma"],
        }
    ]
    unrolled = unroll_training_data(records)
    assert len(unrolled) == 2
    wrapped = tmp_path / "train.json"
    wrapped.write_text(json.dumps({"corpus": {}, "data": unrolled}), encoding="utf-8")
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["query"] == "what?"
    assert lines[0]["pos_doc"] == "alpha"
    assert "gamma" in lines[0]["neg_doc"]


def test_inline_jsonl_unrolls_positives_without_making_them_negative(tmp_path: Path) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps(
            {
                "corpus": {},
                "data": [{"question": "what?", "pos_doc": ["alpha", "beta"], "neg_doc": ["gamma"]}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [line["pos_doc"] for line in lines] == ["alpha", "beta"]
    assert all(line["neg_doc"] == ["gamma"] for line in lines)


def test_inline_jsonl_resolves_corpus_ids_from_parquet(tmp_path: Path) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps({"corpus": {}, "data": [{"question": "q", "pos_doc": [{"id": "d1"}], "neg_doc": [{"id": "d2"}]}]}),
        encoding="utf-8",
    )
    parquet = tmp_path / "corpus" / "train.parquet"
    parquet.parent.mkdir()
    pd.DataFrame({"id": ["d1", "d2"], "text": ["positive text", "negative text"]}).to_parquet(parquet)
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out, parquet)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["pos_doc"] == "positive text"
    assert row["neg_doc"] == ["negative text"]


def test_inline_jsonl_skips_unresolved_corpus_ids(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps(
            {
                "corpus": {},
                "data": [
                    {"question": "drop me", "pos_doc": [{"id": "missing"}], "neg_doc": []},
                    {"question": "drop no negs", "pos_doc": [{"id": "d1"}], "neg_doc": [{"id": "missing"}]},
                    {"question": "keep me", "pos_doc": [{"id": "d1"}, {"id": "missing"}], "neg_doc": [{"id": "d2"}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    parquet = tmp_path / "corpus" / "train.parquet"
    parquet.parent.mkdir()
    pd.DataFrame({"id": ["d1", "d2"], "text": ["positive text", "negative text"]}).to_parquet(parquet)
    out = tmp_path / "training.jsonl"
    with caplog.at_level("WARNING"):
        wrapped_to_inline_jsonl(wrapped, out, parquet)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"query": "keep me", "pos_doc": "positive text", "neg_doc": ["negative text"]}]
    assert "Skipped 3 unresolved corpus document(s)" in caplog.text
    assert "dropped 2 unusable record(s)" in caplog.text


def test_inline_jsonl_skips_unrecognized_document_shapes(tmp_path: Path) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps(
            {
                "corpus": {},
                "data": [
                    {
                        "question": "keep text",
                        "pos_doc": ["alpha", {"docid": 7, "score": 0.4}, 12],
                        "neg_doc": [{"contents": "gamma"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"query": "keep text", "pos_doc": "alpha", "neg_doc": ["gamma"]}]


def test_inline_jsonl_treats_null_document_lists_as_empty(tmp_path: Path) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps(
            {
                "corpus": {},
                "data": [
                    {"question": "drop null pos", "pos_doc": None, "neg_doc": ["gamma"]},
                    {"question": "keep null neg", "pos_doc": ["alpha"], "neg_doc": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"query": "keep null neg", "pos_doc": "alpha", "neg_doc": []}]


def test_inline_jsonl_skips_null_and_empty_document_text(tmp_path: Path) -> None:
    wrapped = tmp_path / "train.json"
    wrapped.write_text(
        json.dumps(
            {
                "corpus": {},
                "data": [
                    {
                        "question": "fallback to contents",
                        "pos_doc": [{"text": None, "contents": "from contents"}],
                        "neg_doc": ["gamma"],
                    },
                    {
                        "question": "parquet none is unresolved",
                        "pos_doc": [{"id": "empty"}],
                        "neg_doc": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    parquet = tmp_path / "corpus" / "train.parquet"
    parquet.parent.mkdir()
    pd.DataFrame({"id": ["empty", "ok"], "text": [None, "neg text"]}).to_parquet(parquet)
    out = tmp_path / "training.jsonl"
    wrapped_to_inline_jsonl(wrapped, out, parquet)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"query": "fallback to contents", "pos_doc": "from contents", "neg_doc": ["gamma"]}]


def test_unroll_skips_null_pos_doc_and_missing_question_id() -> None:
    unrolled = unroll_training_data(
        [
            {"question": "keep", "pos_doc": None, "neg_doc": []},
            {"question": "split", "corpus_id": "c", "pos_doc": ["a", "b"], "neg_doc": ["n"]},
        ]
    )
    assert unrolled[0]["pos_doc"] is None
    assert [row["question_id"] for row in unrolled[1:]] == ["_0", "_1"]
    assert [row["pos_doc"] for row in unrolled[1:]] == [["a"], ["b"]]


def test_move_aux_files_to_additional_keeps_training_and_eval(tmp_path: Path) -> None:
    (tmp_path / "training.jsonl").write_text("{}\n", encoding="utf-8")
    eval_beir = tmp_path / "eval_beir"
    eval_beir.mkdir()
    (eval_beir / "queries.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "train.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cache_embeddings").mkdir()
    (tmp_path / "cache_embeddings" / "x.npz").write_bytes(b"x")

    move_aux_files_to_additional(tmp_path)

    assert (tmp_path / "training.jsonl").is_file()
    assert (tmp_path / "eval_beir" / "queries.jsonl").is_file()
    assert (tmp_path / "additional" / "train.json").is_file()
    assert (tmp_path / "additional" / "cache_embeddings" / "x.npz").is_file()
    assert not (tmp_path / "train.json").exists()
    assert not (tmp_path / "cache_embeddings").exists()
