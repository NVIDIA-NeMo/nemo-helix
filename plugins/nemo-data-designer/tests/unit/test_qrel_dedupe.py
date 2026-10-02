# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nemo_data_designer_plugin.retrieval.conversion import dedupe_beir_qrels, execute_conversion


def test_dedupe_beir_qrels_keeps_the_first_score(tmp_path: Path) -> None:
    path = tmp_path / "test.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\nq13064\td76198\t1\nq1\td1\t1\nq13064\td76198\t2\nq1\td2\t1\n",
        encoding="utf-8",
    )

    assert dedupe_beir_qrels(path) == 1
    assert path.read_text(encoding="utf-8") == ("query-id\tcorpus-id\tscore\nq13064\td76198\t1\nq1\td1\t1\nq1\td2\t1\n")
    assert dedupe_beir_qrels(path) == 0


def test_execute_conversion_rewrites_duplicate_eval_qrels(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "conv"

    def _convert(_config: object) -> SimpleNamespace:
        qrels = out / "eval_beir" / "qrels"
        qrels.mkdir(parents=True)
        (qrels / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n",
            encoding="utf-8",
        )
        return SimpleNamespace(train_file=out / "train.json")

    with patch("data_designer_retrieval_sdg.run_conversion_with_config", side_effect=_convert):
        execute_conversion(
            input_path=src,
            output_dir=out,
            corpus_id="retrieval_sdg",
            quality_threshold=7.0,
            train_ratio=0.8,
            val_ratio=0.0,
            seed=42,
            max_pos_docs=5,
            use_group_id_in_eval=False,
            split_strategy="random",
        )

    qrels = (out / "eval_beir" / "qrels" / "test.tsv").read_text(encoding="utf-8")
    assert qrels == "query-id\tcorpus-id\tscore\nq1\td1\t1\n"
