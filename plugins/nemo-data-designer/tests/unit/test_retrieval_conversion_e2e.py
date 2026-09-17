# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Stage 1 conversion against the real ``data-designer-retrieval-sdg``.

Nothing is mocked here: a synthetic Stage 0 JSONL is fed through
``run_conversion_with_config`` and the resulting BEIR eval files plus the
Customizer-facing inline JSONL are asserted. Stage 0 (LLM calls) and GPU
hard-negative mining cannot run in unit tests, so this is the widest
non-cluster reproduction of the Nemotron recipe.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nemo_data_designer_plugin.jobs.retrieval_prepare import RetrievalPrepareJob
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalPrepareJobConfig, RetrievalPrepareStepConfig
from nemo_data_designer_plugin.retrieval.conversion import execute_conversion
from nemo_data_designer_plugin.retrieval.manifest import write_generation_manifest
from nmp.customization_common.retrieval.inline import wrapped_to_inline_jsonl


def _stage0_record(doc: str, chunks: list[str], questions: list[str]) -> dict:
    return {
        "file_name": [doc],
        "source_id": doc,
        "chunks": [{"chunk_id": idx, "text": text} for idx, text in enumerate(chunks)],
        "deduplicated_qa_pairs": [
            {"question": question, "segment_ids": [idx % len(chunks)]} for idx, question in enumerate(questions)
        ],
        "qa_evaluations": {"evaluations": [{"overall": {"score": 9}} for _ in questions]},
    }


@pytest.fixture
def stage0_jsonl(tmp_path: Path) -> Path:
    records = [
        _stage0_record(
            "alpha.txt",
            ["Photosynthesis converts light into chemical energy.", "Chlorophyll absorbs red and blue light."],
            ["How do plants store light energy?", "Which wavelengths does chlorophyll absorb?"],
        ),
        _stage0_record(
            "beta.txt",
            ["Mitochondria generate ATP.", "The Krebs cycle runs in the mitochondrial matrix."],
            ["Where is ATP produced?", "Where does the Krebs cycle occur?"],
        ),
        _stage0_record(
            "gamma.txt",
            ["Ribosomes translate mRNA into protein.", "tRNA carries amino acids."],
            ["What translates mRNA?", "What carries amino acids?"],
        ),
    ]
    path = tmp_path / "stage0" / "generated.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_conversion_produces_eval_beir_and_training_json(stage0_jsonl: Path, tmp_path: Path) -> None:
    out = tmp_path / "stage1"
    result = execute_conversion(
        input_path=stage0_jsonl,
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

    eval_dir = out / "eval_beir"
    assert (eval_dir / "corpus.jsonl").exists()
    assert (eval_dir / "queries.jsonl").exists()
    assert (eval_dir / "qrels" / "test.tsv").exists()

    qrels = (eval_dir / "qrels" / "test.tsv").read_text(encoding="utf-8").splitlines()
    assert qrels[0].split("\t") == ["query-id", "corpus-id", "score"]
    assert len(qrels) > 1

    train_file = Path(result.train_file)
    assert train_file.exists()
    payload = json.loads(train_file.read_text(encoding="utf-8"))
    assert payload["data"], "conversion produced no training records"
    record = payload["data"][0]
    assert set(record) >= {"question_id", "question", "corpus_id", "pos_doc"}


def test_inline_jsonl_resolves_document_text_from_real_conversion(stage0_jsonl: Path, tmp_path: Path) -> None:
    out = tmp_path / "stage1"
    result = execute_conversion(
        input_path=stage0_jsonl,
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

    inline = wrapped_to_inline_jsonl(Path(result.train_file), out / "training.jsonl")
    rows = [json.loads(line) for line in inline.read_text(encoding="utf-8").splitlines()]
    assert rows, "inline conversion produced no rows"
    for row in rows:
        assert row["query"]
        # pos_doc must be resolved document text, not a bare corpus id.
        assert row["pos_doc"], f"unresolved positive document: {row}"
        assert " " in row["pos_doc"], f"pos_doc looks like an id, not text: {row['pos_doc']!r}"


def test_prepare_job_convert_phase_runs_unmocked(stage0_jsonl: Path, tmp_path: Path) -> None:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    persistent.mkdir()
    ephemeral.mkdir()
    ctx = Mock()
    ctx.workspace = "default"
    ctx.storage.persistent = persistent
    ctx.storage.ephemeral = ephemeral
    ctx.results.save.return_value = SimpleNamespace(model_dump=lambda: {"name": "artifacts"})
    shutil.copytree(stage0_jsonl.parent, persistent / "stage0_sdg")
    write_generation_manifest(
        output_dir=persistent / "stage0_sdg",
        output_path=persistent / "stage0_sdg" / stage0_jsonl.name,
        dataset_name="retrieval_sdg",
    )

    step = RetrievalPrepareStepConfig(
        job_config=RetrievalPrepareJobConfig(sdg_input="stage0_sdg", enable_mining=False),
        phase="convert",
    )
    result = RetrievalPrepareJob().run(step.model_dump(mode="json"), ctx=ctx, sdk=Mock())

    assert result["exit_code"] == 0
    out = persistent / "stage1_data_prep"
    assert (out / "eval_beir" / "corpus.jsonl").exists()
    assert (out / "eval_beir" / "qrels" / "test.tsv").exists()
    assert (out / "additional" / "train.json").exists()
    assert (out / "training.jsonl").exists()

    rows = [json.loads(line) for line in (out / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["query"] and row["pos_doc"] for row in rows)
    ctx.results.save.assert_called_once()


def test_manifest_round_trip_feeds_conversion(stage0_jsonl: Path, tmp_path: Path) -> None:
    stage0_dir = stage0_jsonl.parent
    manifest = write_generation_manifest(
        output_dir=stage0_dir,
        output_path=stage0_jsonl,
        dataset_name="retrieval_sdg",
    )
    result = execute_conversion(
        input_path=manifest,
        output_dir=tmp_path / "from-manifest",
        corpus_id="retrieval_sdg",
        quality_threshold=7.0,
        train_ratio=0.8,
        val_ratio=0.0,
        seed=42,
        max_pos_docs=5,
        use_group_id_in_eval=False,
        split_strategy="random",
    )
    assert Path(result.train_file).exists()
