# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from nhx.automodel.tasks.training.datasets.preparation import (
    DatasetFormatError,
    discover_dataset_files,
    prepare_dataset,
    select_dataset_files,
)
from nhx.customization_common.dataset_prepare import (
    SPLIT_FILES,
    DatasetFormat,
    map_columns,
    split_rows,
    to_training_rows,
    write_split,
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


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["train.jsonl", "validation.jsonl"], (["train.jsonl"], ["validation.jsonl"])),
        (["/training/a.jsonl", "/val/b.jsonl", "README.md"], (["training/a.jsonl"], ["val/b.jsonl"])),
        (["data.jsonl"], (["data.jsonl"], [])),
        (["train.json", "training.jsonl"], (["training.jsonl"], [])),
        (["nested/deep/train.jsonl", "train.jsonl"], (["train.jsonl"], [])),
        (["train_part-0.jsonl", "dev.jsonl", "eval.jsonl", "test.jsonl"], (["train_part-0.jsonl"], ["dev.jsonl"])),
        (["data.jsonl", "data.json", "nested/rows.jsonl"], (["data.jsonl"], [])),
    ],
)
def test_select_dataset_files_picks_what_training_reads(
    paths: list[str], expected: tuple[list[str], list[str]]
) -> None:
    assert select_dataset_files(paths) == expected


def test_select_dataset_files_requires_training_files() -> None:
    with pytest.raises(DatasetFormatError, match="No training files found"):
        select_dataset_files(["validation.jsonl", "README.md"])


def test_prepare_dataset_validates_on_the_datasets_validation_file(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"prompt": f"t{i}", "completion": "a"} for i in range(4)])
    _write_jsonl(tmp_path / "validation.jsonl", [{"prompt": "held-out", "completion": "a"}] * 2)

    prepared = prepare_dataset(tmp_path, output_dir=tmp_path / "merged")

    assert (prepared.train_samples, prepared.validation_samples) == (4, 2)
    assert "held-out" in prepared.validation_file.read_text()
    assert "held-out" not in prepared.train_file.read_text()


def _prepared(tmp_path: Path, dataset_format: DatasetFormat, rows: list[dict], **named: str) -> Path:
    """Write *rows* the way ``nemo customization dataset prepare --format`` uploads them."""
    training_rows, _ = to_training_rows(rows, map_columns(dataset_format, rows[0].keys(), named))
    splits = split_rows(training_rows, val_fraction=0.2, test_fraction=0.0, seed=0)
    write_split(splits.train, tmp_path / SPLIT_FILES["train"])
    write_split(splits.validation, tmp_path / SPLIT_FILES["validation"])
    return tmp_path


def test_data_designer_instruction_output_is_found_and_validates_as_chat(tmp_path: Path) -> None:
    """The Studio "Instruction fine-tuning" template, prepared by the CLI, trains as automodel CHAT."""
    from nhx.automodel.entities.values import TrainingType
    from nhx.automodel.tasks.training.datasets.preparation import DatasetSchema, detect_dataset_schema
    from nhx.automodel.tasks.training.datasets.validation import DatasetValidator

    rows = [{"domain": "d", "instruction": f"q{i}", "response": f"a{i}"} for i in range(10)]

    train_files, val_files = discover_dataset_files(_prepared(tmp_path, DatasetFormat.CHAT, rows))

    assert [f.name for f in train_files] == ["train.jsonl"]
    assert [f.name for f in val_files] == ["validation.jsonl"]
    assert detect_dataset_schema(train_files[0]) == (DatasetSchema.CHAT, None)
    DatasetValidator(TrainingType.SFT).validate_dataset(str(train_files[0]))


def test_prepared_prompt_completion_validates_as_sft(tmp_path: Path) -> None:
    from nhx.automodel.entities.values import TrainingType
    from nhx.automodel.tasks.training.datasets.preparation import DatasetSchema, detect_dataset_schema
    from nhx.automodel.tasks.training.datasets.validation import DatasetValidator

    rows = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)]
    dataset = _prepared(tmp_path, DatasetFormat.PROMPT_COMPLETION, rows, prompt="question", response="answer")

    [train_file], _ = discover_dataset_files(dataset)
    assert detect_dataset_schema(train_file) == (DatasetSchema.SFT, ("prompt", "completion"))
    DatasetValidator(TrainingType.SFT).validate_dataset(str(train_file))


def test_prepared_embedding_rows_are_detected_as_retrieval_triplets(tmp_path: Path) -> None:
    from nhx.automodel.tasks.training.datasets.preparation import DatasetSchema, detect_dataset_schema

    rows = [{"query": f"q{i}", "pos_doc": "p", "neg_doc": ["n1", "n2"]} for i in range(10)]

    [train_file], _ = discover_dataset_files(_prepared(tmp_path, DatasetFormat.EMBEDDING, rows))
    assert detect_dataset_schema(train_file) == (DatasetSchema.EMBEDDING, ("query", "pos_doc", "neg_doc"))
