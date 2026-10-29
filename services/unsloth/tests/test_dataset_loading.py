# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local dataset path resolution in the unsloth SFT driver.

These exercise ``_resolve_local_data_files`` and ``_load_datasets``, which need
only ``pathlib`` and HF ``datasets`` — so the module imports fine on a CPU box
without ``unsloth``/``torch`` installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nhx.unsloth.tasks.training.backends.unsloth_sft import _load_datasets, _resolve_local_data_files


def test_single_file_passthrough(tmp_path: Path) -> None:
    f = tmp_path / "train.jsonl"
    f.write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(f)) == str(f)


def test_directory_expands_to_jsonl(tmp_path: Path) -> None:
    """Container submit hands the loader the download dir, not a file."""
    (tmp_path / "train.jsonl").write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(tmp_path)) == str(tmp_path / "train.jsonl")


def test_directory_prefers_jsonl_over_json(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == str(tmp_path / "a.jsonl")


def test_directory_falls_back_to_json(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == str(tmp_path / "data.json")


def test_directory_returns_sorted_files(tmp_path: Path) -> None:
    for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        str(tmp_path / "c.jsonl"),
    ]


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No training .jsonl/.json files"):
        _resolve_local_data_files(str(tmp_path))


def test_split_train_picks_train_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="train") == str(tmp_path / "train.jsonl")


@pytest.mark.parametrize(
    ("train_name", "val_name"), [("training.jsonl", "eval.jsonl"), ("my_train.jsonl", "dev.jsonl")]
)
def test_any_train_or_val_name_picks_its_split(tmp_path: Path, train_name: str, val_name: str) -> None:
    (tmp_path / train_name).write_text('{"text": "train"}\n')
    (tmp_path / val_name).write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="train") == str(tmp_path / train_name)
    assert _resolve_local_data_files(str(tmp_path), split="validation") == str(tmp_path / val_name)


def test_split_validation_picks_validation_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="validation") == str(
        tmp_path / "validation.jsonl",
    )


def test_no_validation_files_gives_no_validation_split(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="validation") is None
    assert _resolve_local_data_files(str(tmp_path / "train.jsonl"), split="validation") is None


def test_only_a_validation_file_has_no_training_data(tmp_path: Path) -> None:
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    with pytest.raises(FileNotFoundError, match="No training .jsonl/.json files"):
        _resolve_local_data_files(str(tmp_path), split="train")


def _spec(*, apply_chat_template: bool = False, validation_split: float | None = 0.1) -> Any:
    return SimpleNamespace(
        dataset=SimpleNamespace(apply_chat_template=apply_chat_template, text_field="text"),
        schedule=SimpleNamespace(validation_split=validation_split, seed=3407),
    )


def _load(path: Path, spec: Any) -> tuple[Any, Any]:
    from datasets import Dataset

    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, tokenize: messages[0]["content"])
    return _load_datasets(path=str(path), spec=spec, tokenizer=tokenizer, load_dataset=None, Dataset=Dataset)


def _write_rows(path: Path, texts: list[str]) -> None:
    path.write_text("".join(f'{{"text": "{text}"}}\n' for text in texts))


def test_a_validation_file_is_the_validation_data(tmp_path: Path) -> None:
    _write_rows(tmp_path / "train.jsonl", [f"t{i}" for i in range(10)])
    _write_rows(tmp_path / "validation.jsonl", ["v0", "v1"])

    train, validation = _load(tmp_path, _spec())

    assert (len(train), validation["text"]) == (10, ["v0", "v1"])


def test_without_a_validation_file_a_seeded_split_is_held_out(tmp_path: Path) -> None:
    _write_rows(tmp_path / "train.jsonl", [f"t{i}" for i in range(20)])

    train, validation = _load(tmp_path, _spec(validation_split=0.25))
    again_train, again_validation = _load(tmp_path, _spec(validation_split=0.25))

    assert (len(train), len(validation)) == (15, 5)
    assert not set(train["text"]) & set(validation["text"])
    assert (train["text"], validation["text"]) == (again_train["text"], again_validation["text"])


def test_a_single_training_row_cannot_be_split(tmp_path: Path) -> None:
    _write_rows(tmp_path / "train.jsonl", ["only"])
    with pytest.raises(ValueError, match="too few to hold out 10% for validation"):
        _load(tmp_path, _spec())


def test_the_chat_template_renders_both_splits(tmp_path: Path) -> None:
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(4)]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "validation.jsonl").write_text(json.dumps(rows[0]) + "\n")

    train, validation = _load(tmp_path, _spec(apply_chat_template=True))

    assert (train["text"], validation["text"]) == (["q0", "q1", "q2", "q3"], ["q0"])
