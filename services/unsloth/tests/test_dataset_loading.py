# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local dataset path resolution in the unsloth SFT driver.

These exercise ``_resolve_local_data_files`` only, which uses stdlib
``pathlib`` and pulls in no heavy ML deps — so the module imports fine on
a CPU box without ``unsloth``/``torch`` installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.unsloth.tasks.training.backends.unsloth_sft import _resolve_local_data_files


def test_single_file_passthrough(tmp_path: Path) -> None:
    f = tmp_path / "train.jsonl"
    f.write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(f)) == str(f)


def test_directory_expands_to_jsonl(tmp_path: Path) -> None:
    """Container submit hands the loader the download dir, not a file."""
    (tmp_path / "train.jsonl").write_text('{"text": "hi"}\n')
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "train.jsonl")]


def test_directory_prefers_jsonl_over_json(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "a.jsonl")]


def test_directory_falls_back_to_json(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [str(tmp_path / "data.json")]


def test_directory_returns_sorted_files(tmp_path: Path) -> None:
    for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text("{}\n")
    assert _resolve_local_data_files(str(tmp_path)) == [
        str(tmp_path / "a.jsonl"),
        str(tmp_path / "b.jsonl"),
        str(tmp_path / "c.jsonl"),
    ]


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .jsonl/.json files"):
        _resolve_local_data_files(str(tmp_path))


def test_split_train_picks_train_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="train") == str(tmp_path / "train.jsonl")


def test_split_validation_picks_validation_jsonl_only(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"text": "train"}\n')
    (tmp_path / "validation.jsonl").write_text('{"text": "val"}\n')
    assert _resolve_local_data_files(str(tmp_path), split="validation") == str(
        tmp_path / "validation.jsonl",
    )
