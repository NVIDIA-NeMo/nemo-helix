# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dataset preparation (train/validation auto-split)."""

import json
from pathlib import Path

import pytest
from nhx.rl.tasks.training.datasets.preparation import DatasetFormatError, _create_val_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_create_val_split_in_place_does_not_truncate_source(tmp_path: Path) -> None:
    """train_file == output_train (auto-split-from-merged) must not destroy the source.

    Regression: a streaming split that opens output_train for write before reading
    truncates the file when both paths are the same, yielding empty splits.
    """
    rows = [{"prompt": f"q{i}", "chosen": f"good{i}", "rejected": f"bad{i}"} for i in range(10)]
    train = tmp_path / "train.jsonl"
    val = tmp_path / "validation.jsonl"
    _write_jsonl(train, rows)

    # Same path for source and train output — the auto-split-from-merged case.
    train_n, val_n = _create_val_split(train, train, val, val_ratio=0.2, seed=1234)

    assert train_n + val_n == len(rows)
    assert train_n > 0 and val_n > 0, "in-place split produced an empty side (source was truncated)"
    assert len(_read_jsonl(train)) == train_n
    assert len(_read_jsonl(val)) == val_n
    # No leftover temp file from the atomic replace.
    assert not (tmp_path / "train.jsonl.tmp").exists()


def test_create_val_split_distinct_paths(tmp_path: Path) -> None:
    """Distinct source/output paths split correctly and leave the source intact."""
    rows = [{"prompt": f"q{i}", "chosen": f"good{i}", "rejected": f"bad{i}"} for i in range(10)]
    src = tmp_path / "source.jsonl"
    train_out = tmp_path / "merged" / "train.jsonl"
    val_out = tmp_path / "merged" / "validation.jsonl"
    _write_jsonl(src, rows)

    train_n, val_n = _create_val_split(src, train_out, val_out, val_ratio=0.1, seed=1234)

    assert train_n + val_n == len(rows)
    assert train_n > 0 and val_n > 0
    assert len(_read_jsonl(src)) == len(rows), "source file must be left untouched"


def test_create_val_split_is_deterministic(tmp_path: Path) -> None:
    """Same seed → same split, and the local RNG doesn't depend on global state."""
    rows = [{"prompt": f"q{i}", "chosen": f"good{i}", "rejected": f"bad{i}"} for i in range(20)]
    src = tmp_path / "source.jsonl"
    _write_jsonl(src, rows)

    def split_prompts(suffix: str) -> list[str]:
        val = tmp_path / f"val{suffix}.jsonl"
        _create_val_split(src, tmp_path / f"train{suffix}.jsonl", val, val_ratio=0.25, seed=99)
        return sorted(r["prompt"] for r in _read_jsonl(val))

    assert split_prompts("a") == split_prompts("b")


def test_create_val_split_rejects_too_small(tmp_path: Path) -> None:
    """Fewer than 2 rows cannot be split into non-empty train + validation."""
    src = tmp_path / "train.jsonl"
    _write_jsonl(src, [{"prompt": "q", "chosen": "a", "rejected": "b"}])

    with pytest.raises(DatasetFormatError):
        _create_val_split(src, src, tmp_path / "validation.jsonl")
