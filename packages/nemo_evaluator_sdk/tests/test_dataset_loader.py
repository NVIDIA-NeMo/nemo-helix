# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from nemo_evaluator_sdk.datasets.loader import load_dataset_as_dicts


def test_concatenates_parquet_batches_whose_column_types_differ(tmp_path: Path) -> None:
    pq.write_table(pa.table({"prompt": ["a"], "score": pa.array([1], pa.int64())}), tmp_path / "batch_00000.parquet")
    pq.write_table(
        pa.table({"prompt": ["b"], "score": pa.array([2.5], pa.float64())}), tmp_path / "batch_00001.parquet"
    )

    assert load_dataset_as_dicts(tmp_path, None) == [
        {"prompt": "a", "score": 1.0},
        {"prompt": "b", "score": 2.5},
    ]


def test_fills_columns_missing_from_some_parquet_batches(tmp_path: Path) -> None:
    pq.write_table(pa.table({"prompt": ["a"], "notes": ["x"]}), tmp_path / "batch_00000.parquet")
    pq.write_table(pa.table({"prompt": ["b"]}), tmp_path / "batch_00001.parquet")

    assert load_dataset_as_dicts(tmp_path, None) == [
        {"prompt": "a", "notes": "x"},
        {"prompt": "b", "notes": None},
    ]
