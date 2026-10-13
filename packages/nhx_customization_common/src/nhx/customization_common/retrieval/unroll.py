# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand multi-positive retrieval records into one-positive-per-row examples."""

import json
from pathlib import Path
from typing import Any


def unroll_training_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unroll records with multiple positive docs into one-positive records."""
    unrolled: list[dict[str, Any]] = []
    for record in data:
        pos_docs = record.get("pos_doc") or []
        if len(pos_docs) <= 1:
            unrolled.append(record)
            continue
        base_question_id = record.get("question_id", "")
        for idx, pos_doc in enumerate(pos_docs):
            unrolled.append(
                {
                    "question_id": f"{base_question_id}_{idx}",
                    "question": record["question"],
                    "corpus_id": record["corpus_id"],
                    "pos_doc": [pos_doc],
                    "neg_doc": record.get("neg_doc", []),
                }
            )
    return unrolled


def unroll_training_file(input_path: Path, output_path: Path) -> Path:
    training_data = json.loads(input_path.read_text(encoding="utf-8"))
    corpus_info = training_data.get("corpus", {})
    data = training_data.get("data", [])
    output = {"corpus": corpus_info, "data": unroll_training_data(data)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path
