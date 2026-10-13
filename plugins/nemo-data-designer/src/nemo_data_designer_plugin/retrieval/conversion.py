# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nemo_data_designer_plugin.retrieval.manifest import resolve_generation_input

if TYPE_CHECKING:
    from data_designer_retrieval_sdg import ConversionResult

logger = logging.getLogger(__name__)

RETRIEVAL_SDG_SCHEMA_VERSION = 1
_QRELS_HEADER = ["query-id", "corpus-id", "score"]


def execute_conversion(
    input_path: Path,
    output_dir: Path,
    corpus_id: str,
    quality_threshold: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_pos_docs: int,
    use_group_id_in_eval: bool,
    split_strategy: Literal["random", "dedupped", "cluster"],
) -> ConversionResult:
    """Convert Stage 0 JSONL (or a generation manifest) into train/eval BEIR artifacts."""
    try:
        from data_designer_retrieval_sdg import ConversionRunConfig, run_conversion_with_config
    except ImportError as exc:
        raise ImportError("Retrieval prepare requires nemo-data-designer-plugin[retrieval-sdg].") from exc

    resolved = resolve_generation_input(input_path)
    config = ConversionRunConfig(
        schema_version=RETRIEVAL_SDG_SCHEMA_VERSION,
        input_path=resolved,
        corpus_id=corpus_id,
        output_dir=output_dir.resolve(),
        eval_only=False,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        quality_threshold=quality_threshold,
        max_pos_docs=max_pos_docs,
        use_group_id_in_eval=use_group_id_in_eval,
        split_strategy=split_strategy,
    )
    conversion = run_conversion_with_config(config)
    for qrels_path in output_dir.resolve().glob("**/qrels/test.tsv"):
        dropped = dedupe_beir_qrels(qrels_path)
        if dropped:
            logger.info("Dropped %s duplicate qrel rows from %s", dropped, qrels_path)
    return conversion


def dedupe_beir_qrels(path: Path) -> int:
    """Keep the first score for each ``(query-id, corpus-id)`` pair.

    Upstream conversion can emit the same judgment twice. The BEIR loader
    rejects that, so Stage 1 must write a unique-key TSV.
    """
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != _QRELS_HEADER:
            raise ValueError(f"{path}: expected TSV header {_QRELS_HEADER}, got {reader.fieldnames}")
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        dropped = 0
        for row in reader:
            key = (row["query-id"], row["corpus-id"])
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            rows.append(row)
    if dropped == 0:
        return 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_QRELS_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return dropped
