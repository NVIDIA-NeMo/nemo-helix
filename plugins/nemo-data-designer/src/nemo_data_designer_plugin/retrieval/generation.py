# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import data_designer.config as dd
from nemo_data_designer_plugin.retrieval.manifest import write_generation_manifest

if TYPE_CHECKING:
    from data_designer_retrieval_sdg import GenerationPreviewResult, GenerationResult, GenerationRunConfig

RETRIEVAL_SDG_SCHEMA_VERSION = 1


def build_generation_run_config(
    corpus_dir: Path,
    output_dir: Path,
    artifact_path: Path,
    dataset_name: str,
    chat_provider_name: str,
    embed_provider_name: str,
    model_providers: list[dd.ModelProvider],
    file_extensions: list[str] | None,
    min_text_length: int,
    sentences_per_chunk: int,
    num_sections: int,
    num_files: int | None,
    max_artifacts_per_type: int,
    num_pairs: int,
    query_counts: dict[str, int],
    min_hops: int,
    max_hops: int,
    reasoning_counts: dict[str, int],
    min_complexity: int,
    similarity_threshold: float,
    buffer_size: int,
    resume: str,
    num_records: int | None,
    artifact_extraction_model: str,
    qa_generation_model: str,
    quality_judge_model: str,
    embed_model: str,
    multi_doc: bool = False,
    bundle_size: int = 2,
    bundle_strategy: Literal["sequential", "doc_balanced", "interleaved"] = "sequential",
    max_docs_per_bundle: int = 3,
) -> GenerationRunConfig:
    """Translate a platform retrieval generate spec into ``GenerationRunConfig``."""
    try:
        from data_designer_retrieval_sdg import (
            DocumentChunkerSeedSource,
            GenerationPipelineConfig,
            GenerationRunConfig,
        )
    except ImportError as exc:
        raise ImportError("Retrieval generate and preview requires nemo-data-designer-plugin[retrieval-sdg].") from exc

    extensions = file_extensions or [".txt", ".md", ".text", ""]
    seed_source = DocumentChunkerSeedSource(
        path=str(corpus_dir),
        file_pattern="*",
        recursive=True,
        file_extensions=extensions,
        min_text_length=min_text_length,
        sentences_per_chunk=sentences_per_chunk,
        num_sections=num_sections,
        num_files=num_files,
        multi_doc=multi_doc,
        bundle_size=bundle_size,
        bundle_strategy=bundle_strategy,
        max_docs_per_bundle=max_docs_per_bundle,
    )
    pipeline = GenerationPipelineConfig(
        max_artifacts_per_type=max_artifacts_per_type,
        num_pairs=num_pairs,
        query_counts=query_counts,
        min_hops=min_hops,
        max_hops=max_hops,
        reasoning_counts=reasoning_counts,
        min_complexity=min_complexity,
        similarity_threshold=similarity_threshold,
        artifact_extraction_model=artifact_extraction_model,
        artifact_extraction_provider=chat_provider_name,
        qa_generation_model=qa_generation_model,
        qa_generation_provider=chat_provider_name,
        quality_judge_model=quality_judge_model,
        quality_judge_provider=chat_provider_name,
        embed_model=embed_model,
        embed_provider=embed_provider_name,
    )
    return GenerationRunConfig(
        schema_version=RETRIEVAL_SDG_SCHEMA_VERSION,
        seed_source=seed_source,
        output_dir=output_dir,
        artifact_path=artifact_path,
        dataset_name=dataset_name,
        buffer_size=buffer_size,
        resume=resume,  # type: ignore[arg-type]
        model_providers=model_providers,
        pipeline=pipeline,
        num_records=num_records,
        log_level="INFO",
    )


def execute_generation(
    config: GenerationRunConfig, preview: bool = False, num_records: int = 1
) -> GenerationResult | GenerationPreviewResult:
    """Run or preview generation. ``num_records`` sizes the preview and is ignored for a full run."""
    try:
        from data_designer_retrieval_sdg import preview_generation, run_generation
    except ImportError as exc:
        raise ImportError("Retrieval generate and preview requires nemo-data-designer-plugin[retrieval-sdg].") from exc

    if preview:
        return preview_generation(config, num_records)
    result = run_generation(config)
    write_generation_manifest(
        output_dir=config.output_dir,
        output_path=result.output_path,
        dataset_name=result.dataset_name,
    )
    return result
