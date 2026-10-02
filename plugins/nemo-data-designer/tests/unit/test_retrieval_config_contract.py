# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests against the real ``data-designer-retrieval-sdg`` schemas.

These deliberately avoid mocking ``GenerationRunConfig`` / ``ConversionRunConfig``
so upstream field renames or new validators fail here instead of inside a job.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import data_designer.config as dd
import pytest
from nemo_data_designer_plugin.jobs.retrieval_spec import (
    DEFAULT_QUERY_COUNTS,
    DEFAULT_REASONING_COUNTS,
    RetrievalGenerateJobConfig,
)
from nemo_data_designer_plugin.retrieval.conversion import execute_conversion
from nemo_data_designer_plugin.retrieval.generation import build_generation_run_config
from pydantic import ValidationError

_CHAT = "nvidia/nemotron-3-nano-30b-a3b"
_EMBED = "nvidia/nemotron-3-embed-1b"


def _job(**overrides: object) -> RetrievalGenerateJobConfig:
    payload = {
        "corpus": "default/docs",
        "provider": "default/nvidia-build",
        "artifact_extraction_model": _CHAT,
        "qa_generation_model": _CHAT,
        "quality_judge_model": _CHAT,
        "embed_model": _EMBED,
    }
    payload.update(overrides)
    return RetrievalGenerateJobConfig.model_validate(payload)


def _build(job: RetrievalGenerateJobConfig, tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    chat_provider = job.chat_provider or job.provider
    embed_provider = job.embed_provider or job.provider
    provider_names = dict.fromkeys((chat_provider, embed_provider))
    return build_generation_run_config(
        corpus_dir=corpus,
        output_dir=tmp_path / "out",
        artifact_path=tmp_path / "art",
        dataset_name=job.dataset_name or job.corpus_id,
        chat_provider_name=chat_provider,
        embed_provider_name=embed_provider,
        model_providers=[dd.ModelProvider(name=name, endpoint=f"http://igw/{name}") for name in provider_names],
        file_extensions=job.file_extensions,
        min_text_length=job.min_text_length,
        sentences_per_chunk=job.sentences_per_chunk,
        num_sections=job.num_sections,
        num_files=job.num_files,
        max_artifacts_per_type=job.max_artifacts_per_type,
        num_pairs=job.num_pairs,
        query_counts=job.query_counts,
        min_hops=job.min_hops,
        max_hops=job.max_hops,
        reasoning_counts=job.reasoning_counts,
        min_complexity=job.min_complexity,
        similarity_threshold=job.similarity_threshold,
        max_parallel_requests_for_gen=job.max_parallel_requests_for_gen,
        buffer_size=job.buffer_size,
        resume=job.resume,
        num_records=job.num_records,
        artifact_extraction_model=job.artifact_extraction_model,
        qa_generation_model=job.qa_generation_model,
        quality_judge_model=job.quality_judge_model,
        embed_model=job.embed_model,
        multi_doc=job.multi_doc,
        bundle_size=job.bundle_size,
        bundle_strategy=job.bundle_strategy,
        max_docs_per_bundle=job.max_docs_per_bundle,
    )


def test_platform_defaults_match_upstream_count_keys() -> None:
    from data_designer_retrieval_sdg.run_config import DEFAULT_QUERY_COUNTS as UPSTREAM_QUERY
    from data_designer_retrieval_sdg.run_config import DEFAULT_REASONING_COUNTS as UPSTREAM_REASONING

    assert DEFAULT_QUERY_COUNTS == UPSTREAM_QUERY
    assert DEFAULT_REASONING_COUNTS == UPSTREAM_REASONING


def test_spec_builds_a_real_generation_run_config(tmp_path: Path) -> None:
    job = _job(corpus=str(tmp_path))
    config = _build(job, tmp_path)

    assert [p.name for p in config.model_providers] == ["default/nvidia-build"]
    assert config.pipeline.qa_generation_provider == "default/nvidia-build"
    assert config.pipeline.embed_provider == "default/nvidia-build"
    assert config.pipeline.qa_generation_model == _CHAT
    assert config.pipeline.embed_model == _EMBED
    assert type(config.seed_source).__name__ == "DocumentChunkerSeedSource"


def test_chat_and_embedding_providers_can_be_split(tmp_path: Path) -> None:
    job = _job(
        corpus=str(tmp_path),
        provider="default/local-chat",
        embed_provider="default/local-embed",
    )
    config = _build(job, tmp_path)

    assert [p.name for p in config.model_providers] == ["default/local-chat", "default/local-embed"]
    assert config.pipeline.artifact_extraction_provider == "default/local-chat"
    assert config.pipeline.qa_generation_provider == "default/local-chat"
    assert config.pipeline.quality_judge_provider == "default/local-chat"
    assert config.pipeline.embed_provider == "default/local-embed"


def test_quality_controls_reach_generation_config(tmp_path: Path) -> None:
    job = _job(
        corpus=str(tmp_path),
        file_extensions=[".txt", ".text"],
        sentences_per_chunk=8,
        multi_doc=True,
        bundle_size=3,
        bundle_strategy="doc_balanced",
        max_docs_per_bundle=2,
        max_artifacts_per_type=4,
        max_parallel_requests_for_gen=6,
    )
    config = _build(job, tmp_path)

    assert config.seed_source.file_extensions == [".txt", ".text"]
    assert config.seed_source.sentences_per_chunk == 8
    assert config.seed_source.multi_doc is True
    assert config.seed_source.bundle_size == 3
    assert config.seed_source.bundle_strategy == "doc_balanced"
    assert config.seed_source.max_docs_per_bundle == 2
    assert config.pipeline.max_artifacts_per_type == 4
    assert config.pipeline.max_parallel_requests_for_gen == 6


def test_count_distribution_mismatch_is_rejected_at_spec_time() -> None:
    with pytest.raises(ValidationError, match="must sum to num_pairs"):
        _job(num_pairs=9)

    with pytest.raises(ValidationError, match="keys must match"):
        _job(query_counts={"multi_hop": 7})


def test_models_are_required() -> None:
    with pytest.raises(ValidationError):
        RetrievalGenerateJobConfig.model_validate({"corpus": "default/docs", "provider": "default/nvidia-build"})


def test_custom_counts_that_sum_correctly_build_a_real_config(tmp_path: Path) -> None:
    job = _job(
        corpus=str(tmp_path),
        num_pairs=10,
        query_counts={"multi_hop": 4, "structural": 3, "contextual": 3},
        reasoning_counts={
            "factual": 4,
            "relational": 1,
            "inferential": 1,
            "temporal": 1,
            "procedural": 1,
            "causal": 1,
            "visual": 1,
        },
    )
    config = _build(job, tmp_path)
    assert config.pipeline.num_pairs == 10


def test_conversion_defaults_build_a_real_conversion_run_config(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    with patch("data_designer_retrieval_sdg.run_conversion_with_config") as run_conversion:
        execute_conversion(
            input_path=src,
            output_dir=tmp_path / "conv",
            corpus_id="retrieval_sdg",
            quality_threshold=7.0,
            train_ratio=0.8,
            val_ratio=0.0,
            seed=42,
            max_pos_docs=5,
            use_group_id_in_eval=False,
            split_strategy="random",
        )
    config = run_conversion.call_args[0][0]
    assert type(config).__name__ == "ConversionRunConfig"
    assert (config.quality_threshold, config.train_ratio, config.val_ratio, config.seed) == (7.0, 0.8, 0.0, 42)
    assert config.split_strategy == "random"
