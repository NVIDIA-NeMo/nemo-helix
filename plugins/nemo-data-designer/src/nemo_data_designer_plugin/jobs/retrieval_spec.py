# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, Self

import data_designer.config as dd
from nemo_data_designer_plugin.retrieval.manifest import GENERATION_MANIFEST_FILENAME
from pydantic import BaseModel, Field, field_validator, model_validator

# Key sets required by data-designer-retrieval-sdg GenerationPipelineConfig.
DEFAULT_QUERY_COUNTS = {"multi_hop": 3, "structural": 2, "contextual": 2}
DEFAULT_REASONING_COUNTS = {
    "factual": 1,
    "relational": 1,
    "inferential": 1,
    "temporal": 1,
    "procedural": 1,
    "causal": 1,
    "visual": 1,
}


class RetrievalGenerateJobConfig(BaseModel):
    """Submitter-facing spec for Stage 0 retrieval SDG."""

    model_config = {"json_schema_mode_override": "validation"}

    corpus: str = Field(description="Fileset ref (workspace/fileset[#subdir]) or hf:// URI.")
    provider: str = Field(description="Default Inference Gateway provider (name or workspace/name).")
    chat_provider: str | None = Field(
        default=None,
        description="Optional provider override for artifact extraction, Q&A generation, and quality judging.",
    )
    embed_provider: str | None = Field(
        default=None,
        description="Optional provider override for embedding calls.",
    )
    corpus_id: str = "retrieval_sdg"
    dataset_name: str | None = None
    file_extensions: list[str] | None = None
    min_text_length: int = Field(default=50, ge=0)
    sentences_per_chunk: int = Field(default=5, ge=1)
    num_sections: int = Field(default=1, ge=1)
    num_files: int | None = Field(default=None, ge=1)
    max_artifacts_per_type: int = Field(default=2, ge=1)
    num_pairs: int = Field(default=7, ge=1)
    query_counts: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_QUERY_COUNTS))
    reasoning_counts: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_REASONING_COUNTS))
    min_hops: int = Field(default=1, ge=1)
    max_hops: int = Field(default=3, ge=1)
    min_complexity: int = Field(default=2, ge=1, le=5)
    similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    buffer_size: int = Field(default=200, ge=1)
    resume: Literal["never", "always", "if_possible"] = "never"
    num_records: int | None = Field(default=None, ge=1)
    artifact_extraction_model: str = Field(description="Chat model for artifact extraction.")
    qa_generation_model: str = Field(description="Chat model for question/answer generation.")
    quality_judge_model: str = Field(description="Chat model for quality judging.")
    embed_model: str = Field(description="Embedding model for retrieval SDG.")
    hf_token_secret: str | None = None

    @model_validator(mode="after")
    def validate_count_distributions(self) -> Self:
        # GenerationPipelineConfig requires exact key sets summing to num_pairs. Reject
        # here so a bad spec fails at submit instead of inside the running job.
        _validate_counts("query_counts", self.query_counts, DEFAULT_QUERY_COUNTS, self.num_pairs)
        _validate_counts("reasoning_counts", self.reasoning_counts, DEFAULT_REASONING_COUNTS, self.num_pairs)
        if self.max_hops < self.min_hops:
            raise ValueError("max_hops must be greater than or equal to min_hops")
        return self


def _validate_counts(name: str, counts: dict[str, int], expected: dict[str, int], num_pairs: int) -> None:
    actual_keys = set(counts)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} keys must match {sorted(expected_keys)}; missing={missing}, unexpected={unexpected}")
    total = sum(counts.values())
    if total != num_pairs:
        raise ValueError(f"{name} must sum to num_pairs ({num_pairs}); got {total}")


class RetrievalGenerateStepConfig(BaseModel):
    job_config: RetrievalGenerateJobConfig
    model_providers: list[dd.ModelProvider]
    chat_provider_name: str
    embed_provider_name: str


class RetrievalMiningOptions(BaseModel):
    """Automodel hard-negative mining knobs not covered by the prepare job's common fields."""

    hard_neg_margin_type: Literal["perc", "abs"] = "perc"
    query_embedding_batch_size: int = Field(default=16, ge=1)
    document_embedding_batch_size: int = Field(default=16, ge=1)
    corpus_chunk_size: int = Field(default=50000, ge=1)
    load_embeddings_from_cache: bool = False
    use_negatives_from_file: bool = False


class RetrievalPrepareJobConfig(BaseModel):
    """Submitter-facing spec for Stage 1 conversion and optional GPU mining."""

    model_config = {"json_schema_mode_override": "validation"}

    sdg_input: str | None = Field(
        default=None,
        description=(
            "Fileset or hf:// URI to Stage 0 output. Name a file with fileset#path or "
            "hf://org/dataset[@revision]/path; otherwise generation_file is used inside the directory."
        ),
    )
    generation_file: str = Field(
        default=GENERATION_MANIFEST_FILENAME,
        description=(
            "Relative path to the Stage 0 file inside sdg_input when the ref is a directory. "
            "Live generate writes generation_result.json. Skip-SDG dumps set this (e.g. nv_pp_dd_sdg.json) "
            "or put the filename on sdg_input."
        ),
    )
    train_input_file: str | None = Field(
        default=None,
        description="Fileset containing a pre-converted wrapped train.json; skips conversion.",
    )
    corpus_id: str = "retrieval_sdg"
    quality_threshold: float = Field(default=7.0, ge=0.0, le=10.0)
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42
    max_pos_docs: int = Field(default=5, ge=1)
    use_group_id_in_eval: bool = False
    split_strategy: Literal["random", "dedupped", "cluster"] = "random"
    enable_mining: bool = Field(
        default=False,
        description="When true, run GPU hard-negative mining after conversion. Conversion-only is the default.",
    )
    model: str = Field(
        default="nvidia/Nemotron-3-Embed-1B-BF16",
        description="Platform model entity whose fileset contains the mining encoder and tokenizer.",
    )
    hard_negatives_to_mine: int = Field(default=5, ge=1)
    hard_neg_margin: float = Field(default=0.95, gt=0.0, le=1.0)
    mining_batch_size: int = Field(default=128, ge=1)
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    query_max_length: int = Field(default=512, ge=1)
    passage_max_length: int = Field(default=512, ge=1)
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    add_bos_token: bool | None = True
    add_eos_token: bool | None = False
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = Field(default=30, ge=1)
    mining: RetrievalMiningOptions = Field(default_factory=RetrievalMiningOptions)
    hf_token_secret: str | None = None

    @field_validator("generation_file")
    @classmethod
    def validate_generation_file(cls, value: str) -> str:
        posix = PurePosixPath(value.replace("\\", "/"))
        if not value or posix.is_absolute() or any(part == ".." for part in posix.parts):
            raise ValueError("generation_file must be a relative path without '..' components")
        return posix.as_posix()


class RetrievalPrepareStepConfig(BaseModel):
    job_config: RetrievalPrepareJobConfig
    phase: Literal["convert", "mine"] = "convert"
    model_fileset: str | None = None
    model_trust_remote_code: bool = False


class RetrievalRunJobConfig(BaseModel):
    """Convenience spec that chains generate then prepare via the jobs service."""

    model_config = {"json_schema_mode_override": "validation"}

    generate: RetrievalGenerateJobConfig
    prepare: RetrievalPrepareJobConfig = Field(default_factory=RetrievalPrepareJobConfig)


class RetrievalPreviewSpec(BaseModel):
    generate: RetrievalGenerateJobConfig
    num_records: int = Field(default=1, ge=1)
