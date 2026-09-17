# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run hard-negative mining, unroll multi-positive records, and emit training JSONL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from nemo_platform_plugin.job_context import JobContext
from nmp.automodel.tasks.retrieval_mine.launch import run_hard_negative_mining
from nmp.customization_common.retrieval.inline import move_aux_files_to_additional, wrapped_to_inline_jsonl
from nmp.customization_common.retrieval.unroll import unroll_training_file
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class RetrievalMiningOptions(BaseModel):
    """Automodel hard-negative mining knobs not covered by the prepare job's common fields."""

    hard_neg_margin_type: Literal["perc", "abs"] = "perc"
    query_embedding_batch_size: int = Field(default=16, ge=1)
    document_embedding_batch_size: int = Field(default=16, ge=1)
    corpus_chunk_size: int = Field(default=50000, ge=1)
    load_embeddings_from_cache: bool = False
    use_negatives_from_file: bool = False


class RetrievalMineJobConfig(BaseModel):
    """Spec for the retrieval hard-negative mining step."""

    model_config = ConfigDict(extra="ignore")

    model: str = "nvidia/Nemotron-3-Embed-1B-BF16"
    hard_negatives_to_mine: int = Field(default=5, ge=1)
    hard_neg_margin: float = Field(default=0.95, gt=0.0, le=1.0)
    mining_batch_size: int = Field(default=128, ge=1)
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    query_max_length: int = Field(default=512, ge=1)
    passage_max_length: int = Field(default=512, ge=1)
    attn_implementation: str = "sdpa"
    add_bos_token: bool | None = True
    add_eos_token: bool | None = False
    dist_backend: str = "nccl"
    dist_timeout_minutes: int = Field(default=30, ge=1)
    mining: RetrievalMiningOptions = Field(default_factory=RetrievalMiningOptions)

    def to_mining_config(
        self,
        model_path: Path,
        trust_remote_code: bool,
        train_file: Path,
        output_file: Path,
        cache_dir: Path,
    ) -> dict[str, Any]:
        """Build the recipe config the Automodel miner loads."""
        mining: dict[str, Any] = {
            "model_name_or_path": str(model_path),
            "tokenizer_name_or_path": str(model_path),
            "trust_remote_code": trust_remote_code,
            "train_qa_file_path": str(train_file),
            "train_file_output_path": str(output_file),
            "cache_embeddings_dir": str(cache_dir),
            "hard_negatives_to_mine": self.hard_negatives_to_mine,
            "hard_neg_margin": self.hard_neg_margin,
            "mining_batch_size": self.mining_batch_size,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "query_max_length": self.query_max_length,
            "passage_max_length": self.passage_max_length,
            "attn_implementation": self.attn_implementation,
            "add_bos_token": self.add_bos_token,
            "add_eos_token": self.add_eos_token,
        }
        mining.update(self.mining.model_dump(exclude_none=True))
        return {
            "dist_env": {"backend": self.dist_backend, "timeout_minutes": self.dist_timeout_minutes},
            "mining": mining,
        }


class RetrievalMineStepConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_config: RetrievalMineJobConfig
    model_fileset: str
    model_trust_remote_code: bool = False


def work_dir(ctx: JobContext, name: str) -> Path:
    base = ctx.storage.persistent or ctx.storage.ephemeral
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_mine(
    job: RetrievalMineJobConfig,
    output_dir: Path,
    ctx: JobContext,
    model_trust_remote_code: bool,
) -> dict[str, Any]:
    train_file = output_dir / "train.json"
    if not train_file.exists():
        matches = sorted(
            output_dir.rglob("train.json"),
            key=lambda path: (len(path.relative_to(output_dir).parts), path.as_posix()),
        )
        if not matches:
            raise FileNotFoundError(f"train.json not found under {output_dir}")
        train_file = matches[0]
    model_path = output_dir.parent / "model"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Staged model directory not found: {model_path}")
    mined = output_dir / "train_mined.automodel.json"
    mining_config = job.to_mining_config(
        model_path=model_path,
        trust_remote_code=model_trust_remote_code,
        train_file=train_file,
        output_file=mined,
        cache_dir=output_dir / "cache_embeddings",
    )
    # Written next to the outputs so the exact config lands in the job artifacts.
    config_file = output_dir / "mining_config.yaml"
    config_file.write_text(yaml.safe_dump(mining_config, sort_keys=False), encoding="utf-8")
    logger.info("Wrote mining config %s; starting torchrun miner", config_file)
    run_hard_negative_mining(config_file=config_file)
    logger.info("Miner finished; unrolling %s", mined)
    unrolled = unroll_training_file(mined, output_dir / "train_mined.automodel_unrolled.json")
    training_jsonl = output_dir / "training.jsonl"
    wrapped_to_inline_jsonl(
        unrolled,
        training_jsonl,
        output_dir / "corpus" / "train.parquet",
    )
    with training_jsonl.open(encoding="utf-8") as handle:
        if next(handle, None) is None:
            raise ValueError(f"No training rows written to {training_jsonl}")
    move_aux_files_to_additional(output_dir)
    unrolled = output_dir / "additional" / unrolled.name
    artifacts = ctx.results.save(name="artifacts", local_path=output_dir)
    return {
        "exit_code": 0,
        "workspace": ctx.workspace,
        "train_file": str(unrolled),
        "results": {"artifacts": artifacts.model_dump()},
    }
