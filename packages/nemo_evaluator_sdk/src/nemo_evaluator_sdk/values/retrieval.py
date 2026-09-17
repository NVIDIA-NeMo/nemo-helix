# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retrieval target: embedding NIM plus optional reranker."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_evaluator_sdk.values.models import Model

__all__ = ["Retrieval", "Truncation"]

Truncation = Literal["end", "start"]


class Retrieval(BaseModel):
    """Target that ranks a BEIR corpus with dense search and optional reranking."""

    model_config = ConfigDict(extra="forbid")

    embeddings: Model = Field(description="Embedding NIM used to encode queries and passages.")
    reranker: Model | None = Field(default=None, description="Optional ranking NIM applied after dense search.")
    first_stage_k: int = Field(default=100, ge=1, description="Dense-search cutoff before reranking.")
    truncate_long_documents: Truncation | None = Field(
        default="end",
        description="How to cap passages at 65535 characters: keep the start ('end'), the tail ('start'), or error (null).",
        json_schema_extra={"nullable": True},
    )
    batch_size: int = Field(default=32, ge=1, description="Embedding HTTP batch size.")
    embedding_in_flight: int = Field(
        default=2,
        ge=1,
        description="Concurrent embedding POSTs to one NIM so the GPU is not idle between batches.",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Expected embedding width. Omit to accept the model's native width.",
    )
    rankings: dict[str, dict[str, float]] | None = Field(
        default=None,
        exclude=True,
        description="Runtime rankings filled after corpus encode; not part of the public spec.",
    )
