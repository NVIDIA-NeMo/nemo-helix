# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strategy's own config: the YAML named by ``optimize_config`` inside the bundle fileset."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrandsHarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(
        description="JSONL file relative to this YAML, in its directory or below.  Each row is "
        '{"input": str, "expected_output": str} with an optional "id".'
    )
    epochs: int = Field(default=2, ge=1, description="Rollout-and-reflect passes over the dataset.")
    batch_size: int = Field(default=8, ge=1, description="Rows per rollout batch.")
    num_workers: int = Field(default=1, ge=1, description="Parallel agent invocations per batch.")
    max_samples: int | None = Field(default=None, ge=1, description="Use only the first N rows.")
    optimizer_model: str | None = Field(
        default=None,
        description="Gateway model ID for the reflection step that proposes each new prompt.  "
        "Defaults to the agent's own model.",
    )
