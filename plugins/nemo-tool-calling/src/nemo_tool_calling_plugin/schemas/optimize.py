# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``tool-calling`` optimization strategy.

The router forwards every submitted field but ``strategy``. The shared fields keep the router's names, so the
prompt dataset arrives through ``--optimize-config`` / ``--optimize-config-fileset``; the rest travel in
``--spec``.
"""

from __future__ import annotations

from typing import Literal, Self

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_ANALYZER_MODEL = "default/nemotron3-5-super"


class PromptRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class ToolCallingOptimizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent whose tool calls are scored ('name' or 'workspace/name'). Never modified.",
    )
    optimize_config: str | None = Field(
        default=None,
        min_length=1,
        description='Prompt dataset: a JSONL file of {"id": str, "prompt": str} rows, as a path relative to the '
        "root of optimize_config_fileset.",
    )
    optimize_config_fileset: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        description="Fileset holding the prompt dataset ('name' or 'workspace/name').",
    )
    prompts: list[PromptRow] | None = Field(
        default=None,
        min_length=1,
        description="Inline prompt dataset, instead of optimize_config + optimize_config_fileset.",
    )
    runs_per_prompt: int = Field(default=3, ge=1, le=20, description="Runs of the agent per prompt and variant.")
    max_prompts: int | None = Field(default=None, ge=1, description="Use only the first N prompts.")
    max_turns: int = Field(
        default=30,
        ge=1,
        description="runtime.max_turns applied to every variant. A run that hits it counts as an errored run.",
    )
    call_budget: int = Field(
        default=3,
        ge=1,
        description="Tool calls a run may make before it counts as a retry loop (metrics.signals.retry_loop_runs).",
    )
    timeout_seconds: float = Field(default=600.0, gt=0, description="Wall-clock limit for one agent run.")
    concurrency: int = Field(default=1, ge=1, le=16, description="Agent runs in flight at once.")
    analyzer_model: str = Field(
        default=DEFAULT_ANALYZER_MODEL,
        min_length=1,
        description="Inference Gateway model that diagnoses failures and proposes the config fix and schema patch.",
    )
    verify: list[Literal["config-fix", "schema-fix"]] = Field(
        default=["config-fix", "schema-fix"],
        description="Suggestion variants to re-run. Leave one out to skip its runs; it is still suggested.",
    )
    workspace: str = Field(
        default="default",
        description="Workspace the agent and the dataset fileset are resolved in when their references name none.",
    )

    @model_validator(mode="after")
    def _one_dataset(self) -> Self:
        from_fileset = self.optimize_config is not None or self.optimize_config_fileset is not None
        if self.prompts is not None and from_fileset:
            raise ValueError("Pass either prompts or optimize_config + optimize_config_fileset, not both.")
        if self.prompts is None and (self.optimize_config is None or self.optimize_config_fileset is None):
            raise ValueError("A prompt dataset is required: optimize_config + optimize_config_fileset, or prompts.")
        return self
