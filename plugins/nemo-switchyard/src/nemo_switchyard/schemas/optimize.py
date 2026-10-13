# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spec for the ``switchyard`` optimization strategy.

The router (``nemo agents optimize run-strategy``) forwards every submitted field but
``strategy`` here, so this schema decides what is required.  Fields it does not declare --
the router's ``output`` and ``optimize_config`` -- are refused rather than silently dropped:
the artifacts only ever land in the job's own results.
"""

from __future__ import annotations

from typing import Annotated, Literal

from nemo_helix_plugin.refs import ENTITY_REF_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator

RoutingStrategy = Literal["random_routing", "stage_router", "llm_classifier"]
ModelRef = Annotated[str, Field(min_length=1, pattern=ENTITY_REF_PATTERN)]


class SwitchyardOptimizeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(
        min_length=1,
        pattern=ENTITY_REF_PATTERN,
        description="Platform agent to route ('name' or 'workspace/name').  The stored agent is read, never modified.",
    )
    models: list[ModelRef] = Field(
        min_length=2,
        description="Acceptable model entities ('name' or 'workspace/name'), ordered from most capable to most "
        "efficient.  Every pair is routed with the earlier model as the capable one.",
    )
    routing_strategies: list[RoutingStrategy] = Field(
        default=["random_routing"],
        min_length=1,
        description="Switchyard routing strategies to build a VirtualModel for, per model pair.",
    )
    strong_probability: float = Field(
        default=0.5, ge=0, le=1, description="random_routing: share of requests sent to the capable model."
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0, le=1, description="stage_router: confidence below which a request escalates."
    )
    base_threshold: float = Field(
        default=0.5, ge=0, le=1, description="llm_classifier: capability score above which a request escalates."
    )
    judge_model: ModelRef | None = Field(
        default=None, description="llm_classifier judge model; defaults to the capable model of each pair."
    )
    workspace: str = Field(
        default="default",
        description="Workspace the agent, models, and the created VirtualModels are resolved in when a reference "
        "names none.",
    )

    @field_validator("models", "routing_strategies")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("entries must be unique")
        return value
