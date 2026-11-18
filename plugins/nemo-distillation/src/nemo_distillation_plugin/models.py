# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow entity and the graph stored on it."""

from __future__ import annotations

from typing import Any, Literal

from nemo_helix_plugin.entity import NemoEntity
from pydantic import BaseModel, ConfigDict, Field, model_validator

NODE_KINDS = frozenset(
    {
        "convert_atif",
        "customization_job",
        "model_deployment",
        "delete_deployment",
        "agent_eval",
        "experiment",
    }
)

NODE_STATUSES = frozenset({"pending", "running", "succeeded", "failed", "skipped", "cancelled"})
TERMINAL_NODE_STATUSES = frozenset({"succeeded", "failed", "skipped", "cancelled"})
WORKFLOW_STATUSES = frozenset({"pending", "running", "cancelling", "completed", "failed", "cancelled"})

#: The convert job writes its own ``output.json``. Every other kind returns the confirmed
#: output from the external API, and the controller writes the file.
OWN_OUTPUT_KINDS = frozenset({"convert_atif"})

FABRIC_AGENT_CONFIG_FORMAT = "nemo-agents-spec-v1"
CONFIRM_GRACE_SECONDS = 60.0
MAX_RESOURCE_NAME_LEN = 55


class GraphError(ValueError):
    """The graph, its bindings, or the names it needs cannot be built."""


class DeploymentInput(BaseModel):
    """What a deploy node needs in order to create a deployment config."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["vllm", "nim"] = "vllm"
    gpu: int = Field(default=1, gt=0)
    tool_call_config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _tool_calls_need_nim(self) -> DeploymentInput:
        if self.tool_call_config is not None and self.engine != "nim":
            raise ValueError("tool_call_config is only supported with engine='nim'")
        return self


class CandidateSpec(BaseModel):
    """One training candidate. The default graph has a single LoRA candidate named ``lora``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    finetuning_type: Literal["lora", "all_weights"] = "lora"


class DistillInputs(BaseModel):
    """Caller-supplied inputs for the distillation template."""

    model_config = ConfigDict(extra="forbid")

    source_agent: str = Field(min_length=1)
    student_model: str = Field(min_length=1)
    eval_results: list[str] = Field(default_factory=list)
    pass_filter: dict[str, Any] = Field(default_factory=dict)
    candidates: list[CandidateSpec] = Field(default_factory=list)
    deployment: DeploymentInput = Field(default_factory=DeploymentInput)
    prices: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """One node of the expanded graph. ``matrix`` is only set before expansion."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    always_run: bool = False
    matrix: dict[str, list[str]] | Literal["inherit"] | None = None


class RuntimeNode(BaseModel):
    """Live state of one node. ``external`` is assigned at creation and reused on retry."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled"] = "pending"
    attempt: int = 1
    external: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    confirm_since: float | None = None


class Workflow(NemoEntity, entity_type="workflow"):
    """A distillation run: the expanded graph, per-node state, and the fileset the nodes share."""

    template: str = "distillation"
    status: Literal["pending", "running", "cancelling", "completed", "failed", "cancelled"] = "pending"
    source_agent: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    fileset: str = ""
    graph: dict[str, GraphNode] = Field(default_factory=dict)
    nodes: dict[str, RuntimeNode] = Field(default_factory=dict)
    experiment: str | None = None
    stages: list[dict[str, Any]] = Field(default_factory=list)
    results: dict[str, Any] | None = None
