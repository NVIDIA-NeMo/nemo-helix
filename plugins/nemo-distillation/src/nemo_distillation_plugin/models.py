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
        "dataset_eval",
        "agent_deployment",
        "delete_agent_deployment",
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
    gpu: int = Field(default=1, gt=0, description="GPUs one training job for this candidate needs.")


class EvalInput(BaseModel):
    """One evaluation run on every arm.

    ``agent`` is a task-driven agent-eval spec (``tasks`` inline). ``dataset`` is a
    dataset-driven evaluate spec (``dataset``, ``metrics``, ``prompt_template``). The
    per-arm target is filled in when the node starts. Only the original arm's output
    filesets are the convert step's input.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["agent", "dataset"]
    spec: dict[str, Any]

    @model_validator(mode="after")
    def _spec_matches_kind(self) -> EvalInput:
        if self.kind == "agent":
            tasks = self.spec.get("tasks")
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(f"eval {self.name!r} needs a non-empty tasks list")
        else:
            dataset = self.spec.get("dataset")
            metrics = self.spec.get("metrics")
            if not isinstance(dataset, str) or not dataset:
                raise ValueError(f"eval {self.name!r} needs a dataset fileset reference")
            if not isinstance(metrics, list) or not metrics:
                raise ValueError(f"eval {self.name!r} needs a non-empty metrics list")
        return self


class TeacherInput(BaseModel):
    """The teacher model deployment the workflow creates for the original evals."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    engine: Literal["vllm", "nim"] = "vllm"
    gpu: int = Field(default=1, gt=0)
    tool_call_config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _tool_calls_need_nim(self) -> TeacherInput:
        if self.tool_call_config is not None and self.engine != "nim":
            raise ValueError("tool_call_config is only supported with engine='nim'")
        return self


class AgentDeployInput(BaseModel):
    """How the workflow deploys the agent for each arm."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["subprocess", "docker", "k8s", "openshell"] = "k8s"
    image: str = ""
    use_image_entrypoint: bool = False

    @model_validator(mode="after")
    def _container_needs_an_image(self) -> AgentDeployInput:
        if self.mode in {"docker", "k8s", "openshell"} and not self.image.strip():
            raise ValueError(f"agent_deployment.image is required for mode {self.mode!r}")
        return self


class DistillInputs(BaseModel):
    """Caller-supplied inputs for the distillation template."""

    model_config = ConfigDict(extra="forbid")

    source_agent: str = Field(min_length=1)
    student_model: str = Field(min_length=1)
    teacher: TeacherInput | None = Field(
        default=None,
        description="Entry point 1: the workflow deploys this teacher and runs the original evals to produce trajectories.",
    )
    trajectory_filesets: list[str] = Field(
        default_factory=list,
        description="Entry point 2: existing evaluation output filesets (workspace/name). Skips the teacher stage.",
    )
    gpu_budget: int = Field(
        default=1,
        gt=0,
        description=(
            "GPUs the workflow may use at once. Candidate jobs run in budget // gpus-per-job parallel lanes, "
            "so 1 runs everything one job at a time."
        ),
    )
    agent_deployment: AgentDeployInput
    evals: list[EvalInput] = Field(min_length=1)
    eval_results: list[str] = Field(default_factory=list)
    pass_filter: dict[str, Any] = Field(default_factory=dict)
    candidates: list[CandidateSpec] = Field(default_factory=list)
    deployment: DeploymentInput = Field(default_factory=DeploymentInput)
    prices: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _entry_point_and_budget(self) -> DistillInputs:
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.trajectory_filesets):
            raise ValueError("trajectory_filesets entries must be non-empty fileset references")
        if (self.teacher is None) == (not self.trajectory_filesets):
            raise ValueError("provide either teacher or trajectory_filesets, not both and not neither")
        if self.teacher is not None and self.teacher.gpu > self.gpu_budget:
            raise ValueError(f"teacher.gpu ({self.teacher.gpu}) exceeds gpu_budget ({self.gpu_budget})")
        width = self.job_gpus
        if width > self.gpu_budget:
            raise ValueError(f"one candidate job needs {width} GPUs, which exceeds gpu_budget ({self.gpu_budget})")
        return self

    @property
    def job_gpus(self) -> int:
        """GPUs the widest single job (a base or candidate deploy, or a candidate training run) holds."""
        return max([self.deployment.gpu, *[candidate.gpu for candidate in self.candidates]])

    @property
    def parallel_jobs(self) -> int:
        """How many base or candidate jobs the budget lets run at the same time."""
        return max(1, self.gpu_budget // self.job_gpus)


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
