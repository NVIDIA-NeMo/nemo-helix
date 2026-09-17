# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task definitions, semantic views, and run configuration for agent evaluation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.metrics.utils import metric_type_name
from nemo_evaluator_sdk.values.params import RunConfig, RunConfigOnline, RunConfigOnlineModel
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator


class SemanticReducer(str, Enum):
    """Reduction strategy used to combine task-scoped view signals into one score."""

    SINGLE = "single"
    ALL = "all"
    ANY = "any"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"


class ViewSignal(BaseModel):
    """Task-scoped metric output that contributes to a semantic view."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(description="Task-local metric type (metric.type) whose output feeds this signal.")
    output: str = Field(description="Name of the metric output, as declared by the metric's output_spec().")
    weight: float | None = Field(
        default=None,
        description="Relative weight for this signal when the view reducer is 'weighted_mean'; unused otherwise.",
    )

    @field_validator("metric", "output")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("view signal metric and output must not be empty")
        return value


class SemanticView(BaseModel):
    """Task-scoped reporting view that maps a task's own metric outputs into a named score."""

    model_config = ConfigDict(extra="forbid")

    reducer: SemanticReducer = Field(
        description="Strategy used to reduce this view's signals into a single task-level score.",
    )
    signals: list[ViewSignal] = Field(
        min_length=1,
        description="Ordered metric outputs contributing to this view; at least one is required.",
    )


class AgentEvalTask(BaseModel):
    """Standalone agent-eval task: the unit of work being evaluated."""

    # TODO: Tasks may need to define a set of required_capabilities or something that allow the
    # runtime to skip trying to complete a task that isn't possible.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str = Field(description="Stable task identifier, unique within the supplied task collection.")
    intent: str = Field(description="Human-readable description of the desired agent behavior.")
    inputs: dict[str, Any] = Field(
        description="What the agent receives or starts from, e.g. instruction, filesystem seed, or state refs.",
    )
    reference: dict[str, Any] = Field(
        default_factory=dict,
        description="Grader-only ground truth (held-out tests, expected outputs, rubric data). Surfaced to "
        "metrics as row.data['reference'] but never seeded into the agent's workspace or shown to the "
        "agent, so a metric can grade against artifacts the agent cannot influence.",
    )
    metrics: list[Metric] = Field(
        default_factory=list,
        description="Ordered concrete SDK metric instances that score this task; metric types must be unique.",
    )
    views: dict[str, SemanticView] = Field(
        default_factory=dict,
        description="Optional reporting views mapping this task's metric outputs into named semantic scores.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the task.",
    )

    @field_validator("id")
    @classmethod
    def _id_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("task id must not be empty")
        return value

    def agent_prompt(self) -> str:
        """The intent-free prompt handed to the agent under evaluation.

        Exactly the task's natural-language instruction (``inputs["instruction"]``), with no
        runtime-added framing. ``intent`` is deliberately never used: it is the eval-side description
        of the desired behavior (what the grader checks for), so exposing it to the agent is a
        reward-hacking hole.

        Raises ``ValueError`` when ``inputs["instruction"]`` is missing or empty; a task with no
        instruction cannot be evaluated, so the runner fails that task rather than running an agent on
        an empty prompt.
        """
        instruction = self.inputs.get("instruction")
        if instruction:
            return str(instruction)
        raise ValueError(f"task {self.id!r} has no instruction: set inputs['instruction']")

    @field_serializer("metrics", when_used="json")
    def _serialize_metrics(self, metrics: list[Metric]) -> list[dict[str, Any]]:
        """Serialize local metric instances as descriptors for run bundles."""
        serialized: list[dict[str, Any]] = []
        for metric in metrics:
            outputs = [
                {
                    "name": output.name,
                    "description": output.description,
                    "value_schema": output.value_schema.__name__,
                    **({"required": False} if not output.required else {}),
                }
                for output in metric.output_spec()
            ]
            serialized.append({"type": metric_type_name(metric), "outputs": outputs})
        return serialized

    @model_validator(mode="after")
    def _validate_metric_references(self) -> AgentEvalTask:
        metric_types = [metric_type_name(metric) for metric in self.metrics]
        duplicate_metric_types = sorted(
            {metric_type for metric_type in metric_types if metric_types.count(metric_type) > 1}
        )
        if duplicate_metric_types:
            raise ValueError(f"duplicate task metric types: {duplicate_metric_types}")

        outputs_by_metric = {
            metric_type_name(metric): {output.name for output in metric.output_spec()} for metric in self.metrics
        }
        for view_name, view in self.views.items():
            for signal in view.signals:
                if signal.metric not in outputs_by_metric:
                    raise ValueError(f"view {view_name!r} references unknown metric {signal.metric!r}")
                if signal.output not in outputs_by_metric[signal.metric]:
                    raise ValueError(
                        f"view {view_name!r} references unknown output {signal.output!r} for metric {signal.metric!r}"
                    )
        return self


class AgentEvalTaskset(BaseModel):
    """A named set of SDK-native tasks (with optional metadata) to evaluate.

    Produced by an :class:`AgentEvalTasksetLoader`; the evaluator scores
    ``tasks`` directly and never consumes a loader.
    """

    model_config = ConfigDict(extra="forbid")

    tasks: list[AgentEvalTask] = Field(
        default_factory=list,
        min_length=1,
        description="Tasks in this set; at least one is required and task ids must be unique.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form taskset metadata for the run.")

    @model_validator(mode="after")
    def _task_ids_unique(self) -> AgentEvalTaskset:
        ids = [task.id for task in self.tasks]
        duplicates = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate taskset task ids: {duplicates}")
        return self


@runtime_checkable
class AgentEvalTasksetLoader(Protocol):
    """Protocol for adapting an external taskset into agent-eval.

    A loader is a named adapter that loads SDK-native tasks (optionally from an
    external ``source``). Per the design, loaders are resolved upstream into an
    :class:`AgentEvalTaskset`; the evaluator scores those tasks and never consumes
    a loader directly.
    """

    @property
    def name(self) -> str:
        """Stable taskset name used in diagnostics, metadata, and user-facing output."""
        ...

    def load(
        self,
        *,
        source: str | Path | None = None,
        limit: int | None = None,
        evidence_dir: Path | None = None,
    ) -> AgentEvalTaskset:
        """Load tasks into an :class:`AgentEvalTaskset`.

        ``source`` is an optional path/URI to load from, ``limit`` an optional
        positive cap on the number of tasks, and ``evidence_dir`` an optional
        directory holding task evidence inputs.
        """
        ...


class AgentEvalRunConfig(BaseModel):
    """Configuration for a standalone agent-eval run."""

    model_config = ConfigDict(extra="forbid")

    work_dir: Path | None = Field(
        default=None,
        description="Directory the run works in: runtimes write trial evidence beneath it, and it is "
        "the default target for AgentEvalResult.persist so the bundle contains that evidence. Purely "
        "in-memory when omitted.",
    )
    run_id: str | None = Field(default=None, description="Explicit run identifier; generated when omitted.")
    prompt_template: str | dict[str, Any] | None = Field(
        default=None,
        description="Optional prompt template applied when generating trials online.",
    )
    params: RunConfig | RunConfigOnline | RunConfigOnlineModel | None = Field(
        default=None,
        description="Inference/run parameters used when producing trials online.",
    )
    parallelism: int = Field(default=4, ge=1, description="Maximum number of tasks scored concurrently.")
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Caller-supplied tags recorded on the run's metadata (e.g. benchmark, mode, backend, "
        "scenario). Free-form by design and never derived: nothing is inferred from task metadata, so a "
        "label is present only if the caller set it.",
    )
    fail_fast: bool = Field(default=False, description="Stop the run on the first scoring failure when True.")
