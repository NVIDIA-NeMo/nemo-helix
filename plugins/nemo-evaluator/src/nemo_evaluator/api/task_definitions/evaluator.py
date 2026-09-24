# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in task kind: an agent scored by platform metrics."""

from __future__ import annotations

from typing import Any, Literal

from nemo_evaluator.api.fields import MetricInline, MetricRefOrInline, TaskInputs
from nemo_evaluator.api.task_definitions.provenance import TaskProvenance
from nemo_evaluator_sdk.agent_eval.tasks import SemanticView
from pydantic import BaseModel, ConfigDict, Field


class _EvaluatorTaskDefinitionCommon(BaseModel):
    """Definition fields shared by stored tasks and resolved job snapshots."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["evaluator"] = Field(description="Task kind discriminator.")
    intent: str = Field(description="Human-readable description of the desired agent behavior.")
    inputs: TaskInputs = Field(default_factory=TaskInputs, description="Inputs supplied to the task.")
    reference: dict[str, Any] = Field(
        default_factory=dict,
        description="Grader-only ground truth (held-out tests, expected outputs, rubric data). Surfaced to "
        "metrics but never seeded into the agent's workspace or shown to the agent, so a metric can grade "
        "against artifacts the agent cannot influence. Held out from the *agent*, not from the API: anyone "
        "who can read the task can read this.",
    )
    views: dict[str, SemanticView] = Field(
        default_factory=dict,
        description="Optional reporting views mapping metric outputs into named semantic scores.",
    )


class EvaluatorTaskDefinition(_EvaluatorTaskDefinitionCommon):
    """What the agent should do, and how the platform scores it.

    ``metrics`` accepts inline bundles on the way in and holds references once stored: the service
    offloads an inline metric to a content-addressed *derived* metric on create, so a persisted task
    only ever names metrics it does not own. That narrowing is a service invariant rather than a
    type-level one — a single model keeps the API surface small, at the cost of this field being
    wider than what a stored task actually contains.

    Every field here is covered by the revision digest, ``reference`` included: it decides what a
    metric grades against, so two revisions that score differently must not share a digest. Pinning
    a revision therefore fixes the grading, not just the prompt.
    """

    metrics: list[MetricRefOrInline] = Field(
        default_factory=list,
        description="Metrics that score this task — stored-metric references, and inline bundles on "
        "create (normalized to derived stored metrics before the task is persisted).",
    )


class ResolvedEvaluatorTaskDefinition(_EvaluatorTaskDefinitionCommon):
    """Job snapshot with expanded metrics and the original stored revision, if any."""

    provenance: TaskProvenance | None = None
    metrics: list[MetricInline] = Field(default_factory=list)
