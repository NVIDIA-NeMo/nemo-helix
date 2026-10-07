# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trial metric scoring records and scoring diagnostics."""

from __future__ import annotations

from enum import Enum
from typing import Any

from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.metrics.protocol import MetricOutput
from pydantic import BaseModel, ConfigDict, Field


class AgentEvalScoreStatus(str, Enum):
    """Status of metric scoring for one trial."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


#: Diagnostic detail key stamped when a score is ``FAILED`` because the *trial* failed — the agent
#: produced nothing to score — rather than because the metric itself raised. Both are reported as
#: ``FAILED``, but they mean different things to a reader: a failed trial did not pass, a
#: failed metric is a failed *measurement*. Consumers that must tell them apart read this key via
#: :func:`is_trial_failure`.
TRIAL_STATUS_DETAIL = "trial_status"


class AgentEvalDiagnosticSeverity(str, Enum):
    """Severity level for a scoring diagnostic."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AgentEvalDiagnostic(BaseModel):
    """Diagnostic emitted while scoring one trial with one metric."""

    model_config = ConfigDict(extra="forbid")

    severity: AgentEvalDiagnosticSeverity = Field(description="Severity of the diagnostic.")
    message: str = Field(description="Human-readable diagnostic message.")
    source: str | None = Field(
        default=None,
        description="Component or stage that produced the diagnostic, if known.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured supporting details for the diagnostic.",
    )


class AgentEvalTaskScore(BaseModel):
    """Per-task, per-trial, per-metric scoring record: metric outputs, diagnostics, status, and metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier for this score record.")
    run_id: str = Field(description="Identifier of the run this score belongs to.")
    task_id: str = Field(description="Identifier of the task that was scored.")
    trial_id: str = Field(description="Identifier of the trial that was scored.")
    metric_type: str = Field(description="Task-local metric type (metric.type) that produced this score.")
    status: AgentEvalScoreStatus = Field(description="Status of this metric score.")
    outputs: list[MetricOutput] = Field(
        default_factory=list,
        description="Named metric outputs emitted for this trial/metric pair.",
    )
    diagnostics: list[AgentEvalDiagnostic] = Field(
        default_factory=list,
        description="Diagnostics emitted while scoring this trial with this metric.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the score.",
    )


def is_trial_failure(score: AgentEvalTaskScore) -> bool:
    """True when a ``FAILED`` score records a failed *trial* rather than a metric that raised.

    The evaluator short-circuits a failed trial into a failed score without running the metric, and
    stamps :data:`TRIAL_STATUS_DETAIL` on the diagnostic; a metric that raised is stamped with its
    exception type instead. The distinction is what lets pass@k charge a dead rollout to the agent
    while leaving an unusable measurement out of the denominator entirely.

    Matches on the detail's *value*, not merely the key's presence: ``trial_status`` is a natural
    thing for a hand-built score to carry for its own reasons, and only ``failed`` means the attempt
    is one the agent is answerable for.
    """
    return score.status is AgentEvalScoreStatus.FAILED and any(
        diagnostic.details.get(TRIAL_STATUS_DETAIL) == AgentEvalTrialStatus.FAILED.value
        for diagnostic in score.diagnostics
    )
