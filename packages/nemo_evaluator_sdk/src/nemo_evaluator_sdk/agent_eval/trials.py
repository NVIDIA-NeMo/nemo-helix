# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed trial artifacts and runtime/serde interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Protocol, Self, runtime_checkable

from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.metrics.protocol import Metric
from nemo_evaluator_sdk.values.agents import Agent
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FINAL_STATE,
    EVIDENCE_FORMAT_ATIF,
    EVIDENCE_FORMAT_JSON,
    EVIDENCE_INITIAL_STATE,
    EVIDENCE_LOGS,
    EVIDENCE_TRACE,
    EVIDENCE_VERIFIER_LOGS,
    CandidateEvidence,
    EvidenceDescriptor,
)
from nemo_evaluator_sdk.values.models import Model
from nemo_evaluator_sdk.values.results import AggregateScore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    WithJsonSchema,
    field_validator,
    model_validator,
)


class AgentEvalTrialStatus(str, Enum):
    """Lifecycle status for a trial: completed, failed, or partial."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AgentOutput(BaseModel):
    """Captured final output from the evaluated agent, model, or imported baseline for a trial."""

    model_config = ConfigDict(extra="forbid")

    output_text: str | None = Field(
        default=None,
        description="User-visible final text produced by the agent, if any.",
    )
    response: JsonValue | None = Field(
        default=None,
        description="Final response payload produced by the agent, if any. Any JSON value — a "
        "structured object, or a raw JSON string/array for agents that don't return an object.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the agent output.",
    )


# Type recorded when a producer reported a failure but named no usable type. This
# fires only for hand-built or malformed payloads.
UNKNOWN_ERROR_TYPE = "UnknownException"


class TrialError(BaseModel):
    """What went wrong producing one trial, as the producer reported it.

    Present means the *producer* reported a failure. It does **not** imply ``status is FAILED``: an
    errored trial can be marked as :attr:`AgentEvalTrialStatus.PARTIAL` so it is still scored (ex: HarborRuntime).
    It is also unrelated to a score diagnostic's ``exception_type`` detail, which records that the
    *metric* raised - a different event.

    Frozen so callers cannot rewrite ``type`` after construction. These objects are returned as-is
    (not copied), and ``AgentEvalSummary.error_trial_ids`` groups trial ids by that string. Mutating
    ``type`` on a live object would leave the summary keyed on the old value while the trial reports
    a new one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(
        description=(
            "Error class name as the producer reported it, e.g. 'RuntimeError'. Rollup key for "
            f"AgentEvalSummary.error_trial_ids. Falls back to {UNKNOWN_ERROR_TYPE!r} when the "
            "producer reported a failure without a usable type."
        )
    )
    message: str | None = Field(
        default=None,
        description="Short error message, when the producer supplied one.",
    )
    traceback: str | None = Field(
        default=None,
        description=(
            "Formatted traceback, when the producer supplied one. May be truncated by the adapter "
            "that captured it. Note run bundles are portable: this can carry absolute filesystem "
            "paths and other diagnostic text from the machine that ran the trial."
        ),
    )
    # Schema is a bare string, deliberately not `format: date-time`. RFC 3339 date-time requires a
    # UTC offset, and this field may legitimately carry a naive local timestamp (Harbor writes one),
    # so claiming the format would be a promise the value cannot keep -- and a client that trusts it
    # parses a zoneless string into its *own* zone, silently shifting the instant. A plain string
    # says "timestamp as the producer wrote it"; Python callers still get a parsed datetime.
    occurred_at: Annotated[datetime | None, WithJsonSchema({"type": "string"})] = Field(
        default=None,
        description=(
            "When the producer recorded the failure, as it reported it. The SDK does not rewrite "
            "this: an aware value (UTC, offset) is kept, and a naive value stays naive. Harbor's "
            "clock is naive local, e.g. '2026-08-13T17:22:32', while that trial's start in "
            "result.json is UTC ('2026-08-14T00:22:25Z') — the same instant on two clocks, so do "
            "not subtract them or attach a zone Harbor never wrote. A runner that recorded an "
            "offset keeps it. Not RFC 3339 date-time: the offset may be absent."
        ),
    )

    @field_validator("type")
    @classmethod
    def _non_empty_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trial error type must not be empty")
        return value


NonNegativeTokenCount = Annotated[int, Field(strict=True, ge=0)]
FiniteNonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class TrialMeasurements(BaseModel):
    """Validated usage, runtime, and cost measurements for one trial."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    prompt_tokens: NonNegativeTokenCount | None = None
    completion_tokens: NonNegativeTokenCount | None = None
    total_tokens: NonNegativeTokenCount | None = None
    cache_creation_tokens: NonNegativeTokenCount | None = None
    cache_read_tokens: NonNegativeTokenCount | None = None
    runtime_sec: FiniteNonNegativeFloat | None = None
    cost_usd: FiniteNonNegativeFloat | None = None

    @field_validator("runtime_sec", "cost_usd", mode="before")
    @classmethod
    def _require_real_number(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("measurement must be a real number, not a Boolean or string")
        return value

    @model_validator(mode="after")
    def _require_canonical_total(self) -> Self:
        if self.prompt_tokens is None or self.completion_tokens is None:
            if self.total_tokens is not None:
                raise ValueError("total_tokens requires prompt_tokens and completion_tokens")
            return self
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", expected)
        elif self.total_tokens != expected:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class AgentEvalTrial(BaseModel):
    """Durable trial artifact: output, evidence, status, error, measurements, and metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, revalidate_instances="always")

    id: str = Field(description="Stable identifier for this trial.")
    task_id: str = Field(description="Identifier of the AgentEvalTask this trial was produced for.")
    status: AgentEvalTrialStatus = Field(description="Lifecycle status of the trial.")
    output: AgentOutput | None = Field(
        default=None,
        description="Final agent output captured for the trial; required when status is completed.",
    )
    evidence: CandidateEvidence | None = Field(
        default=None,
        description="Named evidence descriptors (final state, traces, logs, ...) captured for the trial.",
    )
    error: TrialError | None = Field(
        default=None,
        description=(
            "What went wrong producing this trial, when the producer reported a failure. Populated "
            "by a runner runtime. Drives AgentEvalSummary.error_trial_ids."
        ),
    )
    measurements: TrialMeasurements = Field(
        default_factory=TrialMeasurements,
        description="Validated token usage, agent runtime, and cost measurements for the trial.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata associated with the trial.",
    )

    @field_validator("id", "task_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("trial id and task_id must not be empty")
        return value

    @model_validator(mode="after")
    def _completed_trial_requires_output(self) -> AgentEvalTrial:
        if self.status == AgentEvalTrialStatus.COMPLETED and self.output is None:
            raise ValueError("completed trial requires output")
        return self

    def get_evidence(self, name: str) -> EvidenceDescriptor | None:
        """Look up a named evidence descriptor for this trial.

        Args:
            name: Evidence key to retrieve.

        Returns:
            The matching descriptor, or ``None`` when the trial has no evidence or
            the key is not present.
        """
        if self.evidence is None:
            return None
        return self.evidence.get(name)


@runtime_checkable
class AgentTaskRunner(Protocol):
    """Online execution interface that runs tasks and produces trials."""

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]: ...

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the settings that shape its results, for run provenance.

        Required: every run has a producer, and the result records it on
        ``AgentEvalResult.metadata.target`` so a run can be understood after the fact. Return a stable
        short ``name`` (``"gym"``, ``"harbor"``) rather than a class name. ``config`` must not contain
        secrets — it is persisted with the run bundle.
        """
        ...


class RunnerInfo(BaseModel):
    """Identity of whatever produced a run's trials, recorded for provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier of the runner/target, e.g. 'gym', 'harbor', or a model name.")
    kind: str = Field(
        default="runner",
        description="What produced the trials: 'runner', 'model', 'agent', or 'imported' for stored trials.",
    )
    version: str | None = Field(default=None, description="Version of the backing tool, when known.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Runner-specific settings that affect results, recorded so a run can be understood "
        "after the fact. Must not contain secrets.",
    )


def callable_identity(target: object) -> str:
    """Module-qualified identity of a callable, for :attr:`RunnerInfo.config`.

    A bare ``__qualname__`` is ambiguous across modules — two runs using different callables that
    share a name would record identical provenance — so qualify it with the defining module.
    """
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or type(target).__name__
    return f"{module}.{name}" if module else name


@runtime_checkable
class RunAggregationsProvider(Protocol):
    """Optional companion to :class:`AgentTaskRunner`: a runner that computed its own run-level
    aggregations (a backend's pass@k, reward profile, environment-specific metrics) exposes them here,
    mapped onto the SDK's typed aggregate scores. The evaluator calls this after ``run_tasks``;
    implementers stash their numbers during the run and convert them here.

    Returned scores are merged into ``summary.scores``, so a backend's own figures sit alongside the
    SDK's and are addressable by name the same way. Implementers must namespace names under
    ``runner.<runner_name>.`` so an imported figure is never mistaken for one the SDK computed. Runners
    with no run-level aggregations simply don't implement this protocol.
    """

    def run_aggregate_scores(self) -> Sequence[AggregateScore]: ...


@runtime_checkable
class TrialAwareMetricsProvider(Protocol):
    """Optional companion to :class:`AgentTaskRunner`: a runner whose metric outputs
    are not fully known until trials exist finalizes them here.

    The evaluator calls this once per task after trial generation and before
    scoring, with only that task's trials. Return the task's full metric list; the
    evaluator rebuilds the task with these metrics and nothing else, re-validates it
    (duplicate metric types, views that reference undeclared outputs), and re-groups
    trials from the final trial list before scoring.

    Do not mutate ``task`` or ``trials``. The evaluator rejects trial ``task_id``
    reassignment after this hook; other mutation remains a convention for in-process
    runner code, matching the standing ``run_tasks`` contract.

    Harbor is the in-tree example. The placeholder ``harbor_reward`` metric cannot
    declare verifier extras until trials exist, because those keys live in
    ``trial.metadata["reward_details"]``. ``scoring_metrics`` unions the primary
    reward with keys observed on the task's trials and replaces only the
    ``harbor_reward`` metric. Two attempts of task A::

        a1: {"reward": 1.0, "format_ok": 1.0}
        a2: {"reward": 0.0}

    become one spec: required ``reward`` plus optional ``format_ok``. Per-task
    scoping is structural: the hook never sees another task's trials.

    Runners whose metrics are known before execution simply don't implement this
    protocol.
    """

    def scoring_metrics(
        self,
        task: AgentEvalTask,
        trials: Sequence[AgentEvalTrial],
    ) -> Sequence[Metric]:
        """Return the metrics to score ``task`` with, given its ``trials``."""
        ...


@runtime_checkable
class AgentTrialSerde(Protocol):
    """Read/write a single stored trial artifact as an :class:`AgentEvalTrial`.

    The offline counterpart to :class:`AgentTaskRunner`: instead of *executing* an
    agent it adapts a stored artifact (a run dir/file) to and from a trial, so prior
    runs can be re-scored. The SDK ships only the protocol; concrete codecs (which
    know a particular on-disk layout) live with their producers.
    """

    def read(self) -> AgentEvalTrial: ...

    def write(self, trial: AgentEvalTrial) -> None: ...


AgentEvalTarget = Model | Agent | AgentTaskRunner


def resolve_trial_status(agent_ok: bool) -> AgentEvalTrialStatus:
    """Map an agent-phase outcome to a *scorable* trial status.

    ``AgentEvaluator`` excludes ``FAILED`` trials from scoring, so an
    executed-but-unsuccessful agent uses ``PARTIAL`` (still scored as ``0`` for
    pass-rate gating); ``FAILED`` is reserved for trial-*production* failures,
    which a runtime surfaces by raising rather than emitting an unscorable trial.
    """
    return AgentEvalTrialStatus.COMPLETED if agent_ok else AgentEvalTrialStatus.PARTIAL


def standard_evidence_descriptors(
    *,
    logs_dir: str | Path,
    final_state_dir: str | Path,
    trace_path: str | Path | None = None,
    trace_format: str | None = None,
    initial_state_ref: str | None = None,
    verifier_logs_dir: str | Path | None = None,
    primary_log: str | None = None,
) -> dict[str, EvidenceDescriptor]:
    """Build the documented evidence map for an agent-eval trial.

    Standard keys: ``initial_state`` (task input filesystem, when staged),
    ``trace`` (trajectory, ATIF-normalized when available), ``logs`` (agent log
    dir), ``final_state`` (workspace), and ``verifier_logs`` (only when present).
    Callers may add their own extension keys to the returned mapping.

    ``trace_format`` is the producer's parser hint. Without it, the hint is
    inferred from the trace basename without reading the file: ``atif`` when the
    name starts with ``atif`` or contains ``.atif.`` (for example
    ``atif-trace.json`` or ``trajectory.atif.json``), otherwise ``json``.
    Producers whose filenames do not follow that pattern can call
    ``read_atif()`` first and pass ``trace_format`` only when parsing succeeds.
    """
    descriptors: dict[str, EvidenceDescriptor] = {}

    if initial_state_ref:
        descriptors[EVIDENCE_INITIAL_STATE] = EvidenceDescriptor(
            kind="filesystem",
            format="dir",
            ref=str(initial_state_ref),
            metadata={"role": EVIDENCE_INITIAL_STATE},
        )

    if trace_path is not None:
        trace_name = Path(trace_path).name.lower()
        is_atif = trace_name.startswith("atif") or ".atif." in trace_name
        descriptors[EVIDENCE_TRACE] = EvidenceDescriptor(
            kind=EVIDENCE_TRACE,
            format=trace_format or (EVIDENCE_FORMAT_ATIF if is_atif else EVIDENCE_FORMAT_JSON),
            ref=str(trace_path),
        )

    logs_metadata = {"primary_log": primary_log} if primary_log else {}
    descriptors[EVIDENCE_LOGS] = EvidenceDescriptor(
        kind="logs",
        format="dir",
        ref=str(logs_dir),
        metadata=logs_metadata,
    )

    descriptors[EVIDENCE_FINAL_STATE] = EvidenceDescriptor(
        kind="filesystem",
        format="dir",
        ref=str(final_state_dir),
        metadata={"role": EVIDENCE_FINAL_STATE},
    )

    if verifier_logs_dir is not None and Path(verifier_logs_dir).exists():
        descriptors[EVIDENCE_VERIFIER_LOGS] = EvidenceDescriptor(
            kind="logs",
            format="dir",
            ref=str(verifier_logs_dir),
            metadata={"role": "verifier"},
        )

    return descriptors
