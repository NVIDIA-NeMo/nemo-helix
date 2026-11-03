# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary mapping: Evaluator vocabulary -> Intake/Experiments wire shapes.

This is the single place where Evaluator domain objects (``AgentEvalTrial``,
``AgentEvalTaskScore``, ``MetricOutput``) become the request bodies Intake and
the Experiments API expect. The Intake write-adapter tickets (D3/D4/D5) obtain
their request shapes and field names *only* from here, so a later rename is a
one-file change.

Design constraints:

* **Pure.** Every function reads SDK types and returns request params. No HTTP,
  no platform client, no imports from the Intake *service* (``nhx.intake.*``).
* **Typed at the boundary.** The returned values are typed-client
  ``TypedDict`` params (``AtifCreateParams`` / ``EvaluatorResultCreateParams``).
  At runtime they are plain dicts the publisher wraps into the typed client's
  request models (``AsyncIntakeClient.create_atif`` / ``create_evaluator_result``);
  statically, ``ty`` checks our field names, literals, and nested shapes against
  the typed intake schema, so an API change there surfaces here as a type error
  instead of drifting silently. We depend on the client SDK (already a plugin dependency),
  never on the Intake service package.
* The well-known evidence-key constants (``initial_state``/``trace``/``logs``/
  ``final_state``/``verifier_logs``) belong with the SDK evidence work (D1).
  Until D1 lands, this module references them as string literals so
  it stays unblocked.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from nemo_helix_plugin.intake.span_attributes import USAGE_SUPERSEDED_ATTRIBUTE
from nemo_helix_plugin.intake.types import (
    AtifAgentParam,
    AtifCreateParams,
    AtifFinalMetricsParam,
    AtifStepParam,
    EvaluationContextParam,
    EvaluatorResultCreateParams,
    EvaluatorResultDataType,
)
from nhx_evals_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial, TrialMeasurements
from nhx_evals_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
from nhx_evals_sdk.values.otlp import (
    fill_missing_root_end_time,
    fill_missing_start_times,
    root_span_id,
    set_root_span_attributes,
    set_root_span_error,
    set_span_attributes,
)
from nhx_evals_sdk.values.protocol import OUTPUT_DETAIL
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span
from pydantic import RootModel

logger = logging.getLogger(__name__)

# --- Shared conventions -----------------------------------------------------

#: ATIF schema version the adapter emits.
ATIF_SCHEMA_VERSION: Literal["ATIF-v1.7"] = "ATIF-v1.7"

#: Default ``agent.version`` when the run target carries none. Neither Model nor
#: Agent has a version field today, and ATIF requires one (design doc §3.9 #6).
DEFAULT_AGENT_VERSION = "unknown"

# Span attribute keys Intake reads identity back from; see its span attribute catalog. The
# session id has two accepted keys and this is the one that wins.
SESSION_ID_ATTRIBUTE = "gen_ai.conversation.id"
EVALUATION_NAME_ATTRIBUTE = "nemo.evaluation.name"
TEST_CASE_NAME_ATTRIBUTE = "nemo.test_case.name"


def session_id_for(run_id: str, trial_id: str) -> str:
    """Return the stable, adapter-minted session id for a trial.

    One session id per Trial keeps ATIF ingest idempotent and lets per-metric
    scores be attached to the same trajectory afterward. This is the single
    source of the convention; callers must not hand-roll it.
    """
    return f"{run_id}:{trial_id}"


def run_task_to_evaluation_context(trial: AgentEvalTrial, *, evaluation_name: str) -> EvaluationContextParam:
    """Build the lean ingest ``evaluation_context`` for a trial.

    Only ``evaluation_name`` and ``test_case_name`` live here. Dataset, group, and free-form
    metadata belong on the Evaluation entity (created separately via the platform SDK), not on
    the per-ingest context.
    """
    return {"evaluation_name": evaluation_name, "test_case_name": trial.task_id}


@dataclass(frozen=True)
class RecordedSteps:
    """A trial's recorded ATIF steps, and whether they carry a duration of their own.

    Intake rolls the root span's window up from step timestamps and ``extra.invocation`` windows,
    so a trajectory conveys a duration only through an invocation window or two producer-recorded
    timestamps. The run start stamped onto untimed steps is not a producer timestamp.
    """

    steps: list[AtifStepParam]
    timed: bool
    agent_version: str | None = None
    """The version the trajectory's producer recorded (``agent.version``), such as codex's ``0.162.1``."""


async def atif_steps_from_trial(trial: AgentEvalTrial, *, started_at: datetime) -> RecordedSteps | None:
    """Return the trial's real ATIF steps, or None when it has no usable ATIF trajectory.

    Reads the ATIF view by name rather than the trial's primary trace, which may be an OTLP
    one. Only runners whose agent emits ATIF attach one; Harbor's
    ``oracle`` and ``nop`` never do, and an unreadable, malformed, or empty trajectory is treated
    the same as an absent one so a publish is never lost to bad evidence.

    The SDK's read model is deliberately more permissive than Intake's ingest schema, so steps that
    parse here can still be rejected there; :func:`nemo_evals.intake.publish.publish_to_intake`
    retries without them rather than enumerating every divergence.
    """
    evidence = trial.evidence
    if evidence is None:
        return None
    try:
        handle = await evidence.trace(EVIDENCE_TRACE, format=EVIDENCE_FORMAT_ATIF)
    except KeyError:
        # Absent is the normal case for a runner whose agent emits no ATIF, and is
        # distinct from a trace that exists but will not read.
        return None
    try:
        trajectory = await handle.trace()
        steps = trajectory.steps
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable ATIF trace for trial %s: %s", trial.id, error)
        return None

    if not steps:
        return None

    payload: list[AtifStepParam] = []
    recorded_timestamps = 0
    has_invocation_window = False
    for index, step in enumerate(steps):
        dumped: AtifStepParam = step.model_dump(mode="json", exclude_none=True)
        # Ingest requires one-based sequential ids, while the SDK read model leaves step_id optional.
        dumped["step_id"] = index + 1
        if step.timestamp is not None:
            recorded_timestamps += 1
        if step.extra is not None and step.extra.get("invocation") is not None:
            has_invocation_window = True
        # Intake keys spans on start_time and falls back to its own ingest clock for a step with
        # no timestamp, which would make re-publish duplicate instead of replace.
        dumped.setdefault("timestamp", started_at.isoformat())
        payload.append(dumped)
    agent_version = trajectory.agent.version if trajectory.agent is not None else None
    return RecordedSteps(payload, timed=has_invocation_window or recorded_timestamps >= 2, agent_version=agent_version)


async def otlp_ingest_for_trial(
    trial: AgentEvalTrial,
    *,
    session_id: str,
    evaluation_name: str,
    measurements: TrialMeasurements,
    started_at: datetime,
    usage_superseded: bool = False,
) -> tuple[bytes, str] | None:
    """Return the trial's OTLP payload stamped with evaluation identity, and its root span id.

    ``None`` when the trial has no readable OTLP trace, or when its spans have no single root
    carrying a usable id: a trial-level score needs one span standing for the whole run.

    ``started_at`` fills in any span that recorded no start time of its own, because Intake
    otherwise stores it against the ingest clock and start time is part of the key its spans
    table replaces on — so a re-publish would insert rather than replace. A root span with no
    end of its own gets the trial's measured runtime, as the ATIF path does, so Intake's latency
    rollup does not read 0 for a run whose duration is known. ``usage_superseded`` marks the root as a trace whose
    model calls the Inference Gateway recorded in the session (``USAGE_SUPERSEDED_ATTRIBUTE``).

    Identity is written onto spans rather than resource attributes because Intake merges the
    layers as ``{**resource, **span}``. An agent that records its own ``gen_ai.conversation.id``
    would otherwise win, and the session id is what Intake's ``ReplacingMergeTree`` is keyed on,
    so a re-publish would insert duplicates instead of replacing.
    """
    evidence = trial.evidence
    if evidence is None:
        return None
    try:
        handle = await evidence.trace(EVIDENCE_TRACE, format=EVIDENCE_FORMAT_OTLP)
    except KeyError:
        return None
    try:
        resource_spans = await handle.resource_spans()
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable OTLP trace for trial %s: %s", trial.id, error)
        return None

    span_id = root_span_id(resource_spans)
    if span_id is None:
        logger.warning(
            "Ignoring OTLP trace for trial %s: it has no single root span with a usable id to score.", trial.id
        )
        return None

    set_span_attributes(
        resource_spans,
        {
            SESSION_ID_ATTRIBUTE: session_id,
            EVALUATION_NAME_ATTRIBUTE: evaluation_name,
            TEST_CASE_NAME_ATTRIBUTE: trial.task_id,
        },
    )
    set_root_span_attributes(
        resource_spans,
        {**_trial_totals(trial, measurements), **({USAGE_SUPERSEDED_ATTRIBUTE: True} if usage_superseded else {})},
    )
    if trial.error is not None:
        set_root_span_error(resource_spans, message=trial.error.message)
    fill_missing_start_times(resource_spans, start_time_unix_nano=int(started_at.timestamp() * 1_000_000_000))
    if measurements.runtime_sec is not None:
        fill_missing_root_end_time(resource_spans, duration_unix_nano=int(measurements.runtime_sec * 1_000_000_000))
    # The Export RPC envelope goes on only here, at the point these actually cross a wire.
    return ExportTraceServiceRequest(resource_spans=resource_spans).SerializeToString(), span_id


class OtlpSpanTooLargeError(ValueError):
    """A single span, with the envelope it needs, is larger than the OTLP request budget."""


def split_otlp_request(payload: bytes, *, max_request_bytes: int) -> list[bytes]:
    """Split a serialized ``ExportTraceServiceRequest`` into requests of at most ``max_request_bytes``.

    A payload that already fits is returned unchanged as the only request. Otherwise spans are
    packed in their original order, and each request repeats the resource and scope envelope
    (resource, scope, schema URLs) of the spans it carries, so every span keeps its ids, parent,
    and metadata. Resource or scope entries that hold no spans are dropped, since there is nothing
    for Intake to store against them.

    Sizes are accumulated from each part's encoded length rather than by re-serializing the growing
    request. Raises :class:`OtlpSpanTooLargeError` before returning anything when one span cannot
    fit in a request on its own, so a caller can fail without sending part of the trace.
    """
    if max_request_bytes <= 0:
        raise ValueError(f"max_request_bytes must be positive, got {max_request_bytes}")
    if len(payload) <= max_request_bytes:
        return [payload]

    request = ExportTraceServiceRequest()
    request.ParseFromString(payload)

    batches: list[bytes] = []
    batch = _OtlpBatch()
    for resource_index, resource_spans in enumerate(request.resource_spans):
        resource = _Envelope.of(resource_spans)
        for scope_index, scope_spans in enumerate(resource_spans.scope_spans):
            scope = _Envelope.of(scope_spans)
            key = (resource_index, scope_index)
            for span in scope_spans.spans:
                span_field = _field_size(span.ByteSize())
                sizes = batch.sizes_with(span_field, resource=resource, scope=scope, key=key)
                if sizes[0] > max_request_bytes and batch.key is not None:
                    batches.append(batch.request.SerializeToString())
                    batch = _OtlpBatch()
                    sizes = batch.sizes_with(span_field, resource=resource, scope=scope, key=key)
                if sizes[0] > max_request_bytes:
                    raise OtlpSpanTooLargeError(
                        f"span {span.span_id.hex()} needs a {sizes[0]}-byte OTLP request, larger than the "
                        f"{max_request_bytes}-byte budget"
                    )
                batch.add(span, sizes, resource=resource, scope=scope, key=key)

    if batch.key is not None:
        batches.append(batch.request.SerializeToString())
    return batches


@dataclass(frozen=True)
class _Envelope[M: (ResourceSpans, ScopeSpans)]:
    """A ResourceSpans or ScopeSpans without its child spans, and that envelope's encoded size.

    Copies the resource or scope and the schema URL. The batch appends the source message's
    child spans, so packing does not copy those span bodies.
    """

    message: M
    size: int

    @classmethod
    def of(cls, message: M) -> _Envelope[M]:
        envelope = type(message)()
        if isinstance(envelope, ResourceSpans) and isinstance(message, ResourceSpans):
            if message.HasField("resource"):
                envelope.resource.CopyFrom(message.resource)
        elif isinstance(envelope, ScopeSpans) and isinstance(message, ScopeSpans):
            if message.HasField("scope"):
                envelope.scope.CopyFrom(message.scope)
        envelope.schema_url = message.schema_url
        return cls(envelope, envelope.ByteSize())


#: ``(request, resource, scope)`` encoded sizes of a batch's open entries.
_BatchSizes = tuple[int, int, int]


@dataclass
class _OtlpBatch:
    """One outgoing request being packed, with the encoded sizes of its open entries."""

    request: ExportTraceServiceRequest = field(default_factory=ExportTraceServiceRequest)
    size: int = 0
    resource_size: int = 0
    scope_size: int = 0
    #: ``(resource index, scope index)`` in the source request of the last ScopeSpans added.
    key: tuple[int, int] | None = None

    def sizes_with(
        self,
        span_field: int,
        *,
        resource: _Envelope[ResourceSpans],
        scope: _Envelope[ScopeSpans],
        key: tuple[int, int],
    ) -> _BatchSizes:
        """Return the sizes after adding a span whose encoded field is ``span_field`` bytes."""
        same_scope = self.key == key
        same_resource = self.key is not None and self.key[0] == key[0]
        scope_size = (self.scope_size if same_scope else scope.size) + span_field
        if not same_resource:
            resource_size = resource.size + _field_size(scope_size)
            return self.size + _field_size(resource_size), resource_size, scope_size
        replaced_scope = _field_size(self.scope_size) if same_scope else 0
        resource_size = self.resource_size - replaced_scope + _field_size(scope_size)
        return self.size - _field_size(self.resource_size) + _field_size(resource_size), resource_size, scope_size

    def add(
        self,
        span: Span,
        sizes: _BatchSizes,
        *,
        resource: _Envelope[ResourceSpans],
        scope: _Envelope[ScopeSpans],
        key: tuple[int, int],
    ) -> None:
        if self.key is None or self.key[0] != key[0]:
            self.request.resource_spans.append(resource.message)
        if self.key != key:
            self.request.resource_spans[-1].scope_spans.append(scope.message)
        self.request.resource_spans[-1].scope_spans[-1].spans.append(span)
        self.size, self.resource_size, self.scope_size = sizes
        self.key = key


def _field_size(length: int) -> int:
    """Encoded size of a length-delimited field numbered 1-15 whose body is ``length`` bytes."""
    return 1 + _varint_size(length) + length


def _varint_size(value: int) -> int:
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def _trial_totals(trial: AgentEvalTrial, measurements: TrialMeasurements) -> dict[str, str | int | float]:
    """The trial-level totals and error that ATIF conveyed outside the step list."""
    # Read as numbers by Intake. ATIF carried these as ``final_metrics``; under OTLP they go on
    # the root span, which is the span that stands for the whole trial.
    totals: dict[str, str | int | float] = {}
    for value, attribute in (
        (measurements.prompt_tokens, "gen_ai.usage.input_tokens"),
        (measurements.completion_tokens, "gen_ai.usage.output_tokens"),
        (measurements.cache_read_tokens, "gen_ai.usage.cached_tokens"),
        (measurements.cost_usd, "gen_ai.usage.cost"),
    ):
        if value is not None:
            totals[attribute] = value
    if trial.error is not None:
        totals["exception.type"] = trial.error.type
        if trial.error.message is not None:
            totals["exception.message"] = trial.error.message
    return totals


def trial_to_atif_ingest(
    trial: AgentEvalTrial,
    *,
    run_id: str,
    evaluation_name: str,
    agent_name: str,
    started_at: datetime,
    agent_version: str = DEFAULT_AGENT_VERSION,
    model_name: str | None = None,
    final_metrics: AtifFinalMetricsParam | None = None,
    ended_at: datetime | None = None,
    steps: RecordedSteps | None = None,
    usage_superseded: bool = False,
) -> AtifCreateParams:
    """Build the ATIF ingest params for a single Trial.

    ``steps`` carries the trial's real ATIF trajectory when it has one (see
    :func:`atif_steps_from_trial`). Without it this falls back to a minimal single-step
    trajectory holding the trial's final output text, which is all a runner that emits no
    trajectory — Harbor's ``oracle``, for one — can offer. ``agent_version`` left at its default takes
    the version the trajectory itself records, when it records one. ``ended_at`` gives a trajectory with
    no duration of its own the trial's runtime as an invocation window on its final step.

    ``started_at`` (the run's start time) stamps any step that carries no timestamp of its own,
    because that is what makes re-ingest idempotent. Intake's ``spans`` table is a ``ReplacingMergeTree``
    keyed on ``(workspace, session_id, start_time, id)``, and a step with no timestamp
    falls back to the server's per-request ingest clock — so the same trajectory sent
    twice lands as two rows that never collapse. An explicit timestamp makes the root
    span's ``start_time`` a function of the run, not of when it was published.

    ``usage_superseded`` marks the trajectory as one whose model calls the Inference Gateway recorded in the session
    (``USAGE_SUPERSEDED_ATTRIBUTE``).
    """
    output_text = trial.output.output_text if trial.output is not None else None
    if agent_version == DEFAULT_AGENT_VERSION and steps is not None and steps.agent_version:
        agent_version = steps.agent_version
    agent: AtifAgentParam = {"name": agent_name, "version": agent_version}
    if model_name is not None:
        agent["model_name"] = model_name
    step: AtifStepParam = {
        "source": "agent",
        "step_id": 1,
        "message": output_text or "",
        "timestamp": started_at,
    }
    step_payload: list[AtifStepParam] = list(steps.steps) if steps is not None else [step]
    timed = steps.timed if steps is not None else False
    if ended_at is not None and not timed:
        final_step = step_payload[-1]
        step_payload[-1] = {
            **final_step,
            "extra": {
                **(final_step.get("extra") or {}),
                "invocation": {
                    "start_timestamp": started_at.timestamp(),
                    "end_timestamp": ended_at.timestamp(),
                },
            },
        }

    body: AtifCreateParams = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id_for(run_id, trial.id),
        "agent": agent,
        "steps": step_payload,
        "evaluation_context": run_task_to_evaluation_context(trial, evaluation_name=evaluation_name),
    }
    extra: dict[str, object] = {}
    if trial.error is not None:
        error: dict[str, object] = {"type": trial.error.type}
        if trial.error.message is not None:
            error["message"] = trial.error.message
        extra["error"] = error
    if usage_superseded:
        extra[USAGE_SUPERSEDED_ATTRIBUTE] = True
    if extra:
        body["extra"] = extra
    if final_metrics is not None:
        body["final_metrics"] = final_metrics
    return body


@dataclass(frozen=True)
class SkippedOutput:
    """A metric output omitted from publish, with the reason it was dropped."""

    name: str
    reason: str


def score_to_evaluator_results(
    score: AgentEvalTaskScore,
    *,
    session_id: str,
    span_id: str,
    output_names: Sequence[str] = (),
) -> tuple[list[EvaluatorResultCreateParams], list[SkippedOutput]]:
    """Map one ``AgentEvalTaskScore`` to ``(rows, skipped)`` for Intake.

    ``rows`` is one evaluator-result param per output: ``name`` is ``"{metric_type}.{output}"``
    (matching the SDK summary's aggregate naming). A scored output's value is coerced into the
    matching ``data_type``, populating exactly one of ``value`` / ``string_value``. An output the
    metric could not score — the whole score is FAILED, or the output is non-finite (NaN/inf) —
    becomes a FAILED row with no value and the diagnostic as its ``comment``, so Intake counts it
    as a failed attempt rather than inferring that from a missing row.

    A FAILED score usually carries no outputs, so ``output_names`` (the metric's declared outputs)
    names the rows to write; any outputs it does carry are added to that set, and when neither
    names anything the metric itself gets one FAILED row.
    ``session_id``/``span_id`` are supplied by the caller — the trajectory span id is resolved
    at publish time, not derivable from the pure score.

    ``skipped`` is kept for callers that report omissions; nothing is omitted today.
    """
    if score.status == AgentEvalScoreStatus.FAILED:
        # Observed outputs first, then declared ones not among them: a partial FAILED score must
        # still produce a row for every output the metric promised.
        names = list(dict.fromkeys([*(output.name for output in score.outputs), *output_names]))
        rows = [
            _failed_row(
                session_id=session_id,
                span_id=span_id,
                name=f"{score.metric_type}.{name}",
                comment=_comment_for_output(score, name),
            )
            for name in names
        ] or [
            _failed_row(
                session_id=session_id,
                span_id=span_id,
                name=score.metric_type,
                comment=_comment_for_output(score, None),
            )
        ]
        return rows, []

    rows: list[EvaluatorResultCreateParams] = []
    skipped: list[SkippedOutput] = []
    for output in score.outputs:
        name = f"{score.metric_type}.{output.name}"
        data_type, value, string_value = _coerce_metric_value(output.value)
        if value is not None and not math.isfinite(value):
            rows.append(
                _failed_row(
                    session_id=session_id,
                    span_id=span_id,
                    name=name,
                    comment=_comment_for_output(score, output.name) or "non-finite value",
                )
            )
            continue
        row: EvaluatorResultCreateParams = {
            "session_id": session_id,
            "span_id": span_id,
            "name": name,
            "data_type": data_type,
        }
        if value is not None:
            row["value"] = value
        if string_value is not None:
            row["string_value"] = string_value
        comment = _comment_for_output(score, output.name)
        if comment is not None:
            row["comment"] = comment
        rows.append(row)
    return rows, skipped


def _failed_row(*, session_id: str, span_id: str, name: str, comment: str | None) -> EvaluatorResultCreateParams:
    # A failed attempt has no value type of its own; NUMERIC keeps it on the score axis Intake rolls up.
    row: EvaluatorResultCreateParams = {
        "session_id": session_id,
        "span_id": span_id,
        "name": name,
        "data_type": "NUMERIC",
        "status": "FAILED",
    }
    if comment is not None:
        row["comment"] = comment
    return row


def _comment_for_output(score: AgentEvalTaskScore, output_name: str | None) -> str | None:
    """Select the diagnostic comment that describes one emitted output.

    A diagnostic naming this output wins; otherwise the first that names no output (the key absent
    or ``None``), which is the one describing the score as a whole.
    """
    for diagnostic in score.diagnostics:
        if output_name is not None and diagnostic.details.get(OUTPUT_DETAIL) == output_name:
            return diagnostic.message
    for diagnostic in score.diagnostics:
        if diagnostic.details.get(OUTPUT_DETAIL) is None:
            return diagnostic.message
    return None


def _coerce_metric_value(value: object) -> tuple[EvaluatorResultDataType, float | None, str | None]:
    """Classify a metric output value into ``(data_type, value, string_value)``.

    Unwraps a Pydantic ``RootModel`` (``.root``) first, then:

    * ``bool`` -> ``BOOLEAN`` with value 1.0/0.0 (checked before ``int``, since
      ``bool`` is a subclass of ``int``);
    * ``int``/``float`` -> ``NUMERIC``;
    * anything else (strings, labels) -> ``TEXT`` via ``str()``.

    CATEGORICAL is intentionally not emitted: a category and free text are
    indistinguishable at the value level today (both arrive as ``str``/``Label``),
    so everything string-valued maps to TEXT until a real signal exists.
    """
    unwrapped = value.root if isinstance(value, RootModel) else value
    if isinstance(unwrapped, bool):
        return "BOOLEAN", (1.0 if unwrapped else 0.0), None
    if isinstance(unwrapped, (int, float)):
        return "NUMERIC", float(unwrapped), None
    return "TEXT", None, str(unwrapped)
