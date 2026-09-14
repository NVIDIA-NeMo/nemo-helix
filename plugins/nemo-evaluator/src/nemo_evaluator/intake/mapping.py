# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary mapping: Evaluator vocabulary -> Intake/Experiments wire shapes.

This is the single place where Evaluator domain objects (``AgentEvalTrial``,
``AgentEvalTaskScore``, ``MetricOutput``) become the request bodies Intake and
the Experiments API expect. The Intake write-adapter tickets (D3/D4/D5) obtain
their request shapes and field names *only* from here, so a later rename is a
one-file change.

Design constraints (see AALGO-289):

* **Pure.** Every function reads SDK types and returns request params. No HTTP,
  no platform client, no imports from the Intake *service* (``nmp.intake.*``).
* **Typed at the boundary.** The returned values are typed-client
  ``TypedDict`` params (``AtifCreateParams`` / ``EvaluatorResultCreateParams``).
  At runtime they are plain dicts the adapter splats into the client
  (``client.intake.ingest.atif.create(**body)``); statically, ``ty`` checks our
  field names, literals, and nested shapes against the real generated schema, so
  an API change that regenerates the SDK surfaces here as a type error instead of
  drifting silently. We depend on the client SDK (already a plugin dependency),
  never on the Intake service package.
* The well-known evidence-key constants (``initial_state``/``trace``/``logs``/
  ``final_state``/``verifier_logs``) belong with the SDK evidence work (D1,
  AALGO-281). Until D1 lands, this module references them as string literals so
  it stays unblocked.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, TrialMeasurements
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
from nemo_evaluator_sdk.values.otlp import (
    fill_missing_start_times,
    root_span_id,
    set_root_span_attributes,
    set_root_span_error,
    set_span_attributes,
)
from nemo_evaluator_sdk.values.protocol import OUTPUT_DETAIL
from nemo_platform_plugin.intake.types import (
    AtifAgentParam,
    AtifCreateParams,
    AtifFinalMetricsParam,
    AtifStepParam,
    EvaluationContextParam,
    EvaluatorResultCreateParams,
    EvaluatorResultDataType,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
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
_SESSION_ID_ATTRIBUTE = "gen_ai.conversation.id"
_EVALUATION_NAME_ATTRIBUTE = "nemo.evaluation.name"
_TEST_CASE_NAME_ATTRIBUTE = "nemo.test_case.name"


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


async def atif_steps_from_trial(trial: AgentEvalTrial, *, started_at: datetime) -> list[AtifStepParam] | None:
    """Return the trial's real ATIF steps, or None when it carries no ATIF trajectory.

    Reads the ATIF view by name rather than the trial's primary trace, which may be an OTLP
    one. Only runners whose agent emits ATIF attach one; Harbor's
    ``oracle`` and ``nop`` never do, and an unreadable or malformed trajectory is treated the
    same as an absent one so a publish is never lost to bad evidence.

    The SDK's read model is deliberately more permissive than Intake's ingest schema, so steps that
    parse here can still be rejected there; :func:`nemo_evaluator.intake.publish.publish_to_intake`
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
        steps = await handle.steps()
    except (OSError, ValueError) as error:
        logger.warning("Ignoring unreadable ATIF trace for trial %s: %s", trial.id, error)
        return None

    payload: list[AtifStepParam] = []
    for index, step in enumerate(steps):
        dumped: AtifStepParam = step.model_dump(mode="json", exclude_none=True)
        # Ingest requires one-based sequential ids, while the SDK read model leaves step_id optional.
        dumped["step_id"] = index + 1
        # Intake keys spans on start_time and falls back to its own ingest clock for a step with
        # no timestamp, which would make re-publish duplicate instead of replace.
        dumped.setdefault("timestamp", started_at.isoformat())
        payload.append(dumped)
    return payload


async def otlp_ingest_for_trial(
    trial: AgentEvalTrial,
    *,
    session_id: str,
    evaluation_name: str,
    measurements: TrialMeasurements,
    started_at: datetime,
) -> tuple[bytes, str] | None:
    """Return the trial's OTLP payload stamped with evaluation identity, and its root span id.

    ``None`` when the trial has no readable OTLP trace, or when its spans have no single root
    carrying a usable id: a trial-level score needs one span standing for the whole run.

    ``started_at`` fills in any span that recorded no start time of its own, because Intake
    otherwise stores it against the ingest clock and start time is part of the key its spans
    table replaces on — so a re-publish would insert rather than replace.

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
            _SESSION_ID_ATTRIBUTE: session_id,
            _EVALUATION_NAME_ATTRIBUTE: evaluation_name,
            _TEST_CASE_NAME_ATTRIBUTE: trial.task_id,
        },
    )
    set_root_span_attributes(resource_spans, _trial_totals(trial, measurements))
    if trial.error is not None:
        set_root_span_error(resource_spans, message=trial.error.message)
    fill_missing_start_times(resource_spans, start_time_unix_nano=int(started_at.timestamp() * 1_000_000_000))
    # The Export RPC envelope goes on only here, at the point these actually cross a wire.
    return ExportTraceServiceRequest(resource_spans=resource_spans).SerializeToString(), span_id


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
    steps: Sequence[AtifStepParam] | None = None,
) -> AtifCreateParams:
    """Build the ATIF ingest params for a single Trial.

    ``steps`` carries the trial's real ATIF trajectory when it has one (see
    :func:`atif_steps_from_trial`). Without it this falls back to a minimal single-step
    trajectory holding the trial's final output text, which is all a runner that emits no
    trajectory — Harbor's ``oracle``, for one — can offer.

    ``started_at`` (the run's start time) stamps any step that carries no timestamp of its own,
    because that is what makes re-ingest idempotent. Intake's ``spans`` table is a ``ReplacingMergeTree``
    keyed on ``(workspace, session_id, start_time, id)``, and a step with no timestamp
    falls back to the server's per-request ingest clock — so the same trajectory sent
    twice lands as two rows that never collapse. An explicit timestamp makes the root
    span's ``start_time`` a function of the run, not of when it was published.
    """
    output_text = trial.output.output_text if trial.output is not None else None
    agent: AtifAgentParam = {"name": agent_name, "version": agent_version}
    if model_name is not None:
        agent["model_name"] = model_name
    step: AtifStepParam = {
        "source": "agent",
        "step_id": 1,
        "message": output_text or "",
        "timestamp": started_at,
    }
    if ended_at is not None:
        # Give the single-step trajectory a real duration so the root span's latency (end - start) is
        # the trial's runtime instead of 0. Intake reads the window start/end from this NAT invocation
        # block (epoch-second timestamps); ``started_at`` stays the start, so re-ingest is still idempotent.
        step["extra"] = {
            "invocation": {
                "start_timestamp": started_at.timestamp(),
                "end_timestamp": ended_at.timestamp(),
            }
        }
    step_payload: list[AtifStepParam] = list(steps) if steps else [step]

    body: AtifCreateParams = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": session_id_for(run_id, trial.id),
        "agent": agent,
        "steps": step_payload,
        "evaluation_context": run_task_to_evaluation_context(trial, evaluation_name=evaluation_name),
    }
    if trial.error is not None:
        error: dict[str, object] = {"type": trial.error.type}
        if trial.error.message is not None:
            error["message"] = trial.error.message
        body["extra"] = {"error": error}
    if final_metrics is not None:
        body["final_metrics"] = final_metrics
    return body


@dataclass(frozen=True)
class SkippedOutput:
    """A metric output omitted from publish, with the reason it was dropped (see cross-team ask X6)."""

    name: str
    reason: str


def score_to_evaluator_results(
    score: AgentEvalTaskScore,
    *,
    session_id: str,
    span_id: str,
) -> tuple[list[EvaluatorResultCreateParams], list[SkippedOutput]]:
    """Map one ``AgentEvalTaskScore`` to ``(rows, skipped)`` for Intake.

    ``rows`` is one evaluator-result param per publishable output: ``name`` is
    ``"{metric_type}.{output}"`` (matching the SDK summary's aggregate naming) and the
    value is coerced into the matching ``data_type``, populating exactly one of ``value``
    / ``string_value``. ``session_id``/``span_id`` are supplied by the caller — the
    trajectory span id is resolved at publish time, not derivable from the pure score.

    ``skipped`` carries the outputs that can't be published, with the reason — so the
    publishable/omitted split has a single source of truth and callers can report the
    omissions instead of silently losing them. A FAILED score yields no rows (every output
    skipped); a completed score's non-finite (NaN/inf) outputs are dropped (NaN isn't
    JSON-representable — the platform client's encoder rejects it — so it can't be sent).

    TODO(X6): once Intake can represent a failed metric result, publish these as failures
    instead of dropping them.
    """
    if score.status == AgentEvalScoreStatus.FAILED:
        skipped = [
            SkippedOutput(name=f"{score.metric_type}.{output.name}", reason="scoring failed")
            for output in score.outputs
        ]
        return [], skipped

    rows: list[EvaluatorResultCreateParams] = []
    skipped: list[SkippedOutput] = []
    for output in score.outputs:
        name = f"{score.metric_type}.{output.name}"
        data_type, value, string_value = _coerce_metric_value(output.value)
        if value is not None and not math.isfinite(value):
            skipped.append(SkippedOutput(name=name, reason="non-finite value"))
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


def _comment_for_output(score: AgentEvalTaskScore, output_name: str) -> str | None:
    """Select the diagnostic comment that describes one emitted output.

    A diagnostic naming this output wins; otherwise the first that names no output at all, which
    is the one describing the score as a whole.
    """
    for diagnostic in score.diagnostics:
        if diagnostic.details.get(OUTPUT_DETAIL) == output_name:
            return diagnostic.message
    for diagnostic in score.diagnostics:
        if OUTPUT_DETAIL not in diagnostic.details:
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
