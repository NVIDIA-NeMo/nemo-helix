# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map ATIF trajectories into Intake spans.

Preserved-only ATIF fields remain in ``raw_attributes`` when Intake has no
dedicated span column for them. That includes agent tool definitions,
``final_metrics.total_steps``, step reasoning effort, and copied-context flags.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from nhx.intake.config import DEFAULT_ATIF_MAX_SUBAGENT_DEPTH, MAX_ATIF_MAX_SUBAGENT_DEPTH
from nhx.intake.spans.domain import (
    NEMO_STEP_ID_ATTRIBUTE,
    EvaluatorResult,
    EvaluatorResultDataType,
    IntakeSpan,
    SpanKind,
    SpanStatus,
)
from nhx.intake.spans.ingest.atif_domain import (
    AtifMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifStepAgent,
    AtifSubagentTrajectoryRef,
    AtifToolCall,
    AtifTrajectory,
)
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.span_attribute_bags import SpanAttributeBags
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx.intake.spans.storage import json_dumps, json_dumps_preserve, stable_id
from pydantic import BaseModel

TrajectoryIdentity = tuple[str, ...]


class AtifTrajectoryDepthError(ValueError):
    """Raised when an embedded ATIF trajectory tree exceeds its configured depth."""


def validate_atif_trajectory_depth(
    trajectory: AtifTrajectory,
    *,
    max_subagent_depth: int = DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
) -> None:
    """Reject trajectory trees deeper than the configured safe maximum."""
    if not 1 <= max_subagent_depth <= MAX_ATIF_MAX_SUBAGENT_DEPTH:
        raise ValueError(
            f"max_subagent_depth must be between 1 and {MAX_ATIF_MAX_SUBAGENT_DEPTH}, got {max_subagent_depth}"
        )
    for _trajectory, _identity, depth in _trajectory_tree(trajectory):
        if depth > max_subagent_depth:
            raise AtifTrajectoryDepthError(
                f"ATIF trajectory depth {depth} exceeds configured maximum {max_subagent_depth}"
            )


def trajectory_to_spans(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    ingested_at: datetime,
    max_subagent_depth: int = DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
) -> list[IntakeSpan]:
    """Map an ATIF trajectory tree into one trace of intake spans."""
    validate_atif_trajectory_depth(trajectory, max_subagent_depth=max_subagent_depth)
    return _trajectory_tree_to_spans(
        workspace=workspace,
        trajectory=trajectory,
        trace_session_id=trajectory.session_id,
        trajectory_identity=(),
        external_parent_span_id=None,
        ingested_at=ingested_at,
        depth=1,
        max_subagent_depth=max_subagent_depth,
    )


def _trajectory_tree_to_spans(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    trace_session_id: str,
    trajectory_identity: TrajectoryIdentity,
    external_parent_span_id: str | None,
    ingested_at: datetime,
    depth: int,
    max_subagent_depth: int,
) -> list[IntakeSpan]:
    """Recursively map a trajectory and its embedded subagents into spans."""
    if depth > max_subagent_depth:
        raise AtifTrajectoryDepthError(f"ATIF trajectory depth {depth} exceeds configured maximum {max_subagent_depth}")
    trajectory_span = _trajectory_to_span(
        workspace=workspace,
        trajectory=trajectory,
        trace_session_id=trace_session_id,
        trajectory_identity=trajectory_identity,
        external_parent_span_id=external_parent_span_id,
        include_evaluation_context=not trajectory_identity,
        ingested_at=ingested_at,
    )
    spans = [trajectory_span]
    spans.extend(
        _evaluator_result_to_span(
            workspace=workspace,
            trajectory=trajectory,
            trace_session_id=trace_session_id,
            trajectory_identity=trajectory_identity,
            parent_span=trajectory_span,
            ingested_at=ingested_at,
        )
    )
    embedded_subagents = {
        subagent.trajectory_id: subagent
        for subagent in trajectory.subagent_trajectories or []
        if subagent.trajectory_id is not None
    }
    expanded_subagent_ids: set[str] = set()
    for index, step in enumerate(trajectory.steps):
        step_ended_at = _step_ended_at(step)
        step_span = _step_to_span(
            workspace=workspace,
            default_session_id=trace_session_id,
            default_agent_name=trajectory.agent.name,
            default_agent_version=trajectory.agent.version,
            default_model_name=trajectory.agent.model_name,
            external_parent_span_id=trajectory_span.external_span_id,
            trajectory_identity=trajectory_identity,
            step=step,
            index=index,
            step_ended_at=step_ended_at,
            ingested_at=ingested_at,
        )
        spans.append(step_span)
        spans.extend(
            _tool_call_to_span(
                workspace=workspace,
                default_session_id=trace_session_id,
                default_agent_name=trajectory.agent.name,
                default_agent_version=trajectory.agent.version,
                default_model_name=trajectory.agent.model_name,
                external_parent_span_id=step_span.external_span_id,
                trajectory_identity=trajectory_identity,
                step=step,
                step_index=index,
                tool_index=tool_index,
                tool_call=tool_call,
                ingested_at=ingested_at,
            )
            for tool_index, tool_call in enumerate(_step_tool_calls(step))
        )
        for result_index, result in _observation_results_with_subagents(step):
            for ref_index, subagent_ref in enumerate(result.subagent_trajectory_ref or []):
                embedded = embedded_subagents.get(subagent_ref.trajectory_id)
                if embedded is not None and subagent_ref.trajectory_id not in expanded_subagent_ids:
                    subagent = embedded
                    assert subagent.trajectory_id is not None
                    expanded_subagent_ids.add(subagent.trajectory_id)
                    spans.extend(
                        _trajectory_tree_to_spans(
                            workspace=workspace,
                            trajectory=subagent,
                            trace_session_id=trace_session_id,
                            trajectory_identity=(*trajectory_identity, "subagent", subagent.trajectory_id),
                            external_parent_span_id=step_span.external_span_id,
                            ingested_at=step_span.start_time,
                            depth=depth + 1,
                            max_subagent_depth=max_subagent_depth,
                        )
                    )
                    continue
                spans.append(
                    _subagent_ref_to_span(
                        workspace=workspace,
                        default_session_id=trace_session_id,
                        default_agent_name=trajectory.agent.name,
                        default_agent_version=trajectory.agent.version,
                        default_model_name=trajectory.agent.model_name,
                        external_parent_span_id=step_span.external_span_id,
                        trajectory_identity=trajectory_identity,
                        step=step,
                        step_index=index,
                        result_index=result_index,
                        ref_index=ref_index,
                        result=result,
                        subagent_ref=subagent_ref,
                        step_ended_at=step_ended_at,
                        ingested_at=ingested_at,
                    )
                )
    for subagent in trajectory.subagent_trajectories or []:
        assert subagent.trajectory_id is not None
        if subagent.trajectory_id in expanded_subagent_ids:
            continue
        spans.extend(
            _trajectory_tree_to_spans(
                workspace=workspace,
                trajectory=subagent,
                trace_session_id=trace_session_id,
                trajectory_identity=(*trajectory_identity, "subagent", subagent.trajectory_id),
                external_parent_span_id=trajectory_span.external_span_id,
                ingested_at=trajectory_span.start_time,
                depth=depth + 1,
                max_subagent_depth=max_subagent_depth,
            )
        )
    return spans


def _trajectory_span_id(
    *,
    workspace: str,
    trace_session_id: str,
    trajectory_identity: TrajectoryIdentity,
) -> str:
    """Return the stable span ID for a trajectory at a tree identity."""
    return stable_id(workspace, trace_session_id, *trajectory_identity, "trajectory", prefix="span")


def _trajectory_to_span(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    trace_session_id: str,
    trajectory_identity: TrajectoryIdentity,
    external_parent_span_id: str | None,
    include_evaluation_context: bool,
    ingested_at: datetime,
) -> IntakeSpan:
    """Map one trajectory node to its root agent span."""
    raw_attributes = _model_dict(trajectory)
    raw_attributes.pop("steps", None)
    raw_attributes.pop("evaluation_context", None)
    raw_attributes.pop("subagent_trajectories", None)
    # ATIF span IDs are trace-native by design: session_id is the trace identity,
    # while evaluation_context is queryable metadata on the root span.
    #
    # Token/cost accounting belongs on the spans that incurred the LLM calls
    # when a source emits per-step metrics. Some ATIF producers only emit
    # trajectory.final_metrics, so use those root totals field-by-field when
    # the matching per-step accounting is absent.
    final_metrics = trajectory.final_metrics
    input_tokens = (
        final_metrics.total_prompt_tokens
        if final_metrics is not None and not _trajectory_has_step_prompt_metrics(trajectory)
        else None
    )
    output_tokens = (
        final_metrics.total_completion_tokens
        if final_metrics is not None and not _trajectory_has_step_completion_metrics(trajectory)
        else None
    )
    cached_tokens = (
        final_metrics.total_cached_tokens
        if final_metrics is not None and not _trajectory_has_step_cached_metrics(trajectory)
        else None
    )
    cost_total_usd = (
        _decimal(final_metrics.total_cost_usd)
        if final_metrics is not None and not _trajectory_has_step_cost_metrics(trajectory)
        else None
    )
    trajectory_error = _dict_or_none((trajectory.extra or {}).get("error"))
    error_type = _non_empty_string(trajectory_error.get("type")) if trajectory_error is not None else None
    error_message = _non_empty_string(trajectory_error.get("message")) if trajectory_error is not None else None
    external_span_id = _trajectory_span_id(
        workspace=workspace,
        trace_session_id=trace_session_id,
        trajectory_identity=trajectory_identity,
    )
    attribute_bags = _span_attributes(
        model=trajectory.agent.model_name,
        agent_name=trajectory.agent.name,
        agent_version=trajectory.agent.version,
        evaluation_context=trajectory.evaluation_context if include_evaluation_context else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        total_tokens=_sum_ints(input_tokens, output_tokens),
        cost_total_usd=cost_total_usd,
        error_type=error_type,
        error_message=error_message,
        raw_attributes=raw_attributes,
    )
    trajectory_started_at = _trajectory_started_at(trajectory, ingested_at)
    # ATIF has no standard trajectory-level outcome field. Intake recognizes the
    # root ``extra.error`` convention; tool failures remain on their TOOL spans.
    return IntakeSpan(
        workspace=workspace,
        session_id=trace_session_id,
        trace_id=trace_session_id,
        source_format="atif",
        external_span_id=external_span_id,
        external_parent_span_id=external_parent_span_id or "",
        kind=SpanKind.AGENT,
        name=trajectory.agent.name,
        status=SpanStatus.ERROR if trajectory_error is not None else SpanStatus.SUCCESS,
        start_time=trajectory_started_at,
        end_time=_clamped_end(trajectory_started_at, _trajectory_ended_at(trajectory)),
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=_trajectory_input(trajectory) or "",
        output=_trajectory_output(trajectory) or "",
        event_ts=ingested_at,
    )


def _step_to_span(
    *,
    workspace: str,
    default_session_id: str,
    default_agent_name: str,
    default_agent_version: str | None,
    default_model_name: str | None,
    external_parent_span_id: str,
    trajectory_identity: TrajectoryIdentity,
    step: AtifStep,
    index: int,
    step_ended_at: datetime | None,
    ingested_at: datetime,
) -> IntakeSpan:
    """Map one ATIF step to a child span in the shared trace."""
    raw_step = _model_dict(step)
    metrics = _step_metrics(step)
    model_name = _step_model_name(step)
    input_tokens = metrics.prompt_tokens if metrics is not None else None
    output_tokens = metrics.completion_tokens if metrics is not None else None
    external_span_id = stable_id(
        workspace,
        default_session_id,
        *trajectory_identity,
        str(index),
        json_dumps(raw_step),
        prefix="span",
    )
    attribute_bags = _span_attributes(
        model=model_name or default_model_name,
        agent_name=default_agent_name,
        agent_version=default_agent_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=metrics.cached_tokens if metrics is not None else None,
        total_tokens=_sum_ints(input_tokens, output_tokens),
        cost_total_usd=_decimal(metrics.cost_usd) if metrics is not None else None,
        raw_attributes=raw_step,
    )
    attribute_bags.number[NEMO_STEP_ID_ATTRIBUTE] = float(step.step_id)
    step_started_at = _step_started_at(step, index, ingested_at)
    return IntakeSpan(
        workspace=workspace,
        session_id=default_session_id,
        trace_id=default_session_id,
        source_format="atif",
        external_span_id=external_span_id,
        external_parent_span_id=external_parent_span_id,
        kind=_step_kind(step),
        name=(default_agent_name if isinstance(step, AtifStepAgent) else f"{step.source}-{step.step_id}"),
        status=SpanStatus.SUCCESS,
        start_time=step_started_at,
        end_time=_clamped_end(step_started_at, step_ended_at),
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=_step_input(step) or "",
        output=_step_output(step) or "",
        event_ts=ingested_at,
    )


def _tool_call_to_span(
    *,
    workspace: str,
    default_session_id: str,
    default_agent_name: str,
    default_agent_version: str | None,
    default_model_name: str | None,
    external_parent_span_id: str,
    trajectory_identity: TrajectoryIdentity,
    step: AtifStep,
    step_index: int,
    tool_index: int,
    tool_call: AtifToolCall,
    ingested_at: datetime,
) -> IntakeSpan:
    """Map one ATIF tool call to a span under its owning step."""
    raw_tool_call = _model_dict(tool_call)
    result = _observation_result_for_tool_call(step, tool_call.tool_call_id)
    error_message = _tool_result_error_message(step, result)
    external_span_id = stable_id(
        workspace,
        default_session_id,
        *trajectory_identity,
        str(step_index),
        "tool",
        str(tool_index),
        json_dumps(raw_tool_call),
        prefix="span",
    )
    attribute_bags = _span_attributes(
        model=_step_model_name(step) or default_model_name,
        agent_name=default_agent_name,
        agent_version=default_agent_version,
        tool_name=tool_call.function_name,
        error_message=error_message,
        raw_attributes={
            "step_id": step.step_id,
            "tool_call": raw_tool_call,
            "observation_result": _model_dict(result) if result is not None else None,
        },
    )
    invocation_started_at, invocation_ended_at = _invocation_window(tool_call.extra)
    tool_started_at = invocation_started_at or _step_started_at(step, step_index, ingested_at)
    return IntakeSpan(
        workspace=workspace,
        session_id=default_session_id,
        trace_id=default_session_id,
        source_format="atif",
        external_span_id=external_span_id,
        external_parent_span_id=external_parent_span_id,
        kind=SpanKind.TOOL,
        name=tool_call.function_name,
        status=SpanStatus.ERROR if _tool_result_is_error(step, result) else SpanStatus.SUCCESS,
        start_time=tool_started_at,
        end_time=_clamped_end(tool_started_at, invocation_ended_at),
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=json_dumps_preserve(raw_tool_call),
        output=_string_or_json(result) or "",
        event_ts=ingested_at,
    )


def _subagent_ref_to_span(
    *,
    workspace: str,
    default_session_id: str,
    default_agent_name: str,
    default_agent_version: str | None,
    default_model_name: str | None,
    external_parent_span_id: str,
    trajectory_identity: TrajectoryIdentity,
    step: AtifStep,
    step_index: int,
    result_index: int,
    ref_index: int,
    result: AtifObservationResult,
    subagent_ref: AtifSubagentTrajectoryRef,
    step_ended_at: datetime | None,
    ingested_at: datetime,
) -> IntakeSpan:
    """Map an unexpanded subagent reference to a delegation span."""
    raw_result = _model_dict(result)
    raw_ref = _model_dict(subagent_ref)
    error_message = _tool_result_error_message(step, result)
    subagent_identity = _subagent_ref_identity(subagent_ref)
    external_span_id = stable_id(
        workspace,
        default_session_id,
        *trajectory_identity,
        str(step_index),
        "subagent",
        str(result_index),
        str(ref_index),
        subagent_identity,
        prefix="span",
    )
    attribute_bags = _span_attributes(
        model=_step_model_name(step) or default_model_name,
        agent_name=default_agent_name,
        agent_version=default_agent_version,
        error_message=error_message,
        raw_attributes={
            "step_id": step.step_id,
            "observation_result": raw_result,
            "subagent_trajectory_ref": raw_ref,
        },
    )
    subagent_started_at = _step_started_at(step, step_index, ingested_at)
    return IntakeSpan(
        workspace=workspace,
        session_id=default_session_id,
        trace_id=default_session_id,
        source_format="atif",
        external_span_id=external_span_id,
        external_parent_span_id=external_parent_span_id,
        kind=SpanKind.AGENT,
        name=f"subagent-{subagent_identity}",
        status=SpanStatus.ERROR if _tool_result_is_error(step, result) else SpanStatus.SUCCESS,
        start_time=subagent_started_at,
        end_time=_clamped_end(subagent_started_at, step_ended_at),
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=json_dumps_preserve(raw_ref),
        output=json_dumps_preserve(raw_result),
        event_ts=ingested_at,
    )


def _evaluator_result_to_span(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    trace_session_id: str,
    trajectory_identity: TrajectoryIdentity,
    parent_span: IntakeSpan,
    ingested_at: datetime,
) -> list[IntakeSpan]:
    """Map a trajectory verifier result to its evaluator span, when present."""
    extra = trajectory.extra or {}
    verifier_result = _dict_or_none(extra.get("verifier_result"))
    if verifier_result is None:
        return []
    verifier = _dict_or_none(extra.get("verifier"))
    score = _evaluator_score(verifier_result)
    input_value = {
        key: value
        for key, value in {
            "session_id": trace_session_id,
            "evaluated_span_id": parent_span.external_span_id,
            "task_id": extra.get("task_id"),
            "task_name": extra.get("task_name"),
            "trial_name": extra.get("trial_name"),
            "trial_uri": extra.get("trial_uri"),
        }.items()
        if value is not None
    }
    output_value = {
        "score": score,
        "verifier_result": verifier_result,
    }
    metadata = {
        key: value
        for key, value in {
            "source": "harbor",
            "event_type": "evaluator_result",
            "name": "harbor.verifier",
            "verifier": verifier,
        }.items()
        if value is not None
    }
    input_json = json_dumps_preserve(input_value)
    output_json = json_dumps_preserve(output_value)
    metadata_json = json_dumps_preserve(metadata)
    raw_attributes = {
        "openinference.span.kind": "EVALUATOR",
        "input.value": input_json,
        "input.mime_type": "application/json",
        "output.value": output_json,
        "output.mime_type": "application/json",
        "metadata": metadata_json,
        "event_type": "evaluator_result",
        "name": "harbor.verifier",
        "score": score,
        "verifier_result": verifier_result,
    }
    started_at = _evaluator_started_at(trajectory) or ingested_at
    ended_at = _evaluator_ended_at(trajectory)
    external_span_id = stable_id(
        workspace,
        trace_session_id,
        *trajectory_identity,
        "evaluator",
        json_dumps(raw_attributes),
        prefix="span",
    )
    attribute_bags = _span_attributes(
        model=trajectory.agent.model_name,
        agent_name=trajectory.agent.name,
        agent_version=trajectory.agent.version,
        raw_attributes=raw_attributes,
    )
    return [
        IntakeSpan(
            workspace=workspace,
            session_id=trace_session_id,
            trace_id=trace_session_id,
            source_format="atif",
            external_span_id=external_span_id,
            external_parent_span_id=parent_span.external_span_id,
            kind=SpanKind.EVALUATOR,
            name="harbor.verifier",
            status=SpanStatus.SUCCESS,
            start_time=started_at,
            end_time=ended_at,
            attributes_string=attribute_bags.string,
            attributes_number=attribute_bags.number,
            attributes_bool=attribute_bags.boolean,
            input=input_json,
            output=output_json,
            event_ts=ingested_at,
        )
    ]


def _span_attributes(
    *,
    model: str | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    evaluation_context: EvaluationContext | None = None,
    tool_name: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_total_usd: Decimal | None = None,
    raw_attributes: dict[str, Any] | None = None,
) -> SpanAttributeBags:
    """Build normalized semantic and raw attribute bags for an ATIF span."""
    semantic_attributes = SpanSemanticAttributes(
        model=model,
        agent_name=agent_name,
        agent_version=agent_version,
        evaluation_name=evaluation_context.evaluation_name if evaluation_context is not None else None,
        test_case_name=evaluation_context.test_case_name if evaluation_context is not None else None,
        tool_name=tool_name,
        error_type=error_type,
        error_message=error_message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens,
        cost_total_usd=cost_total_usd,
    )
    attribute_bags = semantic_attributes.to_bags()
    if raw_attributes is not None:
        attribute_bags.put_json("atif.raw", raw_attributes)
    return attribute_bags


def trajectory_to_evaluator_results(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    spans: list[IntakeSpan],
    ingested_at: datetime,
    max_subagent_depth: int = DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
) -> list[EvaluatorResult]:
    """Extract evaluator_results rows from an ATIF trajectory tree's verifier_result blocks.

    Emits one row per Harbor reward key (``verifier_result.rewards``), named by that
    key, targeting the EVALUATOR-kind span that ``trajectory_to_spans`` produced for
    the verifier. The span preserves the original tree structure; these rows make each
    score queryable by name and value without parsing the span payload.
    """
    validate_atif_trajectory_depth(trajectory, max_subagent_depth=max_subagent_depth)
    results: list[EvaluatorResult] = []
    for current_trajectory, trajectory_identity, _depth in _trajectory_tree(trajectory):
        verifier_result = _dict_or_none((current_trajectory.extra or {}).get("verifier_result"))
        if verifier_result is None:
            continue
        trajectory_span_id = _trajectory_span_id(
            workspace=workspace,
            trace_session_id=trajectory.session_id,
            trajectory_identity=trajectory_identity,
        )
        evaluator_span = next(
            (
                span
                for span in spans
                if span.kind == SpanKind.EVALUATOR and span.external_parent_span_id == trajectory_span_id
            ),
            None,
        )
        if evaluator_span is None:
            continue
        for name, raw_value in _evaluator_rewards(verifier_result):
            data_type, value, string_value = _coerce_evaluator_value(raw_value)
            results.append(
                EvaluatorResult(
                    # Per-key id: the reward name keeps each criterion's row distinct on the
                    # same span, and an identical re-ingest hashes to the same id (dedupe).
                    evaluator_result_id=stable_id(evaluator_span.external_span_id, name, prefix="eval"),
                    span_id=evaluator_span.external_span_id,
                    session_id=trajectory.session_id,
                    workspace=workspace,
                    name=name,
                    value=value,
                    string_value=string_value,
                    data_type=data_type,
                    comment=None,
                    created_by="intake:atif_importer",
                    created_at=ingested_at,
                    ingested_at=ingested_at,
                )
            )
    return results


def _trajectory_tree(
    trajectory: AtifTrajectory,
    trajectory_identity: TrajectoryIdentity = (),
) -> list[tuple[AtifTrajectory, TrajectoryIdentity, int]]:
    """Return trajectories, tree identities, and one-based depths in depth-first order."""
    trajectories: list[tuple[AtifTrajectory, TrajectoryIdentity, int]] = []
    stack = [(trajectory, trajectory_identity, 1)]
    while stack:
        current, current_identity, depth = stack.pop()
        trajectories.append((current, current_identity, depth))
        for subagent in reversed(current.subagent_trajectories or []):
            assert subagent.trajectory_id is not None
            stack.append(
                (
                    subagent,
                    (*current_identity, "subagent", subagent.trajectory_id),
                    depth + 1,
                )
            )
    return trajectories


def _evaluator_rewards(verifier_result: dict[str, Any]) -> list[tuple[str, bool | int | float | str]]:
    """Per-reward ``(name, value)`` pairs from a Harbor ``verifier_result``.

    Harbor writes a ``rewards`` dict (``reward.json``) whose keys are the metric
    identities — one named reward per ``tests/`` subdirectory, or the 1D
    ``{"reward": <score>}`` convention. Each key becomes its own evaluator_result, so
    multi-criterion verifiers keep their per-criterion breakdown and keys (incl.
    namespaced ones like ``v1/correctness``) pass through verbatim. Falls back to a
    single ``reward`` row when only a bare top-level ``score`` scalar is present.
    """
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict):
        return [(name, value) for name, value in rewards.items() if isinstance(value, (int, float, str))]
    score = verifier_result.get("score")
    if isinstance(score, (int, float, str)):
        return [("reward", score)]
    return []


def _coerce_evaluator_value(
    score: bool | int | float | str,
) -> tuple[EvaluatorResultDataType, float | None, str | None]:
    """Coerce an ATIF reward into evaluator result storage columns."""
    if isinstance(score, bool):
        return EvaluatorResultDataType.BOOLEAN, 1.0 if score else 0.0, None
    if isinstance(score, (int, float)):
        return EvaluatorResultDataType.NUMERIC, float(score), None
    return EvaluatorResultDataType.TEXT, None, str(score)


def _subagent_ref_identity(subagent_ref: AtifSubagentTrajectoryRef) -> str:
    """Return the most specific available subagent reference identity."""
    if subagent_ref.trajectory_id is not None:
        return subagent_ref.trajectory_id
    if subagent_ref.trajectory_path is not None:
        return subagent_ref.trajectory_path
    assert subagent_ref.session_id is not None
    return subagent_ref.session_id


def _step_kind(step: AtifStep) -> SpanKind:
    """Map an ATIF step source to its Intake span kind."""
    return SpanKind.LLM if isinstance(step, AtifStepAgent) else SpanKind.AGENT


def _step_input(step: AtifStep) -> str | None:
    """Extract span input from a non-agent step."""
    if isinstance(step, AtifStepAgent):
        return None
    return _string_or_json(step.message)


def _step_output(step: AtifStep) -> str | None:
    """Serialize the message, reasoning, and calls from an agent step."""
    if isinstance(step, AtifStepAgent):
        payload: dict[str, Any] = {}
        if step.message != "":
            payload["message"] = _json_value(step.message)
        if step.reasoning_content is not None:
            payload["reasoning_content"] = step.reasoning_content
        if step.tool_calls is not None:
            payload["tool_calls"] = [_model_dict(tool_call) for tool_call in step.tool_calls]
        return json_dumps_preserve(payload) if payload else None
    return None


def _step_model_name(step: AtifStep) -> str | None:
    """Return the model attached to an agent step, when present."""
    return step.model_name if isinstance(step, AtifStepAgent) else None


def _step_metrics(step: AtifStep) -> AtifMetrics | None:
    """Return metrics attached to an agent step, when present."""
    return step.metrics if isinstance(step, AtifStepAgent) else None


def _trajectory_has_step_prompt_metrics(trajectory: AtifTrajectory) -> bool:
    """Return whether any step reports prompt tokens."""
    return any(
        (metrics := _step_metrics(step)) is not None and metrics.prompt_tokens is not None for step in trajectory.steps
    )


def _trajectory_has_step_completion_metrics(trajectory: AtifTrajectory) -> bool:
    """Return whether any step reports completion tokens."""
    return any(
        (metrics := _step_metrics(step)) is not None and metrics.completion_tokens is not None
        for step in trajectory.steps
    )


def _trajectory_has_step_cached_metrics(trajectory: AtifTrajectory) -> bool:
    """Return whether any step reports cached tokens."""
    return any(
        (metrics := _step_metrics(step)) is not None and metrics.cached_tokens is not None for step in trajectory.steps
    )


def _trajectory_has_step_cost_metrics(trajectory: AtifTrajectory) -> bool:
    """Return whether any step reports cost."""
    return any(
        (metrics := _step_metrics(step)) is not None and metrics.cost_usd is not None for step in trajectory.steps
    )


def _trajectory_input(trajectory: AtifTrajectory) -> str | None:
    """Return the first user message from a trajectory."""
    for step in trajectory.steps:
        if step.source == "user":
            return _string_or_json(step.message)
    return None


def _trajectory_output(trajectory: AtifTrajectory) -> str | None:
    """Return the last agent response from a trajectory, when available."""
    for step in reversed(trajectory.steps):
        if isinstance(step, AtifStepAgent):
            if step.message != "":
                return _string_or_json(step.message)
            return _step_output(step)
    return None


def _step_tool_calls(step: AtifStep) -> list[AtifToolCall]:
    """Return tool calls from an agent step."""
    if not isinstance(step, AtifStepAgent):
        return []
    return step.tool_calls or []


def _step_observation(step: AtifStep) -> AtifObservation | None:
    """Return the observation attached to any step."""
    return step.observation


def _observation_result_for_tool_call(step: AtifStep, tool_call_id: str) -> AtifObservationResult | None:
    """Find the observation result produced by a tool call."""
    observation = _step_observation(step)
    if observation is None:
        return None
    for result in observation.results:
        if result.source_call_id == tool_call_id:
            return result
    return None


def _observation_results_with_subagents(step: AtifStep) -> list[tuple[int, AtifObservationResult]]:
    """Return indexed observation results that reference subagent trajectories."""
    observation = _step_observation(step)
    if observation is None:
        return []
    return [
        (result_index, result)
        for result_index, result in enumerate(observation.results)
        if result.subagent_trajectory_ref
    ]


def _trajectory_started_at(trajectory: AtifTrajectory, ingested_at: datetime) -> datetime:
    """Return the earliest explicit tree timestamp or the ingestion fallback."""
    return _trajectory_explicit_started_at(trajectory) or ingested_at


def _trajectory_explicit_started_at(trajectory: AtifTrajectory) -> datetime | None:
    """Return the earliest explicit timestamp in a trajectory tree."""
    started_candidates: list[datetime | None] = []
    for current, _identity, _depth in _trajectory_tree(trajectory):
        started_candidates.extend(_timestamp(step) for step in current.steps)
        started_candidates.extend(_invocation_window(step.extra)[0] for step in current.steps)
        started_candidates.extend(
            _invocation_window(tool_call.extra)[0] for step in current.steps for tool_call in _step_tool_calls(step)
        )
        started_candidates.append(_evaluator_started_at(current))
    return min((started_at for started_at in started_candidates if started_at is not None), default=None)


def _trajectory_ended_at(trajectory: AtifTrajectory) -> datetime | None:
    """Return the latest explicit timestamp in a trajectory tree."""
    ended_candidates: list[datetime | None] = []
    for current, _identity, _depth in _trajectory_tree(trajectory):
        ended_candidates.extend(_timestamp(step) for step in current.steps)
        ended_candidates.extend(_invocation_window(step.extra)[1] for step in current.steps)
        ended_candidates.extend(
            _invocation_window(tool_call.extra)[1] for step in current.steps for tool_call in _step_tool_calls(step)
        )
        ended_candidates.append(_evaluator_ended_at(current))
    return max((ended_at for ended_at in ended_candidates if ended_at is not None), default=None)


def _evaluator_started_at(trajectory: AtifTrajectory) -> datetime | None:
    """Return the verifier start timestamp, when valid."""
    verifier = _dict_or_none((trajectory.extra or {}).get("verifier"))
    return _datetime_from_value(verifier.get("started_at") if verifier is not None else None)


def _evaluator_ended_at(trajectory: AtifTrajectory) -> datetime | None:
    """Return the verifier finish timestamp, when valid."""
    verifier = _dict_or_none((trajectory.extra or {}).get("verifier"))
    return _datetime_from_value(verifier.get("finished_at") if verifier is not None else None)


def _step_started_at(step: AtifStep, index: int, ingested_at: datetime) -> datetime:
    """Derive a step start from invocation, timestamp, or fallback order."""
    invocation_started_at, _ = _invocation_window(step.extra)
    return invocation_started_at or _timestamp(step) or (ingested_at + timedelta(milliseconds=index))


def _step_ended_at(step: AtifStep) -> datetime | None:
    """Return only a producer-supplied invocation end.

    Step timestamps and adjacent steps establish ordering, not duration. An
    absent invocation end therefore remains absent instead of being inferred
    from a sibling step or evaluator timestamp.
    """
    return _invocation_window(step.extra)[1]


def _clamped_end(start_time: datetime, end_time: datetime | None) -> datetime | None:
    """Drop ends that precede the start (out-of-order producer timestamps)."""
    if end_time is not None and end_time < start_time:
        return None
    return end_time


def _invocation_window(extra: dict[str, Any] | None) -> tuple[datetime | None, datetime | None]:
    """Timing from the NAT ``extra["invocation"]`` contract (AtifInvocationInfo:
    epoch-second ``start_timestamp``/``end_timestamp``). Tolerant by spec —
    absent or malformed blocks yield no timing, never an ingest error."""
    invocation = _dict_or_none((extra or {}).get("invocation"))
    if invocation is None:
        return (None, None)
    return (
        _datetime_from_epoch(invocation.get("start_timestamp")),
        _datetime_from_epoch(invocation.get("end_timestamp")),
    )


def _datetime_from_epoch(value: Any) -> datetime | None:
    """Parse a numeric epoch timestamp as UTC without raising."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _timestamp(step: AtifStep) -> datetime | None:
    """Parse an optional step timestamp."""
    if step.timestamp is None:
        return None
    return _datetime_from_value(step.timestamp)


def _datetime_from_value(value: Any) -> datetime | None:
    """Parse an ISO 8601 value as an aware datetime without raising."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _tool_result_is_error(step: AtifStep, result: AtifObservationResult | None) -> bool:
    """Recognize supported ATIF producer error markers."""
    # These markers are emitted by the Claude-Code Harbor trajectories we ingest today.
    # ATIF itself does not define a normalized tool-result error field.
    metadata = _matched_tool_result_metadata(step, result)
    if metadata.get("is_error") is True:
        return True
    if _step_extra_bool(step, "tool_result_is_error") and _is_only_observation_result(step, result):
        return True
    content = _result_text(result)
    return content is not None and "[error]" in content.lower()


def _tool_result_error_message(step: AtifStep, result: AtifObservationResult | None) -> str | None:
    """Extract an error message from a failed observation result."""
    if not _tool_result_is_error(step, result):
        return None
    content = _result_text(result)
    if content is not None:
        return content
    raw_tool_result = _matched_tool_result_metadata(step, result).get("raw_tool_result")
    if isinstance(raw_tool_result, dict) and isinstance(raw_tool_result.get("content"), str):
        return raw_tool_result["content"]
    return None


def _matched_tool_result_metadata(step: AtifStep, result: AtifObservationResult | None) -> dict[str, Any]:
    """Return tool-result metadata that corresponds to an observation."""
    metadata = _step_extra_dict(step, "tool_result_metadata")
    if not metadata or result is None:
        return {}
    raw_tool_result = metadata.get("raw_tool_result")
    if isinstance(raw_tool_result, dict):
        raw_tool_use_id = raw_tool_result.get("tool_use_id")
        if isinstance(raw_tool_use_id, str):
            return metadata if result.source_call_id == raw_tool_use_id else {}
    return metadata if _is_only_observation_result(step, result) else {}


def _is_only_observation_result(step: AtifStep, result: AtifObservationResult | None) -> bool:
    """Return whether a result is the step's sole observation result."""
    observation = _step_observation(step)
    return (
        result is not None
        and observation is not None
        and len(observation.results) == 1
        and observation.results[0] is result
    )


def _result_text(result: AtifObservationResult | None) -> str | None:
    """Return scalar text from an observation result."""
    if result is None or not isinstance(result.content, str):
        return None
    return result.content


def _step_extra_bool(step: AtifStep, key: str) -> bool:
    """Read a strictly true boolean from step extras."""
    return step.extra is not None and step.extra.get(key) is True


def _step_extra_dict(step: AtifStep, key: str) -> dict[str, Any]:
    """Read a dictionary value from step extras."""
    value = step.extra.get(key) if step.extra is not None else None
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    """Narrow a value to a dictionary or None."""
    return value if isinstance(value, dict) else None


def _non_empty_string(value: Any) -> str | None:
    """Return a stripped non-empty string without coercing arbitrary metadata."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _evaluator_score(verifier_result: dict[str, Any]) -> bool | int | float | str | None:
    """Extract the primary scalar score from a verifier result."""
    score = verifier_result.get("score")
    if isinstance(score, (int, float, str)):
        return score
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, (int, float, str)):
        return reward
    if len(rewards) == 1:
        only_reward = next(iter(rewards.values()))
        if isinstance(only_reward, (int, float, str)):
            return only_reward
    return None


def _string_or_json(value: Any) -> str | None:
    """Preserve strings and serialize other JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json_dumps_preserve(_json_value(value))


def _json_value(value: Any) -> Any:
    """Recursively convert Pydantic models into JSON-compatible values."""
    if isinstance(value, BaseModel):
        return _model_dict(value)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _model_dict(model: BaseModel) -> dict[str, Any]:
    """Serialize a Pydantic model while omitting null fields."""
    return model.model_dump(mode="json", exclude_none=True)


def _decimal(value: Any) -> Decimal | None:
    """Coerce a value to Decimal without raising for invalid input."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _sum_ints(*values: int | None) -> int | None:
    """Sum present integer values or return None when all are absent."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)
