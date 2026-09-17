# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt a dataset-driven (row) evaluation result into the shape ``publish_to_intake`` consumes.

Row evaluation and agent evaluation converge at the Intake boundary: a published trajectory is a
single step carrying the final output text (see ``mapping.trial_to_atif_ingest``), and a row's
``sample`` is already ``{"output_text": ..., "response": ...}`` — the field names of ``AgentOutput``.
So rather than a second mapping and a second publish loop, a row result is adapted to an
``AgentEvalResult`` and goes through the same publisher, inheriting its idempotency guarantees.

The row vocabulary maps as: one row -> one trial, one (row, metric key) -> one score. A row's
test case identity is its content hash by default, so repeated rows become repeated trials of a
single test case — which is what the trial/task split already expresses.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nemo_evaluator_sdk.agent_eval.scores import (
    AgentEvalDiagnostic,
    AgentEvalDiagnosticSeverity,
    AgentEvalScoreStatus,
    AgentEvalTaskScore,
)
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, TrialMeasurements
from nemo_evaluator_sdk.agent_eval.usage_keys import (
    CACHE_CREATION_INPUT_TOKENS_KEY,
    CACHE_READ_INPUT_TOKENS_KEY,
    CACHED_TOKENS_KEY,
    COMPLETION_TOKEN_KEYS,
    PROMPT_TOKEN_KEYS,
    SEPARATE_CACHE_KEYS,
    USAGE_COUNT_KEYS,
    USAGE_DETAILS_KEYS,
    _first_nonnegative_int,
    _first_usage_details,
)
from nemo_evaluator_sdk.values.dataset_schemas import _KNOWN_BINDING_FIELDS
from nemo_evaluator_sdk.values.multi_metric_results import BenchmarkEvaluationResult
from nemo_evaluator_sdk.values.results import EvaluationResult, RowScore

logger = logging.getLogger(__name__)

#: Key ``sample`` carries when generation itself failed, rather than the metric.
_INFERENCE_ERROR = "inference_error"

#: Canonical evaluator fields that ``field_mapping`` copies into a row alongside its own columns.
_CANONICAL_FIELDS = frozenset(_KNOWN_BINDING_FIELDS)


class RowIdentityError(ValueError):
    """A row's published identity is unusable — missing, or shared with another row."""


def _canonical_row_hash(row: RowScore) -> str:
    """Stable ``sha256`` of a row's dataset content, excluding field-mapping's canonical aliases.

    Mirrors ``gym_runtime._canonical_row_hash``: identity is the row content alone, so a row keeps
    its id across dataset revisions and reorderings and a changed row becomes a new test case. The
    aliases are excluded because they duplicate values already in the row, so hashing them would
    change the id whenever only the ``field_mapping`` changed.
    """
    payload = {key: value for key, value in row.item.items() if key not in _CANONICAL_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _task_id(row: RowScore, index: int, test_case_id_field: str | None) -> str:
    """The row's test case identity — stable run over run, so rollups line up."""
    if test_case_id_field is None:
        return _canonical_row_hash(row)
    if test_case_id_field not in row.item:
        raise RowIdentityError(
            f"Row {index} has no {test_case_id_field!r} column; "
            f"available columns: {sorted(row.item)}. "
            "Fix `publication.intake.test_case_id_field` or remove it to identify rows by content."
        )
    return str(row.item[test_case_id_field])


def _output(row: RowScore) -> AgentOutput | None:
    """The row's generated output, or ``None`` when generation failed or produced nothing."""
    if row.sample.get(_INFERENCE_ERROR):
        return None
    output_text = row.sample.get("output_text")
    response = row.sample.get("response")
    if output_text is None and response is None:
        return None
    return AgentOutput(output_text=output_text, response=response)


def _token_measurements(row: RowScore) -> TrialMeasurements:
    """Normalize the generation response's ``usage`` block into typed measurements.

    A row's generation response is the only place its token usage survives: ``row.requests`` mixes
    the generation call with each metric's judge calls, so summing that would credit judge tokens to
    the agent. ``TrialMeasurements`` derives the canonical total from the parts.

    OpenAI's cache detail is already included in prompt tokens. Anthropic's cache
    creation/read fields are separate and are added to input tokens exactly once.

    No key list covers every provider, and a renamed key would fail the same silent way this
    function exists to fix, so a usage block that yields nothing is logged with the keys it actually
    carried. An unrecognized schema is then a log line naming what to add, not an unexplained blank.
    """
    response = row.sample.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if not isinstance(usage, Mapping):
        return TrialMeasurements()
    _warn_invalid_usage_counts(usage, row_index=row.row_index)

    separate_cache = any(key in usage for key in SEPARATE_CACHE_KEYS)
    details = _first_usage_details(usage)
    inclusive_cache = details is not None and CACHED_TOKENS_KEY in details
    completion = _first_nonnegative_int(usage, *COMPLETION_TOKEN_KEYS)

    if separate_cache and inclusive_cache:
        logger.warning(
            "Row %r generation usage contains both inclusive cache details and separate cache fields; "
            "omitting ambiguous prompt/cache measurements",
            row.row_index,
        )
        return TrialMeasurements(completion_tokens=completion)

    cache_read = _first_nonnegative_int(usage, CACHE_READ_INPUT_TOKENS_KEY) if separate_cache else None
    cache_creation = _first_nonnegative_int(usage, CACHE_CREATION_INPUT_TOKENS_KEY) if separate_cache else None
    invalid_separate_cache = any(
        key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None
        for key in SEPARATE_CACHE_KEYS
    )
    prompt = _first_nonnegative_int(usage, *PROMPT_TOKEN_KEYS)
    if separate_cache and prompt is not None:
        prompt = None if invalid_separate_cache else prompt + (cache_creation or 0) + (cache_read or 0)
    elif details is not None and CACHED_TOKENS_KEY in details:
        cache_read = _first_nonnegative_int(details, CACHED_TOKENS_KEY)

    captured = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
    }
    recorded = {key: value for key, value in captured.items() if value is not None}
    if not recorded:
        logger.warning(
            "Row %r has no recognized token counts in the generation response's usage block; keys present: %s",
            row.row_index,
            sorted(str(key) for key in usage),
        )
    return TrialMeasurements(**recorded)


def _warn_invalid_usage_counts(usage: Mapping[str, Any], *, row_index: int | None) -> None:
    for key in USAGE_COUNT_KEYS:
        if key in usage and usage[key] is not None and _first_nonnegative_int(usage, key) is None:
            logger.warning(
                "Row %r has invalid generation usage %s=%r; expected a non-negative integer and omitted it",
                row_index,
                key,
                usage[key],
            )
    for details_key in USAGE_DETAILS_KEYS:
        details = usage.get(details_key)
        if (
            isinstance(details, Mapping)
            and CACHED_TOKENS_KEY in details
            and details[CACHED_TOKENS_KEY] is not None
            and _first_nonnegative_int(details, CACHED_TOKENS_KEY) is None
        ):
            logger.warning(
                "Row %r has invalid generation usage %s.cached_tokens=%r; expected a non-negative integer and omitted it",
                row_index,
                details_key,
                details[CACHED_TOKENS_KEY],
            )


def _scores(row: RowScore, *, run_id: str, task_id: str, trial_id: str) -> list[AgentEvalTaskScore]:
    """One score per metric key on the row; ``metrics`` values are already ``MetricOutput``."""
    errors = row.metric_errors or {}
    diagnostics = row.metric_diagnostics or {}
    scores: list[AgentEvalTaskScore] = []
    for metric_key, outputs in row.metrics.items():
        error = errors.get(metric_key)
        # Error first: this is the leading general diagnostic, and the failure is what a reader
        # needs to see when a score can eventually be published as a failed measurement.
        row_diagnostics = [
            AgentEvalDiagnostic(
                severity=AgentEvalDiagnosticSeverity.WARNING,
                message=item.message,
                details=dict(item.details or {}),
            )
            for item in diagnostics.get(metric_key, [])
        ]
        if error:
            row_diagnostics.insert(0, AgentEvalDiagnostic(severity=AgentEvalDiagnosticSeverity.ERROR, message=error))
        scores.append(
            AgentEvalTaskScore(
                id=f"{run_id}:{trial_id}:{metric_key}",
                run_id=run_id,
                task_id=task_id,
                trial_id=trial_id,
                metric_type=metric_key,
                status=AgentEvalScoreStatus.FAILED if error else AgentEvalScoreStatus.COMPLETED,
                outputs=list(outputs),
                diagnostics=row_diagnostics,
            )
        )
    return scores


def row_result_to_agent_eval_result(
    result: EvaluationResult | BenchmarkEvaluationResult,
    *,
    run_id: str,
    started_at: datetime,
    test_case_id_field: str | None = None,
) -> AgentEvalResult:
    """Adapt a row evaluation result for ``publish_to_intake``.

    ``run_id`` and ``started_at`` come from the job: a row result carries neither, and both must be
    stable across a re-publish or the trajectory lands as a duplicate span rather than replacing the
    previous one.

    Reads the top-level ``row_scores`` only. ``BenchmarkEvaluationResult`` repeats every row under
    ``per_metric[key].row_scores`` as well; walking those would publish each row once per metric.
    """
    trials: list[AgentEvalTrial] = []
    scores: list[AgentEvalTaskScore] = []
    first_seen: dict[str, int] = {}
    occurrences: dict[str, int] = {}
    for index, row in enumerate(result.row_scores):
        task_id = _task_id(row, index, test_case_id_field)
        repeat = occurrences.get(task_id, 0)
        # A named column that repeats is a misconfiguration — the submitter said it identifies rows.
        # Identical content under the content hash is just the same test case evaluated twice.
        if repeat and test_case_id_field is not None:
            raise RowIdentityError(
                f"Rows {first_seen[task_id]} and {index} share test case id {task_id!r} from column "
                f"{test_case_id_field!r}. Name a column whose values are unique per row."
            )
        occurrences[task_id] = repeat + 1
        first_seen.setdefault(task_id, index)
        # Session ids are `{run_id}:{trial id}`, so repeats need distinct trial ids or the second
        # trajectory would replace the first. They keep one `task_id`, which is what rollups group on.
        trial_id = task_id if repeat == 0 else f"{task_id}#{repeat + 1}"
        output = _output(row)
        trials.append(
            AgentEvalTrial(
                id=trial_id,
                task_id=task_id,
                status=AgentEvalTrialStatus.COMPLETED if output is not None else AgentEvalTrialStatus.FAILED,
                output=output,
                measurements=_token_measurements(row),
            )
        )
        scores.extend(_scores(row, run_id=run_id, task_id=task_id, trial_id=trial_id))

    return AgentEvalResult(
        run_id=run_id,
        tasks=[],
        trials=trials,
        scores=scores,
        summary=AgentEvalSummary(scores=result.aggregate_scores),
        metadata=RunMetadata(started_at=started_at),
    )
