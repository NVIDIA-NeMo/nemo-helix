# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of the per-session inputs for Evaluation comparison.

One query per request covers both evaluations and reads every session — the
pairing pipeline in ``nhx.intake.experiments.compare`` needs each attempt, so
unlike the session pages there is no LIMIT.
"""

from __future__ import annotations

import math
from typing import Any

from nhx.intake.experiments.compare import (
    CompareMetric,
    EvaluationMetricSession,
    InvalidEvaluationMetricError,
)
from nhx.intake.repository.clickhouse.evaluation_session import SESSION_USAGE, usage_by_session_sql
from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.clickhouse.trace import current_spans_sql
from nhx.intake.repository.evaluation_compare import EvaluationCompareRepository
from nhx.intake.spans.storage import float_or_none, str_or_none


class ClickHouseEvaluationCompareRepository(EvaluationCompareRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def get_metric_sessions(
        self,
        *,
        workspace: str,
        evaluation_names: list[str],
        metric: CompareMetric,
    ) -> list[EvaluationMetricSession]:
        evaluation_names = list(dict.fromkeys(evaluation_names))
        if not evaluation_names:
            return []

        evaluation_names_sql, evaluation_parameters = _evaluation_name_parameters(evaluation_names)
        parameters: dict[str, Any] = {"workspace": workspace, **evaluation_parameters}
        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)

        evaluator_name = metric.evaluator_name
        if evaluator_name is not None:
            parameters["evaluator_name"] = evaluator_name
            statement, query_name = (
                _evaluator_sessions_sql(
                    trace_index_table=trace_index_table,
                    evaluator_results_table=self._executor.table(ClickHouseTable.EVALUATOR_RESULTS),
                    evaluation_names_sql=evaluation_names_sql,
                ),
                "evaluation_compare.evaluator_sessions",
            )
        elif metric.id == "latency_ms":
            statement, query_name = (
                _latency_sessions_sql(trace_index_table=trace_index_table, evaluation_names_sql=evaluation_names_sql),
                "evaluation_compare.latency_sessions",
            )
        elif metric.id in _USAGE_VALUE_EXPR:
            # The usage pipeline reads span attribute bags by parameterized key, so its
            # parameters must be bound like the session reads do.
            parameters.update(SESSION_USAGE.parameters)
            statement, query_name = (
                _usage_sessions_sql(
                    trace_index_table=trace_index_table,
                    spans_table=self._executor.table(ClickHouseTable.SPANS),
                    evaluation_names_sql=evaluation_names_sql,
                    metric=metric,
                ),
                "evaluation_compare.usage_sessions",
            )
        else:
            raise InvalidEvaluationMetricError(
                f"Unsupported metric {metric.id!r}: use 'evaluators.<name>', 'cost_usd', 'latency_ms', or 'tokens'."
            )

        rows = await self._executor.fetch_all(
            ClickHouseQuery(name=query_name, statement=statement, parameters=parameters)
        )
        return [_session(record) for record in rows]


def _evaluation_name_parameters(evaluation_names: list[str]) -> tuple[str, dict[str, str]]:
    # Only the generated ``%(evaluation_name_N)s`` placeholders are interpolated into the SQL; the
    # names themselves travel as bound parameters.
    parameters = {f"evaluation_name_{index}": evaluation_name for index, evaluation_name in enumerate(evaluation_names)}
    return ", ".join(f"%({name})s" for name in parameters), parameters


def _scoped_sessions_sql(trace_index_table: str, evaluation_names_sql: str) -> str:
    # Every session of both evaluations, deduped like the session reads. No LIMIT: the
    # comparison needs every attempt, not one 1,000-row page.
    return f"""
        SELECT
            workspace,
            evaluation_name,
            session_id,
            test_case_name,
            latency_ms
        FROM {trace_index_table} FINAL
        WHERE workspace = %(workspace)s
            AND is_deleted = 0
            AND evaluation_name IN ({evaluation_names_sql})
        ORDER BY root_started_at ASC, root_span_id ASC
        LIMIT 1 BY workspace, session_id, evaluation_name
    """


def _latency_sessions_sql(*, trace_index_table: str, evaluation_names_sql: str) -> str:
    # The root-span latency lives on trace_index itself; NULL means the session never
    # finished, and the pairing pipeline omits it there.
    return f"""
        SELECT
            sessions.evaluation_name AS evaluation_name,
            sessions.test_case_name AS test_case_name,
            sessions.latency_ms AS value
        FROM (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        ) AS sessions
        ORDER BY sessions.evaluation_name, sessions.session_id
    """


def _evaluator_sessions_sql(
    *,
    trace_index_table: str,
    evaluator_results_table: str,
    evaluation_names_sql: str,
) -> str:
    # One row per session that has any result for the named evaluator. A session whose
    # evaluator only failed still joins (avgIf over no scored rows is NaN, mapped to a
    # NULL value) and scores 0 in the pairing rules; a session with no result at all
    # stays absent and is omitted from the mean.
    return f"""
        WITH scoped_sessions AS (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        )
        SELECT
            sessions.evaluation_name AS evaluation_name,
            sessions.test_case_name AS test_case_name,
            avgIf(results.value, isNotNull(results.value)) AS value,
            max(results.status = 'FAILED') AS failed
        FROM scoped_sessions AS sessions
        INNER JOIN (
            SELECT workspace, session_id, value, status
            FROM {evaluator_results_table} FINAL
            WHERE workspace = %(workspace)s
                AND (workspace, session_id) IN (
                    SELECT DISTINCT workspace, session_id
                    FROM scoped_sessions
                )
                AND name = %(evaluator_name)s
                AND (
                    (data_type IN ('NUMERIC', 'BOOLEAN') AND value IS NOT NULL)
                    OR status = 'FAILED'
                )
        ) AS results
            ON sessions.workspace = results.workspace
            AND sessions.session_id = results.session_id
        GROUP BY
            sessions.workspace,
            sessions.session_id,
            sessions.evaluation_name,
            sessions.test_case_name
        ORDER BY sessions.evaluation_name, sessions.session_id
    """


_USAGE_VALUE_EXPR: dict[str, str] = {
    # summed_metric_columns already scales cost from stored micro-dollars once.
    "cost_usd": "usage.cost_usd",
    "tokens": (
        "if("
        "isNull(usage.input_tokens) AND isNull(usage.output_tokens), "
        "NULL, "
        "coalesce(usage.input_tokens, 0) + coalesce(usage.output_tokens, 0))"
    ),
}


def _usage_sessions_sql(
    *,
    trace_index_table: str,
    spans_table: str,
    evaluation_names_sql: str,
    metric: CompareMetric,
) -> str:
    # Sessions keep their row even when no span reported the metric (value NULL), so
    # the pairing pipeline can omit that attempt rather than count it as zero.
    current_scoped_spans = current_spans_sql(
        spans_table,
        extra_where_sql=(
            "(span_versions.workspace, span_versions.session_id) IN (SELECT workspace, session_id FROM scoped_sessions)"
        ),
    )
    return f"""
        WITH
        scoped_sessions AS (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        ),
        current_scoped_spans AS (
            {current_scoped_spans}
        ),
        session_usage AS (
            {usage_by_session_sql("current_scoped_spans")}
        )
        SELECT
            sessions.evaluation_name AS evaluation_name,
            sessions.test_case_name AS test_case_name,
            {_USAGE_VALUE_EXPR[metric.id]} AS value
        FROM scoped_sessions AS sessions
        LEFT JOIN session_usage AS usage
            ON sessions.workspace = usage.workspace
            AND sessions.session_id = usage.session_id
        ORDER BY sessions.evaluation_name, sessions.session_id
    """


def _session(record: dict[str, Any]) -> EvaluationMetricSession:
    value = float_or_none(record.get("value"))
    if value is not None and math.isnan(value):
        # avgIf over no scored rows yields NaN (a failed-only session); the pairing
        # rules score it 0 via the failed flag, not via a NaN mean.
        value = None
    return EvaluationMetricSession(
        evaluation_name=record["evaluation_name"],
        test_case_name=str_or_none(record.get("test_case_name")),
        value=value,
        failed=bool(record.get("failed")),
    )
