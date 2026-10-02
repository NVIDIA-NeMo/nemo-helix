# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Evaluation rollup reads."""

from __future__ import annotations

from typing import Any

from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.evaluation_rollup import EvaluationRollup, EvaluationRollupRepository, ScoreRollup
from nhx.intake.spans.span_attribute_catalog import COST_SCALE, SpanAttributeField, spec_for_field
from nhx.intake.spans.storage import float_or_none


class ClickHouseEvaluationRollupRepository(EvaluationRollupRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def get_rollups(self, *, workspace: str, evaluation_names: list[str]) -> dict[str, EvaluationRollup]:
        evaluation_names = list(dict.fromkeys(evaluation_names))
        rollups = {
            evaluation_name: EvaluationRollup(evaluation_name=evaluation_name) for evaluation_name in evaluation_names
        }
        if not evaluation_names:
            return rollups

        evaluation_names_sql, evaluation_parameters = _evaluation_name_parameters(evaluation_names)
        parameters = {"workspace": workspace, **evaluation_parameters}
        trace_index_table = self._executor.table(ClickHouseTable.TRACE_INDEX)

        for row in await self._executor.fetch_all(
            ClickHouseQuery(
                name="evaluation_rollups.run_counts",
                statement=_run_counts_sql(trace_index_table, evaluation_names_sql),
                parameters=parameters,
            )
        ):
            rollups[row["evaluation_name"]].run_count = int(row["run_count"])
            rollups[row["evaluation_name"]].test_case_count = int(row["test_case_count"])

        for row in await self._executor.fetch_all(
            ClickHouseQuery(
                name="evaluation_rollups.scores",
                statement=_score_rollups_sql(
                    trace_index_table=trace_index_table,
                    evaluator_results_table=self._executor.table(ClickHouseTable.EVALUATOR_RESULTS),
                    evaluation_names_sql=evaluation_names_sql,
                ),
                parameters=parameters,
            )
        ):
            rollups[row["evaluation_name"]].evaluator_scores[row["evaluator_name"]] = ScoreRollup(
                sum=float_or_none(row["sum"]),
                mean=float_or_none(row["mean"]),
                median=float_or_none(row["median"]),
                p90=float_or_none(row["p90"]),
                p95=float_or_none(row["p95"]),
                p99=float_or_none(row["p99"]),
                count=int(row["count"]),
            )

        for row in await self._executor.fetch_all(
            ClickHouseQuery(
                name="evaluation_rollups.metrics",
                statement=_metric_rollups_sql(
                    trace_index_table=trace_index_table,
                    spans_table=self._executor.table(ClickHouseTable.SPANS),
                    evaluation_names_sql=evaluation_names_sql,
                ),
                parameters={
                    **parameters,
                    "cost_key": spec_for_field(SpanAttributeField.COST_TOTAL_USD).bag_key,
                    "input_tokens_key": spec_for_field(SpanAttributeField.INPUT_TOKENS).bag_key,
                    "output_tokens_key": spec_for_field(SpanAttributeField.OUTPUT_TOKENS).bag_key,
                    "model_key": spec_for_field(SpanAttributeField.MODEL).bag_key,
                    "agent_name_key": spec_for_field(SpanAttributeField.AGENT_NAME).bag_key,
                    "agent_version_key": spec_for_field(SpanAttributeField.AGENT_VERSION).bag_key,
                },
            )
        ):
            rollup = rollups[row["evaluation_name"]]
            rollup.model_names = _string_list(row["model_names"])
            rollup.agent_names = _string_list(row["agent_names"])
            rollup.agent_versions = _string_list(row["agent_versions"])
            rollup.cost_usd = _score_rollup(row, "cost")
            rollup.latency_ms = _score_rollup(row, "latency")
            rollup.tokens = _score_rollup(row, "tokens")

        return rollups


def _evaluation_name_parameters(evaluation_names: list[str]) -> tuple[str, dict[str, str]]:
    parameters = {f"evaluation_name_{index}": evaluation_name for index, evaluation_name in enumerate(evaluation_names)}
    return ", ".join(f"%({name})s" for name in parameters), parameters


def _scoped_sessions_sql(trace_index_table: str, evaluation_names_sql: str) -> str:
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


def _run_counts_sql(trace_index_table: str, evaluation_names_sql: str) -> str:
    # run_count is every ingested session; test_case_count is the distinct test cases those sessions
    # belong to. Sessions with no test_case_name aren't attributable to a test case, so they don't count
    # toward test_case_count (and are excluded from the test-case-weighted rollups below).
    return f"""
        WITH scoped_sessions AS (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        )
        SELECT
            evaluation_name,
            count() AS run_count,
            uniqExactIf(test_case_name, test_case_name != '') AS test_case_count
        FROM scoped_sessions
        GROUP BY evaluation_name
        ORDER BY evaluation_name ASC
    """


# Quantile name -> ClickHouse probability argument, shared by every distribution rollup.
_STAT_QUANTILES = {"median": "0.5", "p90": "0.9", "p95": "0.95", "p99": "0.99"}


def _stat_columns(value_expr: str, *, prefix: str = "", guarded: bool = False) -> str:
    """Build the sum/mean/median/p90/p95/p99/count column list for one value expression.

    ``guarded`` wraps each aggregate so that an empty or all-NULL set yields NULL instead of
    ``sumIf``'s 0 / ``avgIf``'s NaN; use it when ``value_expr`` can be NULL per input row.
    """

    label = f"{prefix}_" if prefix else ""
    if guarded:
        not_null = f"isNotNull({value_expr})"
        guard = f"countIf({not_null}) = 0"

        def stat(expr: str) -> str:
            return f"if({guard}, NULL, {expr})"

        columns = [
            f"{stat(f'sumIf({value_expr}, {not_null})')} AS {label}sum",
            f"{stat(f'avgIf({value_expr}, {not_null})')} AS {label}mean",
            *(
                f"{stat(f'quantileExactIf({q})({value_expr}, {not_null})')} AS {label}{name}"
                for name, q in _STAT_QUANTILES.items()
            ),
            f"countIf({not_null}) AS {label}count",
        ]
    else:
        columns = [
            f"sum({value_expr}) AS {label}sum",
            f"avg({value_expr}) AS {label}mean",
            *(f"quantileExact({q})({value_expr}) AS {label}{name}" for name, q in _STAT_QUANTILES.items()),
            f"count() AS {label}count",
        ]
    return ",\n            ".join(columns)


def _score_rollups_sql(*, trace_index_table: str, evaluator_results_table: str, evaluation_names_sql: str) -> str:
    """Test-case-weighted score distribution per (evaluation, evaluator).

    A pipeline of CTEs, each built by its own helper below — read top to bottom:
      scoped_sessions    — the in-scope sessions (deduped)
      session_scores     — stage 1: one value per (session, evaluator)
      test_case_sessions — attempts per test case (the fixed denominator)
      evaluators         — the evaluator axis of the per-test-case grid
      test_case_scores   — stage 2: one value per (test case, evaluator), zero-filled
    The final SELECT takes the distribution (sum/mean/quantiles/count) across test cases. Sessions with
    no test_case_name can't be attributed to a test case and are dropped.
    """
    return f"""
        WITH
        scoped_sessions AS (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        ),
        session_scores AS (
            {_session_scores_cte(evaluator_results_table)}
        ),
        test_case_sessions AS (
            {_test_case_sessions_cte()}
        ),
        evaluators AS (
            {_evaluators_cte(evaluator_results_table)}
        ),
        test_case_scores AS (
            {_test_case_scores_cte()}
        )
        SELECT
            evaluation_name,
            evaluator_name,
            {_stat_columns("value")}
        FROM test_case_scores
        GROUP BY evaluation_name, evaluator_name
        ORDER BY evaluation_name ASC, evaluator_name ASC
    """


def _sessions_join_scored_results(evaluator_results_table: str, *, columns: str) -> str:
    """Join scoped ``sessions`` to their scored evaluator_results (NUMERIC/BOOLEAN, non-null values).

    ``columns`` is the projection taken from evaluator_results ("name, value" or "name"). The inner
    subquery pre-filters to scoped sessions so ClickHouse prunes evaluator_results before the join, and
    the trailing WHERE keeps only sessions that carry a test_case_name — the ones the rollup is over.
    """
    return f"""FROM scoped_sessions AS sessions
            INNER JOIN (
                SELECT workspace, session_id, {columns}
                FROM {evaluator_results_table} FINAL
                WHERE workspace = %(workspace)s
                    AND (workspace, session_id) IN (
                        SELECT DISTINCT workspace, session_id
                        FROM scoped_sessions
                    )
                    AND data_type IN ('NUMERIC', 'BOOLEAN')
                    AND value IS NOT NULL
            ) AS results
                ON sessions.workspace = results.workspace
                AND sessions.session_id = results.session_id
            WHERE sessions.test_case_name != ''"""


def _session_scores_cte(evaluator_results_table: str) -> str:
    """Stage 1 — reduce each (session, evaluator) to one value by averaging its result rows.

    Averaging first means a session that emitted the same evaluator more than once counts once, so it
    can't inflate the test case's sum downstream.
    """
    return f"""
            SELECT
                sessions.evaluation_name AS evaluation_name,
                sessions.test_case_name AS test_case_key,
                results.name AS evaluator_name,
                avg(results.value) AS value
            {_sessions_join_scored_results(evaluator_results_table, columns="name, value")}
            GROUP BY sessions.evaluation_name, sessions.session_id, sessions.test_case_name, results.name"""


def _test_case_sessions_cte() -> str:
    """Per test case, the number of sessions (attempts) — the fixed denominator for its mean.

    Counting distinct sessions (not scored rows) means unscored attempts still count toward the
    denominator, so missing scores land as implicit zeros in the test case's mean.
    """
    return """
            SELECT
                evaluation_name,
                test_case_name AS test_case_key,
                count(DISTINCT session_id) AS session_count
            FROM scoped_sessions
            WHERE test_case_name != ''
            GROUP BY evaluation_name, test_case_name"""


def _evaluators_cte(evaluator_results_table: str) -> str:
    """Distinct (evaluation, evaluator) pairs — the evaluator axis of the per-test-case grid.

    Sourced from evaluator_results rather than ``FROM session_scores``: ClickHouse re-executes a CTE on
    every reference, so reading session_scores here would redundantly re-run its averaging just to
    recover evaluator names.
    """
    return f"""
            SELECT DISTINCT
                sessions.evaluation_name AS evaluation_name,
                results.name AS evaluator_name
            {_sessions_join_scored_results(evaluator_results_table, columns="name")}"""


def _test_case_scores_cte() -> str:
    """Stage 2 — one value per (test case, evaluator): summed scores over the test case's session count.

    The fixed denominator (``sum(scores)/session_count``) makes unscored sessions implicit zeros. Crossing
    every test case with every evaluator (INNER JOIN evaluators) and coalescing keeps a fully-unscored
    test case in the distribution as 0 instead of dropping it.
    """
    return """
            SELECT
                test_cases.evaluation_name AS evaluation_name,
                test_cases.test_case_key AS test_case_key,
                evaluators.evaluator_name AS evaluator_name,
                coalesce(sum(scores.value), 0) / test_cases.session_count AS value
            FROM test_case_sessions AS test_cases
            INNER JOIN evaluators ON evaluators.evaluation_name = test_cases.evaluation_name
            LEFT JOIN session_scores AS scores
                ON scores.evaluation_name = test_cases.evaluation_name
                AND scores.test_case_key = test_cases.test_case_key
                AND scores.evaluator_name = evaluators.evaluator_name
            GROUP BY
                test_cases.evaluation_name, test_cases.test_case_key, evaluators.evaluator_name, test_cases.session_count"""


def _current_session_span_metrics_sql(spans_table: str) -> str:
    """Latest-version spans for the scoped sessions, projected to *only* the metric inputs.

    The shared ``current_spans_sql`` emits every span column, including the full ``attributes_*`` Maps.
    The metric rollup then joins that to the sessions, so the join hash table would hold every deduped
    span with its maps — gigabytes on real agent workloads (long trajectories × many spans), which is
    what trips ClickHouse's memory limit. Here we ``argMax`` just the cost and token values (plus a
    present-flag each, to keep an absent metric distinct from a real 0) and the model/agent strings, so
    the join carries a handful of scalars per span instead. Deduping by span identity picks each span's
    latest version.
    """
    return f"""
        (
            SELECT
                workspace,
                argMax(session_id, (event_ts, is_deleted)) AS dedup_session_id,
                argMax(attributes_number[%(cost_key)s], (event_ts, is_deleted)) AS cost_value,
                argMax(has(mapKeys(attributes_number), %(cost_key)s), (event_ts, is_deleted)) AS cost_present,
                argMax(attributes_number[%(input_tokens_key)s], (event_ts, is_deleted)) AS input_tokens_value,
                argMax(has(mapKeys(attributes_number), %(input_tokens_key)s), (event_ts, is_deleted)) AS input_tokens_present,
                argMax(attributes_number[%(output_tokens_key)s], (event_ts, is_deleted)) AS output_tokens_value,
                argMax(has(mapKeys(attributes_number), %(output_tokens_key)s), (event_ts, is_deleted)) AS output_tokens_present,
                argMax(attributes_string[%(model_key)s], (event_ts, is_deleted)) AS model_name,
                argMax(attributes_string[%(agent_name_key)s], (event_ts, is_deleted)) AS agent_name,
                argMax(attributes_string[%(agent_version_key)s], (event_ts, is_deleted)) AS agent_version,
                argMax(is_deleted, (event_ts, is_deleted)) AS del_flag
            FROM {spans_table}
            WHERE workspace = %(workspace)s
                AND (workspace, session_id) IN (SELECT DISTINCT workspace, session_id FROM scoped_sessions)
            GROUP BY workspace, source_format, trace_id, external_span_id, id
        )
    """


def _metric_rollups_sql(*, trace_index_table: str, spans_table: str, evaluation_names_sql: str) -> str:
    # Two-level rollup: per-attempt cost/latency, then averaged per test case (avg per attempt — the
    # number must not scale with k), then the distribution across test cases (test-case-weighted).
    # Attempts with no cost/latency are excluded from a test case's average rather than counted as zero;
    # sessions with no test_case_name aren't attributable to a test case, so they're dropped.
    return f"""
        WITH
        scoped_sessions AS (
            {_scoped_sessions_sql(trace_index_table, evaluation_names_sql)}
        ),
        current_session_spans AS {_current_session_span_metrics_sql(spans_table)},
        session_costs AS (
            SELECT
                sessions.evaluation_name AS evaluation_name,
                sessions.test_case_name AS test_case_key,
                sessions.latency_ms AS latency_ms,
                if(
                    countIf(spans.cost_present) = 0,
                    NULL,
                    sumIf(spans.cost_value, spans.cost_present) / {COST_SCALE}
                ) AS cost_usd,
                if(
                    countIf(spans.input_tokens_present) = 0 AND countIf(spans.output_tokens_present) = 0,
                    NULL,
                    sumIf(spans.input_tokens_value, spans.input_tokens_present)
                        + sumIf(spans.output_tokens_value, spans.output_tokens_present)
                ) AS tokens,
                groupUniqArrayIf(spans.model_name, spans.model_name != '') AS model_names,
                groupUniqArrayIf(spans.agent_name, spans.agent_name != '') AS agent_names,
                groupUniqArrayIf(spans.agent_version, spans.agent_version != '') AS agent_versions
            FROM scoped_sessions AS sessions
            LEFT JOIN current_session_spans AS spans
                ON sessions.workspace = spans.workspace
                AND sessions.session_id = spans.dedup_session_id
                AND spans.del_flag = 0
            WHERE sessions.test_case_name != ''
            GROUP BY sessions.evaluation_name, sessions.session_id, sessions.test_case_name, sessions.latency_ms
        ),
        test_case_metrics AS (
            SELECT
                evaluation_name,
                test_case_key,
                if(countIf(isNotNull(cost_usd)) = 0, NULL, avgIf(cost_usd, isNotNull(cost_usd))) AS cost_usd,
                if(countIf(isNotNull(latency_ms)) = 0, NULL, avgIf(latency_ms, isNotNull(latency_ms))) AS latency_ms,
                if(countIf(isNotNull(tokens)) = 0, NULL, avgIf(tokens, isNotNull(tokens))) AS tokens,
                arrayDistinct(arrayFlatten(groupArray(model_names))) AS model_names,
                arrayDistinct(arrayFlatten(groupArray(agent_names))) AS agent_names,
                arrayDistinct(arrayFlatten(groupArray(agent_versions))) AS agent_versions
            FROM session_costs
            GROUP BY evaluation_name, test_case_key
        )
        SELECT
            evaluation_name,
            arraySort(arrayDistinct(arrayFlatten(groupArray(model_names)))) AS model_names,
            arraySort(arrayDistinct(arrayFlatten(groupArray(agent_names)))) AS agent_names,
            arraySort(arrayDistinct(arrayFlatten(groupArray(agent_versions)))) AS agent_versions,
            {_stat_columns("cost_usd", prefix="cost", guarded=True)},
            {_stat_columns("latency_ms", prefix="latency", guarded=True)},
            {_stat_columns("tokens", prefix="tokens", guarded=True)}
        FROM test_case_metrics
        GROUP BY evaluation_name
        ORDER BY evaluation_name ASC
    """


def _score_rollup(row: dict[str, Any], prefix: str) -> ScoreRollup | None:
    count = int(row[f"{prefix}_count"])
    if count == 0:
        return None
    return ScoreRollup(
        sum=float_or_none(row[f"{prefix}_sum"]),
        mean=float_or_none(row[f"{prefix}_mean"]),
        median=float_or_none(row[f"{prefix}_median"]),
        p90=float_or_none(row[f"{prefix}_p90"]),
        p95=float_or_none(row[f"{prefix}_p95"]),
        p99=float_or_none(row[f"{prefix}_p99"]),
        count=count,
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    return sorted(str(item) for item in value if str(item))
