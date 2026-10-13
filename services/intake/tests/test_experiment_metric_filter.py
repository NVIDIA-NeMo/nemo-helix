# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric filtering on the evaluations list (Option A app-merge).

Two layers: pure helpers (split/validate/match) and endpoint wiring. The shared ``client`` fixture
overrides the rollup repository to ``None`` (ClickHouse unavailable), so a metric filter that passes
field validation must surface as 503 rather than silently dropping every row.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.intake.api.v2.experiments.endpoints import (
    _METRIC_STATS,
    _extract_metric_predicates,
    _is_metric_field,
    _is_valid_metric_path,
    _matches_metric_predicates,
    _operation_references_metric,
)
from nhx.intake.api.v2.experiments.schemas import EvaluationResponse, EvaluatorAggregate, MetricStatFilters

EVALUATIONS = "/apis/intake/v2/workspaces/default/evaluations"
EXPERIMENTS = "/apis/intake/v2/workspaces/default/experiments"


def _exp(
    name: str,
    *,
    run_count: int = 0,
    cost_mean: float | None = None,
    tokens_mean: float | None = None,
) -> EvaluationResponse:
    return EvaluationResponse(
        id=name,
        name=name,
        workspace="default",
        experiment_ids=["grp"],
        dataset_name="ds",
        run_count=run_count,
        cost_usd=EvaluatorAggregate(mean=cost_mean) if cost_mean is not None else None,
        tokens=EvaluatorAggregate(mean=tokens_mean) if tokens_mean is not None else None,
    )


def _cmp(field: str, op: FilterOperator, value: object) -> ComparisonOperation:
    return ComparisonOperation(operator=op, field=field, value=value)


# ----------------------------- pure helpers -----------------------------


def test_is_metric_field_classifies_by_head() -> None:
    assert _is_metric_field("run_count")
    assert _is_metric_field("cost_usd.mean")
    assert _is_metric_field("cost_usd.bogus")  # intentionally loose: extracted, then rejected
    assert _is_metric_field("evaluators.harbor.verifier.mean")
    assert not _is_metric_field("data.name")
    assert not _is_metric_field("name")


def test_metric_stat_filters_match_runtime_stats() -> None:
    # The stats enumerated in the OpenAPI-visible schema must mirror the runtime grammar, or the spec
    # would advertise stats the server rejects (or omit ones it accepts).
    assert set(MetricStatFilters.model_fields) == set(_METRIC_STATS)


def test_is_valid_metric_path() -> None:
    assert _is_valid_metric_path("run_count")
    assert _is_valid_metric_path("cost_usd.p95")
    assert _is_valid_metric_path("evaluators.harbor.verifier.mean")
    assert not _is_valid_metric_path("cost_usd.bogus")
    assert not _is_valid_metric_path("cost_usd")  # missing stat
    assert not _is_valid_metric_path("evaluators.reward")  # missing stat


def test_extract_splits_metric_from_entity_predicates() -> None:
    tree = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            _cmp("data.name", FilterOperator.EQ, "foo"),
            _cmp("cost_usd.mean", FilterOperator.LTE, "0.5"),
            _cmp("run_count", FilterOperator.GTE, "3"),
        ],
    )
    entity_op, predicates = _extract_metric_predicates(tree)
    # Only the entity predicate is forwarded to the store.
    assert isinstance(entity_op, ComparisonOperation)
    assert entity_op.field == "data.name"
    assert {p.field for p in predicates} == {"cost_usd.mean", "run_count"}
    assert all(isinstance(p.threshold, float) for p in predicates)


def test_extract_single_metric_comparison() -> None:
    entity_op, predicates = _extract_metric_predicates(_cmp("cost_usd.mean", FilterOperator.GT, "0.1"))
    assert entity_op is None
    assert predicates[0].field == "cost_usd.mean"


def test_extract_rejects_bad_stat() -> None:
    with pytest.raises(HTTPException) as exc:
        _extract_metric_predicates(_cmp("cost_usd.bogus", FilterOperator.GTE, "1"))
    assert exc.value.status_code == 400


def test_extract_rejects_non_numeric_operator() -> None:
    with pytest.raises(HTTPException) as exc:
        _extract_metric_predicates(_cmp("cost_usd.mean", FilterOperator.LIKE, "x"))
    assert exc.value.status_code == 400


def test_extract_rejects_non_numeric_value() -> None:
    with pytest.raises(HTTPException) as exc:
        _extract_metric_predicates(_cmp("cost_usd.mean", FilterOperator.GTE, "not-a-number"))
    assert exc.value.status_code == 400


def test_extract_flattens_nested_and() -> None:
    # A metric comparison nested inside a sub-AND is still AND-combined, so it must be accepted.
    tree = LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            _cmp("data.name", FilterOperator.EQ, "foo"),
            LogicalOperation(
                operator=FilterOperator.AND,
                operations=[_cmp("cost_usd.mean", FilterOperator.LTE, "0.5")],
            ),
        ],
    )
    entity_op, predicates = _extract_metric_predicates(tree)
    assert [p.field for p in predicates] == ["cost_usd.mean"]
    # The entity predicate survives; the metric one is stripped out for in-app evaluation.
    assert entity_op is not None
    assert not _operation_references_metric(entity_op)


def test_extract_rejects_metric_under_or() -> None:
    tree = LogicalOperation(
        operator=FilterOperator.OR,
        operations=[
            _cmp("cost_usd.mean", FilterOperator.GTE, "0.5"),
            _cmp("data.name", FilterOperator.EQ, "foo"),
        ],
    )
    with pytest.raises(HTTPException) as exc:
        _extract_metric_predicates(tree)
    assert exc.value.status_code == 400


def test_matches_predicates_excludes_missing_metric() -> None:
    cheap = _exp("cheap", cost_mean=0.2)
    pricey = _exp("pricey", cost_mean=0.9)
    norun = _exp("norun")  # no cost rollup
    _, predicates = _extract_metric_predicates(_cmp("cost_usd.mean", FilterOperator.LTE, "0.5"))
    assert _matches_metric_predicates(cheap, predicates)
    assert not _matches_metric_predicates(pricey, predicates)
    assert not _matches_metric_predicates(norun, predicates)  # missing metric never matches


def test_matches_predicates_resolves_tokens_metric() -> None:
    # Regression: tokens.mean must resolve off response.tokens, not fall through to the evaluator
    # branch (which returned None → "missing metric never matches" → filtered everything out).
    wordy = _exp("wordy", tokens_mean=942803)
    terse = _exp("terse", tokens_mean=200)
    _, gte = _extract_metric_predicates(_cmp("tokens.mean", FilterOperator.GTE, "500"))
    assert _matches_metric_predicates(wordy, gte)
    assert not _matches_metric_predicates(terse, gte)
    _, lte = _extract_metric_predicates(_cmp("tokens.mean", FilterOperator.LTE, "500"))
    assert not _matches_metric_predicates(wordy, lte)
    assert _matches_metric_predicates(terse, lte)


# ----------------------------- endpoint wiring -----------------------------


def _make_evaluation(client: TestClient, name: str = "exp-1", group: str = "grp-1") -> None:
    group_resp = client.post(EXPERIMENTS, json={"name": group})
    assert group_resp.status_code == 201, group_resp.text
    exp_resp = client.post(
        EVALUATIONS,
        json={"name": name, "experiment_group_id": group_resp.json()["id"], "dataset_name": "ds"},
    )
    assert exp_resp.status_code == 201, exp_resp.text


def test_metric_filter_passes_validation_and_503s_without_rollups(client: TestClient) -> None:
    # If the namespace declaration works, these paths get past field validation and reach the
    # metric-filter path, which 503s because the (mocked) rollup repository is None. Needs a non-empty
    # result set: an empty group has nothing to hydrate and correctly returns 200 empty.
    _make_evaluation(client)
    for param in (
        {"filter[cost_usd.mean][gte]": "0.5"},
        {"filter[latency_ms.p95][lte]": "1000"},
        {"filter[evaluators.harbor.verifier.mean][gte]": "0.8"},
        {"filter[run_count][gte]": "5"},
    ):
        response = client.get(EVALUATIONS, params=param)
        assert response.status_code == 503, (param, response.text)


def test_metric_filter_bad_stat_returns_400(client: TestClient) -> None:
    response = client.get(EVALUATIONS, params={"filter[cost_usd.bogus][gte]": "0.5"})
    assert response.status_code == 400, response.text


def test_metric_filter_non_numeric_value_returns_400(client: TestClient) -> None:
    response = client.get(EVALUATIONS, params={"filter[cost_usd.mean][gte]": "abc"})
    assert response.status_code == 400, response.text


def test_metric_filter_under_or_returns_400(client: TestClient) -> None:
    json_filter = '{"$or": [{"cost_usd.mean": {"$gte": 0.5}}, {"name": {"$eq": "x"}}]}'
    response = client.get(EVALUATIONS, params={"filter": json_filter})
    assert response.status_code == 400, response.text
