# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-memory evaluation sort (Option A app-merge)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from nhx.intake.api.v2.experiments.endpoints import (
    _parse_session_sort_keys,
    _parse_sort_keys,
    _sort_evaluations,
    _validate_sort_field,
)
from nhx.intake.api.v2.experiments.schemas import EvaluationResponse, EvaluatorAggregate


def _exp(
    name: str,
    *,
    run_count: int = 0,
    cost_mean: float | None = None,
    evaluators: dict[str, float] | None = None,
) -> EvaluationResponse:
    return EvaluationResponse(
        id=name,
        name=name,
        workspace="default",
        experiment_ids=["grp"],
        dataset_name="ds",
        run_count=run_count,
        cost_usd=EvaluatorAggregate(mean=cost_mean) if cost_mean is not None else None,
        aggregate_scores={key: EvaluatorAggregate(mean=value) for key, value in (evaluators or {}).items()} or None,
    )


def _names(responses: list[EvaluationResponse]) -> list[str]:
    return [r.name for r in responses]


def test_sort_by_evaluator_mean_descending() -> None:
    rows = [
        _exp("a", evaluators={"reward": 0.4}),
        _exp("b", evaluators={"reward": 0.9}),
        _exp("c", evaluators={"reward": 0.6}),
    ]
    ordered = _sort_evaluations(rows, keys=[("evaluators.reward.mean", True)])
    assert _names(ordered) == ["b", "c", "a"]


def test_sort_by_cost_ascending() -> None:
    rows = [_exp("a", cost_mean=2.0), _exp("b", cost_mean=0.5), _exp("c", cost_mean=1.0)]
    ordered = _sort_evaluations(rows, keys=[("cost_usd.mean", False)])
    assert _names(ordered) == ["b", "c", "a"]


def test_sort_by_run_count() -> None:
    rows = [_exp("a", run_count=3), _exp("b", run_count=10), _exp("c", run_count=1)]
    assert _names(_sort_evaluations(rows, keys=[("run_count", True)])) == ["b", "a", "c"]


def test_evaluator_name_with_dots_resolves() -> None:
    # "harbor.verifier" contains a dot; the stat is the last segment.
    rows = [_exp("a", evaluators={"harbor.verifier": 0.2}), _exp("b", evaluators={"harbor.verifier": 0.8})]
    ordered = _sort_evaluations(rows, keys=[("evaluators.harbor.verifier.mean", True)])
    assert _names(ordered) == ["b", "a"]


def test_missing_metric_sorts_last_in_both_directions() -> None:
    rows = [_exp("scored", cost_mean=1.0), _exp("unscored")]  # unscored has no cost
    assert _names(_sort_evaluations(rows, keys=[("cost_usd.mean", True)])) == ["scored", "unscored"]
    assert _names(_sort_evaluations(rows, keys=[("cost_usd.mean", False)])) == ["scored", "unscored"]


def test_ties_broken_by_name() -> None:
    rows = [_exp("c", cost_mean=1.0), _exp("a", cost_mean=1.0), _exp("b", cost_mean=1.0)]
    # Equal values -> deterministic ascending-name order, regardless of sort direction.
    assert _names(_sort_evaluations(rows, keys=[("cost_usd.mean", True)])) == ["a", "b", "c"]


def test_entity_field_sort() -> None:
    rows = [_exp("b"), _exp("a"), _exp("c")]
    assert _names(_sort_evaluations(rows, keys=[("name", False)])) == ["a", "b", "c"]


def test_validate_accepts_entity_and_metric_fields() -> None:
    for field in (
        "name",
        "created_at",
        "pinned_at",
        "run_count",
        "cost_usd.mean",
        "latency_ms.p95",
        "evaluators.harbor.verifier.mean",
    ):
        _validate_sort_field(field)  # no raise


def test_validate_rejects_unknown_field() -> None:
    for field in ("bogus", "cost_usd.nope", "evaluators.reward", "evaluators..mean"):
        with pytest.raises(HTTPException) as exc:
            _validate_sort_field(field)
        assert exc.value.status_code == 400


def test_parse_sort_keys_single_field() -> None:
    keys, metric = _parse_sort_keys("-cost_usd.mean")
    assert keys == [("cost_usd.mean", True)]
    assert metric is True


def test_parse_sort_keys_multi_field_preserves_order_and_per_field_direction() -> None:
    keys, metric = _parse_sort_keys("-evaluators.reward.mean,cost_usd.mean")
    assert keys == [("evaluators.reward.mean", True), ("cost_usd.mean", False)]
    assert metric is True


def test_parse_sort_keys_entity_only_is_not_metric_backed() -> None:
    keys, metric = _parse_sort_keys("name,-created_at")
    assert keys == [("name", False), ("created_at", True)]
    assert metric is False


def test_parse_sort_keys_metric_flag_true_if_any_key_is_a_metric() -> None:
    _, metric = _parse_sort_keys("name,-cost_usd.mean")
    assert metric is True


def test_parse_sort_keys_tolerates_whitespace() -> None:
    keys, _ = _parse_sort_keys(" name , -cost_usd.mean ")
    assert keys == [("name", False), ("cost_usd.mean", True)]


def test_parse_sort_keys_rejects_unknown_field_in_any_position() -> None:
    with pytest.raises(HTTPException) as exc:
        _parse_sort_keys("name,bogus.field")
    assert exc.value.status_code == 400


def test_parse_sort_keys_rejects_empty() -> None:
    for value in ("", "  ", ",", " , "):
        with pytest.raises(HTTPException) as exc:
            _parse_sort_keys(value)
        assert exc.value.status_code == 400


def test_parse_session_sort_uses_test_case_name_and_accepts_deprecated_alias() -> None:
    assert _parse_session_sort_keys("-test_case_name") == [("test_case_name", True)]
    assert _parse_session_sort_keys("-test_case_id") == [("test_case_name", True)]


def test_multi_field_sort_ranks_by_first_key_then_tiebreak() -> None:
    # Switchyard's default ranking: reward desc, then cost asc as the tiebreak.
    rows = [
        _exp("a", evaluators={"reward": 0.9}, cost_mean=2.0),
        _exp("b", evaluators={"reward": 0.9}, cost_mean=1.0),
        _exp("c", evaluators={"reward": 0.5}, cost_mean=0.1),
    ]
    keys, _ = _parse_sort_keys("-evaluators.reward.mean,cost_usd.mean")
    ordered = _sort_evaluations(rows, keys=keys)
    # a and b tie on reward (0.9) -> cheaper (b) wins the tiebreak; c has lower reward -> last.
    assert _names(ordered) == ["b", "a", "c"]
