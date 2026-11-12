# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``RunConfig.group_by``: aggregate scores per group of rows."""

from __future__ import annotations

import json

import pytest
from nhx_evals_sdk.execution.benchmark_execution import evaluate_benchmark
from nhx_evals_sdk.execution.grouping import GroupKey, resolve_group_by
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nhx_evals_sdk.values.params import RunConfig
from nhx_evals_sdk.values.results import RowScore
from pydantic import ValidationError


class _LabeledMetric:
    """Scores each row from its item: ``score`` from ``item["score"]``, the ``task`` label from ``item["task"]``.

    ``bonus`` is optional and emitted only when the item has one, to check that optional outputs are
    aggregated over the rows that have them, per group as overall.
    """

    def __init__(self, name: str = "m") -> None:
        self._name = name
        self.calls = 0

    @property
    def type(self) -> str:
        return self._name

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.continuous_score("score"),
            MetricOutputSpec.boolean("passed"),
            MetricOutputSpec.continuous_score("bonus", required=False),
            MetricOutputSpec.label("task"),
        ]

    def metric(self, item: dict, sample: dict, trace=None) -> float:
        del sample, trace
        return float(item["score"])

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        self.calls += 1
        item = input.row.data
        outputs = [
            MetricOutput(name="score", value=float(item["score"])),
            MetricOutput(name="passed", value=item["score"] >= 0.5),
            MetricOutput(name="task", value=item["task"]),
        ]
        if "bonus" in item:
            outputs.append(MetricOutput(name="bonus", value=float(item["bonus"])))
        return MetricResult(outputs=outputs)


class _ConstantMetric:
    """A second metric with no labels of its own, for grouping one metric by another's label."""

    @property
    def type(self) -> str:
        return "c"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("value")]

    def metric(self, item: dict, sample: dict, trace=None) -> float:
        del sample, trace
        return float(item["score"]) * 10

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        return MetricResult(outputs=[MetricOutput(name="value", value=float(input.row.data["score"]) * 10)])


ROWS = [
    {"task": "span", "score": 1.0, "bonus": 2.0, "meta": {"level": 1}},
    {"task": "count", "score": 0.0, "meta": {"level": 2}},
    {"task": "span", "score": 0.5, "meta": {"level": 2}},
    {"task": "arithmetic", "score": 0.25, "bonus": 1.0, "meta": {"level": 1}},
    {"task": "span", "score": 0.0, "meta": {}},
]


async def _run(rows: list[dict], *, group_by: list[str] | None = None, metrics=None, aggregate_fields=None):
    return await evaluate_benchmark(
        metrics=metrics or [("m", _LabeledMetric())],
        rows=rows,
        target=None,
        params=RunConfig(parallelism=2, group_by=group_by),
        aggregate_fields=aggregate_fields,
    )


def _dump(scores) -> list[dict]:
    return [score.model_dump(mode="json") for score in scores]


async def test_each_label_group_equals_a_run_over_its_rows() -> None:
    result = await _run(ROWS, group_by=["m.task"])

    groups = result.aggregate_scores.groups
    assert groups is not None
    assert [(group.by, group.value, group.row_count) for group in groups] == [
        ("m.task", "arithmetic", 1),
        ("m.task", "count", 1),
        ("m.task", "span", 3),
    ]
    for group in groups:
        subset = await _run([row for row in ROWS if row["task"] == group.value])
        assert _dump(group.scores) == _dump(subset.aggregate_scores.scores)
    # The overall scores are those of a run without group_by.
    assert _dump(result.aggregate_scores.scores) == _dump((await _run(ROWS)).aggregate_scores.scores)


async def test_groups_by_a_nested_row_field_and_skips_rows_without_it() -> None:
    result = await _run(ROWS, group_by=["item.meta.level"])

    groups = result.aggregate_scores.groups
    assert groups is not None
    # The last row has no level, so it joins no group; numbers become their JSON text.
    assert [(group.value, group.row_count) for group in groups] == [("1", 2), ("2", 2)]
    span_and_arithmetic = await _run([ROWS[0], ROWS[3]])
    assert _dump(groups[0].scores) == _dump(span_and_arithmetic.aggregate_scores.scores)


async def test_several_entries_list_their_groups_in_group_by_order() -> None:
    result = await _run(ROWS, group_by=["item.meta.level", "m.task"])

    groups = result.aggregate_scores.groups
    assert groups is not None
    assert [group.by for group in groups] == ["item.meta.level"] * 2 + ["m.task"] * 3


async def test_without_group_by_the_result_serializes_as_before() -> None:
    result = await _run(ROWS)

    assert result.aggregate_scores.groups is None
    assert "groups" not in json.loads(result.aggregate_scores.model_dump_json())
    assert "groups" not in result.per_metric["m"].aggregate_scores.model_dump(mode="json")


async def test_a_label_from_one_metric_groups_every_metric() -> None:
    result = await _run(ROWS, group_by=["m.task"], metrics=[("m", _LabeledMetric()), ("c", _ConstantMetric())])

    groups = result.aggregate_scores.groups
    assert groups is not None
    span = next(group for group in groups if group.value == "span")
    assert [score.name for score in span.scores] == ["m.score", "m.passed", "m.bonus", "c.value"]
    assert span.scores[-1].mean == pytest.approx((10.0 + 5.0 + 0.0) / 3)
    # Each metric's own result carries its own groups.
    per_metric_groups = result.per_metric["c"].aggregate_scores.groups
    assert per_metric_groups is not None
    assert [score.name for group in per_metric_groups for score in group.scores] == ["c.value"] * 3


async def test_aggregate_fields_trim_group_scores_as_they_trim_the_overall_scores() -> None:
    result = await _run(ROWS, group_by=["m.task"], aggregate_fields=("mean",))

    dumped = json.loads(result.aggregate_scores.model_dump_json())
    assert set(dumped["scores"][0]) == set(dumped["groups"][0]["scores"][0]) == {"name", "count", "mean"}


async def test_an_entry_that_names_no_label_fails_before_any_row_is_scored() -> None:
    metric = _LabeledMetric()

    with pytest.raises(ValueError, match=r"'m\.score' names no label output.*'m\.task'"):
        await _run(ROWS, group_by=["m.score"], metrics=[("m", metric)])
    assert metric.calls == 0


def test_metric_keys_with_dots_resolve_by_their_whole_key() -> None:
    keys = resolve_group_by(["custom.ref.task"], [("custom.ref", _LabeledMetric())])

    assert keys == [GroupKey(entry="custom.ref.task", metric_key="custom.ref", output="task")]


def test_a_nan_label_joins_no_group() -> None:
    row = RowScore(
        row_index=0,
        item={},
        sample={},
        metrics={"m": [MetricOutput(name="task", value=float("nan"))]},
        requests=[],
    )

    assert GroupKey(entry="m.task", metric_key="m", output="task").value(row) is None


@pytest.mark.parametrize(
    ("group_by", "message"),
    [
        (["m.task", "m.task"], "unique"),
        ([" "], "must not be empty"),
        (["item."], "needs a field path"),
        (["item.meta..level"], "needs a field path"),
    ],
)
def test_run_config_rejects_malformed_entries(group_by: list[str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RunConfig(group_by=group_by)
