# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate a run's scores per group of rows, for ``RunConfig.group_by``.

A ``group_by`` entry names what the rows are grouped by: a label output of one of the run's metrics, named
as aggregate scores are (``<metric key>.<output>``), or a field of the row (``item.<path>``). Each group's
scores come from :func:`aggregate_metrics` over the group's rows, so they are computed and named exactly as
the run's overall scores are. Corpus-level scores are not grouped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from nhx_evals_sdk.metrics.aggregation import aggregate_metrics, rubric_definitions_from_metric
from nhx_evals_sdk.metrics.protocol import Metric, MetricResult
from nhx_evals_sdk.values.protocol import Label
from nhx_evals_sdk.values.results import AggregateFieldName, AggregateGroup, RowScore

ITEM_PREFIX = "item."


@dataclass(frozen=True)
class GroupKey:
    """One resolved ``group_by`` entry: a metric's label output, or a path into the row."""

    entry: str
    metric_key: str | None = None
    output: str | None = None
    item_path: tuple[str, ...] = ()

    def value(self, row: RowScore) -> str | None:
        """The row's value for this entry, or ``None`` when it has none (the row joins no group)."""
        if self.metric_key is not None:
            for output in row.metrics.get(self.metric_key, ()):
                if output.name == self.output:
                    # A failed row's label is NaN (a float), so it belongs to no group.
                    return output.value if isinstance(output.value, str) and output.value else None
            return None
        value: object = row.item
        for part in self.item_path:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        if isinstance(value, str):
            return value or None
        if isinstance(value, bool | int | float):
            return json.dumps(value)
        return None


def resolve_group_by(entries: Sequence[str] | None, metrics: Sequence[tuple[str, Metric]]) -> list[GroupKey]:
    """Resolve ``group_by`` entries against the run's ``(metric key, metric)`` pairs, before any row runs.

    A row field can't be checked until the rows are read, so any ``item.<path>`` resolves.

    Raises:
        ValueError: An entry names no label output of the run's metrics and isn't an ``item.`` path.
    """
    if not entries:
        return []
    labels = {
        f"{metric_key}.{spec.name}": (metric_key, spec.name)
        for metric_key, metric in metrics
        for spec in metric.output_spec()
        if issubclass(spec.value_schema, Label)
    }
    keys: list[GroupKey] = []
    for entry in entries:
        if entry.startswith(ITEM_PREFIX):
            keys.append(GroupKey(entry=entry, item_path=tuple(entry.removeprefix(ITEM_PREFIX).split("."))))
        elif entry in labels:
            metric_key, output = labels[entry]
            keys.append(GroupKey(entry=entry, metric_key=metric_key, output=output))
        else:
            available = ", ".join(repr(label) for label in sorted(labels)) or "none"
            raise ValueError(
                f"group_by entry {entry!r} names no label output of this run's metrics "
                f"(label outputs: {available}); to group by a row field, write 'item.<field>'"
            )
    return keys


@dataclass(frozen=True)
class RowGroup:
    """The positions, in the run's rows, of the rows that share one value of one entry."""

    key: GroupKey
    value: str
    positions: tuple[int, ...]


def group_rows(keys: Sequence[GroupKey], row_scores: Sequence[RowScore]) -> list[RowGroup]:
    """Each entry's groups, in ``group_by`` order and then by value."""
    groups: list[RowGroup] = []
    for key in keys:
        by_value: dict[str, list[int]] = {}
        for position, row in enumerate(row_scores):
            value = key.value(row)
            if value is not None:
                by_value.setdefault(value, []).append(position)
        groups.extend(RowGroup(key, value, tuple(by_value[value])) for value in sorted(by_value))
    return groups


def aggregate_metric_groups(
    groups: Sequence[RowGroup],
    *,
    metric_key: str,
    metric: Metric,
    results: Sequence[MetricResult | None],
) -> list[AggregateGroup]:
    """One metric's aggregates for each group, its scores named ``<metric key>.<output>``.

    ``results`` line up with the run's rows. A row's NaN result counts as it does in the overall scores.
    """
    output_spec = metric.output_spec()
    rubric_definitions = rubric_definitions_from_metric(metric) or None
    aggregated_groups: list[AggregateGroup] = []
    for group in groups:
        group_results = [result for position in group.positions if (result := results[position]) is not None]
        aggregated = aggregate_metrics(group_results, output_spec, rubric_definitions=rubric_definitions)
        aggregated_groups.append(
            AggregateGroup(
                by=group.key.entry,
                value=group.value,
                row_count=len(group.positions),
                scores=[score.model_copy(update={"name": f"{metric_key}.{score.name}"}) for score in aggregated.scores],
            )
        )
    return aggregated_groups


def filter_groups(
    groups: list[AggregateGroup], aggregate_fields: tuple[AggregateFieldName, ...] | None
) -> list[AggregateGroup]:
    """``groups`` with their scores trimmed to ``aggregate_fields``, as the overall scores are."""
    if not aggregate_fields:
        return groups
    return [group.with_fields(frozenset(aggregate_fields)) for group in groups]


def merge_metric_groups(per_metric: Sequence[Sequence[AggregateGroup]]) -> list[AggregateGroup]:
    """Combine several metrics' groups, built from the same :func:`group_rows`, into one list."""
    if not per_metric:
        return []
    return [
        first.model_copy(update={"scores": [score for groups in per_metric for score in groups[index].scores]})
        for index, first in enumerate(per_metric[0])
    ]
