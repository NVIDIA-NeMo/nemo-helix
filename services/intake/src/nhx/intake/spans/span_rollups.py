# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ClickHouse expressions for span-derived token and cost rollups.

Agent and root spans repeat the usage of the model calls beneath them, so a sum over every span counts the same
tokens and cost two or three times. Instead, each trace sums a metric over the first of these tiers that reports it:

1. Inference Gateway call spans (``nemo.inference_gateway.call``): the usage the provider reported for each call
   the gateway served. When a trace has them, the agent's own spans for those same calls don't count again.
2. Model calls: non-root ``LLM`` and ``EMBEDDING`` spans, other non-root spans of unknown kind, and every ATIF
   span. ATIF ingest already reconciles step and trajectory usage, so an ATIF root only carries a metric when no
   step reported it. A root model span is not a second call: it is the publisher total in the next tier.
3. The trace's other roots.
4. Any other span.

A metric that no span in the trace reports is NULL. A stored 0 counts as reported, so it blocks the next tier.

A session adds up its traces, except that gateway call spans win across the whole session: when any of its traces
totals a metric from gateway call spans, only those traces count for that metric. An agent's uploaded trajectory
lands in a trace of its own, beside the trial trace that holds the gateway's spans for the same calls.

Queries apply the rule at three levels:

- Per span: select ``usage_sql().span_columns`` beside the span's own columns.
- Per trace: group those rows by ``source_format`` and ``trace_id``, and select ``usage_sql().trace_columns``.
- Per session: add up the trace rows with ``summed_metric_columns``, which converts cost to USD once.
"""

from dataclasses import dataclass
from typing import Any

from nhx.intake.spans.span_attribute_catalog import COST_SCALE, SpanAttributeField, spec_for_field

METRIC_ATTRIBUTE_FIELDS = {
    "input_tokens": SpanAttributeField.INPUT_TOKENS,
    "output_tokens": SpanAttributeField.OUTPUT_TOKENS,
    "cached_tokens": SpanAttributeField.CACHED_TOKENS,
    "total_tokens": SpanAttributeField.TOTAL_TOKENS,
    "cost_usd": SpanAttributeField.COST_TOTAL_USD,
    "cost_input_usd": SpanAttributeField.COST_INPUT_USD,
    "cost_output_usd": SpanAttributeField.COST_OUTPUT_USD,
}
_ALL_METRICS = tuple(METRIC_ATTRIBUTE_FIELDS)

# SQL templates. Python fills in {metric}; the ClickHouse client binds %(<metric>_key)s to its attribute name.

# Per span and metric: whether the span reports the metric, and its value, which is 0 when it does not.
_REPORTS = "has(mapKeys(attributes_number), %({metric}_key)s) AS reports_{metric}"
_VALUE = "attributes_number[%({metric}_key)s] AS {metric}_value"

# Per trace and metric: the sum over the lowest tier that reports the metric. When no span reports it, minIf
# returns 0, no branch matches, and the total is NULL. A span that does not report the metric adds 0, so each sum
# filters on the tier alone. multiIf rather than a simple CASE: ClickHouse runs it faster.
_TRACE_TOTAL = """\
multiIf(
    minIf(usage_tier, reports_{metric}) = 1, sumIf({metric}_value, usage_tier = 1),
    minIf(usage_tier, reports_{metric}) = 2, sumIf({metric}_value, usage_tier = 2),
    minIf(usage_tier, reports_{metric}) = 3, sumIf({metric}_value, usage_tier = 3),
    minIf(usage_tier, reports_{metric}) = 4, sumIf({metric}_value, usage_tier = 4),
    NULL
)"""

# Per trace and metric: whether the trace's total came from gateway call spans.
_FROM_GATEWAY = "minIf(usage_tier, reports_{metric}) = 1 AS {metric}_from_gateway"


@dataclass(frozen=True)
class UsageSql:
    """The usage rule for a set of metrics: its per-span and per-trace SELECT columns, and their parameters."""

    span_columns: str
    """``usage_tier``, then ``reports_<metric>`` and ``<metric>_value`` for each metric. They read the span's
    ``source_format``, ``kind``, ``external_parent_span_id``, ``attributes_bool`` and ``attributes_number`` without a
    table alias."""
    trace_columns: str
    """For each metric, the trace's total named after it, or NULL, then ``<metric>_from_gateway``: whether that total
    came from gateway call spans. ``summed_metric_columns`` reads both."""
    parameters: dict[str, Any]
    """Each metric's attribute name, bound as ``%(<metric>_key)s``."""


GATEWAY_CALL_ATTRIBUTE = "nemo.inference_gateway.call"

# Per span: the usage tier number. Callers alias it as ``usage_tier``.
# 1 is gateway call spans, 2 is model calls, 3 is other roots, 4 is everything else. A root LLM or EMBEDDING span
# is tier 3 so it cannot be summed with the calls beneath it. Reads ``source_format``, ``kind``,
# ``external_parent_span_id`` and ``attributes_bool``.
USAGE_TIER_EXPRESSION = f"""\
multiIf(
    attributes_bool['{GATEWAY_CALL_ATTRIBUTE}'], 1,
    source_format = 'atif'
        OR (
            external_parent_span_id != ''
            AND kind IN ('LLM', 'EMBEDDING', 'UNKNOWN')
        ), 2,
    external_parent_span_id = '', 3,
    4
)"""
_USAGE_TIER = f"{USAGE_TIER_EXPRESSION} AS usage_tier"


def usage_sql(metrics: tuple[str, ...] = _ALL_METRICS, *, scaled: bool = True) -> UsageSql:
    """The usage rule for ``metrics``.

    ``scaled=False`` leaves cost in stored units, for queries that add up traces with ``summed_metric_columns``,
    which then converts it once.
    """
    span_columns = [_USAGE_TIER]
    for metric in metrics:
        span_columns += [_REPORTS.format(metric=metric), _VALUE.format(metric=metric)]

    trace_columns = []
    for metric in metrics:
        total = _TRACE_TOTAL.format(metric=metric)
        if scaled:
            total = _scaled(total, metric)
        trace_columns += [f"{total} AS {metric}", _FROM_GATEWAY.format(metric=metric)]

    return UsageSql(
        span_columns=_select_list(span_columns, indent=16),
        trace_columns=_select_list(trace_columns, indent=12),
        parameters={f"{metric}_key": spec_for_field(METRIC_ATTRIBUTE_FIELDS[metric]).bag_key for metric in metrics},
    )


def summed_metric_columns(source_alias: str, metrics: tuple[str, ...] = _ALL_METRICS) -> str:
    """Per session: add up the unscaled trace columns of ``source_alias``.

    NULL when no trace reports a metric. Cost is converted from stored micro-dollars once, here.
    """
    totals = [f"{_scaled(session_total(source_alias, metric), metric)} AS {metric}" for metric in metrics]
    return _select_list(totals, indent=12)


def session_total(source_alias: str, metric: str) -> str:
    """Per session: one metric's unscaled total over the trace rows of ``source_alias``, or NULL.

    Only the traces whose total came from gateway call spans count, when there are any.
    """
    column = f"{source_alias}.{metric}"
    from_gateway = f"{source_alias}.{metric}_from_gateway"
    return (
        f"multiIf(countIf({from_gateway}) > 0, sumIf({column}, {from_gateway}), "
        f"countIf(isNotNull({column})) = 0, NULL, sum({column}))"
    )


def _scaled(expression: str, metric: str) -> str:
    """``expression`` in the metric's reported unit: cost is stored as integer micro-dollars (``COST_SCALE``)."""
    if spec_for_field(METRIC_ATTRIBUTE_FIELDS[metric]).scale is None:
        return expression
    return f"{expression} / {COST_SCALE}"


def _select_list(columns: list[str], *, indent: int) -> str:
    """Join SELECT ``columns`` one per line, for a query that places the first column ``indent`` spaces in."""
    return ",\n".join(columns).replace("\n", "\n" + " " * indent)
