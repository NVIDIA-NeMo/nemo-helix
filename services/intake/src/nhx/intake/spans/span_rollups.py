# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ClickHouse expressions for span-derived token and cost rollups.

Agent and root spans repeat the usage of the model calls beneath them, so a sum over every span counts the same
tokens and cost two or three times. Instead, each trace sums a metric over the first of these tiers that reports it:

1. Inference Gateway call spans (``GATEWAY_CALL_ATTRIBUTE``): the usage the provider reported for each call the
   gateway served. An agent that sends a ``traceparent`` gets them beneath its own spans for the same calls, which
   then don't count again. They win for a metric only when every gateway call span in the trace recorded usage and
   one reports the metric; otherwise the trace counts the agent's own spans and none of the gateway's, so a call
   whose response carried no usage is neither dropped nor counted twice.
2. Model calls: non-root ``LLM`` and ``EMBEDDING`` spans, other non-root spans of unknown kind, and every ATIF
   span. ATIF ingest already reconciles step and trajectory usage, so an ATIF root only carries a metric when no
   step reported it. A root model span is not a second call: it is the publisher total in the next tier.
3. The trace's other roots.
4. Any other span.

A metric that no span in the trace reports is NULL. A stored 0 counts as reported, so it blocks the next tier.

A session adds up its traces, except for a trace marked ``USAGE_SUPERSEDED_ATTRIBUTE``: the gateway recorded its calls
in another trace of the session, so once any trace totals a metric from gateway call spans, marked traces don't add
that metric again. The marked trace keeps its own totals, and a metric the gateway doesn't report still counts from it.
Queries apply the rule at three levels:

- Per span: select ``usage_sql().span_columns`` beside the span's own columns.
- Per trace: group those rows by ``source_format`` and ``trace_id``, and select ``usage_sql().trace_columns``.
- Per session: add up the trace rows with ``summed_metric_columns``, which converts cost to USD once.
"""

from dataclasses import dataclass
from typing import Any

from nemo_helix_plugin.intake.span_attributes import GATEWAY_CALL_ATTRIBUTE, USAGE_SUPERSEDED_ATTRIBUTE
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
    {from_gateway}, sumIf({metric}_value, usage_tier = 1),
    minIf(usage_tier, reports_{metric} AND usage_tier > 1) = 2, sumIf({metric}_value, usage_tier = 2),
    minIf(usage_tier, reports_{metric} AND usage_tier > 1) = 3, sumIf({metric}_value, usage_tier = 3),
    minIf(usage_tier, reports_{metric} AND usage_tier > 1) = 4, sumIf({metric}_value, usage_tier = 4),
    NULL
)"""
# Per trace and metric: whether its total came from gateway call spans; and per trace, whether it is marked.
_GATEWAY_COVERS = (
    "(countIf(usage_tier = 1 AND reports_{metric}) > 0 AND countIf(usage_tier = 1 AND NOT reports_usage) = 0)"
)
# Per span: whether it reports any token usage at all. Reads the input and output token keys.
_REPORTS_USAGE = (
    "(has(mapKeys(attributes_number), %(input_tokens_key)s) OR has(mapKeys(attributes_number), %(output_tokens_key)s))"
    " AS reports_usage"
)
_FROM_GATEWAY = "{from_gateway} AS {metric}_from_gateway"
_SUPERSEDED = "max(usage_superseded) AS usage_superseded"


@dataclass(frozen=True)
class UsageSql:
    """The usage rule for a set of metrics: its per-span and per-trace SELECT columns, and their parameters."""

    span_columns: str
    """``usage_tier``, ``usage_superseded`` and ``reports_usage``, then ``reports_<metric>`` and ``<metric>_value`` for each metric.
    They read the span's ``source_format``, ``kind``, ``external_parent_span_id``, ``attributes_bool`` and
    ``attributes_number`` without a table alias."""
    trace_columns: str
    """For each metric, the trace's total named after it, or NULL, and ``<metric>_from_gateway``: whether that total
    came from gateway call spans. Then ``usage_superseded``: whether the trace is marked. ``summed_metric_columns``
    reads them all."""
    parameters: dict[str, Any]
    """Each metric's attribute name, bound as ``%(<metric>_key)s``."""


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
# Per span: whether it marks its trace's usage as superseded by the gateway. Callers alias it as ``usage_superseded``.
USAGE_SUPERSEDED_EXPRESSION = f"attributes_bool['{USAGE_SUPERSEDED_ATTRIBUTE}']"
_USAGE_SUPERSEDED = f"{USAGE_SUPERSEDED_EXPRESSION} AS usage_superseded"


def usage_sql(metrics: tuple[str, ...] = _ALL_METRICS, *, scaled: bool = True) -> UsageSql:
    """The usage rule for ``metrics``.

    ``scaled=False`` leaves cost in stored units, for queries that add up traces with ``summed_metric_columns``,
    which then converts it once.
    """
    span_columns = [_USAGE_TIER, _USAGE_SUPERSEDED, _REPORTS_USAGE]
    for metric in metrics:
        span_columns += [_REPORTS.format(metric=metric), _VALUE.format(metric=metric)]

    trace_columns = []
    for metric in metrics:
        from_gateway = _GATEWAY_COVERS.format(metric=metric)
        total = _TRACE_TOTAL.format(metric=metric, from_gateway=from_gateway)
        if scaled:
            total = _scaled(total, metric)
        trace_columns += [f"{total} AS {metric}", _FROM_GATEWAY.format(metric=metric, from_gateway=from_gateway)]
    trace_columns.append(_SUPERSEDED)

    return UsageSql(
        span_columns=_select_list(span_columns, indent=16),
        trace_columns=_select_list(trace_columns, indent=12),
        parameters={
            f"{metric}_key": spec_for_field(METRIC_ATTRIBUTE_FIELDS[metric]).bag_key
            for metric in {*metrics, "input_tokens", "output_tokens"}
        },
    )


def summed_metric_columns(source_alias: str, metrics: tuple[str, ...] = _ALL_METRICS) -> str:
    """Per session: add up the unscaled trace columns of ``source_alias``.

    NULL when no trace reports a metric. Cost is converted from stored micro-dollars once, here.
    """
    totals = [f"{_scaled(session_total(source_alias, metric), metric)} AS {metric}" for metric in metrics]
    return _select_list(totals, indent=12)


def session_total(source_alias: str, metric: str) -> str:
    """Per session: one metric's unscaled total over the trace rows of ``source_alias``, or NULL.

    Marked traces don't count once any trace totals the metric from gateway call spans.
    """
    column = f"{source_alias}.{metric}"
    counted = f"{source_alias}.{metric}_from_gateway OR NOT {source_alias}.usage_superseded"
    return (
        f"multiIf(countIf({source_alias}.{metric}_from_gateway) > 0, sumIf({column}, {counted}), "
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
