# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyst tools over Intake spans: ``fetch_spans``, ``get_span``, ``fetch_scores``.

Spans are the LLM calls, tool invocations, and agent steps inside a trace.
These tools are thin pass-throughs over ``client.intake.spans`` — the analyst
supplies the raw Intake filter and composes the results in Nooa CodeAct.
"""

from typing import Any

from nemo_insights_plugin.analyst.deps import AnalystDeps
from nemo_platform_plugin.intake.types import SpanMode

# Sentinel the analyst can pass as ``filter["agent_name"]`` to query spans across
# all agents instead of the run's default agent under test.
ALL_AGENTS = "__all__"


def _effective_span_filter(
    filter: dict[str, Any] | None,
    agent: str,
) -> dict[str, Any] | None:
    """Default span queries to the run's agent unless the caller opts out."""
    if filter is None:
        return {"agent_name": agent} if agent else None

    effective = dict(filter)
    agent_name = effective.get("agent_name")
    if agent_name == ALL_AGENTS:
        effective.pop("agent_name")
    elif "agent_name" not in effective and agent:
        effective["agent_name"] = agent
    return effective or None


async def fetch_spans(
    deps: AnalystDeps,
    filter: dict[str, Any] | None = None,
    group_by: str | None = None,
    sort: str | None = None,
    mode: SpanMode = "detailed",
    limit: int | None = None,
) -> dict[str, Any]:
    """List the AUT's spans from Intake, or roll them up into groups.

    One tool, two modes:

    - **Grouped** (pass ``group_by``, e.g. ``group_by="session_id"``): rolls
      the matching spans up server-side into one row per group and returns
      ``{"groups": [...], "grouped_by": str, "count": int, "total": int,
      "truncated": bool}``, where each group is
      ``{"group": {<by-field>: value, ...}, "span_count": int}``. ``total`` is
      the server's full distinct-group count. **Start here** for initial
      exploration: grouping by ``session_id`` recovers the AUT's sessions (its
      "traces") so you fan out across **many** of them in one shot — seeing 100
      sessions is far more informative than 100 spans drawn from 2 sessions.
    - **Flat** (omit ``group_by``): returns the individual spans as
      ``{"spans": [...], "count": int, "truncated": bool}``. Use this once you
      have specific sessions worth opening up — scope it with a ``session_id``
      or ``trace_id`` filter so it doesn't bunch up in a few heavy sessions.

    In both modes ``truncated`` means more matched than ``limit`` — narrow the
    filter or raise ``limit``.

    Args:
        filter: Raw Intake span filter pushed to the server. Supported keys:
            ``agent_name`` (e.g. "codex"), ``status`` ("success"/"error"/"cancelled"/"unknown"),
            ``kind`` ("LLM"/"TOOL"/"AGENT"/"CHAIN"/"EVALUATOR"/...),
            ``session_id``, ``trace_id``, ``parent_span_id`` (direct children
            of a span), ``model``, ``provider``, ``tool_name``, ``source``,
            ``project``, ``agent_id``, ``evaluation_id``, ``test_case_id``, and
            ``started_at`` (a range, e.g. ``{"gte": "2026-06-01T00:00:00"}``).
            Those are the only keys Intake serves. Every one takes a single
            exact value; ``started_at`` is the only key that takes a range, and
            no key accepts ``$in``.
            ``agent_name`` defaults to the run's agent under test when omitted
            from ``filter``; pass an explicit value to query another agent, or
            ``"__all__"`` to disable agent scoping. (There is no span-id
            filter; resolve a known span id with ``get_span``.)
        group_by: When set, the span field(s) to group by — switches to
            grouped mode. Only ``session_id`` and ``trace_id`` are groupable;
            pass one (e.g. "session_id") or both comma-separated
            ("session_id,trace_id"). Omit for a flat span list.
        sort: Sort field. Defaults to "-started_at" (newest first) for flat
            mode and "-span_count" (heaviest groups first) for grouped mode.
        mode: "summary" omits input/output; "detailed" includes everything.
            Ignored in grouped mode.
        limit: Max rows to pull (clamped to the run's ceiling). Defaults to 100
            in grouped mode and 50 in flat mode.
    """
    assert deps.backend is not None
    effective_filter = _effective_span_filter(filter, deps.agent)
    if group_by is not None:
        return await deps.backend.list_span_groups(
            workspace=deps.workspace,
            filter=effective_filter or None,
            group_by=group_by,
            sort=sort or "-span_count",
            limit=min(limit or 100, deps.max_results),
            since=deps.since,
            evaluation_id=deps.evaluation_id,
        )
    return await deps.backend.list_spans(
        workspace=deps.workspace,
        filter=effective_filter or None,
        sort=sort or "-started_at",
        mode=mode,
        limit=min(limit or 50, deps.max_results),
        since=deps.since,
        evaluation_id=deps.evaluation_id,
    )


async def get_span(deps: AnalystDeps, span_id: str) -> dict[str, Any]:
    """Fetch a single span by id (e.g. a span id cited by an annotation).

    Args:
        span_id: Intake span id.
    """
    assert deps.backend is not None
    return await deps.backend.get_span(workspace=deps.workspace, span_id=span_id)


async def fetch_scores(deps: AnalystDeps, span_id: str) -> dict[str, Any]:
    """Fetch evaluator results (scores) attached to a span.

    Evaluator results are the verifier/judge outputs for a span — each has a
    ``name``, a numeric ``value`` and/or ``string_value``, and an optional
    ``comment``. For terminal-bench/eval traces the score lives on the
    EVALUATOR span; pass that span's id (or any span you want scores for).
    Returns ``{"evaluator_results": [...], "count": int}``.

    Args:
        span_id: Intake span id to read evaluator results for.
    """
    assert deps.backend is not None
    return await deps.backend.list_scores(workspace=deps.workspace, span_id=span_id)
