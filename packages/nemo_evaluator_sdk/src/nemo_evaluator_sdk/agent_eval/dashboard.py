# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small HTML dashboard for standalone agent-eval result bundles."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalTaskScore
from nemo_evaluator_sdk.values.results import AggregateScore
from pydantic import BaseModel


def write_dashboard(result: AgentEvalResult, output_path: str | Path) -> Path:
    """Write an HTML dashboard and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(result), encoding="utf-8")
    return path


def render_dashboard(result: AgentEvalResult) -> str:
    """Render a compact generic report for metric outputs."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Eval Report</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f7f8fa; --fg:#172033; --muted:#64748b; --line:#d8dee9; --panel:#fff; --soft:#f8fafc; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --fg:#f8fafc; --muted:#a4aebc; --line:#374151; --panel:#1f2937; --soft:#111827; }} }}
    * {{ box-sizing: border-box; }}
    body {{ background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; }}
    header {{ background: var(--panel); border-bottom: 1px solid var(--line); padding: 28px 32px 20px; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 24px 32px 48px; }}
    h1 {{ margin: 0 0 4px; }}
    h2 {{ margin-top: 28px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 160px; }}
    .card strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ background: var(--panel); border: 1px solid var(--line); border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: var(--soft); color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    code {{ background: var(--soft); border-radius: 4px; padding: 1px 4px; }}
    pre {{ background: var(--soft); border-radius: 6px; max-height: 320px; overflow: auto; padding: 10px; white-space: pre-wrap; }}
    .output-list {{ display: grid; gap: 8px; }}
    .output-row strong {{ display: block; margin-bottom: 4px; }}
  </style>
</head>
<body>
<header>
  <h1>Agent Eval Report</h1>
  <div class="muted">Run {_e(result.run_id)} · {_e(result.summary.task_count)} tasks · {_e(result.summary.trial_count)} trials</div>
</header>
<main>
  <div class="cards">
    <div class="card"><span class="muted">Tasks</span><strong>{_e(result.summary.task_count)}</strong></div>
    <div class="card"><span class="muted">Trials</span><strong>{_e(result.summary.trial_count)}</strong></div>
    <div class="card"><span class="muted">Metric Scores</span><strong>{_e(result.summary.score_count)}</strong></div>
  </div>
  <h2>Metric Rollups</h2>
  {_metric_rollups(result)}
  <h2>Scores</h2>
  {_score_table(result.scores)}
</main>
</body>
</html>
"""


def _metric_rollups(result: AgentEvalResult) -> str:
    aggregated = result.summary.scores.scores
    if not aggregated:
        return '<p class="muted">No numeric metric outputs to summarize.</p>'
    rows: list[str] = []
    for score in sorted(aggregated, key=lambda item: item.name):
        rows.append(
            "<tr>"
            f"<td><code>{_e(score.name)}</code></td>"
            f"<td>{_format_score(score.headline_value)}</td>"
            f"<td>{_format_score(_median(score))}</td>"
            f"<td>{_format_score(score.sample_std_dev)}</td>"
            f"<td>{_count(score.count)}</td>"
            f"<td>{_e(score.nan_count)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Name</th><th>Value</th><th>Median</th><th>Std dev</th>"
        "<th>Count</th><th>NaN</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _median(score: AggregateScore) -> float | None:
    """The median, whether it arrived as a field or only inside a percentile distribution.

    Reading `percentiles.p50` alone would blank the column for every imported aggregate: a backend that
    reports a median without a full distribution (Gym does) sets `median` and nothing else, which is the
    case the field was added for. Natively computed scores populate both, identically.
    """
    if score.median is not None:
        return score.median
    percentiles = getattr(score, "percentiles", None)
    return percentiles.p50 if percentiles is not None else None


def _count(count: int | None) -> str:
    """Sample size, or an em dash when the producer didn't report one (imported aggregates).

    Tests for None specifically: a genuine 0 means every sample was NaN, which is worth seeing.
    """
    return "&mdash;" if count is None else _e(count)


def _score_table(scores: list[AgentEvalTaskScore]) -> str:
    if not scores:
        return '<p class="muted">No metric scores.</p>'
    rows = [
        "<tr>"
        f"<td><code>{_e(score.task_id)}</code></td>"
        f"<td><code>{_e(score.trial_id)}</code></td>"
        f"<td><code>{_e(score.metric_type)}</code></td>"
        f"<td>{_outputs(score)}</td>"
        "</tr>"
        for score in scores
    ]
    return (
        "<table><thead><tr><th>Task</th><th>Trial</th><th>Metric</th><th>Outputs</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _outputs(score: AgentEvalTaskScore) -> str:
    chunks = []
    for output in score.outputs:
        chunks.append(
            f'<div class="output-row"><strong>{_e(output.name)}</strong><pre>{_e(_jsonish(output.value))}</pre></div>'
        )
    return '<div class="output-list">' + "".join(chunks) + "</div>"


def _jsonish(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        # ValueError covers circular references; fall back to a plain string rather than crash rendering.
        return str(value)


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)
