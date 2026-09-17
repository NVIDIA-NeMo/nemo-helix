# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic candidate-vs-baseline gate over a run (example-local CI policy).

This lives in the example, not the SDK: turning scores into a pass/fail decision
is a CI policy concern, while the SDK stays responsible for generating trials and
running scorers/metrics. Adds the pass-rate/token/runtime-tie-breaker gate on top
of the persisted run bundle. Note ``pass_rate`` here is a per-task pass/fail count
against a reward threshold — deliberately different from
:class:`~nemo_evaluator_sdk.agent_eval.results.AgentEvalSummary`'s mean-per-output.
Token/runtime are read via
:class:`~nemo_evaluator_sdk.agent_eval.trials.TrialMeasurements`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.scores import AgentEvalScoreStatus, AgentEvalTaskScore
from nemo_evaluator_sdk.agent_eval.trials import TrialMeasurements
from pydantic import BaseModel

# Metric outputs, in priority order, that represent a task's pass/reward signal.
DEFAULT_REWARD_OUTPUTS: tuple[str, ...] = ("verifier_reward", "agent_phase_success")


@dataclass(frozen=True)
class GateThresholds:
    """Knobs controlling the candidate gate (defaults are the strict CI policy)."""

    min_pass_rate: float = 1.0
    require_token_metrics: bool = False
    max_pass_rate_drop: float = 0.0
    max_token_regression_pct: float = 0.0
    max_runtime_regression_pct: float = 0.0


@dataclass
class GateCheck:
    name: str
    passed: bool
    details: str


@dataclass
class GateReport:
    gate_passed: bool
    summary: dict[str, Any]
    checks: list[GateCheck] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "gate_passed": self.gate_passed,
            "summary": self.summary,
            "checks": [asdict(check) for check in self.checks],
        }


def evaluate_gate(
    result: AgentEvalResult,
    *,
    thresholds: GateThresholds | None = None,
    baseline_summary: dict[str, Any] | None = None,
    reward_outputs: tuple[str, ...] = DEFAULT_REWARD_OUTPUTS,
) -> GateReport:
    """Summarize a run and apply gate checks, optionally against a baseline."""
    thresholds = thresholds or GateThresholds()
    summary = summarize_run(result, reward_outputs=reward_outputs)
    checks = run_gate_checks(summary, thresholds=thresholds, baseline_summary=baseline_summary)
    return GateReport(gate_passed=all(check.passed for check in checks), summary=summary, checks=checks)


def write_gate_report(report: GateReport, output_dir: str | Path, *, filename: str = "gate.json") -> Path:
    """Persist the gate report alongside the run bundle."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    gate_path = path / filename
    gate_path.write_text(json.dumps(report.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate_path


def load_baseline_summary(path: str | Path) -> dict[str, Any]:
    """Load + normalize a baseline summary (raw summary or a prior gate.json)."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Baseline summary must be a JSON object: {source}")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    _validate_baseline_summary(summary, source)
    return summary


def summarize_run(
    result: AgentEvalResult,
    *,
    reward_outputs: tuple[str, ...] = DEFAULT_REWARD_OUTPUTS,
) -> dict[str, Any]:
    """Aggregate pass-rate, token, and runtime for one run.

    Token/runtime are read via :class:`TrialMeasurements`; pass-rate uses only
    completed canonical score outputs. Tasks without one are reported as unscored.
    """
    trials_by_task = {trial.task_id: trial for trial in result.trials}
    reward_by_task = _rewards_by_task(result.scores, reward_outputs)
    task_ids = sorted({task.id for task in result.tasks} | set(trials_by_task))

    passed = 0
    unscored_task_ids: list[str] = []
    token_sum = 0
    token_count = 0
    token_unavailable: list[str] = []
    runtime_sum = 0.0
    runtime_count = 0
    runtime_unavailable: list[str] = []

    for task_id in task_ids:
        trial = trials_by_task.get(task_id)
        measurements = trial.measurements if trial is not None else TrialMeasurements()

        reward_value = reward_by_task.get(task_id)
        if reward_value is None:
            unscored_task_ids.append(task_id)
        elif reward_value >= 1.0:
            passed += 1

        if measurements.total_tokens is not None:
            token_sum += measurements.total_tokens
            token_count += 1
        else:
            token_unavailable.append(task_id)

        if measurements.runtime_sec is not None:
            runtime_sum += measurements.runtime_sec
            runtime_count += 1
        else:
            runtime_unavailable.append(task_id)

    total = len(task_ids)
    return {
        "run_id": result.run_id,
        "labels": result.metadata.labels,
        "total_tasks": total,
        "passed_tasks": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "unscored_task_ids": sorted(unscored_task_ids),
        "task_names": task_ids,
        "total_tokens_sum": token_sum if token_count else None,
        "avg_total_tokens": (token_sum / token_count) if token_count else None,
        "token_metrics_coverage": (token_count / total) if total else 0.0,
        "token_metrics_available_tasks": token_count,
        "token_metrics_unavailable_tasks": sorted(token_unavailable),
        "runtime_sec_sum": runtime_sum if runtime_count else None,
        "avg_runtime_sec": (runtime_sum / runtime_count) if runtime_count else None,
        "runtime_metrics_coverage": (runtime_count / total) if total else 0.0,
        "runtime_metrics_available_tasks": runtime_count,
        "runtime_metrics_unavailable_tasks": sorted(runtime_unavailable),
    }


def run_gate_checks(
    summary: dict[str, Any],
    *,
    thresholds: GateThresholds,
    baseline_summary: dict[str, Any] | None = None,
) -> list[GateCheck]:
    """Apply absolute + relative (vs baseline) gate checks to a summary."""
    checks: list[GateCheck] = []
    total_tasks = int(summary["total_tasks"])
    pass_rate = float(summary["pass_rate"])

    checks.append(GateCheck("non_empty_result_set", total_tasks > 0, f"total_tasks={total_tasks}"))
    checks.append(
        GateCheck(
            "min_pass_rate",
            pass_rate >= thresholds.min_pass_rate,
            f"pass_rate={pass_rate:.3f}, min_pass_rate={thresholds.min_pass_rate:.3f}",
        )
    )

    if thresholds.require_token_metrics:
        token_coverage = float(summary["token_metrics_coverage"])
        runtime_coverage = float(summary["runtime_metrics_coverage"])
        checks.append(
            GateCheck(
                "token_metrics_available_for_all_tasks",
                token_coverage == 1.0,
                f"token_metrics_coverage={token_coverage:.3f}",
            )
        )
        checks.append(
            GateCheck(
                "runtime_metrics_available_for_all_tasks",
                runtime_coverage == 1.0,
                f"runtime_metrics_coverage={runtime_coverage:.3f}",
            )
        )

    if baseline_summary is not None:
        checks.extend(_baseline_checks(summary, baseline_summary, thresholds))

    return checks


def _baseline_checks(
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    thresholds: GateThresholds,
) -> list[GateCheck]:
    checks: list[GateCheck] = []
    pass_rate = float(summary["pass_rate"])
    total_tokens_sum = summary["total_tokens_sum"]
    runtime_sec_sum = summary["runtime_sec_sum"]

    # Regression checks only make sense when both runs measured the same tasks.
    baseline_tasks = baseline_summary.get("task_names")
    candidate_tasks = summary.get("task_names")
    task_sets_comparable = True
    if isinstance(baseline_tasks, list) and isinstance(candidate_tasks, list):
        comparable = sorted(baseline_tasks) == sorted(candidate_tasks)
        task_sets_comparable = comparable
        checks.append(
            GateCheck(
                "baseline_candidate_task_sets_match",
                comparable,
                (
                    f"both runs measured {len(candidate_tasks)} tasks"
                    if comparable
                    else f"baseline={sorted(baseline_tasks)} candidate={sorted(candidate_tasks)}; "
                    "regression checks short-circuited"
                ),
            )
        )
    else:
        checks.append(
            GateCheck(
                "baseline_candidate_task_sets_match",
                True,
                "task_names not present on baseline and/or candidate; skipping equality guard",
            )
        )

    if not task_sets_comparable:
        return checks

    baseline_pass_rate = float(baseline_summary.get("pass_rate", 0.0))
    checks.append(
        GateCheck(
            "no_pass_rate_regression_vs_baseline",
            pass_rate >= baseline_pass_rate - thresholds.max_pass_rate_drop,
            f"pass_rate={pass_rate:.3f}, baseline={baseline_pass_rate:.3f}, max_drop={thresholds.max_pass_rate_drop:.3f}",
        )
    )

    baseline_tokens = baseline_summary.get("total_tokens_sum")
    if isinstance(total_tokens_sum, int) and isinstance(baseline_tokens, int):
        max_allowed = baseline_tokens * (1.0 + thresholds.max_token_regression_pct / 100.0)
        checks.append(
            GateCheck(
                "tokens_not_worse_than_baseline",
                total_tokens_sum <= max_allowed,
                f"total_tokens_sum={total_tokens_sum}, baseline={baseline_tokens}, "
                f"max_regression_pct={thresholds.max_token_regression_pct:.2f}",
            )
        )
    else:
        checks.append(
            GateCheck(
                "tokens_not_worse_than_baseline",
                False,
                "Missing token totals for candidate or baseline; cannot run deterministic token comparison.",
            )
        )

    # Runtime is only a tie-breaker when token totals match exactly.
    baseline_runtime = baseline_summary.get("runtime_sec_sum")
    tokens_tied = (
        isinstance(total_tokens_sum, int) and isinstance(baseline_tokens, int) and total_tokens_sum == baseline_tokens
    )
    if not tokens_tied:
        checks.append(
            GateCheck(
                "runtime_tie_breaker_not_worse_than_baseline",
                True,
                "Not applicable (token totals differ from baseline).",
            )
        )
    elif isinstance(runtime_sec_sum, int | float) and isinstance(baseline_runtime, int | float):
        max_allowed_runtime = float(baseline_runtime) * (1.0 + thresholds.max_runtime_regression_pct / 100.0)
        checks.append(
            GateCheck(
                "runtime_tie_breaker_not_worse_than_baseline",
                float(runtime_sec_sum) <= max_allowed_runtime,
                f"runtime_sec_sum={float(runtime_sec_sum):.3f}, baseline={float(baseline_runtime):.3f}, "
                f"max_regression_pct={thresholds.max_runtime_regression_pct:.2f}",
            )
        )
    else:
        checks.append(
            GateCheck(
                "runtime_tie_breaker_not_worse_than_baseline",
                False,
                "Token totals tied with baseline but runtime totals missing; cannot run tie-breaker.",
            )
        )

    return checks


def _rewards_by_task(scores: list[AgentEvalTaskScore], reward_outputs: tuple[str, ...]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for score in scores:
        if score.status == AgentEvalScoreStatus.FAILED:
            continue
        for output_name in reward_outputs:
            value = _numeric_output(score, output_name)
            if value is not None:
                # Highest-priority output wins; don't overwrite with later metrics.
                rewards.setdefault(score.task_id, value)
                break
    return rewards


def _numeric_output(score: AgentEvalTaskScore, name: str) -> float | None:
    for output in score.outputs:
        if output.name == name:
            return _reward_value(output.value)
    return None


def _reward_value(value: Any) -> float | None:
    # A boolean reward signal (e.g. agent_phase_success) maps to 1.0/0.0.
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, BaseModel):
        root = getattr(value, "root", None)
        if isinstance(root, bool):
            return 1.0 if root else 0.0
        if isinstance(root, int | float):
            return float(root)
    return None


def _validate_baseline_summary(summary: dict[str, Any], source: Path) -> None:
    missing = [key for key in ("pass_rate", "total_tokens_sum", "runtime_sec_sum") if key not in summary]
    if missing:
        raise ValueError(
            f"Baseline summary {source} is missing required key(s): {', '.join(missing)}. "
            "Expected a raw summary object or a gate.json with a `summary`."
        )
    if not isinstance(summary.get("pass_rate"), int | float):
        raise ValueError(f"Baseline summary {source} has invalid `pass_rate`; expected a number.")
