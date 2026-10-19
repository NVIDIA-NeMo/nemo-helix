# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run agent-eval tasks through a workflow runtime, using the trials-based SDK.

It drives the example-local ``AgentEvalPipeline`` two ways:

* **online** — generate trials by running :class:`WorkflowAgentRuntime`
  (the agent), score them, and apply the deterministic gate; or
* **offline** — re-score the ``trials.jsonl`` of a prior run with no agent
  execution (``--rescore-dir``).

Run it as a module from the repository root::

    python -m packages.nhx_evals_sdk.examples.run_agent_eval.run_agent_eval --task all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

if __package__ in {None, ""}:
    raise SystemExit(
        "Run this example as a module from the repository root:\n"
        "  python -m packages.nhx_evals_sdk.examples.run_agent_eval.run_agent_eval --task all"
    )

from nhx_evals_sdk.agent_eval.results import AgentEvalResult

from .gating import GateThresholds
from .pipeline import AgentEvalPipeline, PipelineConfig
from .workflow_runtime import (
    WorkflowAgentRuntime,
    WorkflowRuntimeConfig,
    example_tasks,
    load_stored_trials,
    tasks_by_id,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "run-agent-eval-output"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _pipeline(min_pass_rate: float) -> AgentEvalPipeline:
    return AgentEvalPipeline(
        config=PipelineConfig(
            parallelism=2,
            write_dashboard=True,
            write_gate=True,
            gate_thresholds=GateThresholds(min_pass_rate=min_pass_rate),
        ),
    )


async def run_online(task_names: list[str], *, output_dir: Path, min_pass_rate: float) -> AgentEvalResult:
    tasks = tasks_by_id(task_names)
    runtime = WorkflowAgentRuntime(WorkflowRuntimeConfig())
    return await _pipeline(min_pass_rate).run_tasks(
        tasks,
        target=runtime,
        labels={"example": "run-agent-eval", "mode": "online"},
        output_dir=output_dir,
    )


async def rescore(rescore_dirs: list[Path], *, output_dir: Path, min_pass_rate: float) -> AgentEvalResult:
    trials = [trial for run_dir in rescore_dirs for trial in load_stored_trials(run_dir)]
    needed = {trial.task_id for trial in trials}
    tasks = [task for task in example_tasks() if task.id in needed]
    return await _pipeline(min_pass_rate).score_trials(
        tasks,
        trials=trials,
        labels={"example": "run-agent-eval", "mode": "offline"},
        output_dir=output_dir,
    )


def _print_result(result: AgentEvalResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    for metric_type, true_count, total in _boolean_true_rates(result):
        print(f"  {metric_type}: {true_count}/{total} true")
    for score in result.summary.scores.scores:
        if score.headline_value is not None:
            print(f"  {score.name}: {score.headline_value:.3f}")
    _print_measurements(result)
    if result.work_dir is not None:
        print(f"work_dir: {result.work_dir}")
        print(f"gate: {result.work_dir / 'gate.json'}")


def _print_measurements(result: AgentEvalResult) -> None:
    """Print token/runtime totals recorded on the trials."""
    measurements = [trial.measurements for trial in result.trials]
    total_tokens = [m.total_tokens for m in measurements if m.total_tokens is not None]
    runtimes = [m.runtime_sec for m in measurements if m.runtime_sec is not None]
    if total_tokens:
        print(f"  total_tokens: {sum(total_tokens)} across {len(total_tokens)}/{len(measurements)} trials")
    if runtimes:
        print(f"  runtime_sec: {sum(runtimes):.1f} across {len(runtimes)}/{len(measurements)} trials")


def _boolean_true_rates(result: AgentEvalResult) -> list[tuple[str, int, int]]:
    """Tally True/total per metric output for the boolean signals this example emits."""
    tallies: dict[str, list[int]] = {}
    for score in result.scores:
        for output in score.outputs:
            if isinstance(output.value, bool):
                tally = tallies.setdefault(f"{score.metric_type}.{output.name}", [0, 0])
                tally[0] += int(output.value)
                tally[1] += 1
    return [(name, true_count, total) for name, (true_count, total) in sorted(tallies.items())]


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Run agent-eval tasks through a workflow runtime (trials API).")
    parser.add_argument(
        "--task",
        default="all",
        help="Example task id to run, or 'all' (default). Available: " + ", ".join(task.id for task in example_tasks()),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Run bundle output directory.")
    parser.add_argument(
        "--rescore-dir",
        type=Path,
        action="append",
        default=None,
        help="Re-score the trials.jsonl of a prior run dir offline (repeatable); skips agent execution.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=1.0, help="Gate threshold for the pass rate.")
    parser.add_argument("--list-tasks", action="store_true", help="List available example tasks and exit.")
    args = parser.parse_args()

    if args.list_tasks:
        for task in example_tasks():
            print(f"{task.id}: {task.intent}")
        return 0

    _configure_logging()

    if args.rescore_dir:
        result = await rescore(args.rescore_dir, output_dir=args.output_dir, min_pass_rate=args.min_pass_rate)
    else:
        task_names = [task.id for task in example_tasks()] if args.task == "all" else [args.task]
        result = await run_online(task_names, output_dir=args.output_dir, min_pass_rate=args.min_pass_rate)

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
