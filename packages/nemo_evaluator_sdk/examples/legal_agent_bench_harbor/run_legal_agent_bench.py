# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Harvey Labs Legal Agent Benchmark (LAB) through NeMo Evaluator.

LAB (https://github.com/harveyai/harvey-labs) is a **Harbor** benchmark, and NeMo
Evaluator runs Harbor task suites natively — so running LAB is just the SDK's
Harbor runner (:func:`run_harbor_eval` / :class:`HarborAgentTaskRunner`) pointed at
LAB's generated Harbor suite. The whole run is one
:class:`~nemo_evaluator_sdk.agent_eval.evaluator.AgentEvaluator` call in a single
local process, then the same run can be submitted as a governed NeMo Platform job.

The LAB-specific pieces — downloading the pinned LAB source, generating the Harbor
tasks, LAB's Harbor agent, and LAB's rubric verifier — are Harbor-native and used
unchanged, because the SDK already speaks Harbor.

Two modes:

* ``--mode reward``     — score LAB's official reward only (``HarborRewardMetric``:
  the ``full_task`` all-criteria score). Minimal plumbing: one ``run_harbor_eval``.
* ``--mode components`` — additionally attach :class:`LabCriteriaMetric` and a
  ``legal_quality`` view, turning LAB's rubric into per-criterion component scores
  in the *same* run, rather than having to pick a single reward per run.

Prerequisites:

* Python >= 3.12 and a running Docker daemon.
* A NeMo Platform source checkout (see SETUP.md for toolchain prerequisites).
  From the repository root, install Harbor with
  ``uv sync --frozen --package nemo-evaluator-sdk --extra harbor``. The SDK is not
  published as a standalone PyPI package. Harbor is imported lazily.
* A prepared LAB Harbor suite on disk — a directory of task folders. Generate it
  with the bundled, self-contained ``prepare_lab_suite.py`` (pinned download +
  Harbor-task generation; see the example README).
* An agent: ``--agent-name`` (a built-in Harbor agent) or ``--agent-import-path``
  (your own). The agent model is passed via ``--model``; judge credentials reach
  the in-container verifier via each task's ``[verifier.env]`` (baked by
  ``prepare_lab_suite.py --judge-*``).

Run from the repository root::

    python -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_harbor.run_legal_agent_bench \\
        --dataset-path /path/to/lab-harbor-suite \\
        --agent-import-path legal_harbor_agent:LegalAgentBenchHarborAgent \\
        --agent-dir /path/to/legal_agent_bench \\
        --model your-policy-model \\
        --mode components \\
        --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    discover_harbor_tasks,
    run_harbor_eval,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, SemanticReducer, SemanticView, ViewSignal

from .lab_criteria_metric import LabCriteriaMetric

logger = logging.getLogger(__name__)


def _non_negative_int(raw: str) -> int:
    """argparse type for ``--limit``: a negative value would silently slice tasks off the END."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return value


def _build_config(args: argparse.Namespace) -> HarborRuntimeConfig:
    """Map the CLI onto Harbor's runtime config."""
    use_builtin = bool(args.agent_name)  # e.g. --agent-name oracle for a wiring smoke test
    return HarborRuntimeConfig(
        jobs_dir=Path(args.jobs_dir),
        job_name=args.job_name,  # pin a stable name to reuse the job-dir cache; omit for a fresh run
        agent_name=args.agent_name if use_builtin else None,
        agent_import_path=None if use_builtin else args.agent_import_path,  # LAB's Harbor agent, reused as-is
        agent_dir=None if use_builtin else (Path(args.agent_dir) if args.agent_dir else None),
        agent_model_name=args.model,  # the policy model handed to LAB's agent
        n_attempts=args.n_attempts,
        n_concurrent_trials=args.concurrency,  # Harbor-side concurrency (async, in-process)
        # LAB tasks are heavy (document tooling + a rubric judge); give the phases room.
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
    )


def _legal_quality_view() -> SemanticView:
    """Blend LAB's official reward and the criterion pass-rate into one tracked score."""
    return SemanticView(
        reducer=SemanticReducer.MEAN,
        signals=[
            ViewSignal(metric="harbor_reward", output="reward"),  # attached by discover_harbor_tasks
            ViewSignal(metric="lab_criteria", output="criteria_pass_rate"),  # attached below
        ],
    )


def _selected_task_names(args: argparse.Namespace) -> list[str] | None:
    if args.limit is None:
        return None
    tasks = discover_harbor_tasks(args.dataset_path)
    return [task.id for task in tasks[: args.limit]]


async def _run_reward_only(args: argparse.Namespace) -> AgentEvalResult:
    """Minimal plumbing: discover + run + score LAB's official reward in one call."""
    run_config = AgentEvalRunConfig(work_dir=Path(args.output_dir), parallelism=args.parallelism)
    return await run_harbor_eval(
        _build_config(args),
        args.dataset_path,
        task_names=_selected_task_names(args),
        run_config=run_config,
    )


async def _run_with_components(args: argparse.Namespace) -> AgentEvalResult:
    """Explicit form: keep HarborRewardMetric, add the per-criterion metric and a view."""
    tasks = discover_harbor_tasks(args.dataset_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    for task in tasks:
        # discover_harbor_tasks already attached HarborRewardMetric; append the rubric metric.
        task.metrics = [*task.metrics, LabCriteriaMetric()]
        task.views = {**task.views, "legal_quality": _legal_quality_view()}

    # Restrict the Harbor run itself to the selected tasks (not just the scoring).
    runner = HarborAgentTaskRunner(config=_build_config(args), task_names=[task.id for task in tasks])
    run_config = AgentEvalRunConfig(work_dir=Path(args.output_dir), parallelism=args.parallelism)
    return await AgentEvaluator().run(tasks=tasks, target=runner, config=run_config)


async def _main(args: argparse.Namespace) -> None:
    if args.mode == "components":
        result = await _run_with_components(args)
    else:
        result = await _run_reward_only(args)

    # Storing is explicit; both modes persist into the run's work_dir.
    result.persist()

    print(f"run_id: {result.run_id}  tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    print("Aggregate scores:")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: {aggregate.headline_value}")
    print(f"\nRun bundle (run.json, trials.jsonl, scores.jsonl, summary.json, report.html): {args.output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-path", required=True, help="Prepared LAB Harbor suite (a directory of task folders)."
    )
    parser.add_argument("--mode", choices=("reward", "components"), default="components")
    parser.add_argument(
        "--agent-import-path",
        default="legal_harbor_agent:LegalAgentBenchHarborAgent",
        help="Import path (module:Class) of LAB's Harbor agent.",
    )
    parser.add_argument(
        "--agent-dir",
        default=None,
        help="Directory holding the agent module when it is a loose file (not an installed package).",
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Use a built-in Harbor agent instead of the custom one (e.g. 'oracle' to smoke-test wiring).",
    )
    parser.add_argument("--model", default=None, help="Policy model slug handed to LAB's agent.")
    parser.add_argument(
        "--jobs-dir",
        default="./results/legal_agent_bench/harbor_jobs",
        help="Where Harbor writes its <job_name>/ results tree (also doubles as a re-run cache).",
    )
    parser.add_argument("--job-name", default=None, help="Pin a stable job name to enable the job-dir cache.")
    parser.add_argument(
        "--output-dir",
        default="./results/legal_agent_bench/run",
        help="Where the agent-eval run bundle + report.html are written.",
    )
    parser.add_argument(
        "--limit", type=_non_negative_int, default=None, help="Score only the first N tasks (handy for smoke runs)."
    )
    parser.add_argument("--n-attempts", type=int, default=1, help="Harbor trials per task.")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent Harbor trials.")
    parser.add_argument("--parallelism", type=int, default=4, help="Tasks scored concurrently by the evaluator.")
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None, help="Agent-phase timeout multiplier.")
    parser.add_argument(
        "--verifier-timeout-multiplier", type=float, default=None, help="Verifier-phase timeout multiplier."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(_parse_args()))
