# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run agent-eval tasks through a workflow runtime, using the trials-based SDK.

It drives the example-local ``AgentEvalPipeline`` two ways:

* **online** — generate trials by running :class:`WorkflowAgentRuntime`
  (the agent), score them, and apply the deterministic gate; or
* **offline** — re-score the ``trials.jsonl`` of a prior run with no agent
  execution (``--rescore-dir``).

Run it as a module from the repository root::

    python -m packages.nemo_evaluator_sdk.examples.run_agent_eval.run_agent_eval --task all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

if __package__ in {None, ""}:
    raise SystemExit(
        "Run this example as a module from the repository root:\n"
        "  python -m packages.nemo_evaluator_sdk.examples.run_agent_eval.run_agent_eval --task all"
    )

from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.metrics.protocol import Metric

from .aut_runtime import AutConfig, NatAutRuntime
from .gating import GateThresholds
from .pipeline import AgentEvalPipeline, PipelineConfig
from .platform_runtime import (
    NatWorkflowConfig,
    NatWorkflowRuntime,
    VerifierRewardMetric,
    agentic_task_from_dir,
    ensure_task_image,
)
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


def _pipeline(min_pass_rate: float, *, extra_metrics: tuple[Metric, ...] = ()) -> AgentEvalPipeline:
    return AgentEvalPipeline(
        config=PipelineConfig(
            parallelism=2,
            write_dashboard=True,
            write_gate=True,
            gate_thresholds=GateThresholds(min_pass_rate=min_pass_rate),
        ),
        extra_metrics=extra_metrics,
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


async def run_agentic_task(
    task_name: str,
    *,
    output_dir: Path,
    min_pass_rate: float,
    nmp_base_url: str,
    agent_model: str | None,
    skip_build: bool,
    verify: bool,
    backend: str,
    aut_agent_name: str | None,
    aut_agent_config: Path | None,
    seed_providers: bool,
) -> AgentEvalResult:
    """Run a real ``tests/agentic-use`` task: BUILD → AGENT → VERIFY → score → gate.

    ``backend='workflow'`` runs the task-local ``nat run`` workflow; ``backend='aut'``
    drives a deployed platform agent-under-test (the canonical ``nat_runner`` path).
    """
    task = agentic_task_from_dir(task_name)
    runtime: NatWorkflowRuntime | NatAutRuntime
    if backend == "aut":
        if not aut_agent_name:
            raise ValueError("--backend aut requires --aut-agent-name")
        runtime = NatAutRuntime(
            AutConfig(
                aut_agent_name=aut_agent_name,
                aut_agent_config=aut_agent_config,
                aut_seed_providers=seed_providers,
                agent_model=agent_model,
                nmp_base_url=nmp_base_url,
                nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
                inference_nvidia_api_key=os.environ.get("INFERENCE_NVIDIA_API_KEY"),
                anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
                run_verify=verify,
            ),
        )
    else:
        runtime = NatWorkflowRuntime(
            NatWorkflowConfig(
                nmp_base_url=nmp_base_url,
                nvidia_api_key=os.environ.get("NVIDIA_API_KEY"),
                agent_model=agent_model,
                run_verify=verify,
            ),
        )
    extra_metrics: tuple[Metric, ...] = (VerifierRewardMetric(),) if verify else ()
    return await _pipeline(min_pass_rate, extra_metrics=extra_metrics).run_tasks(
        [task],
        target=runtime,
        labels={"example": "agentic-use", "task": task_name, "backend": backend},
        output_dir=output_dir,
        prepare_task=lambda t: ensure_task_image(t, skip_build=skip_build),
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
    """Print token/runtime totals (the same measurements nat_runner records)."""
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
    parser.add_argument(
        "--agentic-task",
        default=None,
        help="Run a real tests/agentic-use/<name> task end to end via the NAT workflow runtime "
        "(requires Docker, nmp-agentic-base, and a running NeMo Platform).",
    )
    parser.add_argument(
        "--backend",
        choices=("workflow", "aut"),
        default="workflow",
        help="Agentic-task backend: 'workflow' (task-local nat run) or 'aut' (deployed agent-under-test).",
    )
    parser.add_argument("--aut-agent-name", default=None, help="Name of the deployed agent-under-test (aut backend).")
    parser.add_argument(
        "--aut-agent-config",
        type=Path,
        default=None,
        help="Path to the AUT agent NAT config; created/recreated on the platform if needed.",
    )
    parser.add_argument(
        "--no-seed-providers",
        action="store_true",
        help="Skip seeding inference providers from providers.yaml (aut backend).",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip the BUILD phase (image must exist).")
    parser.add_argument("--verify", action="store_true", help="Run the pytest VERIFY phase for the agentic task.")
    parser.add_argument("--nmp-base-url", default=os.environ.get("NMP_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--agent-model", default=os.environ.get("NAT_AGENT_MODEL"), help="Model for the agent.")
    args = parser.parse_args()

    if args.agentic_task and args.backend == "aut" and not args.aut_agent_name:
        parser.error("--backend aut requires --aut-agent-name")

    if args.list_tasks:
        for task in example_tasks():
            print(f"{task.id}: {task.intent}")
        return 0

    _configure_logging()

    if args.agentic_task:
        try:
            result = await run_agentic_task(
                args.agentic_task,
                output_dir=args.output_dir,
                min_pass_rate=args.min_pass_rate,
                nmp_base_url=args.nmp_base_url,
                agent_model=args.agent_model,
                skip_build=args.skip_build,
                verify=args.verify,
                backend=args.backend,
                aut_agent_name=args.aut_agent_name,
                aut_agent_config=args.aut_agent_config,
                seed_providers=not args.no_seed_providers,
            )
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            print(f"agentic-task run failed: {exc}")
            print("Real tasks need Docker, the nmp-agentic-base image, and a running NeMo Platform (see README).")
            return 1
    elif args.rescore_dir:
        result = await rescore(args.rescore_dir, output_dir=args.output_dir, min_pass_rate=args.min_pass_rate)
    else:
        task_names = [task.id for task in example_tasks()] if args.task == "all" else [args.task]
        result = await run_online(task_names, output_dir=args.output_dir, min_pass_rate=args.min_pass_rate)

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
