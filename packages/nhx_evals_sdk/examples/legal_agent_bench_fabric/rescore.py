# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-grade a stored LAB run bundle with an ALTERNATIVE judge — WITHOUT re-running the agent.

This is the payoff of NeMo Helix Evals's decoupled execution/scoring: a run persists each trial's
deliverables as durable ``workspace`` filesystem evidence, so you can feed the **stored trials** straight
back into ``AgentEvaluator().run(trials=...)`` with a *different* judge attached and get a full re-scored
bundle (``scores.jsonl``, ``report.html``, aggregates, diagnostics) — no agent invocations, no credits.

We reuse the SDK's own imported-trials path rather than calling the metric by hand:

* **rebuild the tasks from source** with ``build_lab_tasks`` pointed at the *alternative* judge — so each
  task carries its correct metric (metrics belong to tasks; they are not reloaded from the bundle);
* **hydrate only the trials** from the bundle's ``trials.jsonl`` (their ``workspace`` evidence still points
  at the stored deliverables — the one thing you can't regenerate without re-running the agent);
* ``AgentEvaluator().run(tasks=…, trials=…)`` matches them by ``task_id``, re-scores, and writes a fresh bundle.

Run from the repository root, e.g. re-grade an existing run with llama-3.3-70b on inference-api::

    NVIDIA_API_KEY=... \\
    .venv/bin/python -m packages.nhx_evals_sdk.examples.legal_agent_bench_fabric.rescore \\
        --run-dir ./results/lab-fabric \\
        --judge-model nvidia/meta/llama-3.3-70b-instruct \\
        --judge-base-url https://inference-api.nvidia.com/v1 --judge-api-key-env NVIDIA_API_KEY \\
        --judge-parallel 4 --judge-min-interval 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.persistence import read_trials
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig

from .prepare_lab_taskset import build_lab_tasks, ensure_lab_source

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _original_pass_rates(run_dir: Path) -> dict[str, float | None]:
    """task_id -> criteria_pass_rate from the bundle's original scores.jsonl (None if it errored)."""
    rates: dict[str, float | None] = {}
    scores = run_dir / "scores.jsonl"
    if scores.is_file():
        for row in _read_jsonl(scores):
            outputs = {o["name"]: o["value"] for o in row.get("outputs") or []}
            rates[row.get("task_id")] = outputs.get("criteria_pass_rate")
    return rates


async def _main(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    if not (run_dir / "trials.jsonl").is_file():
        raise FileNotFoundError(f"{run_dir} is not a run bundle (no trials.jsonl)")

    source_root = ensure_lab_source(args.source_dir, allow_download=not args.no_download)

    # Hydrate ONLY the trials from the bundle — the agent's stored deliverables are the one thing you can't
    # regenerate without re-running. The TASKS (and their metrics) are rebuilt from source via the benchmark's
    # own task builder, pointed at the ALTERNATIVE judge; run() matches trials to tasks by task_id. This is
    # why metrics never need re-attaching — each task already carries its correct metric.
    trials = read_trials(run_dir)  # SDK loader: hydrate stored trials + resolve evidence refs
    tasks = build_lab_tasks(
        source_root,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key_env=args.judge_api_key_env,
        judge_parallel=args.judge_parallel,
        judge_min_interval=args.judge_min_interval,
        task_ids={trial.task_id for trial in trials},
    )
    original = _original_pass_rates(run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir.parent / f"{run_dir.name}-rescored"

    endpoint = args.judge_base_url or "LAB native"
    print(
        f"Re-scoring {len(trials)} stored trials from {run_dir} with judge: {args.judge_model} @ {endpoint}\n",
        flush=True,
    )

    # The SDK's imported-trials path: no agent runs — it just scores the stored trials with our metric.
    result = await AgentEvaluator().run(
        tasks=tasks,
        trials=trials,
        config=AgentEvalRunConfig(work_dir=output_dir, parallelism=args.parallelism),
    )
    # Storing is explicit: writes the full bundle (run.json / trials.jsonl / scores.jsonl /
    # summary.json / report.html).
    result.persist()

    print(f"{'task (area)':30s} {'original':>10s} {'rescored':>10s}")
    print("-" * 54)
    for score in result.scores:
        if score.metric_type != "lab_rubric" or score.status.value != "completed":
            continue
        outputs = {o.name: o.value for o in score.outputs}
        orig = original.get(score.task_id)
        orig_str = f"{orig:.2f}" if isinstance(orig, (int, float)) else "—"
        print(
            f"{score.task_id.split('__')[0][:30]:30s} {orig_str:>10s} {outputs.get('criteria_pass_rate', 0.0):>10.2f}"
        )
    print("-" * 54)
    for aggregate in result.summary.scores.scores:
        if aggregate.name == "lab_rubric.criteria_pass_rate":
            print(f"mean criteria_pass_rate: {aggregate.mean}")
    print(f"\nRe-scored bundle (with report.html): {output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="A run bundle written by run_legal_agent_bench_fabric.")
    p.add_argument("--source-dir", default="./data/lab-source", help="LAB source (scorer + prompts).")
    p.add_argument("--no-download", action="store_true", help="Fail if LAB source is not already present.")
    p.add_argument("--judge-model", required=True, help="Alternative judge model to re-grade with.")
    p.add_argument("--judge-base-url", default=None, help="OpenAI-compatible judge endpoint (else LAB's native Judge).")
    p.add_argument("--judge-api-key-env", default="OPENAI_API_KEY", help="Env var holding the judge endpoint key.")
    p.add_argument("--judge-parallel", type=int, default=1, help="Concurrent judge calls per task.")
    p.add_argument(
        "--judge-min-interval",
        type=float,
        default=2.0,
        help="Min seconds between judge calls (throttle). ~2 for build.nvidia.com; ~0.3 for inference-api.",
    )
    p.add_argument("--parallelism", type=int, default=1, help="Tasks scored concurrently.")
    p.add_argument("--output-dir", default=None, help="Re-scored bundle dir (default <run-dir>-rescored).")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main(_parse_args()))
