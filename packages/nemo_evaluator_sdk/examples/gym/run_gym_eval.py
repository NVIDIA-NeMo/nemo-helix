# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live end-to-end: run an existing NeMo Gym environment through NeMo Evaluator.

Points the Gym runner (:class:`GymAgentTaskRunner`) at an installed Gym checkout and runs a dataset
(mcqa's bundled ``example.jsonl`` by default) via the two-step Gym flow — ``gym env start`` brings up
the resources-server + agent + model servers, then ``gym eval run --no-serve --input`` collects
rollouts against them — and scores the result with :class:`AgentEvaluator`. Gym owns execution *and*
scoring; :class:`GymRewardMetric` surfaces the per-attempt reward (Evaluator does not re-derive it).

One Gym dataset → one run; each distinct row → one task (id = content hash of the row); each attempt
(``num_repeats``) → one trial. The runner materializes the tasks into a normalized dataset with
``_ng_task_index`` stamped per row and hands *that* to Gym, which honors the stamp — so rollouts join
back to tasks by an explicit map rather than by position (see the README).

Prerequisites (see README.md):

* **NeMo Gym installed in this environment** (``pip install nemo-gym``), plus the target env's own
  deps — each env ships its own ``requirements.txt`` (mcqa needs ``tiktoken``). Environments ship in
  the wheel, so no checkout is needed for the bundled example data.
* A **gitignored ``env.yaml``** holding the model credentials the collector calls
  (``policy_base_url`` / ``policy_api_key`` / ``policy_model_name``) in the directory you run from.
  This SDK never handles secrets — the ``gym`` subprocess reads them from that file.

Run from the directory holding ``env.yaml``::

    uv run python -m packages.nemo_evaluator_sdk.examples.gym.run_gym_eval

Pass ``--output-dir`` to write the bundle somewhere stable, then read it with ``inspect_results.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.runtimes.gym import (
    GymAgentTaskRunner,
    GymRuntimeConfig,
    discover_gym_tasks,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig


def _non_negative_int(raw: str) -> int:
    """argparse type for ``--limit``: a negative value would silently slice tasks off the END."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value}")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset jsonl (default: the installed package's "
        "resources_servers/<resources-server>/data/example.jsonl).",
    )
    parser.add_argument("--resources-server", default="mcqa", help="Gym resources-server (environment) name.")
    parser.add_argument("--agent", default="simple_agent", help="Agent to collect rollouts with.")
    parser.add_argument(
        "--agent-config",
        default="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        help="Repo-relative agent config passed to `gym env start`.",
    )
    parser.add_argument(
        "--model-type",
        default="inference_provider",
        help="`inference_provider` speaks OpenAI-compatible chat; `openai_model` uses the Responses API.",
    )
    parser.add_argument("--num-repeats", type=int, default=2, help="Attempts per task (each becomes a trial).")
    parser.add_argument(
        "--limit", type=_non_negative_int, default=None, help="Run only the first N tasks (handy for smoke runs)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the run bundle is written (default: a fresh temporary directory). Each run needs its own "
        "directory — the runner refuses to reuse one that already holds Gym rollout output.",
    )
    return parser.parse_args(argv)


def _packaged_dataset(resources_server: str) -> Path:
    """The environment's bundled ``example.jsonl`` as installed by the ``nemo-gym`` wheel.

    Gym ships ``resources_servers`` beside ``nemo_gym`` in site-packages, configs and example data
    included, so this resolves without a checkout. It only works when Gym is installed in *this*
    interpreter — with Gym in a separate venv there is nothing to import, and the caller passes
    ``--dataset`` instead.
    """
    try:
        from importlib import resources
    except ImportError as exc:  # pragma: no cover - importlib.resources is stdlib
        raise SystemExit(f"cannot resolve a packaged dataset: {exc}") from exc
    try:
        dataset = Path(str(resources.files(f"resources_servers.{resources_server}") / "data" / "example.jsonl"))
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"resources_servers.{resources_server} is not importable here, so its bundled dataset cannot be "
            "located. Install Gym in this environment, or pass --dataset with the path to the jsonl you "
            "want to run."
        ) from exc
    # An importable environment does not guarantee bundled data: only git-tracked files ship in the
    # wheel, so an environment whose splits are downloaded at runtime has no example.jsonl. Checking
    # here keeps the guidance identical to the import failure, rather than letting
    # `discover_gym_tasks` raise a bare FileNotFoundError on a path the caller never chose.
    if not dataset.is_file():
        raise SystemExit(
            f"{resources_server} ships no bundled dataset at {dataset}. Pass --dataset with the path to "
            "the jsonl you want to run."
        )
    return dataset


async def _main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="gym-eval-"))
    dataset = args.dataset or _packaged_dataset(args.resources_server)
    tasks = discover_gym_tasks(dataset)
    print(f"discovered {len(tasks)} tasks from {dataset}")
    if args.limit is not None:
        # The runner materializes Gym's input from these tasks, so this bounds the rollout too.
        tasks = tasks[: args.limit]
        print(f"limited to {len(tasks)} task(s)")

    runner = GymAgentTaskRunner(
        config=GymRuntimeConfig(
            agent=args.agent,
            agent_config=args.agent_config,
            resources_server=args.resources_server,
            model_type=args.model_type,
            num_repeats=args.num_repeats,
        )
    )

    result = await AgentEvaluator().run(
        tasks=tasks,
        target=runner,
        config=AgentEvalRunConfig(work_dir=output_dir, parallelism=1),
    )
    # Storing the run is its own step. Defaults to the run's work_dir, so the bundle contains the
    # evidence the trials point at.
    location = result.persist()

    print("=== RESULT ===")
    print(f"tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    print("aggregate scores:")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: {aggregate.headline_value}")
    print(f"\nRun bundle (run.json, trials.jsonl, scores.jsonl, report.html): {location.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parse_args())))
