# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Harbor dataset through the SDK's native Harbor runtime.

The point of this example is how *little* plumbing the caller needs: apart from
imports, running a whole Harbor local dataset is two lines — build a
:class:`HarborRuntimeConfig`, call :func:`run_harbor_eval`. The SDK builds and
runs Harbor's ``JobConfig`` and scores the results; the caller never imports
``harbor`` or assembles a job.

It runs the bundled ``hello_world_dataset`` with Harbor's deterministic **oracle**
agent, so no LLM/API key is needed. Requires ``harbor`` installed and a working
Docker daemon.

Run it as a module from the repository root::

    uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_harbor_example
    uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_harbor_example --n-attempts 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborRuntimeConfig,
    run_harbor_eval,
)

logger = logging.getLogger(__name__)

# A Harbor "local dataset" is a directory whose immediate subdirectories are task
# folders. Point the runtime at it and every task is discovered, run, and scored.
HELLO_WORLD_DATASET_DIR = Path(__file__).resolve().parent / "hello_world_dataset"


async def _main(jobs_dir: Path, *, n_attempts: int, job_name: str | None) -> None:
    # The entire caller-side plumbing: a config and one call.
    # n_attempts>1 runs the same verifier criteria per attempt so summary can emit pass@k.
    config = HarborRuntimeConfig(
        jobs_dir=jobs_dir,
        job_name=job_name,
        agent_name="oracle",
        n_attempts=n_attempts,
        n_concurrent_trials=1,
        quiet=False,
    )
    result = await run_harbor_eval(config, HELLO_WORLD_DATASET_DIR)

    print(f"run_id: {result.run_id}  tasks: {result.summary.task_count}  trials: {result.summary.trial_count}")
    for aggregate in result.summary.scores.scores:
        print(f"  {aggregate.name}: {aggregate.headline_value}")
    for score in result.scores:
        reward = score.outputs[0].value if score.outputs else None
        print(f"  {score.task_id}: reward={reward} status={score.status.value}")
    for trial in result.trials:
        if trial.error is not None:
            print(f"  {trial.id}: error={trial.error.type}: {trial.error.message}")


if __name__ == "__main__":
    if __package__ in {None, ""}:
        raise SystemExit(
            "Run this example as a module from the repository root:\n"
            "  uv run python -m packages.nemo_evaluator_sdk.examples.harbor.run_harbor_example"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "harbor-example-output",
        help="Directory Harbor writes its job results into.",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=1,
        help="Harbor trials per task (same verifier criteria each attempt; enables pass@k when >1).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Pin a stable Harbor job name to reuse the job-dir cache across debug runs.",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.jobs_dir, n_attempts=args.n_attempts, job_name=args.job_name))
