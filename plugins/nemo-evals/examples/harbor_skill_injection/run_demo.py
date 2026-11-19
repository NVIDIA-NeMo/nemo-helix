# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the same Harbor task with and without the bundled Supercool Coding Guidelines skill."""

import asyncio
from pathlib import Path

from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig, run_harbor_eval
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus

DATASET = Path(__file__).with_name("dataset")
AGENT = "nemo_evals.examples.harbor_skill_demo:SupercoolGuidelinesAgent"
REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL = REPO_ROOT / "packages/nhx_evals_sdk/src/nhx_evals_sdk/agent_eval/example_skills/supercool-guidelines"


async def main() -> None:
    runs: tuple[tuple[str, list[Path], int], ...] = (
        ("baseline", [], 0),
        ("with-skill", [SKILL], 1),
    )
    for name, skills, expected in runs:
        result = await run_harbor_eval(
            HarborRuntimeConfig(
                jobs_dir=Path("temp/harbor-skill-demo").resolve(),
                agent_import_path=AGENT,
                skills=skills,
            ),
            DATASET,
        )
        (trial,) = result.trials
        if trial.status != AgentEvalTrialStatus.COMPLETED or trial.error is not None:
            raise RuntimeError(f"{name} did not complete: {trial.status}, {trial.error}")
        reward = trial.metadata["reward"]
        print(f"{name} reward={reward}", flush=True)
        if reward != expected:
            raise RuntimeError(f"{name} expected reward {expected}, got {reward}")


if __name__ == "__main__":
    asyncio.run(main())
