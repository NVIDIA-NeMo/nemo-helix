# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the same Harbor task with and without the bundled mock skill."""

import asyncio
from pathlib import Path

from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig, run_harbor_eval
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus

DATASET = Path(__file__).with_name("dataset")
AGENT = "nemo_evals.examples.harbor_skill_demo:SupercoolGuidelinesAgent"


async def main() -> None:
    for name, env_vars, expected in (
        ("baseline", {}, 0),
        ("with-skill", {"NHX_HARBOR_MOCK_SKILL": "1"}, 1),
    ):
        result = await run_harbor_eval(
            HarborRuntimeConfig(
                jobs_dir=Path("temp/harbor-skill-demo").resolve(),
                agent_import_path=AGENT,
                env_vars=env_vars,
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
