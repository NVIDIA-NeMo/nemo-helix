# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draw a simulation's personas without running it: a simulate spec on stdin, the drawn rows on stdout.

Runs as its own process because Data Designer reads its persona folder once, at import, from
``DATA_DESIGNER_MANAGED_ASSETS_PATH``, which the caller sets. The draw makes no model calls.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from nemo_user_sim_plugin import scenarios
from nemo_user_sim_plugin.config import UserSimConfig
from nemo_user_sim_plugin.gateway import write_models_file
from nemo_user_sim_plugin.jobs.spec import SimulateSpec
from nemo_user_sim_plugin.personas import require_personas
from usersim.engine.external import materialize_episode_inputs


def main() -> None:
    spec = SimulateSpec.model_validate_json(sys.stdin.read())
    require_personas(spec.locale)
    config = UserSimConfig.get()
    with tempfile.TemporaryDirectory() as work:
        if fetched := os.environ.get(scenarios.FETCHED_ENV):  # custom scenarios, fetched by the service
            scenarios.install(scenarios.read_local(Path(fetched)), Path(work) / "scenarios")
        # Model names only feed trajectory IDs here; nothing is called, so the endpoint is never used.
        models = write_models_file(
            Path(work) / "models.toml",
            gateway_url="http://unused",
            models={
                "user_model": config.user_model,
                "assistant_model": spec.assistant_model,
                "judge_model": config.judge_model,
                "summary_model": config.summary_model,
                "evaluator_model": config.evaluator_model,
            },
        )
        rows = materialize_episode_inputs(
            locale=spec.locale,
            num_rows=spec.users,
            probe_mix={probe: 1 / len(spec.probes) for probe in spec.probes},
            random_seed=spec.seed,
            models_path=models,
            max_turns=spec.max_turns,
        )
    json.dump(rows, sys.stdout, default=str)


if __name__ == "__main__":
    main()
