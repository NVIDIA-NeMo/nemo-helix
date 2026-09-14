# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nat`` implementation of ``nemo agents optimize --strategy`` — wraps the existing HPO engine."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext

from nemo_optimization.bundle import _agent_problems, _optimizer_problems
from nemo_optimization.jobs.optimize import _staged_dataset
from nemo_optimization.preflight import preflight_validate_llm_models
from nemo_optimization.router import OptimizeRouter


class NatOptimizationStrategy:
    """Numeric/categorical HPO via the existing OptimizeRouter/Optuna/GA backends."""

    name: ClassVar[str] = "nat"

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        problems = [*_agent_problems(config, agent=agent), *_optimizer_problems(config)]
        if problems:
            raise ValueError("; ".join(problems))

    def run(
        self,
        *,
        agent_config: dict[str, Any] | None,
        source_agent_config: dict[str, Any] | None = None,
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform | None = None,
    ) -> dict[str, Any]:
        del source_agent_config
        preflight_validate_llm_models(config, workspace=workspace, sdk=sdk, agent_config=agent_config)
        with _staged_dataset(config, workspace=workspace, ctx=ctx, sdk=sdk) as staged_config:
            return OptimizeRouter.dispatch(agent_config=agent_config, optimize_config=staged_config, ctx=ctx, sdk=sdk)
