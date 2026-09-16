# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nat`` agent optimization: numeric/categorical HPO over the agent's parameters.

Runs the existing Optuna/GA study machinery against a Fabric translation of the source
agent, then overlays the winning parameters back onto the stored ``nemo-agents-spec-v1``
config via :func:`apply_tuned_params_to_spec` — the study tunes Fabric-shaped dotted
paths (e.g. ``models.default.temperature``), which do not exist verbatim on the spec the
platform persists.

Every declared search-space path is checked against both halves of that contract
*before* the study starts (:func:`search_space_path_problem`).  A path that the study
cannot really apply would otherwise run every trial with identical parameters, let
Optuna pick a "winner" among identical scores, and register an agent whose tuned value
was never measured — after paying for a full study of real model calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.run_dependencies import LocalRunError

from nemo_optimization.agents import to_fabric_agent_package
from nemo_optimization.bundle import _agent_problems, _optimizer_problems
from nemo_optimization.dataset import _staged_dataset
from nemo_optimization.fabric import build_optimize_payload
from nemo_optimization.preflight import preflight_validate_llm_models
from nemo_optimization.router import OptimizeRouter
from nemo_optimization.spec_overlay import apply_tuned_params_to_spec, search_space_path_problem


class NatAgentOptimizeJob(AgentOptimizeJob):
    """Numeric/categorical HPO via the Optuna and GA backends."""

    name: ClassVar[str] = "agent_optimize"
    strategy: ClassVar[str] = "nat"
    description: ClassVar[str] = "Tune an agent's numeric and categorical parameters."
    task_module: ClassVar[str] = "nemo_optimization.tasks.agent_optimize"

    def optimize(
        self,
        *,
        source_agent_config: dict[str, Any],
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> dict[str, Any]:
        # The base class already resolved an Agent under Test (source_agent_config), so the
        # only thing left for the old validate_config's _agent_problems check to catch is a
        # config that looks like legacy NAT workflow YAML instead of the optimizer/eval
        # overlay it is supposed to be.
        problems = [
            *_agent_problems(config, agent=source_agent_config.get("name")),
            *_optimizer_problems(config),
        ]
        if problems:
            raise ValueError("; ".join(problems))

        # The study runs against the Fabric package the harness actually executes, not the
        # stored platform spec.
        agent_config = to_fabric_agent_package(source_agent_config, label=str(source_agent_config.get("name")))

        # Cheapest check first: refusing an untunable search space here costs seconds,
        # whereas the same refusal after the study costs a study's worth of model calls.
        _refuse_untunable_search_space(config, agent_config=agent_config, spec_config=source_agent_config)

        preflight_validate_llm_models(config, workspace=workspace, sdk=sdk, agent_config=agent_config)
        with _staged_dataset(config, workspace=workspace, ctx=ctx, sdk=sdk) as staged_config:
            result = OptimizeRouter.dispatch(agent_config=agent_config, optimize_config=staged_config, ctx=ctx, sdk=sdk)

        by_path = result.get("best_params_by_path")
        if not isinstance(by_path, dict) or not by_path:
            raise LocalRunError(
                "The study finished without reporting tuned parameters, so there is nothing to apply to the agent."
            )
        return apply_tuned_params_to_spec(source_agent_config, by_path)


def _refuse_untunable_search_space(
    config: dict[str, Any],
    *,
    agent_config: dict[str, Any],
    spec_config: Mapping[str, Any],
) -> None:
    """Refuse the whole run when any declared search-space path cannot be tuned end to end."""
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):  # already reported by _optimizer_problems
        return
    prompt = optimizer.get("prompt")
    if isinstance(prompt, Mapping) and prompt.get("enabled"):
        # Prompt GA owns its own dimensions; the Optuna path (and this contract) does not run.
        return

    # Imported here so loading this job class (which `nemo agents optimize` does for
    # every run, whatever the strategy) does not pull in numpy/Optuna.
    from nemo_optimization.backends.optuna.search_space import parse_search_space

    search_space = parse_search_space(optimizer)
    # The study mutates the merged payload, not the bare agent package: an overlay's
    # extra models and its eval/optimizer blocks are addressable by a search-space path too.
    payload = build_optimize_payload(agent_config=agent_config, optimize_config=config)

    problems = [
        f"  - {name!r} (path {spec.path!r}): {problem}"
        for name, spec in search_space.items()
        if (problem := search_space_path_problem(spec.path, fabric_payload=payload, spec_config=spec_config))
    ]
    if problems:
        raise LocalRunError(
            "optimizer.search_space declares parameter(s) this study cannot tune end to end:\n"
            + "\n".join(problems)
            + "\nSearch-space paths are Fabric-shaped: each one must address a value the study can "
            "really change in the Fabric package it runs, and that can be written back onto the "
            "stored agent afterwards. Refusing now rather than after the study keeps you from "
            "paying for trials that all run identically."
        )
