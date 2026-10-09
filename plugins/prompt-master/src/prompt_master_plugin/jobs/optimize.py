# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``prompt-master`` -- rewrite a platform agent's system prompt with the Prompt Master skill.

Reached as ``nemo agents optimize run-strategy --strategy prompt-master``: the router job in
nemo-agent-optimization-plugin finds this class through the
``nemo_agent_optimization_strategy`` class variable below and delegates ``compile`` and
``run`` to it, so this job's single step is what the platform actually runs.

The run is one Fabric invocation.  A Deep Agents optimizer carrying the vendored Prompt
Master skill is handed the target agent's current system prompt as inert data and asked for
one improved prompt.  The result is written back as a complete ``nemo-agents-spec-v1`` config
(``agent.yaml``) next to a run summary and registered as the job's result.  The stored agent
is never modified.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar

import yaml
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import (
    InternalServerError,
    NemoResponseValidationError,
    NemoTransportError,
    NotFoundError,
)
from nemo_helix_plugin.errors import LocalRunError
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    ExecutorSpec,
    HelixJobSpec,
    HelixJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.image import get_qualified_image
from nemo_helix_plugin.refs import parse_entity_ref
from prompt_master_plugin.runner import PromptMasterOutcome, build_optimizer_agent, run_prompt_master
from prompt_master_plugin.schemas.optimize import PromptMasterOptimizeSpec
from pydantic import BaseModel

logger = logging.getLogger(__name__)

STRATEGY_NAME = "prompt-master"

#: Task entry point for the step, in both executor flavours.  The subprocess backend takes one
#: flat command; the cpu backend splits it into a container entrypoint + command.
TASK_MODULE = "prompt_master_plugin.tasks.optimize"
TASK_ENTRYPOINT = ["python", "-m"]
TASK_COMMAND = [TASK_MODULE]

#: Image for the cpu (docker / kubernetes_job) fallback.  This plugin is not part of the stock
#: ``cpu-tasks`` dependency group, so the image a deployment registers under that profile must
#: have ``prompt-master-plugin`` installed for ``python -m prompt_master_plugin.tasks.optimize``
#: to import there (see the README).
TASK_IMAGE = "nhx-tasks"

#: Name the artifacts are registered under in the job's results.
RESULT_NAME = "prompt_master"
OPTIMIZED_AGENT_FILENAME = "agent.yaml"
SUMMARY_FILENAME = "prompt-master-result.json"

#: The only stored-agent format whose ``instructions.system.content`` this strategy rewrites.
SUPPORTED_CONFIG_FORMAT = "nemo-agents-spec-v1"


class PromptMasterOptimizeJob(NemoJob):
    """Optimize a platform agent's system prompt with the bundled Prompt Master skill."""

    name: ClassVar[str] = STRATEGY_NAME
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name=STRATEGY_NAME,
        description="Rewrite the agent's system prompt using https://github.com/nidhinjs/prompt-master.",
    )
    description: ClassVar[str] = "Optimize a platform agent's system prompt with Prompt Master."
    spec_schema: ClassVar[type[BaseModel]] = PromptMasterOptimizeSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        *,
        workspace: str,
        spec: PromptMasterOptimizeSpec,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncNemoClient,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        del entity_client, job_name, options
        config = spec.model_dump(mode="json")
        config["workspace"] = workspace
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name=STRATEGY_NAME,
                    executor=await _resolve_executor(profile=profile or "default", async_sdk=async_sdk),
                    config=config,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient) -> dict[str, Any]:
        spec = PromptMasterOptimizeSpec.model_validate(config)
        overrides = None
        if spec.optimize_config is not None:
            with resolve_staged_config(
                spec.optimize_config,
                spec.optimize_config_fileset,
                workspace=spec.workspace,
                ctx=ctx,
                sdk=sdk,
                kind="prompt-master-config",
            ) as config_path:
                overrides = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        optimizer = build_optimizer_agent(overrides, workspace=spec.workspace)

        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        source = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)

        runtime_dir = ctx.storage.ephemeral / STRATEGY_NAME / "fabric"
        logger.info(
            "Running Prompt Master on agent %s with optimizer model %s", agent_label, optimizer.models["default"].model
        )
        outcome = run_prompt_master(optimizer, source, runtime_dir)
        optimized = with_system_prompt(source, outcome.optimized_prompt)

        results_dir = ctx.storage.ephemeral / STRATEGY_NAME / "results"
        write_artifacts(
            results_dir,
            agent=agent_label,
            optimizer=optimizer,
            source=source,
            optimized=optimized,
            outcome=outcome,
        )
        result_ref = ctx.results.save(RESULT_NAME, results_dir)
        return {
            "status": "completed",
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "optimized_prompt": outcome.optimized_prompt,
            "result": result_ref.model_dump(mode="json"),
        }


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    """The stored ``nemo-agents-spec-v1`` config of agent *workspace*/*name*."""
    try:
        config = client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data().config
    except NotFoundError as exc:
        raise LocalRunError(f"Agent '{workspace}/{name}' does not exist; there is nothing to optimize.") from exc
    if config.get("config_format") != SUPPORTED_CONFIG_FORMAT:
        raise LocalRunError(
            f"Agent '{workspace}/{name}' is not a {SUPPORTED_CONFIG_FORMAT!r} agent; the prompt-master strategy "
            "rewrites instructions.system.content of those only."
        )
    logger.info("Resolved agent %s/%s", workspace, name)
    return config


def with_system_prompt(agent_config: dict[str, Any], prompt: str) -> dict[str, Any]:
    """A deep copy of *agent_config* whose ``instructions.system.content`` is *prompt*."""
    optimized = copy.deepcopy(agent_config)
    optimized["instructions"]["system"]["content"] = prompt
    return optimized


def write_artifacts(
    results_dir: Path,
    *,
    agent: str,
    optimizer: AgentConfig,
    source: dict[str, Any],
    optimized: dict[str, Any],
    outcome: PromptMasterOutcome,
) -> None:
    """Write the optimized agent config and a run summary into a fresh *results_dir*.

    ``agent.yaml`` is a complete ``nemo-agents-spec-v1`` config, so it can be registered as-is
    with ``nemo agents create --agent-config``.  The summary keeps both prompts and Prompt
    Master's full reply side by side for review.
    """
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    (results_dir / OPTIMIZED_AGENT_FILENAME).write_text(yaml.safe_dump(optimized, sort_keys=False), encoding="utf-8")
    summary = {
        "strategy": STRATEGY_NAME,
        "agent": agent,
        "optimizer_model": optimizer.models["default"].model,
        "original_prompt": source["instructions"]["system"]["content"],
        "optimized_prompt": outcome.optimized_prompt,
        "optimized_agent_config": OPTIMIZED_AGENT_FILENAME,
        "response": outcome.response,
    }
    (results_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


async def _resolve_executor(*, profile: str, async_sdk: AsyncNemoClient) -> ExecutorSpec:
    """Pick the executor for *profile* from the backends the platform actually registered.

    ``subprocess`` is preferred: the one-shot Fabric run needs a venv carrying the Deep Agents
    harness adapter, which the platform host's own environment has.  Deployments that register
    no subprocess backend (Helm / Minikube) get the ``cpu`` provider instead, which the platform
    maps to whichever container backend it registered for that profile.
    """
    try:
        profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise HelixJobDependencyUnavailableError(
            f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
            "unavailable.  Retry the submission."
        ) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(
            provider="subprocess",
            profile=profile,
            command=[*TASK_ENTRYPOINT, *TASK_COMMAND],
        )

    # Jobs keys execution profiles by (provider, profile); "cpu" is whatever container backend
    # the deployment registered under that name.
    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(
                image=get_qualified_image(TASK_IMAGE),
                entrypoint=TASK_ENTRYPOINT,
                command=TASK_COMMAND,
            ),
        )

    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise HelixJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the "
        f"prompt-master step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )
