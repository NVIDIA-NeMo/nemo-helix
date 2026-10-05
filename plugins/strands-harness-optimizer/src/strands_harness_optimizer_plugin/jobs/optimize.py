# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``strands-harness-optimizer`` -- tune a platform agent's system prompt with Strands Harness Optimizer.

Reached as ``nemo agents optimize run-strategy --strategy strands-harness-optimizer``: the router
job in nemo-agent-optimization-plugin finds this class through the
``nemo_agent_optimization_strategy`` class variable and delegates ``compile`` and ``run`` to it.

The run converts the stored ``nemo-agents-spec-v1`` config into a Strands agent on the same
model (reached through the Inference Gateway), runs harness-optimizer's rollout-and-reflect loop
over the bundle's dataset, and writes the resulting prompt back as a complete ``agent.yaml``
next to a run summary in the job's results.  The stored agent is never modified.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import yaml
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config
from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy, rewrite_gateway_models
from nemo_agents_plugin.utils import inject_fabric_gateway_url
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
from pydantic import BaseModel
from strands_harness_optimizer_plugin.config import StrandsHarnessConfig
from strands_harness_optimizer_plugin.schemas.optimize import StrandsHarnessOptimizeSpec

logger = logging.getLogger(__name__)

STRATEGY_NAME = "strands-harness-optimizer"

#: Task entry point for the step, in both executor flavours.  The subprocess backend takes one
#: flat command; the cpu backend splits it into a container entrypoint + command.
TASK_MODULE = "strands_harness_optimizer_plugin.tasks.optimize"
TASK_ENTRYPOINT = ["python", "-m"]
TASK_COMMAND = [TASK_MODULE]

#: Image for the cpu (docker / kubernetes_job) fallback.  This plugin is opt-in, so the image a
#: deployment registers under that profile must have it installed for the task to import.
TASK_IMAGE = "nhx-tasks"

RESULT_NAME = "strands_harness_optimizer"
OPTIMIZED_AGENT_FILENAME = "agent.yaml"
SUMMARY_FILENAME = "strands-harness-optimizer-result.json"

SUPPORTED_CONFIG_FORMAT = "nemo-agents-spec-v1"

#: The gateway ignores the bearer the Strands client sends: the loopback proxy authenticates
#: each call with the job's identity, and an auth-disabled platform accepts anything.
GATEWAY_API_KEY = "not-used"


class StrandsHarnessOptimizeJob(NemoJob):
    """Optimize a platform agent's system prompt with Strands Harness Optimizer."""

    name: ClassVar[str] = STRATEGY_NAME
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name=STRATEGY_NAME,
        description="Tune the agent's system prompt with Strands Harness Optimizer's rollout-and-reflect "
        "loop against a labelled dataset (https://github.com/strands-labs/harness-optimizer).",
    )
    description: ClassVar[str] = "Optimize a platform agent's system prompt with Strands Harness Optimizer."
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = StrandsHarnessOptimizeSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        *,
        workspace: str,
        spec: StrandsHarnessOptimizeSpec,
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
        # Lazy so job discovery (`nemo plugins list`, the router) imports this module without Strands.
        from strands_harness_optimizer_plugin.strands_bridge import (
            load_dataset,
            optimize_system_prompt,
            with_system_prompt,
        )

        spec = StrandsHarnessOptimizeSpec.model_validate(config)
        with resolve_staged_config(
            spec.optimize_config,
            spec.optimize_config_fileset,
            workspace=spec.workspace,
            ctx=ctx,
            sdk=sdk,
            kind="strands-harness-optimizer-config",
        ) as yaml_path:
            settings = StrandsHarnessConfig.model_validate(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))
            rows = load_dataset(_dataset_path(yaml_path, settings.dataset))[: settings.max_samples]

        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        source = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)
        agent = AgentConfig.model_validate(inject_fabric_gateway_url(source, spec.workspace))

        logger.info("Optimizing agent %s over %d rows for %d epoch(s)", agent_label, len(rows), settings.epochs)
        with platform_auth_proxy() as origin:
            outcome = optimize_system_prompt(
                source,
                rows,
                base_url=_gateway_base_url(rewrite_gateway_models(agent, origin)),
                api_key=GATEWAY_API_KEY,
                settings=settings,
            )

        results_dir = ctx.storage.ephemeral / STRATEGY_NAME / "results"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True)
        optimized = with_system_prompt(source, outcome.optimized_prompt)
        (results_dir / OPTIMIZED_AGENT_FILENAME).write_text(
            yaml.safe_dump(optimized, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "dataset_size": len(rows),
            "epochs": settings.epochs,
            "epoch_stats": outcome.epoch_stats,
            "original_prompt": outcome.original_prompt,
            "optimized_prompt": outcome.optimized_prompt,
            "optimized_agent_config": OPTIMIZED_AGENT_FILENAME,
        }
        (results_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        result_ref = ctx.results.save(RESULT_NAME, results_dir)
        return {
            "status": "completed",
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "optimized_prompt": outcome.optimized_prompt,
            "result": result_ref.model_dump(mode="json"),
        }


def _dataset_path(yaml_path: Path, dataset: str) -> Path:
    # ``dataset`` comes from the uploaded YAML: keep it inside the YAML's directory so a ``..``
    # or absolute path cannot make the task read arbitrary files from its host.
    root = yaml_path.parent.resolve()
    path = (root / dataset).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"dataset {dataset!r} resolves outside the directory of {yaml_path.name}")
    return path


def _gateway_base_url(agent: AgentConfig) -> str:
    model = agent.models.get("default") or agent.harnesses[agent.default_harness].model
    # A direct provider URL would be called with the placeholder API key and fail every rollout.
    if model is None or not model.base_url or not urlsplit(model.base_url).path.startswith("/apis/inference-gateway/"):
        raise LocalRunError(f"Agent '{agent.name}' has no default model bound to the Inference Gateway.")
    return model.base_url


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    """The stored ``nemo-agents-spec-v1`` config of agent *workspace*/*name*."""
    try:
        config = client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data().config
    except NotFoundError as exc:
        raise LocalRunError(f"Agent '{workspace}/{name}' does not exist; there is nothing to optimize.") from exc
    if config.get("config_format") != SUPPORTED_CONFIG_FORMAT:
        raise LocalRunError(
            f"Agent '{workspace}/{name}' is not a {SUPPORTED_CONFIG_FORMAT!r} agent; this strategy rewrites "
            "instructions.system.content of those only."
        )
    if not config.get("instructions", {}).get("system", {}).get("content", "").strip():
        raise LocalRunError(f"Agent '{workspace}/{name}' has no instructions.system.content to optimize.")
    return config


async def _resolve_executor(*, profile: str, async_sdk: AsyncNemoClient) -> ExecutorSpec:
    """Pick the executor for *profile* from the backends the platform actually registered.

    ``subprocess`` is preferred: it runs in the platform host's own venv, which is where this
    opt-in plugin gets installed.  Deployments without one (Helm / Minikube) get the ``cpu``
    provider, which the platform maps to whichever container backend it registered for that profile.
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
        f"{STRATEGY_NAME} step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )
