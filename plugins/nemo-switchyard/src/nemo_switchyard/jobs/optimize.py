# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``switchyard`` -- route a platform agent across model pairs with Switchyard VirtualModels.

Reached as ``nemo agents optimize run-strategy --strategy switchyard``: the router job in
nemo-agent-optimization-plugin finds this class through the ``nemo_agent_optimization_strategy``
class variable and delegates ``compile`` and ``run`` to it.

For every (capable, efficient) pair of the acceptable models and every routing strategy the run
creates one Inference Gateway VirtualModel carrying a single ``nemo-switchyard`` request
middleware call, then saves a copy of the agent's ``nemo-agents-spec-v1`` config pointed at that
VirtualModel into the job's results.  The stored agent is never modified and no VirtualModel is
deleted; one that already exists is accepted only when it routes exactly as requested.

Keep this module light: it is imported by every ``discover_jobs()`` call, so it must not pull in
``nemo_switchyard.middleware`` (Rust bindings) or ``nemo_agents_plugin``.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from typing import Any, ClassVar

import yaml
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
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
from nemo_helix_plugin.inference_middleware_models import MiddlewareCall, VirtualModelInferenceConfig
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
from nemo_helix_plugin.virtual_models.client import VirtualModelsClient
from nemo_helix_plugin.virtual_models.types import CreateVirtualModelRequest
from nemo_switchyard.routing import Combination, build_combinations, rewrite_agent_config
from nemo_switchyard.schemas.optimize import SwitchyardOptimizeSpec
from pydantic import BaseModel

logger = logging.getLogger(__name__)

STRATEGY_NAME = "switchyard"
MIDDLEWARE_NAME = "nemo-switchyard"

#: Task entry point for the step, in both executor flavours.  The subprocess backend takes one
#: flat command; the cpu backend splits it into a container entrypoint + command.
TASK_MODULE = "nemo_switchyard.tasks.optimize"
TASK_ENTRYPOINT = ["python", "-m"]
TASK_COMMAND = [TASK_MODULE]
TASK_IMAGE = "nhx-tasks"

RESULT_NAME = "switchyard"
INDEX_FILENAME = "switchyard-result.json"

SUPPORTED_CONFIG_FORMAT = "nemo-agents-spec-v1"


class SwitchyardOptimizeJob(NemoJob):
    """Create Switchyard VirtualModels over model pairs and emit an agent config per route."""

    name: ClassVar[str] = STRATEGY_NAME
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name=STRATEGY_NAME,
        description="Route the agent across pairs of acceptable models with Switchyard VirtualModels, "
        "one agent config per model pair and routing strategy.",
    )
    description: ClassVar[str] = "Build Switchyard-routed variants of a platform agent."
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = SwitchyardOptimizeSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        *,
        workspace: str,
        spec: SwitchyardOptimizeSpec,
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
        spec = SwitchyardOptimizeSpec.model_validate(config)
        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        source = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)
        combos = build_combinations(spec, agent_name=agent_ref.name)
        # Rewritten before the loop so an agent config with nothing to route creates no VirtualModel.
        rewritten = [rewrite_agent_config(source, f"{spec.workspace}/{combo.virtual_model}") for combo in combos]

        virtual_models_client = client_from_platform(sdk, VirtualModelsClient)
        results_dir = ctx.storage.ephemeral / STRATEGY_NAME / "results"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True)

        index = []
        for combo, agent_config in zip(combos, rewritten, strict=True):
            logger.info("Creating VirtualModel %s/%s (%s)", spec.workspace, combo.virtual_model, combo.config_type)
            ensure_virtual_model(virtual_models_client, workspace=spec.workspace, combo=combo)
            filename = f"agent-{combo.virtual_model}.yaml"
            (results_dir / filename).write_text(yaml.safe_dump(agent_config, sort_keys=False), encoding="utf-8")
            index.append({**asdict(combo), "agent_config": filename})
        summary = {"strategy": STRATEGY_NAME, "agent": agent_label, "workspace": spec.workspace, "combinations": index}
        (results_dir / INDEX_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        result_ref = ctx.results.save(RESULT_NAME, results_dir)
        return {
            "status": "completed",
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "virtual_models": [combo.virtual_model for combo in combos],
            "result": result_ref.model_dump(mode="json"),
        }


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    try:
        config = client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data().config
    except NotFoundError as exc:
        raise LocalRunError(f"Agent '{workspace}/{name}' does not exist; there is nothing to route.") from exc
    if config.get("config_format") != SUPPORTED_CONFIG_FORMAT:
        raise LocalRunError(
            f"Agent '{workspace}/{name}' is not a {SUPPORTED_CONFIG_FORMAT!r} agent; the switchyard strategy "
            "rewrites the model blocks of those only."
        )
    return config


def ensure_virtual_model(client: VirtualModelsClient, *, workspace: str, combo: Combination) -> None:
    """Create the VirtualModel; accept an existing one only when it already routes exactly as requested."""
    request = virtual_model_request(combo)
    existing = client.create_virtual_model(workspace=workspace, body=request, exist_ok=True).data()
    if [m.model for m in existing.models] != combo.models or existing.request_middleware != request.request_middleware:
        raise LocalRunError(
            f"VirtualModel '{workspace}/{combo.virtual_model}' already exists but routes differently from this "
            f"request; delete it (`nemo inference virtual-models delete {combo.virtual_model}`) or change the agent "
            "or model list, then re-run."
        )


def virtual_model_request(combo: Combination) -> CreateVirtualModelRequest:
    return CreateVirtualModelRequest(
        name=combo.virtual_model,
        models=[VirtualModelInferenceConfig(model=ref) for ref in combo.models],
        request_middleware=[MiddlewareCall(name=MIDDLEWARE_NAME, config_type=combo.config_type, config=combo.config)],
    )


async def _resolve_executor(*, profile: str, async_sdk: AsyncNemoClient) -> ExecutorSpec:
    """Prefer the platform host's own venv (subprocess); fall back to the cpu container profile."""
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
        f"switchyard step has nowhere to run.  Available profiles: {available or ['<none>']}."
    )
