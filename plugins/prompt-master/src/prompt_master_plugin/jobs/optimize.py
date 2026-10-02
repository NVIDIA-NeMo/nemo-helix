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
(``agent.yaml``) next to a run summary, registered as the job's result and -- when ``output``
names one -- published to a fileset or local directory.  The stored agent is never modified.
"""

from __future__ import annotations

import copy
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import yaml
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config, upload_to_fileset
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
from nemo_helix_plugin.refs import FilesetRef, LocalDir, classify_output_target, parse_entity_ref
from prompt_master_plugin.config import PromptMasterConfig, load_prompt_master_config
from prompt_master_plugin.runner import PromptMasterOutcome, run_prompt_master
from prompt_master_plugin.schemas.optimize import (
    FILESET_REQUIRED,
    PromptMasterOptimizeSpec,
    PromptMasterOptimizeSubmitSpec,
)
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
    #: Marks this job as an agent optimization strategy, names it for
    #: ``nemo agents optimize run-strategy --strategy``, and says what it optimizes for
    #: ``list-strategies``.  Declaring the variable is the whole contract -- nothing to
    #: subclass, and no strategy-specific entry-point group to join.
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name=STRATEGY_NAME,
        description="Rewrite the agent's system prompt with the Prompt Master skill in one Fabric run.",
    )
    description: ClassVar[str] = "Optimize a platform agent's system prompt with Prompt Master."
    container: ClassVar[str] = "cpu-tasks"
    job_collection_path: ClassVar[str | None] = None
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = PromptMasterOptimizeSpec
    input_spec_schema: ClassVar[type[BaseModel]] = PromptMasterOptimizeSubmitSpec

    @classmethod
    async def to_spec(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        input_spec: PromptMasterOptimizeSubmitSpec,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncNemoClient,
        is_local: bool,
    ) -> PromptMasterOptimizeSpec:
        del entity_client, async_sdk, is_local
        payload = input_spec.model_dump(mode="json")
        payload["workspace"] = workspace
        return PromptMasterOptimizeSpec.model_validate(payload)

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
        # ``compile`` is the remote submission path only, so requiring the fileset here keeps
        # platform execution remote-safe even for a caller that bypassed the submit schema.
        if spec.optimize_config_fileset is None:
            raise HelixJobCompilationError(FILESET_REQUIRED)

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

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient | None = None) -> dict[str, Any]:
        spec = PromptMasterOptimizeSpec.model_validate(config)
        if sdk is None:
            raise LocalRunError(
                "The prompt-master strategy needs a platform client to read the agent it optimizes. "
                "Set NHX_BASE_URL, or pass sdk=... to NemoJobScheduler.run_local."
            )

        with resolve_staged_config(
            spec.optimize_config,
            spec.optimize_config_fileset,
            workspace=spec.workspace,
            ctx=ctx,
            sdk=sdk,
            kind="prompt-master-config",
        ) as config_path:
            optimizer = load_prompt_master_config(config_path)

        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        source = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)

        runtime_dir = ctx.storage.ephemeral / STRATEGY_NAME / "fabric"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running Prompt Master on agent %s with optimizer model %s", agent_label, optimizer.model.model)
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
        published = publish(results_dir, spec.output, workspace=spec.workspace, sdk=sdk)

        result: dict[str, Any] = {
            "status": "completed",
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "optimized_prompt": outcome.optimized_prompt,
            "result": result_ref.model_dump(mode="json"),
        }
        if published is not None:
            result["output"] = published
        return result


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    """The stored ``nemo-agents-spec-v1`` config of agent *workspace*/*name*."""
    try:
        agent = client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data()
    except NotFoundError as exc:
        raise LocalRunError(f"Agent '{workspace}/{name}' does not exist; there is nothing to optimize.") from exc
    config = agent.config
    if not isinstance(config, Mapping) or not config:
        raise LocalRunError(f"Agent '{workspace}/{name}' has an empty or invalid stored config; cannot optimize it.")
    config_format = config.get("config_format")
    if config_format != SUPPORTED_CONFIG_FORMAT:
        raise LocalRunError(
            f"Agent '{workspace}/{name}' has config_format {config_format!r}; the prompt-master strategy "
            f"rewrites instructions.system.content of {SUPPORTED_CONFIG_FORMAT!r} agents only."
        )
    logger.info("Resolved agent %s/%s", workspace, name)
    return dict(config)


def with_system_prompt(agent_config: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    """A deep copy of *agent_config* whose ``instructions.system.content`` is *prompt*."""
    optimized = copy.deepcopy(dict(agent_config))
    optimized.setdefault("instructions", {}).setdefault("system", {})["content"] = prompt
    return optimized


def write_artifacts(
    results_dir: Path,
    *,
    agent: str,
    optimizer: PromptMasterConfig,
    source: Mapping[str, Any],
    optimized: Mapping[str, Any],
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
    (results_dir / OPTIMIZED_AGENT_FILENAME).write_text(
        yaml.safe_dump(dict(optimized), sort_keys=False),
        encoding="utf-8",
    )
    summary = {
        "strategy": STRATEGY_NAME,
        "agent": agent,
        "optimizer_model": optimizer.model.model_dump(exclude_none=True),
        "original_prompt": _system_prompt(source),
        "optimized_prompt": outcome.optimized_prompt,
        "optimized_agent_config": OPTIMIZED_AGENT_FILENAME,
        "response": outcome.response,
    }
    (results_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _system_prompt(agent_config: Mapping[str, Any]) -> str | None:
    instructions = agent_config.get("instructions")
    system = instructions.get("system") if isinstance(instructions, Mapping) else None
    content = system.get("content") if isinstance(system, Mapping) else None
    return content if isinstance(content, str) else None


def publish(results_dir: Path, output: str | None, *, workspace: str, sdk: NemoClient) -> dict[str, str] | None:
    """Copy the artifacts to *output* and return a pointer for the job result.

    The job's own results (``ctx.results.save``) already land in the job's fileset on the
    platform; *output* is the stable, caller-named location a remote client can read back
    or hand to a follow-up command.  Returns ``None`` when no target was requested.
    """
    if output is None:
        return None

    if classify_output_target(output) is LocalDir:
        local = Path(output).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        shutil.copytree(results_dir, local, dirs_exist_ok=True)
        logger.info("Published Prompt Master results to local dir %s", local)
        return {"type": "local_dir", "path": str(local)}

    ref = parse_entity_ref(FilesetRef(output), default_workspace=workspace)
    upload_to_fileset(results_dir, fileset=ref.name, workspace=ref.workspace, sdk=sdk)
    logger.info("Published Prompt Master results to fileset %s/%s", ref.workspace, ref.name)
    return {"type": "fileset", "fileset": f"{ref.workspace}/{ref.name}"}


def _profiles_unavailable(profile: str) -> HelixJobDependencyUnavailableError:
    """A retryable failure while resolving the backend for *profile*."""
    return HelixJobDependencyUnavailableError(
        f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
        "unavailable.  Retry the submission."
    )


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
        raise _profiles_unavailable(profile) from exc

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
