# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tool-calling``: score a platform agent's tool calls, diagnose them, and verify a prompt fix and schema patch.

Reached as ``nemo agents optimize run-strategy --strategy tool-calling``. The stored agent is never modified: its
files are staged into the job, every variant runs in-process through Fabric, and the suggestions land in the job's
results (see CONTRACT.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar

import httpx
from nemo_agent_optimization_plugin.schemas.strategies import OptimizationStrategy
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.environment import ensure_local_workspace_dir
from nemo_agents_plugin.jobs.fileset_io import resolve_staged_config
from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy
from nemo_agents_plugin.runner.fabric_artifact_staging import stage_fabric_ethos_dir
from nemo_agents_plugin.utils import get_base_url
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
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.intake.client import IntakeClient
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
from nemo_tool_calling_plugin.analyzer import GatewayAnalyzer
from nemo_tool_calling_plugin.pipeline import PipelineInputs, Prompt, run_pipeline
from nemo_tool_calling_plugin.runner import FabricVariantRunner, fetch_tool_definitions
from nemo_tool_calling_plugin.schemas.optimize import ToolCallingOptimizeSpec
from nemo_tool_calling_plugin.traces import IntakeSpanSource, TracedBaseline, TraceWindow, load_traced_baseline
from pydantic import BaseModel

logger = logging.getLogger(__name__)

STRATEGY_NAME = "tool-calling"
TASK_MODULE = "nemo_tool_calling_plugin.tasks.optimize"
TASK_ENTRYPOINT = ["python", "-m"]
TASK_COMMAND = [TASK_MODULE]
TASK_IMAGE = "nhx-tasks"
RESULT_NAME = "tool_calling"
SUPPORTED_CONFIG_FORMAT = "nemo-agents-spec-v1"
PLATFORM_LIST_TIMEOUT_SECONDS = 15.0


class ToolCallingOptimizeJob(NemoJob):
    """Find and fix the tool calls a platform agent gets wrong."""

    name: ClassVar[str] = STRATEGY_NAME
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name=STRATEGY_NAME,
        title="Tool-call repair",
        description="Score the tool calls in the agent's recent traces, lint its MCP tool schemas, and verify an "
        "LLM-suggested prompt fix and tool-schema patch by replaying the traced prompts.",
    )
    description: ClassVar[str] = "Score a platform agent's tool calls and verify a prompt fix and schema patch."
    generate_legacy_verbs: ClassVar[bool] = False
    spec_schema: ClassVar[type[BaseModel]] = ToolCallingOptimizeSpec

    @classmethod
    async def compile(  # ty: ignore[invalid-method-override]  (narrows the spec types)
        cls,
        *,
        workspace: str,
        spec: ToolCallingOptimizeSpec,
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
        spec = ToolCallingOptimizeSpec.model_validate(config)
        agent_ref = parse_entity_ref(spec.agent, default_workspace=spec.workspace)
        agent_label = f"{agent_ref.workspace}/{agent_ref.name}"
        stored = fetch_agent_config(sdk, workspace=agent_ref.workspace, name=agent_ref.name)

        root = ctx.storage.ephemeral / STRATEGY_NAME
        if root.exists():
            shutil.rmtree(root)
        template_dir = root / "agent"
        template_dir.mkdir(parents=True)
        asyncio.run(_stage_agent_files(sdk, agent_ref.workspace, agent_ref.name, stored, template_dir))
        agent_config = AgentConfig.model_validate(stored)
        workspace_dir = ensure_local_workspace_dir(agent_config, template_dir) or template_dir

        tools, problems = asyncio.run(fetch_tool_definitions(agent_config, cwd=workspace_dir))
        for problem in problems:
            logger.warning(problem)
        logger.info("Agent %s publishes %d tool(s): %s", agent_label, len(tools), [tool.name for tool in tools])

        traced: TracedBaseline | None = None
        if spec.uses_traces:
            traced = load_traced_baseline(
                IntakeSpanSource(client_from_platform(sdk, IntakeClient), workspace=agent_ref.workspace),
                agent_name=agent_config.telemetry.agent_name or agent_config.name,
                tool_names=[tool.name for tool in tools],
                window=TraceWindow.resolve(since=spec.since, until=spec.until, hours=spec.trace_window_hours),
                max_sessions=spec.max_sessions,
                max_replay_prompts=spec.max_replay_prompts,
            )
            prompts = traced.prompts
            logger.info(
                "Scored %d traced session(s); replaying %d distinct prompt(s)", len(traced.sessions), len(prompts)
            )
        else:
            prompts = _load_prompts(spec, ctx=ctx, sdk=sdk)[: spec.max_prompts]

        results_dir = root / "results"
        with platform_auth_proxy() as origin:
            platform_url = origin or get_base_url()
            gateway_url = f"{platform_url}/apis/inference-gateway/v2/workspaces/{spec.workspace}/openai/-/v1"
            models, providers = _platform_models(platform_url, spec.workspace)
            runner = FabricVariantRunner(
                template_dir=template_dir,
                work_root=root / "runs",
                workspace=spec.workspace,
                runs_per_prompt=spec.runs_per_prompt,
                max_turns=spec.max_turns,
                timeout_seconds=spec.timeout_seconds,
                concurrency=spec.concurrency,
                proxy_origin=origin,
                tool_names={tool.name for tool in tools},
            )
            result = run_pipeline(
                PipelineInputs(
                    agent=agent_label,
                    agent_config=stored,
                    prompts=prompts,
                    runs_per_prompt=spec.runs_per_prompt,
                    tools=tools,
                    overlay_dir=root / "overlays",
                    models=models,
                    providers=providers,
                    verify=tuple(spec.verify),
                    call_budget=spec.call_budget,
                    traced_baseline=traced.runs if traced is not None else None,
                    trace_window=traced.window.to_json() if traced is not None else None,
                ),
                runner=runner,
                analyzer=GatewayAnalyzer(base_url=gateway_url, model=spec.analyzer_model),
                results_dir=results_dir,
            )

        result_ref = ctx.results.save(RESULT_NAME, results_dir)
        return {
            "status": "completed",
            "strategy": STRATEGY_NAME,
            "agent": agent_label,
            "recommendation": result["recommendation"],
            "variants": {variant["id"]: variant["metrics"] for variant in result["variants"]},
            "result": result_ref.model_dump(mode="json"),
        }


def parse_prompts(lines: list[str]) -> list[Prompt]:
    prompts: list[Prompt] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("prompt") or row.get("input") or row.get("instruction")
        if not isinstance(text, str) or not text.strip():
            raise LocalRunError(f"Dataset line {number} has no 'prompt'.")
        prompts.append(Prompt(id=str(row.get("id") or f"prompt-{number}"), prompt=text))
    if not prompts:
        raise LocalRunError("The prompt dataset is empty.")
    return prompts


def _load_prompts(spec: ToolCallingOptimizeSpec, *, ctx: JobContext, sdk: NemoClient) -> list[Prompt]:
    if spec.prompts is not None:
        return [Prompt(id=row.id, prompt=row.prompt) for row in spec.prompts]
    if spec.optimize_config is None:
        raise LocalRunError("No prompt dataset was given.")
    with resolve_staged_config(
        spec.optimize_config,
        spec.optimize_config_fileset,
        workspace=spec.workspace,
        ctx=ctx,
        sdk=sdk,
        kind="tool-calling-dataset",
    ) as path:
        return parse_prompts(path.read_text(encoding="utf-8").splitlines())


def fetch_agent_config(sdk: NemoClient, *, workspace: str, name: str) -> dict[str, Any]:
    try:
        config = client_from_platform(sdk, AgentsClient).get_agent(name=name, workspace=workspace).data().config
    except NotFoundError as exc:
        raise LocalRunError(f"Agent '{workspace}/{name}' does not exist.") from exc
    if config.get("config_format") != SUPPORTED_CONFIG_FORMAT:
        raise LocalRunError(f"Agent '{workspace}/{name}' is not a {SUPPORTED_CONFIG_FORMAT!r} agent.")
    return config


async def _stage_agent_files(
    sdk: NemoClient, workspace: str, name: str, config: dict[str, Any], base_dir: Path
) -> None:
    async with sdk.to_async() as async_client:
        await stage_fabric_ethos_dir(
            workspace=workspace,
            agent_name=name,
            agent_config=config,
            base_dir=base_dir,
            files_client=AsyncFilesClient.from_client(async_client),
        )


def _platform_models(platform_url: str, workspace: str) -> tuple[list[str], list[str]]:
    """Served gateway model IDs and registered provider names, for the analyzer. Best effort."""
    paths = {
        "models": (f"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models", "id"),
        "providers": (f"/apis/models/v2/workspaces/{workspace}/providers?page_size=1000", "name"),
    }
    found: dict[str, list[str]] = {}
    with httpx.Client(timeout=PLATFORM_LIST_TIMEOUT_SECONDS) as client:
        for label, (path, key) in paths.items():
            try:
                response = client.get(f"{platform_url}{path}")
                response.raise_for_status()
                found[label] = sorted(str(item[key]) for item in response.json().get("data", []) if key in item)
            except (httpx.HTTPError, ValueError) as error:
                logger.warning("Could not list platform %s: %s", label, error)
                found[label] = []
    return found["models"], found["providers"]


async def _resolve_executor(*, profile: str, async_sdk: AsyncNemoClient) -> ExecutorSpec:
    """``subprocess`` when the platform registered one for *profile* (this plugin lives in the host venv), else ``cpu``."""
    try:
        profiles = (await client_from_platform(async_sdk, AsyncJobsClient).get_execution_profiles()).data()
    except (NemoTransportError, NemoResponseValidationError, InternalServerError) as exc:
        raise HelixJobDependencyUnavailableError(
            f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily unavailable."
        ) from exc

    if any(
        isinstance(candidate, SubprocessJobExecutionProfile) and candidate.profile == profile for candidate in profiles
    ):
        return SubprocessExecutionProviderSpec(
            provider="subprocess", profile=profile, command=[*TASK_ENTRYPOINT, *TASK_COMMAND]
        )
    if any(candidate.provider == "cpu" and candidate.profile == profile for candidate in profiles):
        return CPUExecutionProviderSpec(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(
                image=get_qualified_image(TASK_IMAGE), entrypoint=TASK_ENTRYPOINT, command=TASK_COMMAND
            ),
        )
    available = sorted({f"{candidate.provider}/{candidate.profile}" for candidate in profiles})
    raise HelixJobCompilationError(
        f"No 'subprocess' or 'cpu' execution profile named {profile!r} is registered, so the {STRATEGY_NAME} step "
        f"has nowhere to run. Available profiles: {available or ['<none>']}."
    )
