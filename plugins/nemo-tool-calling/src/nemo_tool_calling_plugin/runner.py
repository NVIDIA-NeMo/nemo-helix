# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run agent-config variants in-process through Fabric and capture every tool call from the Relay stream."""

from __future__ import annotations

import asyncio
import copy
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.environment import ensure_local_workspace_dir
from nemo_agents_plugin.fabric.mcp_status import probe_mcp_server
from nemo_agents_plugin.fabric.runtime import FabricOneShotRequest, StreamingMethod, stream_fabric_agent_once
from nemo_agents_plugin.fabric.streaming import resolve_streaming_method
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_agents_plugin.fabric.validation import plan_fabric_config
from nemo_agents_plugin.jobs.gateway_proxy import rewrite_gateway_models, rewrite_intake_otlp_endpoints
from nemo_agents_plugin.utils import inject_fabric_gateway_url
from nemo_helix_plugin.agents.post_run_actions import FabricRuntimeResult
from nemo_tool_calling_plugin.analyzer import ToolDefinition
from nemo_tool_calling_plugin.pipeline import Prompt, VariantId
from nemo_tool_calling_plugin.scoring import RunRecord, calls_from_messages, calls_from_relay_records, only_tools

logger = logging.getLogger(__name__)

SUCCEEDED = "succeeded"


def with_max_turns(config: dict[str, Any], max_turns: int) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    runtime = updated.get("runtime")
    updated["runtime"] = {**(runtime if isinstance(runtime, dict) else {}), "max_turns": max_turns}
    return updated


def _error_text(result: FabricRuntimeResult) -> str:
    error = result.error
    if isinstance(error, Mapping):
        return f"{error.get('code') or result.status}: {error.get('message') or ''}".strip()
    return str(error or result.status)


class FabricVariantRunner:
    """Run each prompt ``runs_per_prompt`` times per variant, each run in a fresh copy of the staged agent files."""

    def __init__(
        self,
        *,
        template_dir: Path,
        work_root: Path,
        workspace: str,
        runs_per_prompt: int,
        max_turns: int,
        timeout_seconds: float,
        concurrency: int,
        proxy_origin: str | None,
        tool_names: set[str],
    ) -> None:
        self._template_dir = template_dir
        self._work_root = work_root
        self._workspace = workspace
        self._runs_per_prompt = runs_per_prompt
        self._max_turns = max_turns
        self._timeout_seconds = timeout_seconds
        self._concurrency = concurrency
        self._proxy_origin = proxy_origin
        self._tool_names = tool_names

    def runtime_config(self, agent_config: dict[str, Any]) -> AgentConfig:
        resolved = inject_fabric_gateway_url(with_max_turns(agent_config, self._max_turns), self._workspace)
        config = AgentConfig.model_validate(resolved)
        return rewrite_intake_otlp_endpoints(rewrite_gateway_models(config, self._proxy_origin), self._proxy_origin)

    def run(self, variant: VariantId, agent_config: dict[str, Any], prompts: Sequence[Prompt]) -> list[RunRecord]:
        return asyncio.run(self._run_all(variant, self.runtime_config(agent_config), prompts))

    async def _run_all(self, variant: VariantId, config: AgentConfig, prompts: Sequence[Prompt]) -> list[RunRecord]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def bounded(prompt: Prompt, attempt: int) -> RunRecord:
            async with semaphore:
                return await self._run_one(variant, config, prompt, attempt)

        jobs = [bounded(prompt, attempt) for prompt in prompts for attempt in range(1, self._runs_per_prompt + 1)]
        return list(await asyncio.gather(*jobs))

    async def _run_one(self, variant: VariantId, config: AgentConfig, prompt: Prompt, attempt: int) -> RunRecord:
        base_dir = self._work_root / variant / f"{prompt.id}-{attempt}"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        await asyncio.to_thread(shutil.copytree, self._template_dir, base_dir)
        records: list[Mapping[str, Any]] = []
        result: FabricRuntimeResult | None = None
        error: str | None = None
        try:
            result = await self._stream(config, base_dir, prompt.prompt, records)
            if result.status != SUCCEEDED:
                error = _error_text(result)
        except Exception as exc:  # noqa: BLE001 -- one failed run is a scored outcome, not a failed job
            error = f"{type(exc).__name__}: {exc}"
        calls = calls_from_relay_records(records)
        output = result.output if result is not None else None
        if not calls and isinstance(output, Mapping):
            calls = calls_from_messages(output.get("messages") or [])
        run = RunRecord(prompt_id=prompt.id, attempt=attempt, calls=only_tools(calls, self._tool_names), error=error)
        logger.info(
            "[%s] %s #%d: %d call(s), %d rejected%s",
            variant,
            prompt.id,
            attempt,
            len(run.calls),
            sum(call.tool_error for call in run.calls),
            f", error: {error}" if error else "",
        )
        return run

    async def _stream(
        self, config: AgentConfig, base_dir: Path, prompt: str, records: list[Mapping[str, Any]]
    ) -> FabricRuntimeResult:
        fabric_config = translate_agent_config(config, proxy_origin=self._proxy_origin)
        await asyncio.to_thread(ensure_local_workspace_dir, config, base_dir)
        plan = await plan_fabric_config(fabric_config, base_dir=base_dir)
        method = resolve_streaming_method(plan, fabric_config)
        if method is StreamingMethod.RELAY:
            fabric_config = fabric_config.model_copy(deep=True).enable_relay()
        request = FabricOneShotRequest(
            fabric_config=fabric_config, base_dir=base_dir, input=prompt, timeout_seconds=self._timeout_seconds
        )
        async with asyncio.timeout(self._timeout_seconds):
            async with stream_fabric_agent_once(request, streaming_method=method) as stream:
                async for record in stream.records():
                    records.append(record)
                return await stream.result()


async def fetch_tool_definitions(config: AgentConfig, *, cwd: Path) -> tuple[list[ToolDefinition], list[str]]:
    """The tools each declared MCP server publishes, probed the way the runtime launches them."""
    servers = config.mcp.servers if config.mcp is not None else {}
    tools: list[ToolDefinition] = []
    problems: list[str] = []
    for name, spec in servers.items():
        status = await probe_mcp_server(name, spec, config, cwd=cwd)
        if status.state != "running":
            problems.append(
                f"MCP server {name!r} is {status.state}: {status.error or ''} {status.stderr or ''}".strip()
            )
            continue
        tools.extend(
            ToolDefinition(
                server=name, name=tool.name, description=tool.description or "", input_schema=tool.input_schema
            )
            for tool in status.tools
            if tool.exposed
        )
    return tools, problems
