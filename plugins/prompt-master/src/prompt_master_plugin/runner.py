# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the bundled Prompt Master skill through a Fabric agent."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.invocation import AgentConfigInvocationRequest, invoke_agent_config_request_once
from nemo_agents_plugin.fabric.runtime import FabricRuntimeExecutionError
from nemo_agents_plugin.fabric.translator import FabricTranslationError
from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy, rewrite_gateway_models
from nemo_agents_plugin.utils import inject_fabric_gateway_url
from prompt_master_plugin.skills import skills_dir

logger = logging.getLogger(__name__)

_PROMPT_BLOCK = re.compile(r"```[^\n]*\n(?P<prompt>.*?)\n```", flags=re.DOTALL)
#: The strategy line that closes Prompt Master's reply ("🎯 Target: <tool>,💡 <why>").  The prompt
#: is everything before it; the emoji may carry a variation selector and the label may be bold.
_TARGET_LINE = re.compile(r"🎯\ufe0f?\s*\**\s*Target", flags=re.IGNORECASE)
#: A prompt the model fenced as a whole, so the fence can come off.
_WRAPPED_BLOCK = re.compile(r"\A```[^\n]*\n(?P<prompt>.*)\n```\Z", flags=re.DOTALL)
#: A horizontal rule some models put between the prompt and the strategy line.
_TRAILING_RULE = re.compile(r"\n[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*\Z")

#: How much of an unparseable reply the error message carries.  The job log only ever sees
#: the exception, so this is the one place the model's actual words survive for diagnosis.
_RESPONSE_PREVIEW_CHARS = 300

#: The Fabric local workspace, relative to a run's base directory.
_WORKSPACE = "workspace"
#: Where the vendored skills library is staged inside that workspace, and the same place as the
#: harness names it.  The Deep Agents adapter roots its filesystem at the workspace in virtual
#: mode, so a host path outside it resolves to nothing -- and the harness only warns, then runs
#: on with no skills.  ``.agents/skills`` is the convention the evaluator SDK and the packaging
#: validator already use for workspace-rooted harnesses.
WORKSPACE_SKILLS_DIR = Path(".agents/skills")
WORKSPACE_SKILLS_SOURCE = "/.agents/skills"
#: The optimizer agent.  A run's ``optimize_config`` is a partial file in the same format,
#: deep-merged over it.  Its ``skills.paths`` names :data:`WORKSPACE_SKILLS_SOURCE`.
OPTIMIZER_AGENT_YAML = Path(__file__).with_name("agent.yaml")


class PromptMasterExecutionError(RuntimeError):
    """Raised when Fabric or Prompt Master does not produce an optimized prompt."""


@dataclass(frozen=True, slots=True)
class PromptMasterOutcome:
    """What one Prompt Master run produced."""

    #: The copyable prompt block, ready to become the agent's ``instructions.system.content``.
    optimized_prompt: str
    #: Prompt Master's full reply (strategy line included), kept for the run summary.
    response: str


def build_optimizer_agent(
    overrides: Mapping[str, Any] | None,
    *,
    workspace: str,
    platform_base_url: str | None = None,
) -> AgentConfig:
    """The bundled optimizer agent with *overrides* merged in, bound to *workspace*'s Inference Gateway.

    *overrides* is a partial agent.yaml whose fields replace the bundled ones.  The gateway is
    on the platform at *platform_base_url* (the job's ``NHX_BASE_URL`` when ``None``), and the
    translator hands the harness the placeholder gateway credential, so neither file names a
    provider URL or an API key.
    """
    agent = _merge(yaml.safe_load(OPTIMIZER_AGENT_YAML.read_text(encoding="utf-8")), overrides or {})
    return AgentConfig.model_validate(inject_fabric_gateway_url(agent, workspace, platform_base_url))


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _bundled_skill_library() -> Path:
    """The resolved skills library holding the vendored Prompt Master skill.

    The Deep Agents harness treats a missing skills source as a warning and runs on with no
    skills at all, so a broken path would only surface after a wasted model call as a reply
    without the copyable prompt block.  Refuse up front instead.
    """
    library = skills_dir().resolve()
    manifest = library / "prompt-master" / "SKILL.md"
    if not manifest.is_file():
        raise PromptMasterExecutionError(
            f"The bundled Prompt Master skill is missing: expected {manifest} (prompt-master/SKILL.md under the "
            f"skills library {library}). Reinstall the prompt-master plugin."
        )
    return library


def stage_skills(base_dir: Path) -> Path:
    """Copy the vendored skills library into the Fabric workspace under *base_dir*.

    Returns the staged library, which the harness reaches as :data:`WORKSPACE_SKILLS_SOURCE`.
    A previous staging is replaced wholesale so no run sees stale files.
    """
    destination = base_dir / _WORKSPACE / WORKSPACE_SKILLS_DIR
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(_bundled_skill_library(), destination)
    return destination


def build_optimization_input(agent_config: Mapping[str, Any]) -> str:
    """Build the fully specified one-shot task sent to Prompt Master."""
    prompt = agent_config.get("instructions", {}).get("system", {}).get("content", "")
    if not prompt.strip():
        raise PromptMasterExecutionError("The selected agent must define non-empty instructions.system.content.")
    harness = agent_config["default_harness"]
    model = agent_config.get("models", {}).get("default") or agent_config["harnesses"][harness].get("model", {})
    return (
        "Use the prompt-master skill to improve the existing system prompt below.\n"
        f"Target tool: a NeMo Fabric agent using the {harness} harness and {model.get('model', 'unknown')} model.\n"
        "Preserve the prompt's intent, safety boundaries, and supported capabilities. "
        "Remove ambiguity and wasted tokens; add explicit output, scope, and success criteria only "
        "when they follow from the existing prompt. Do not invent tools, permissions, context, or requirements.\n"
        "This is a non-interactive run. Produce the optimized prompt now using Prompt Master's required output format.\n\n"
        f"<existing_prompt>{prompt}</existing_prompt>"
    )


async def optimize_prompt(
    optimizer: AgentConfig,
    *,
    agent_config: Mapping[str, Any],
    base_dir: Path,
    proxy_origin: str | None = None,
) -> PromptMasterOutcome:
    """Run Prompt Master through Fabric once and return its copyable prompt block.

    *proxy_origin* is the loopback auth proxy the optimizer's inference calls travel through,
    or ``None`` to reach the gateway directly (see :func:`run_prompt_master`).
    """
    stage_skills(base_dir)
    try:
        result = await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(
                agent_config=rewrite_gateway_models(optimizer, proxy_origin),
                input=build_optimization_input(agent_config),
                base_dir=base_dir,
                timeout_seconds=optimizer.runtime.timeout_seconds,
            )
        )
    except (FabricRuntimeExecutionError, FabricTranslationError) as exc:
        raise PromptMasterExecutionError(f"Prompt Master Fabric run failed: {exc}") from exc
    if result.status != "succeeded" or not result.response:
        detail = result.error or result.response or "no response"
        raise PromptMasterExecutionError(f"Prompt Master Fabric run failed: {detail}")
    _log_run(result, result.response)
    return PromptMasterOutcome(optimized_prompt=extract_optimized_prompt(result.response), response=result.response)


def extract_optimized_prompt(response: str) -> str:
    """Extract the optimized prompt from Prompt Master's reply.

    Prompt Master's output format is the prompt followed by a ``🎯 Target: ...,💡 ...`` strategy
    line; no fence is promised.  Everything before the first strategy marker is the prompt.  If
    the model wrapped that in a fenced code block anyway, the fence comes off; fences *inside* an
    unwrapped prompt stay, since they are the prompt's own examples.  Without a strategy line the
    first fenced block is accepted.  Anything else fails, quoting the reply for the job log.
    """
    marker = _TARGET_LINE.search(response)
    if marker is not None:
        head = _TRAILING_RULE.sub("", response[: marker.start()].strip()).strip()
        wrapped = _WRAPPED_BLOCK.match(head)
        prompt = wrapped.group("prompt") if wrapped else head
    else:
        block = _PROMPT_BLOCK.search(response)
        if block is None:
            raise PromptMasterExecutionError(
                "Prompt Master response did not contain a copyable prompt block or a 🎯 Target line. "
                f"Response began: {_preview(response)}"
            )
        prompt = block.group("prompt")
    prompt = prompt.strip()
    if not prompt:
        raise PromptMasterExecutionError("Prompt Master response contained an empty prompt.")
    return prompt


def _log_run(result: Any, response: str) -> None:
    """Record what the optimizer did, so a bad reply can be diagnosed from the job log alone.

    Deep Agents discloses a skill progressively: the model sees its name and description, and
    only learns Prompt Master's output format by reading SKILL.md through a tool call.  Whether
    that read happened is the first question after a reply the parser rejects, and the reply
    itself is the second.  Both are logged before parsing so a failure still leaves them behind.
    """
    reads = _skill_reads(result)
    if reads:
        logger.info("Optimizer read the skill manifest through a tool call: %s", ", ".join(reads))
    else:
        logger.warning(
            "Optimizer never read a SKILL.md through a tool call; Prompt Master's output format may not have "
            "reached the model."
        )
    logger.info("Prompt Master response (%d chars):\n%s", len(response), response)


def _skill_reads(result: Any) -> list[str]:
    """Every SKILL.md path the optimizer named in a tool call, in order, from the run's transcript."""
    output = getattr(result, "output", None)
    messages = output.get("messages") if isinstance(output, Mapping) else None
    reads: list[str] = []
    for message in messages if isinstance(messages, list) else []:
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        for call in calls if isinstance(calls, list) else []:
            args = call.get("args") if isinstance(call, Mapping) else None
            for value in args.values() if isinstance(args, Mapping) else []:
                if isinstance(value, str) and value.endswith("SKILL.md"):
                    reads.append(f"{call.get('name', 'tool')}({value})")
    return reads


def _preview(response: str) -> str:
    """The first :data:`_RESPONSE_PREVIEW_CHARS` of *response* on one line, for an error message."""
    flattened = " ".join(response.split())
    if len(flattened) <= _RESPONSE_PREVIEW_CHARS:
        return flattened
    return flattened[:_RESPONSE_PREVIEW_CHARS].rstrip() + "..."


def run_prompt_master(optimizer: AgentConfig, agent_config: Mapping[str, Any], base_dir: Path) -> PromptMasterOutcome:
    """Execute one Prompt Master run against the target agent's stored config.

    The optimizer model carries no credential of its own, so its inference calls travel
    through a loopback proxy that authenticates each one with the job's identity -- the same
    arrangement ``agents.execute`` jobs use.  On a platform with auth disabled the proxy yields
    no origin and the harness reaches the gateway directly.
    """
    with platform_auth_proxy() as proxy_origin:
        return asyncio.run(
            optimize_prompt(optimizer, agent_config=agent_config, base_dir=base_dir, proxy_origin=proxy_origin)
        )
