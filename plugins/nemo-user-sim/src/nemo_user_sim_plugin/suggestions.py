# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infer which probes fit an agent from what it is: its config and, when it has one, its ETHOS.md.

Customers shouldn't have to know UserSim's probe catalog. The agent's instructions, tools and stated
constraints say what kind of users it will meet and how it could fail; a model maps that onto probes.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from nemo_agents_plugin.entities import ETHOS_FILENAME, Agent, ethos_fileset_name
from nemo_agents_plugin.ethos_parse import EthosParseError, parse_ethos
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_user_sim_plugin.agents import AgentResolutionError
from nemo_user_sim_plugin.config import UserSimConfig
from nemo_user_sim_plugin.probes import PROBES
from nemo_user_sim_plugin.types.endpoints import ProbeSuggestions, Suggestion

logger = logging.getLogger(__name__)

# The Ethos sections that describe how the agent behaves today, not where it's headed or how it's measured.
BEHAVIOR_SECTIONS = ("Role", "Purpose & Outcomes", "Scope", "Tools", "Behavior", "Principles", "Constraints")
MAX_SUGGESTIONS = 3
FALLBACK = {"name": "general_open_ended", "reason": "The agent's files didn't point to a narrower probe."}


def agent_profile(config: dict, ethos: str | None) -> str:
    """What the agent is, in plain text: its instructions, tools and, from ETHOS.md, its behavior sections."""
    instructions = ((config.get("instructions") or {}).get("system") or {}).get("content") or ""
    servers = ", ".join(((config.get("mcp") or {}).get("servers") or {}).keys())
    lines = [f"Agent: {config.get('name', '')}", f"Instructions: {instructions}", f"Tool servers: {servers or 'none'}"]
    if ethos:
        try:
            sections = parse_ethos(ethos).sections
            lines += [
                f"{title}: {sections[title]}"
                for title in BEHAVIOR_SECTIONS
                if sections.get(title, "").strip() not in ("", "_(none)_")
            ]
        except EthosParseError:
            lines.append(f"ETHOS.md: {ethos[:4000]}")
    return "\n".join(lines)


def _prompt(profile: str) -> str:
    catalog = "\n".join(f"- {probe.name}: tests for {probe.tests}. Suits: {probe.suits}." for probe in PROBES)
    return (
        "Pick the probes that would find this agent's most likely failures with real users.\n\n"
        f"{profile}\n\nProbes:\n{catalog}\n\n"
        f"Choose 1 to {MAX_SUGGESTIONS}, most important first. Only pick a probe whose 'tests for' plainly matches "
        "something this agent does. Give each a one-sentence reason that names something specific about this "
        'agent. Answer with JSON only: {"probes": [{"name": "...", "reason": "..."}]}'
    )


async def suggest_probes(profile: str, ask: Callable[[str], Awaitable[str]]) -> list[dict]:
    """Up to three catalog probes with reasons. Never empty: an unusable answer falls back to general questions."""
    known = {probe.name for probe in PROBES}
    try:
        picked = json.loads(await ask(_prompt(profile)))["probes"]
        suggested = [
            {"name": item["name"], "reason": str(item.get("reason", ""))}
            for item in picked
            if item.get("name") in known
        ]
    except Exception as exc:
        logger.warning("Couldn't infer probes for the agent: %s", exc)
        suggested = []
    return suggested[:MAX_SUGGESTIONS] or [FALLBACK]


async def probes_for_agent(client: AsyncNemoClient, entities: Any, workspace: str, agent: str) -> ProbeSuggestions:
    """Read the registered agent and its ETHOS.md, if any, and ask the evaluator model which probes fit."""
    try:
        config = (await entities.get(Agent, name=agent, workspace=workspace)).config
    except Exception as exc:
        raise AgentResolutionError(f"Agent '{agent}' wasn't found in workspace '{workspace}': {exc}") from exc
    try:
        files = AsyncFilesClient.from_client(client)
        response = await files.download_file(workspace=workspace, name=ethos_fileset_name(agent), path=ETHOS_FILENAME)
        ethos = (await response.read()).decode()
    except NotFoundError:  # ETHOS.md is optional
        ethos = None
    settings = UserSimConfig.get()
    gateway = AsyncModelsClient.from_client(client).get_async_openai_client(workspace=workspace)

    async def ask(prompt: str) -> str:
        response = await gateway.chat.completions.create(
            model=settings.evaluator_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body=settings.model_settings.get("evaluator_model", {}).get("extra_body"),
        )
        return response.choices[0].message.content or ""

    suggested = await suggest_probes(agent_profile(config, ethos), ask)
    return ProbeSuggestions(
        agent=agent,
        probes=[Suggestion(**probe) for probe in suggested],
        read=["agent config"] + ([ETHOS_FILENAME] if ethos else []),
    )
