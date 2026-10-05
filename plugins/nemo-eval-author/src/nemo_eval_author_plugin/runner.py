# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the bundled eval-author-first-eval skill once through a Fabric Deep Agents run."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.invocation import AgentConfigInvocationRequest, invoke_agent_config_request_once
from nemo_agents_plugin.fabric.runtime import FabricRuntimeExecutionError
from nemo_agents_plugin.fabric.translator import FabricTranslationError
from nemo_agents_plugin.jobs.gateway_proxy import platform_auth_proxy, rewrite_gateway_models
from nemo_agents_plugin.utils import inject_fabric_gateway_url

#: The author agent.  A run's ``author_config`` is a partial file in the same format, deep-merged over it.
AUTHOR_AGENT_YAML = Path(__file__).with_name("agent.yaml")
#: The vendored skills library: a directory of skill directories, each holding a SKILL.md.
SKILLS_DIR = Path(__file__).with_name("vendor") / "skills"
#: Where the library is staged inside the Fabric workspace; agent.yaml's ``skills.paths`` names it
#: from the workspace root, which the Deep Agents backend treats as ``/``.
WORKSPACE_SKILLS_DIR = Path(".agents/skills")


class EvalAuthorExecutionError(RuntimeError):
    """Raised when the Fabric run does not complete."""


def build_author_agent(
    overrides: Mapping[str, Any] | None, *, workspace: str, platform_base_url: str | None = None
) -> AgentConfig:
    """The bundled author agent with *overrides* merged in, bound to *workspace*'s Inference Gateway."""
    agent = _merge(yaml.safe_load(AUTHOR_AGENT_YAML.read_text(encoding="utf-8")), overrides or {})
    return AgentConfig.model_validate(inject_fabric_gateway_url(agent, workspace, platform_base_url))


def _merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def stage_skills(workspace_dir: Path) -> Path:
    """Copy the vendored skills library into the fresh *workspace_dir*."""
    destination = workspace_dir / WORKSPACE_SKILLS_DIR
    shutil.copytree(SKILLS_DIR, destination)
    return destination


def build_task_input(agent_label: str) -> str:
    return (
        f"Use the eval-author-first-eval skill to author the first evaluation suite for the NeMo Helix "
        f"agent '{agent_label}'. Its configuration is /agent.yaml and its Ethos, if one exists, is /ETHOS.md; "
        "any other files in the workspace are the agent's repository. "
        "This is a non-interactive run: do not ask questions, make reasonable assumptions, record them, "
        "and finish with a short summary of what you produced."
    )


async def _invoke(author: AgentConfig, *, input: str, base_dir: Path, proxy_origin: str | None) -> str:
    try:
        result = await invoke_agent_config_request_once(
            AgentConfigInvocationRequest(
                agent_config=rewrite_gateway_models(author, proxy_origin),
                input=input,
                base_dir=base_dir,
                timeout_seconds=author.runtime.timeout_seconds,
            )
        )
    except (FabricRuntimeExecutionError, FabricTranslationError) as exc:
        raise EvalAuthorExecutionError(f"Eval Author Fabric run failed: {exc}") from exc
    if result.status != "succeeded":
        raise EvalAuthorExecutionError(
            f"Eval Author Fabric run failed: {result.error or result.response or 'no response'}"
        )
    return str(result.response or "")


def run_author(author: AgentConfig, *, input: str, base_dir: Path) -> str:
    """Run the author once and return its final response text.

    The author model carries no credential of its own, so its inference calls travel through a
    loopback proxy that authenticates each one with the job's identity, as ``agents.execute`` does.
    """
    with platform_auth_proxy() as proxy_origin:
        return asyncio.run(_invoke(author, input=input, base_dir=base_dir, proxy_origin=proxy_origin))
