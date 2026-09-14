# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform agent resolution for optimize studies."""

from __future__ import annotations

import logging
from typing import Any

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.run_dependencies import LocalRunError

from nemo_optimization.fabric import FABRIC_AGENT_SCHEMA_VERSION, is_fabric_agent_config

logger = logging.getLogger(__name__)

_PLATFORM_AGENT_FORMAT = "nemo-agents-spec-v1"


def resolve_agent_config(
    agent: str | None,
    *,
    workspace: str,
    sdk: NeMoPlatform | None,
) -> dict[str, Any] | None:
    """Fetch a platform-managed agent's config and return a Fabric agent package.

    Stored agents use ``nemo-agents-spec-v1``; optimize requires
    ``fabric.agent/v1alpha1``. Platform specs are translated here.
    """
    if agent is None:
        return None

    if "://" in agent:
        raise LocalRunError(
            "Endpoint URL / URI optimize mode has been removed. Pass a platform-managed "
            "Fabric agent name (e.g. --agent hermes-optimize-chatonly or "
            "--agent default/hermes-optimize-chatonly), not an http(s):// or file:// URL. "
            "Or include an inline Fabric agent package in optimize_config."
        )

    if "/" in agent:
        ws, name = agent.split("/", 1)
    else:
        ws, name = workspace, agent

    if sdk is None:
        raise LocalRunError(
            f"An optimize study with --agent {agent!r} requires a platform SDK to fetch the "
            "stored agent config. Set NEMO_BASE_URL or pass sdk via NemoJobScheduler.run_local(sdk=...)."
        )

    agent_dict = sdk.agents.get(name=name, workspace=ws)
    agent_config = agent_dict["config"] if isinstance(agent_dict, dict) else getattr(agent_dict, "config", {})
    if not isinstance(agent_config, dict) or not agent_config:
        raise RuntimeError(f"Agent '{ws}/{name}' has an empty or invalid stored config; cannot optimize it.")
    logger.info("Resolved agent %r to platform agent %s/%s", agent, ws, name)
    return _to_fabric_agent_package(agent_config, label=f"{ws}/{name}")


def _to_fabric_agent_package(agent_config: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Normalize a stored agent config into a Fabric agent package mapping."""
    if is_fabric_agent_config(agent_config):
        return dict(agent_config)

    config_format = agent_config.get("config_format")
    if config_format != _PLATFORM_AGENT_FORMAT:
        raise LocalRunError(
            f"Agent {label!r} has unsupported config_format {config_format!r}. "
            f"Expected {_PLATFORM_AGENT_FORMAT!r} or schema_version {FABRIC_AGENT_SCHEMA_VERSION!r}."
        )

    try:
        from nemo_agents_plugin.agent_config import AgentConfig
        from nemo_agents_plugin.fabric.gateway_credentials import bind_platform_gateway_model_credential
        from nemo_agents_plugin.fabric.translator import translate_agent_config
    except ImportError as exc:  # pragma: no cover - agents plugin always present for CLI path
        raise LocalRunError(
            "Resolving a platform agent for optimize requires nemo-agents-plugin "
            "(nemo agents optimize / NemoJobScheduler with agents installed)."
        ) from exc

    platform_cfg = AgentConfig.model_validate(agent_config)
    fabric_mapping = translate_agent_config(platform_cfg).to_mapping()
    # Translator emits models.default from the selected harness; keep any extra
    # named models (e.g. judge) from the platform agent for eval overlays.
    extras = {
        name: bind_platform_gateway_model_credential(model.model_dump(exclude_none=True))
        for name, model in platform_cfg.models.items()
        if name != "default"
    }
    if extras:
        fabric_mapping.setdefault("models", {}).update(extras)
    return fabric_mapping
