# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builds Fabric agent packages for optimize studies.

No longer resolves agents: platform agent lookup lives on
``nemo_agent_optimization_plugin.job_base.fetch_agent_config``.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_platform_plugin.run_dependencies import LocalRunError

from nemo_optimization.fabric import FABRIC_AGENT_SCHEMA_VERSION, is_fabric_agent_config

logger = logging.getLogger(__name__)

_PLATFORM_AGENT_FORMAT = "nemo-agents-spec-v1"


def to_fabric_agent_package(agent_config: dict[str, Any], *, label: str) -> dict[str, Any]:
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
