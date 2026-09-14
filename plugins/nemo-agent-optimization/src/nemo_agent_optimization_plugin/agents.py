# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform agent resolution for optimize studies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nemo_optimization.agents import _to_fabric_agent_package
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.run_dependencies import LocalRunError

logger = logging.getLogger(__name__)


def resolve_agent_config(
    agent: str | None,
    *,
    workspace: str,
    sdk: NeMoPlatform | None,
) -> dict[str, Any] | None:
    """Load a local or platform-managed agent config as a Fabric package.

    Local ``agent.yaml`` files and stored agents use ``nemo-agents-spec-v1``;
    optimize strategies consume ``fabric.agent/v1alpha1``.
    """
    if agent is None:
        return None

    local_path = Path(agent).expanduser()
    if local_path.is_file():
        try:
            from nemo_agents_plugin.agent_config import load_agent_config
        except ImportError as exc:  # pragma: no cover - agents plugin always present for CLI path
            raise LocalRunError("Loading a local agent.yaml requires nemo-agents-plugin.") from exc
        local_config = load_agent_config(local_path).model_dump(mode="json", exclude_none=True)
        logger.info("Resolved agent %r from local config %s", agent, local_path)
        return _to_fabric_agent_package(local_config, label=str(local_path))

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
