# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which agent installs can be done once per environment image and reused, and the key each is reused under.

An *agent layer* is an environment image with an agent already installed. Every trial of that environment with that
agent can start from it: Harbor's installed agents, and :class:`FabricInstalledAgent`, find the install done and skip
it. Local Docker runs and Kubernetes runs build their layers differently but key them the same way:
:meth:`AgentInstall.layer_key` over the environment image and :func:`agent_install`'s description of the install.

Only what changes the installed files is in the key: the agent, its version and install options, and the Harbor
version whose install code runs. The model, prompts, turn limits and secrets aren't, so trials that differ only in
those share a layer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import JsonValue

#: The SDK's installed Fabric agent, which a registered platform agent runs as.
FABRIC_INSTALLED_AGENT = "nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent:FabricInstalledAgent"
#: Bumped when the way a layer is built changes what it holds, so older layers are no longer reused.
LAYER_FORMAT = 1


@dataclass(frozen=True)
class AgentInstall:
    """One agent install that can be reused: which agent, and every input that changes what it installs."""

    agent: str
    """``harbor:<name>`` for a Harbor installed agent, ``fabric`` for :class:`FabricInstalledAgent`."""
    inputs: Mapping[str, str] = field(default_factory=dict)
    harbor_version: str = ""

    def layer_key(self, environment_image: str) -> str:
        """The key of this install's layer on ``environment_image``, an image ID or digest."""
        document = {
            "format": LAYER_FORMAT,
            "environment": environment_image,
            "agent": self.agent,
            "inputs": dict(sorted(self.inputs.items())),
            "harbor": self.harbor_version,
        }
        return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:24]


def agent_install(
    *,
    agent_name: str | None,
    agent_import_path: str | None,
    agent_dir: Path | None,
    agent_kwargs: Mapping[str, JsonValue],
    today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
) -> AgentInstall | None:
    """The reusable install a Harbor run's agent does in setup, or None when it can't be reused.

    Harbor's installed agents (``codex``, ``claude-code``, ...) and the registered-agent :class:`FabricInstalledAgent`
    can be reused. Agents that install nothing (``oracle``, ``nop``), a loose ``agent_dir``, and any other
    ``import_path`` can't: their setup may do anything, so it runs in every trial.

    An agent whose ``version`` isn't pinned installs whatever is newest, so its layer is keyed by the UTC date: it is
    reused for a day, then installed again.
    """
    harbor_version = importlib.metadata.version("harbor")
    if agent_dir is not None:
        return None
    if agent_import_path is not None:
        if agent_import_path != FABRIC_INSTALLED_AGENT:
            return None
        return _fabric_install(agent_kwargs, harbor_version)
    if agent_name is None or not _is_installed_agent(agent_name):
        return None
    version = agent_kwargs.get("version")
    pinned = str(version) if version is not None else f"unpinned:{today().isoformat()}"
    return AgentInstall(agent=f"harbor:{agent_name}", inputs={"version": pinned}, harbor_version=harbor_version)


def _fabric_install(agent_kwargs: Mapping[str, JsonValue], harbor_version: str) -> AgentInstall | None:
    from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent import (
        DEFAULT_FABRIC_PYTHON_VERSION,
        DEFAULT_UV_VERSION,
        fabric_install_spec,
    )

    package = agent_kwargs.get("fabric_package")
    if not isinstance(package, str) or not package.strip():
        return None
    spec = fabric_install_spec(
        package,
        str(agent_kwargs.get("fabric_python_version") or DEFAULT_FABRIC_PYTHON_VERSION),
        str(agent_kwargs.get("fabric_uv_version") or DEFAULT_UV_VERSION),
    )
    return AgentInstall(agent="fabric", inputs={"install": spec}, harbor_version=harbor_version)


def _is_installed_agent(agent_name: str) -> bool:
    """Whether Harbor's built-in ``agent_name`` installs itself in setup, which is what a layer can do ahead of time."""
    from harbor.agents.factory import AgentFactory
    from harbor.agents.installed.base import BaseInstalledAgent
    from harbor.models.trial.config import AgentConfig

    try:
        agent_class = AgentFactory.get_agent_class_from_config(AgentConfig(name=agent_name))
    except (ValueError, KeyError):
        return False
    return issubclass(agent_class, BaseInstalledAgent)
