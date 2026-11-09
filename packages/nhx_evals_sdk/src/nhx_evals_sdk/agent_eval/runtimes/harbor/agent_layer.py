# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which agent installs can be done once per environment image and reused, and the key each is reused under.

An *agent layer* is an environment image with an agent already installed. Every trial of that environment with that
agent can start from it: Harbor's installed agents, and :class:`FabricInstalledAgent`, find the install done and skip
it. Local Docker runs and Kubernetes runs build their layers differently but key them the same way:
:meth:`AgentInstall.layer_key` over the environment image and :func:`agent_install`'s description of the install.

Only what changes the installed files is in the key: the agent, its version and install options (and where a
Fabric agent installs to), the proxy and package-index settings its install runs with, and the Harbor version whose
install code runs. The model, prompts, turn limits and secrets aren't, so trials that differ only in those share a
layer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import JsonValue

#: The SDK's installed Fabric agent, which a registered platform agent runs as.
FABRIC_INSTALLED_AGENT = "nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent:FabricInstalledAgent"
#: Bumped when the way a layer is built changes what it holds, so older layers are no longer reused.
LAYER_FORMAT = 1
AGENT_LAYERS_ENV = "NHX_HARBOR_AGENT_LAYERS"
"""``0`` turns agent layers off for local Docker trials."""
AGENT_LAYER_DOCKER_ENVIRONMENT = (
    "nhx_evals_sdk.agent_eval.runtimes.harbor.local_agent_layer:AgentLayerDockerEnvironment"
)
AGENT_LAYER_KWARG = "agent_layer"
"""The environment kwarg naming the trial's agent: ``{"name", "import_path", "kwargs", "env"}``."""
#: The kwarg holding a Fabric agent's config bundle, a directory its setup uploads; never part of a layer.
FABRIC_CONFIG_BUNDLE_KWARG = "fabric_config_bundle"
#: Harbor's installed agents whose install a layer can hold. Each skips its install when the version it would install
#: is already there, and installs from its version alone. Harbor's other installed agents install again in every
#: trial, so a layer would save them nothing, and some write run options or credentials during their install.
LAYERED_HARBOR_AGENTS = frozenset({"codex", "claude-code"})
#: Proxy, package-index and TLS settings an agent's install runs with: a layer is built with them and keyed by them.
#: No other variable of the agent's environment, such as an API key, reaches a layer's build.
INSTALL_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "PIP_CERT",
        "PIP_CLIENT_CERT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def agent_layers_enabled() -> bool:
    return os.environ.get(AGENT_LAYERS_ENV, "").strip() != "0"


def agent_layer_kwarg(
    *, name: str | None, import_path: str | None, kwargs: dict[str, Any], env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The :data:`AGENT_LAYER_KWARG` value for a trial's agent; of its environment, only what its install runs with."""
    return {"name": name, "import_path": import_path, "kwargs": kwargs, "env": install_env(env)}


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
    agent_env: Mapping[str, str] | None = None,
    today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
) -> AgentInstall | None:
    """The reusable install a Harbor run's agent does in setup, or None when it can't be reused.

    The Harbor agents in :data:`LAYERED_HARBOR_AGENTS` and the registered-agent :class:`FabricInstalledAgent` can be
    reused. Every other agent can't: one that installs nothing (``oracle``, ``nop``) has nothing to reuse, a loose
    ``agent_dir`` or any other ``import_path`` may do anything in setup, and Harbor's other installed agents install
    again in every trial.

    An agent whose ``version`` isn't pinned installs whatever is newest, so its layer is keyed by the UTC date: it is
    reused for a day, then installed again. ``agent_env`` is the agent's environment; only its
    :data:`INSTALL_ENV_NAMES` are part of the install.

    An install given a URL with a username or password, such as a private package index, isn't reused either: a
    layer records its install's inputs, and anyone who can read the layer could read that password.
    """
    harbor_version = importlib.metadata.version("harbor")
    if agent_dir is not None:
        return None
    env = install_env(agent_env)
    if has_url_credentials(dict(agent_kwargs)) or has_url_credentials(env):
        return None
    if agent_import_path is not None:
        if agent_import_path != FABRIC_INSTALLED_AGENT:
            return None
        return _fabric_install(agent_kwargs, env, harbor_version)
    if agent_name not in LAYERED_HARBOR_AGENTS:
        return None
    version = agent_kwargs.get("version")
    pinned = str(version) if version is not None else f"unpinned:{today().isoformat()}"
    return AgentInstall(
        agent=f"harbor:{agent_name}", inputs={"version": pinned, **_env_inputs(env)}, harbor_version=harbor_version
    )


#: A URL carrying userinfo, such as ``https://user:token@pypi.example/simple``.  # trufflehog:ignore
_URL_CREDENTIALS = re.compile(r"[a-z][a-z0-9+.-]*://[^/?#\s@]+@", re.IGNORECASE)


def has_url_credentials(value: object) -> bool:
    """Whether any string in ``value`` holds a URL with a username or password."""
    if isinstance(value, str):
        return _URL_CREDENTIALS.search(value) is not None
    if isinstance(value, dict):
        return any(has_url_credentials(item) for item in value.values())
    if isinstance(value, list):
        return any(has_url_credentials(item) for item in value)
    return False


def install_env(agent_env: Mapping[str, str] | None) -> dict[str, str]:
    """The part of an agent's environment its install runs with: its :data:`INSTALL_ENV_NAMES`."""
    return {name: value for name, value in sorted((agent_env or {}).items()) if name in INSTALL_ENV_NAMES}


def _env_inputs(env: Mapping[str, str]) -> dict[str, str]:
    return {f"env:{name}": value for name, value in env.items()}


def _fabric_install(
    agent_kwargs: Mapping[str, JsonValue], env: Mapping[str, str], harbor_version: str
) -> AgentInstall | None:
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
    inputs = {"install": spec, **_env_inputs(env)}
    # Setup looks for the install where this run's agent puts it, so a layer installed elsewhere is no use to it.
    if venv := agent_kwargs.get("fabric_venv_path"):
        inputs["venv"] = str(venv)
    return AgentInstall(agent="fabric", inputs=inputs, harbor_version=harbor_version)
