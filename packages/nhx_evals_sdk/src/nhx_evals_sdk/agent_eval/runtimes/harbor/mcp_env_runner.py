# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo Fabric's Harbor runner with stdio MCP servers' ``${NAME}`` env filled in from its environment.

:class:`~nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent.NemoFabricAgent` uploads this file into the
task container and runs it in place of ``python -m nemo_fabric.integrations.harbor.runner``, with the same
arguments. A stdio MCP server inherits only its own ``env``, so a secret it needs is named there as a
``${NAME}`` template. Harbor puts the value in the runner's environment, and only this process fills it in,
in memory: the spec Harbor persists and uploads keeps the template.

A harness that writes the server's ``env`` to a file of its own, and fills in ``${NAME}`` itself when it
starts the server, keeps its templates: filling them in here would put the value in that file.

It imports nothing but the standard library and ``nemo_fabric``, which is all the container has.
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping
from typing import Any, Final

from nemo_fabric import FabricConfig, FabricConfigError, RunResult
from nemo_fabric.integrations.harbor import runner
from nemo_fabric.integrations.harbor.models import FabricRunPayload

#: The whole value of an env entry that names a variable, as registered-agent resolution writes it.
MCP_ENV_TEMPLATE: Final = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

#: Harness adapters that write each stdio MCP server's ``env`` to a file Harbor keeps with the trial, and fill
#: in ``${NAME}`` there from their own environment when they start the server. Their templates stay as they
#: are. nemo-fabric 0.7.0's Hermes adapter writes the env to Hermes's ``config.yaml`` under the trial's agent
#: logs, and hermes-agent 0.19.0 expands ``${NAME}`` in each MCP server's config when it starts it.
ADAPTERS_THAT_FILL_MCP_ENV: Final = frozenset({"nvidia.fabric.hermes"})


def unbound_mcp_env_templates(config: Mapping[str, Any], bound: Collection[str]) -> dict[str, list[str]]:
    """Each stdio MCP server's ``${NAME}`` env templates whose variable isn't in ``bound``, sorted.

    Args:
        config: A Fabric config as JSON, such as ``FabricConfig.model_dump(mode="json")`` or a registered
            agent's stored config.
        bound: The variables the run sets.

    Returns:
        The variables each server lacks, by server name; servers that lack none are left out.
    """
    servers = (config.get("mcp") or {}).get("servers") or {}
    unbound: dict[str, list[str]] = {}
    for name, server in servers.items():
        if not isinstance(server, Mapping) or server.get("transport") != "stdio":
            continue
        variables = {
            match.group(1)
            for value in (server.get("env") or {}).values()
            if isinstance(value, str) and (match := MCP_ENV_TEMPLATE.match(value))
        }
        if missing := sorted(variables - set(bound)):
            unbound[name] = missing
    return unbound


def fill_mcp_env(config: FabricConfig, environ: Mapping[str, str] = os.environ) -> None:
    """Replace each stdio MCP server's ``${NAME}`` env value with ``NAME``'s value from ``environ``, in place.

    For an adapter in :data:`ADAPTERS_THAT_FILL_MCP_ENV`, the templates stay, and only their variables are
    checked.

    Raises:
        FabricConfigError: A template names a variable ``environ`` doesn't set.
    """
    fill = config.harness is None or config.harness.adapter_id not in ADAPTERS_THAT_FILL_MCP_ENV
    for name, server in (config.mcp.servers if config.mcp is not None else {}).items():
        if server.transport != "stdio":
            continue
        for key, value in server.env.items():
            match = MCP_ENV_TEMPLATE.match(value)
            if match is None:
                continue
            variable = match.group(1)
            if variable not in environ:
                raise FabricConfigError(
                    f"MCP server {name!r} reads {variable} for its env {key}, but the Harbor runner's "
                    "environment doesn't set it",
                    stage="configuration",
                    code="harbor_mcp_env_unset",
                )
            if fill:
                server.env[key] = environ[variable]


def run_through(runner_module: Any = runner) -> None:
    """Run ``runner_module.main`` with a ``run`` that fills in the payload's stdio MCP server env first.

    ``main`` parses the arguments and reports failures as it always does. It looks ``run`` up in its module
    when it calls it, so replacing the module's function is the seam.
    """
    original = runner_module.run

    async def run(payload: FabricRunPayload) -> RunResult:
        fill_mcp_env(payload.config)
        return await original(payload)

    runner_module.run = run
    runner_module.main()


if __name__ == "__main__":
    run_through()
