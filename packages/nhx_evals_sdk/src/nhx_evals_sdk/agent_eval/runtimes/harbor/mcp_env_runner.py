# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo Fabric's Harbor runner with stdio MCP servers' ``${NAME}`` env filled in from its environment.

:class:`~nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent.NemoFabricAgent` uploads this file into the
task container and runs it in place of ``python -m nemo_fabric.integrations.harbor.runner``, with the same
arguments. A stdio MCP server inherits only its own ``env``, so a secret it needs is named there as a
``${NAME}`` template. Harbor puts the value in the runner's environment, and only this process fills it in,
in memory: the spec Harbor persists and uploads keeps the template.

It imports nothing but the standard library and ``nemo_fabric``, which is all the container has.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from nemo_fabric import FabricConfig, FabricConfigError, RunResult
from nemo_fabric.integrations.harbor import runner
from nemo_fabric.integrations.harbor.models import FabricRunPayload

#: The whole value of an env entry that names a variable, as registered-agent resolution writes it.
_TEMPLATE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def fill_mcp_env(config: FabricConfig, environ: Mapping[str, str] = os.environ) -> None:
    """Replace each stdio MCP server's ``${NAME}`` env value with ``NAME``'s value from ``environ``, in place.

    Raises:
        FabricConfigError: A template names a variable ``environ`` doesn't set.
    """
    for name, server in (config.mcp.servers if config.mcp is not None else {}).items():
        if server.transport != "stdio":
            continue
        for key, value in server.env.items():
            match = _TEMPLATE.match(value)
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
