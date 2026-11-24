# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility fixes for the Fabric Deep Agents runtime bundled in the image."""

import os
from collections.abc import Callable
from typing import Any

# The wrappers this module has installed, so applying a compatibility patch twice is a no-op.
_INSTALLED: set[Callable[..., Any]] = set()


def apply_deepagents_mcp_env_compatibility() -> None:
    """Preserve the adapter environment when Fabric starts stdio MCP tools."""
    from nemo_fabric_adapters.deepagents import adapter as adapter_module

    adapter: Any = adapter_module
    original = adapter._mcp_connection
    if original in _INSTALLED:
        return

    def mcp_connection(name: str, spec: Any) -> dict[str, Any]:
        connection = original(name, spec)
        if connection.get("transport") == "stdio":
            connection["env"] = {**os.environ, **connection.get("env", {})}
        return connection

    _INSTALLED.add(mcp_connection)
    adapter._mcp_connection = mcp_connection
