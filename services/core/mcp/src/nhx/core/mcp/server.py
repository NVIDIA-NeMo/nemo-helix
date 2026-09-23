# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform MCP server aggregator for NeMo Helix.

This module provides the main MCP server that aggregates all service-specific
MCP servers into a unified interface for AI agents. It follows the same pattern
as the REST API platform aggregator, where each service owns its own MCP tools
and the platform composes them all together.

Architecture:
    services/core/entities/mcp/     -> Workspace tools (list_workspaces, etc.)
    services/core/jobs/mcp/         -> Job monitoring tools (future)
    services/core/models/mcp/       -> Model discovery tools (future)
    services/core/mcp/              -> This aggregator (composes all above)

This enables:
- 1:1 mapping of service:MCP server (matches REST API architecture)
- Each service can be deployed with its own MCP server
- Platform can compose them all for unified deployment
- Code maintainers can see clear service boundaries

"""

from __future__ import annotations

import logging

from fastmcp import FastMCP
from nhx.core.entities.mcp import create_server as create_entities_mcp

logger = logging.getLogger(__name__)


def create_server(base_url: str | None = None) -> FastMCP:
    """
    Create and configure the platform MCP aggregator.

    This server aggregates tools from all service-specific MCP servers,
    similar to how the REST API platform aggregates service routers.

    Args:
        base_url: Optional NeMo Helix base URL

    Returns:
        Configured FastMCP server instance with all service tools
    """
    # Initialize platform aggregator
    platform = FastMCP(
        "NeMo Helix",
        website_url="https://docs.nvidia.com/nemo-helix",
    )

    # Mount per service MCP servers
    platform.mount(create_entities_mcp(base_url))

    # Future: aggregate additional service MCP servers as they're implemented
    # from nhx.core.jobs.mcp import create_server as create_jobs_mcp
    # jobs_mcp = create_jobs_mcp(base_url)
    # platform.mount(jobs_mcp)
    #
    # from nhx.core.models.mcp import create_server as create_models_mcp
    # models_mcp = create_models_mcp(base_url)
    # platform.mount(models_mcp)

    return platform
