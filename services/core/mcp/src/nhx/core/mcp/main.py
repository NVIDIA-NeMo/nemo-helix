#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for the NeMo MCP server."""

from __future__ import annotations

import argparse
import logging
import os

from nhx.core.mcp.auth import AUTH_TOKEN_ENV_VAR, MCPAuthConfigError, resolve_http_auth
from nhx.core.mcp.server import create_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="NeMo MCP Server - Model Context Protocol server for NeMo Helix")
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for NeMo Helix API (default: NHX_BASE_URL env var)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )

    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help=(
            f"Serve the HTTP transport without a bearer token ({AUTH_TOKEN_ENV_VAR} unset). "
            "Loopback hosts only; for local debugging."
        ),
    )
    parser.add_argument(
        "--enable-destructive-tools",
        action="store_true",
        help="Expose tools that delete platform resources, such as delete_workspace (off by default)",
    )

    args = parser.parse_args()

    # Resolve auth before building the server so a bad combination fails fast
    auth = None
    if args.allow_unauthenticated:
        if args.transport != "streamable-http":
            parser.error("--allow-unauthenticated only applies to --transport streamable-http")
        if os.environ.get(AUTH_TOKEN_ENV_VAR):
            parser.error(
                f"--allow-unauthenticated cannot be combined with {AUTH_TOKEN_ENV_VAR}: "
                "unset the token to serve without authentication, or drop the flag to require it."
            )
    if args.transport == "streamable-http":
        try:
            auth = resolve_http_auth(args.host, allow_unauthenticated=args.allow_unauthenticated)
        except MCPAuthConfigError as e:
            parser.error(str(e))
        if auth is None:
            if args.enable_destructive_tools:
                parser.error(
                    "--enable-destructive-tools cannot be combined with --allow-unauthenticated: "
                    f"set {AUTH_TOKEN_ENV_VAR} to expose destructive tools over HTTP."
                )
            logger.warning("Serving the HTTP transport WITHOUT authentication on %s", args.host)

    # Create server
    logger.info("Creating NeMo MCP server...")
    server = create_server(
        base_url=args.base_url,
        auth=auth,
        enable_destructive_tools=args.enable_destructive_tools,
    )

    # Run server with specified transport
    if args.transport == "stdio":
        logger.info("Starting MCP server with stdio transport...")
        server.run(transport="stdio")
    else:
        logger.info(f"Starting MCP server with HTTP transport on {args.host}:{args.port}...")
        server.run(transport=args.transport, host=args.host, port=args.port)  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
