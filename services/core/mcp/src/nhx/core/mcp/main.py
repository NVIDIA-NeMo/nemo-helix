#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main entry point for the NeMo MCP server."""

from __future__ import annotations

import argparse
import logging

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

    args = parser.parse_args()

    # Create server
    logger.info("Creating NeMo MCP server...")
    server = create_server(base_url=args.base_url)

    # Run server with specified transport
    if args.transport == "stdio":
        logger.info("Starting MCP server with stdio transport...")
        server.run(transport="stdio")
    else:
        logger.info(f"Starting MCP server with HTTP transport on {args.host}:{args.port}...")
        server.run(transport=args.transport, host=args.host, port=args.port)  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
