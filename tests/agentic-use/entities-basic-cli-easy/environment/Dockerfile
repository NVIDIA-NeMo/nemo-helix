# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use the pre-built nhx-agentic-base image
# Build the base image first with: docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
FROM nhx-agentic-base:latest

# Disable MCP to force CLI usage
RUN rm -f /app/.mcp.json
