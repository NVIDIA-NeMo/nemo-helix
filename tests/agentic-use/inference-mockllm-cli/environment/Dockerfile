# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM nhx-agentic-base:latest

# Disable MCP to force CLI usage
RUN rm -f /app/.mcp.json

# Enable mock provider mode for the Inference Gateway.
# Providers whose name starts with "igw-mock-" will return mock responses
# instead of proxying to real backends.
ENV NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX=igw-mock-
