# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric-native MCP boundary for NeMo Studio Assistant tools."""

from fastmcp import FastMCP
from nemo_studio_assistant.register import (
    APPROVAL_MODE_CHAT,
    approval_mode,
    ask_user_question,
    check_status,
    deploy_guardrail,
    job_progress,
    nemo_api,
    search_docs,
    select_agent,
    select_dataset_file,
    select_eval_config,
    select_model,
    studio_link,
)

STUDIO_TOOLS = (
    deploy_guardrail,
    nemo_api,
    check_status,
    select_agent,
    select_model,
    select_dataset_file,
    select_eval_config,
    job_progress,
    studio_link,
    ask_user_question,
    search_docs,
)

# Terminal tools need no Studio session: no pickers, progress cards, or Studio links.
TERMINAL_TOOLS = (
    deploy_guardrail,
    nemo_api,
    check_status,
    search_docs,
)


def create_server() -> FastMCP:
    server = FastMCP("NeMo Studio Assistant")
    tools = TERMINAL_TOOLS if approval_mode() == APPROVAL_MODE_CHAT else STUDIO_TOOLS
    for tool in tools:
        server.tool(tool)
    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
