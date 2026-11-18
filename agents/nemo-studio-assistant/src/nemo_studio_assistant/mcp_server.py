# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric-native MCP boundary for NeMo Studio Assistant tools."""

from fastmcp import FastMCP
from nemo_studio_assistant.docs_search import search_docs
from nemo_studio_assistant.register import (
    ask_user_question,
    check_status,
    deploy_guardrail,
    job_progress,
    nemo_api,
    select_agent,
    select_dataset_file,
    select_eval_config,
    select_model,
    studio_link,
)

TOOLS = (
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


def create_server() -> FastMCP:
    server = FastMCP("NeMo Studio Assistant")
    for tool in TOOLS:
        server.tool(tool)
    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
