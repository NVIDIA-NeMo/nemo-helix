# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal research agent powered by NAT and Tavily.

Thin wrapper around `nat run`: loads .env, then executes the workflow defined in
`workflow.yml` via NAT's Python API. The same workflow can be invoked directly
with:

    nat run --config_file workflow.yml --input "..."
"""

import asyncio
import pathlib
import sys
from contextvars import ContextVar

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
from nat.runtime.loader import load_workflow

CONFIG_FILE = pathlib.Path(__file__).parent / "workflow.yml"

_NAT_PROFILER_HANDLER: ContextVar[BaseCallbackHandler | None] = ContextVar("nat_profiler_handler", default=None)
register_configure_hook(_NAT_PROFILER_HANDLER, inheritable=True)


async def run(query: str) -> str:
    async with load_workflow(CONFIG_FILE) as workflow:
        _NAT_PROFILER_HANDLER.set(LangchainProfilerHandler())
        async with workflow.run(query) as runner:
            return await runner.result(to_type=str)


def main() -> None:
    load_dotenv()

    query = " ".join(sys.argv[1:]).strip() or input("Research query: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)

    output = asyncio.run(run(query))
    print(output)


if __name__ == "__main__":
    main()
