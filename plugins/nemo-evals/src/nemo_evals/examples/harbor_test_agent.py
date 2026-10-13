# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable Harbor wrapper for the bundled taskset notebook example."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from harbor import AgentContext, BaseAgent, BaseEnvironment

logger = logging.getLogger(__name__)

# The notebook is a source-checkout example. Resolve its runtime assets from the
# installed editable plugin so evaluator workers do not need a custom PYTHONPATH.
AGENT_DIR = Path(__file__).resolve().parents[3] / "examples/harbor_taskset/agent"


class WrappedAgent(BaseAgent):
    """Upload the example agent's source files into a Harbor task container."""

    @staticmethod
    def name() -> str:
        return "hello-harbor-agent"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Upload the dependency-free runtime files."""
        for name in ("agent.py", "main.py", "tracing.py"):
            source = AGENT_DIR / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Agent source must be a regular file: {source}")
            await environment.upload_file(source, f"/app/{name}")
        logger.info("[setup] uploaded agent sources to /app")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute the agent on ``instruction`` inside the task container."""
        session_id = self.session_id or "local"
        proc = await environment.exec(
            f"cd /app && python main.py --prompt {shlex.quote(instruction.strip())} "
            f"--session-id {shlex.quote(session_id)}"
        )
        context.metadata = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.return_code,
        }
        context.n_input_tokens = 7
        context.n_output_tokens = 3
        context.n_cache_tokens = 1
        if proc.return_code != 0:
            raise RuntimeError(f"Agent process failed with exit code {proc.return_code}: {proc.stderr or proc.stdout}")
