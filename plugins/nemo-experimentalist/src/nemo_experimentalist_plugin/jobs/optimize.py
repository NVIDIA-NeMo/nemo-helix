# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``experimentalist`` agent optimization — name reserved, not yet implemented.

The open design question is the config shape: the Experimentalist takes either
an Insight ref or a Harbor-compatible eval config, and which of those becomes
the optimization config is not settled. Until it is, run the Experimentalist
through its own CLI.
"""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext


class ExperimentalistOptimizeJob(AgentOptimizeJob):
    """Improve an agent's source or harness. Not implemented yet."""

    name: ClassVar[str] = "agent_optimize"
    strategy: ClassVar[str] = "experimentalist"
    description: ClassVar[str] = "Improve an agent's source or harness (not implemented yet)."
    task_module: ClassVar[str] = "nemo_experimentalist_plugin.tasks.agent_optimize"

    def optimize(
        self,
        *,
        source_agent_config: dict[str, Any],
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> dict[str, Any]:
        del source_agent_config, config, ctx, workspace, sdk
        raise NotImplementedError(
            "The experimentalist optimization strategy is not implemented yet. "
            "Run the Experimentalist directly with `nemo agents experimentalist` instead."
        )
