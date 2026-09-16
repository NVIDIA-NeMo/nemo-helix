# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``prompt-master`` agent optimization: rewrite an agent's system prompt."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

from nemo_agent_optimization_plugin.job_base import AgentOptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from prompt_master_plugin.config import PromptMasterConfig
from prompt_master_plugin.runner import run_prompt_master


class PromptMasterOptimizeJob(AgentOptimizeJob):
    """Optimize the agent's system instructions."""

    name: ClassVar[str] = "agent_optimize"
    strategy: ClassVar[str] = "prompt-master"
    description: ClassVar[str] = "Optimize an agent's system prompt."
    task_module: ClassVar[str] = "prompt_master_plugin.tasks.agent_optimize"

    def optimize(
        self,
        *,
        source_agent_config: dict[str, Any],
        config: dict[str, Any],
        ctx: JobContext,
        workspace: str,
        sdk: NeMoPlatform,
    ) -> dict[str, Any]:
        del workspace, sdk
        parsed = PromptMasterConfig.model_validate(config)
        runtime_dir = ctx.storage.ephemeral / "prompt-master"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        optimized_prompt = run_prompt_master(parsed, source_agent_config, runtime_dir)

        optimized = copy.deepcopy(source_agent_config)
        optimized.setdefault("instructions", {}).setdefault("system", {})["content"] = optimized_prompt
        return optimized
