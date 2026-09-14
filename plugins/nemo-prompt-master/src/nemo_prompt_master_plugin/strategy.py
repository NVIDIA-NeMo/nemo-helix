# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt Master implementation of ``nemo agents optimize --strategy``."""

from __future__ import annotations

import copy
import json
from typing import Any, ClassVar

import yaml
from nemo_agent_optimization_plugin.strategies import PRIMARY_ARTIFACT_KEY
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from nemo_prompt_master_plugin.config import PromptMasterConfig
from nemo_prompt_master_plugin.runner import run_prompt_master

RESULT_NAME = "prompt_master_results"


class PromptMasterStrategy:
    """Optimize a resolved Fabric agent's system instructions."""

    name: ClassVar[str] = "prompt-master"

    def validate_config(self, config: dict[str, Any], *, agent: str | None) -> None:
        if agent is None:
            raise ValueError("The prompt-master strategy requires --agent.")
        PromptMasterConfig.model_validate(config)

    def run(
        self,
        *,
        agent_config: dict[str, Any],
        source_agent_config: dict[str, Any] | None = None,
        config: dict[str, Any],
        ctx: JobContext,
        sdk: NeMoPlatform | None = None,
    ) -> dict[str, Any]:
        del sdk
        parsed = PromptMasterConfig.model_validate(config)
        runtime_dir = ctx.storage.ephemeral / "prompt-master"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        optimized_prompt = run_prompt_master(parsed, agent_config, runtime_dir)

        optimized_config = copy.deepcopy(source_agent_config or agent_config)
        optimized_config.setdefault("instructions", {}).setdefault("system", {})["content"] = optimized_prompt

        output_dir = ctx.storage.persistent / "results" / RESULT_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        optimized_path = output_dir / "optimized_config.yml"
        optimized_path.write_text(
            yaml.safe_dump(optimized_config, sort_keys=False),
            encoding="utf-8",
        )

        agent_name = str(optimized_config.get("name") or optimized_config.get("metadata", {}).get("name") or "unknown")
        summary = {
            "status": "completed",
            "strategy": self.name,
            "agent": agent_name,
            "optimized_config": optimized_path.name,
        }
        (output_dir / "prompt_master_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        ref = ctx.results.save(RESULT_NAME, output_dir)
        return {
            **summary,
            "result": ref.model_dump(mode="json"),
            PRIMARY_ARTIFACT_KEY: str(optimized_path),
        }
