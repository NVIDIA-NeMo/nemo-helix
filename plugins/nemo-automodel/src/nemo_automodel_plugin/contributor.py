# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel customization contributor.

Shared shape lives in :class:`nmp.customization_common.contributor.base.BaseContributor`;
this subclass supplies the backend-specific values + the SDK resource classes the
customization hub composes under ``client.customization.automodel``.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.customization_contributor import (
    CustomizationCLISummary,
    CustomizationContributorSDKResources,
)
from nmp.customization_common.contributor.base import BaseContributor

from nemo_automodel_plugin.config import AutomodelPluginConfig, generate_automodel_id, get_config
from nemo_automodel_plugin.jobs.jobs import AutomodelJob

_CLI_HELP = """Fine-tune a model with Automodel: SFT, LoRA, or knowledge distillation.

The platform runs the job on a GPU execution profile. That profile's backend
is docker or kubernetes_job, depending on how the platform was set up. Run
'nemo jobs list-execution-profiles' to see the profiles on this platform.

Multi-node training (parallelism.num_nodes above 1) needs a kubernetes_job or
volcano_job backend. On a single node, set the GPU count with
parallelism.num_gpus_per_node.

The job JSON follows the AutomodelJobInput schema:
  model        base model entity, as 'name' or 'workspace/name'
  dataset      training fileset, plus an optional validation fileset
  training     sft or distillation, and lora or all_weights
  schedule     epochs, max_steps, seed, validation interval
  batch        global_batch_size, micro_batch_size, sequence packing
  optimizer    learning_rate, weight_decay, warmup_steps, decay shape
  parallelism  nodes, GPUs per node, tensor and pipeline parallel sizes

Optional blocks: output names the model entity written at the end of the run,
and integrations enables Weights & Biases reporting.

For DPO or GRPO, use the rl backend.

Run 'nemo customization automodel explain' for the full schema and defaults."""


class AutomodelContributor(BaseContributor):
    """Registers Automodel routes/CLI under the customization router."""

    name: ClassVar[str] = "automodel"
    job_cls: ClassVar[type] = AutomodelJob
    cli_help: ClassVar[str] = _CLI_HELP
    cli_summary: ClassVar[CustomizationCLISummary] = CustomizationCLISummary(
        trains="SFT and LoRA fine-tuning, and knowledge distillation.",
        runs_on=(
            "a GPU execution profile, on the docker or kubernetes_job backend. "
            "Multi-node needs kubernetes_job or volcano_job."
        ),
        job_json="model, dataset, training, schedule, batch, optimizer, parallelism.",
        use_when="you need SFT, LoRA, or knowledge distillation on one or more GPUs.",
        command="nemo customization automodel submit job.json",
    )
    jobs_router_description: ClassVar[str] = "Automodel training jobs."

    generate_job_name = staticmethod(generate_automodel_id)

    def _get_config(self) -> AutomodelPluginConfig:
        return get_config()

    def apply_cli_overrides(self, app: typer.Typer) -> None:
        from nemo_automodel_plugin.cli.inputs import apply_automodel_job_cli_overrides

        apply_automodel_job_cli_overrides(app)

    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        from nemo_automodel_plugin.sdk.resources import AsyncAutomodelCustomization, AutomodelCustomization

        return CustomizationContributorSDKResources(
            sync_resource=AutomodelCustomization,
            async_resource=AsyncAutomodelCustomization,
        )
