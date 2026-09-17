# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unsloth customization contributor.

Registered under ``nemo.customization.contributors`` (key ``unsloth``).
The customization router hub (``nemo-customizer-plugin``) discovers this
class at startup and:

- merges :meth:`get_routers` into ``/apis/customization/...`` (HTTP authz is
  derived from the ``@path_rule``-decorated routes those routers carry)
- adds :meth:`get_cli` under ``nemo customization unsloth``
- composes :meth:`get_sdk_resources` under ``client.customization.unsloth``

The shared shape lives in :class:`nmp.customization_common.contributor.base.BaseContributor`.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.customization_contributor import (
    CustomizationCLISummary,
    CustomizationContributorSDKResources,
)
from nmp.customization_common.contributor.base import BaseContributor

from nemo_unsloth_plugin.config import UnslothPluginConfig, generate_unsloth_id, get_config
from nemo_unsloth_plugin.jobs.jobs import UnslothJob

_CLI_HELP = """Fine-tune a model with Unsloth on one GPU: SFT with LoRA or full weights.

The platform runs the job on a GPU execution profile. That profile's backend
is docker or kubernetes_job, depending on how the platform was set up. Run
'nemo jobs list-execution-profiles' to see what this platform offers.

One job trains on one GPU. hardware.gpus only picks which GPU to use; it does
not split the model across several GPUs.

The job JSON follows the UnslothJobInput schema:
  model      base model entity, max_seq_length, 4-bit or 8-bit loading
  dataset    training fileset path, plus an optional validation_path
  training   lora or all_weights, and the LoRA rank, alpha, dropout
  schedule   epochs, max_steps, warmup, scheduler, logging cadence
  batch      per_device_train_batch_size, gradient_accumulation_steps
  optimizer  learning_rate, weight_decay, optimizer algorithm
  hardware   which GPU to use, and training precision

Optional blocks: output names the result and picks save_method (lora,
merged_16bit, merged_4bit), deployment_config deploys the trained model, and
integrations turns on Weights & Biases reporting.

For tensor, pipeline, or multi-node parallelism, use the automodel backend.
For DPO or GRPO, use the rl backend.

Run 'nemo customization unsloth explain' for the full schema and defaults."""


class UnslothContributor(BaseContributor):
    """Registers Unsloth routes/CLI under the customization router (SFT only, container submit)."""

    name: ClassVar[str] = "unsloth"
    job_cls: ClassVar[type] = UnslothJob
    cli_help: ClassVar[str] = _CLI_HELP
    cli_summary: ClassVar[CustomizationCLISummary] = CustomizationCLISummary(
        trains="SFT fine-tuning with LoRA or with full weights.",
        runs_on="a GPU execution profile, on the docker or kubernetes_job backend. One job uses one GPU.",
        job_json="model, dataset, training, schedule, batch, optimizer, hardware.",
        pick_when="you asked for Unsloth, or want its 4-bit LoRA path on one GPU.",
        command="nemo customization unsloth submit job.json",
    )
    jobs_router_description: ClassVar[str] = "Unsloth GPU fine-tuning jobs (container submit)."

    generate_job_name = staticmethod(generate_unsloth_id)

    def _get_config(self) -> UnslothPluginConfig:
        return get_config()

    def apply_cli_overrides(self, app: typer.Typer) -> None:
        from nemo_unsloth_plugin.cli.inputs import apply_unsloth_job_cli_overrides

        apply_unsloth_job_cli_overrides(app)

    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        from nemo_unsloth_plugin.sdk.resources import AsyncUnslothCustomization, UnslothCustomization

        return CustomizationContributorSDKResources(
            sync_resource=UnslothCustomization,
            async_resource=AsyncUnslothCustomization,
        )
