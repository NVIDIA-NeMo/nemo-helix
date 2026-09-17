# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-RL customization contributor.

Registered under ``nemo.customization.contributors`` (key ``rl``). The
customization router hub (``nemo-customizer-plugin``) discovers this class and
merges its routes/CLI/authz/SDK. Shared shape lives in
:class:`nmp.customization_common.contributor.base.BaseContributor`.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_platform_plugin.customization_contributor import (
    CustomizationCLISummary,
    CustomizationContributorSDKResources,
)
from nmp.customization_common.contributor.base import BaseContributor

from nemo_rl_plugin.config import RlPluginConfig, generate_rl_id, get_config
from nemo_rl_plugin.jobs.jobs import RlJob

# A single command serves both algorithms, selected by `training.type` in the job JSON,
# so there is no `grpo` subcommand to find. The help text names both algorithms up front,
# because this is where a user looking for GRPO first checks whether it is supported.
_CLI_HELP = """Align a model with NeMo-RL, using DPO or GRPO.

The platform runs the job on the kubernetes_job backend only, starting a Ray
cluster for it. There is no docker path. Run 'nemo jobs list-execution-profiles'
to see the profiles on this platform.

A single command covers both algorithms: training.type in the job JSON selects
between 'dpo' and 'grpo'. DPO trains on preference pairs, full weights only.
GRPO trains against a NeMo Gym environment, and can also train a LoRA adapter
through training.finetuning_type.

The job JSON follows the RlJobInput schema:
  model        base model entity, as 'name' or 'workspace/name'
  dataset      fileset reference. For DPO, one fileset holding training.jsonl
               and validation.jsonl of prompt, chosen and rejected rows. For
               GRPO, Gym rollout rows in training.jsonl
  environment  Gym environment fileset. Required for GRPO, unused for DPO
  training     type 'dpo' or 'grpo', plus that method's hyperparameters

Optional blocks: output names the model entity written at the end of the run,
and integrations enables Weights & Biases reporting.

For SFT or LoRA fine-tuning, use the automodel or unsloth backend.

Run 'nemo customization rl explain' for the full schema and defaults."""


class RlContributor(BaseContributor):
    """Registers NeMo-RL routes/CLI under the customization router (DPO + GRPO, Kubernetes only)."""

    name: ClassVar[str] = "rl"
    job_cls: ClassVar[type] = RlJob
    cli_help: ClassVar[str] = _CLI_HELP
    cli_summary: ClassVar[CustomizationCLISummary] = CustomizationCLISummary(
        trains="DPO on preference pairs, or GRPO on a NeMo Gym environment.",
        runs_on="a Ray cluster on the kubernetes_job backend only. There is no docker path.",
        job_json="model, dataset, training, and environment for GRPO.",
        use_when="you need DPO or GRPO. No other backend supports them.",
        command="nemo customization rl submit job.json",
    )
    jobs_router_description: ClassVar[str] = "NeMo-RL DPO and GRPO training jobs (Ray on Kubernetes)."

    generate_job_name = staticmethod(generate_rl_id)

    def _get_config(self) -> RlPluginConfig:
        return get_config()

    def apply_cli_overrides(self, app: typer.Typer) -> None:
        from nemo_rl_plugin.cli.inputs import apply_rl_job_cli_overrides

        apply_rl_job_cli_overrides(app)

    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        from nemo_rl_plugin.sdk.resources import AsyncRlCustomization, RlCustomization

        return CustomizationContributorSDKResources(
            sync_resource=RlCustomization,
            async_resource=AsyncRlCustomization,
        )
