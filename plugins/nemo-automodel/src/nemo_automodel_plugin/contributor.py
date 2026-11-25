# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel customization contributor.

Shared shape lives in :class:`nhx.customization_common.contributor.base.BaseContributor`;
this subclass supplies the backend-specific values + the SDK resource classes the
customization hub composes under ``client.customization.automodel``.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.commands import _add_submit_command
from nemo_helix_plugin.customization_contributor import (
    CustomizationCLISummary,
    CustomizationContributorSDKResources,
)
from nemo_helix_plugin.jobs.api_factory import JobRouteOption
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.scheduler import NemoJobScheduler
from nemo_helix_plugin.service import RouterSpec
from nhx.customization_common.cli.renderer import CustomizationSubmitRenderer
from nhx.customization_common.contributor.base import BaseContributor
from nhx.customization_common.contributor.config import generate_job_id
from nhx.customization_common.service.platform_client import CUSTOMIZATION_JOB_SOURCE

from nemo_automodel_plugin.config import AutomodelPluginConfig, generate_automodel_id, get_config
from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nemo_automodel_plugin.jobs.merge import AutomodelMergeJob

_CLI_HELP = """Fine-tune a model with Automodel: SFT, LoRA, or knowledge distillation.

The platform runs the job on a GPU execution profile. On a single node, that
profile's backend is docker or kubernetes_job, depending on how the platform
was set up; set the GPU count with parallelism.num_gpus_per_node.

Multi-node training (parallelism.num_nodes above 1) runs on a volcano_job or
kubernetes_job backend, so it needs a Kubernetes platform. Run
'nemo jobs list-execution-profiles' to see the profiles on this platform.

The job JSON follows the AutomodelJobInput schema:
  model        base model entity, as 'name' or 'workspace/name'
  dataset      fileset of training files and optional validation files
  training     sft or distillation, and lora or all_weights
  schedule     epochs, max_steps, seed, validation interval
  batch        global_batch_size, micro_batch_size, sequence packing
  optimizer    learning_rate, weight_decay, warmup_steps, decay shape
  parallelism  nodes, GPUs per node, tensor and pipeline parallel sizes

Optional blocks: output names the model entity written at the end of the run,
and integrations enables Weights & Biases reporting.

To merge a trained LoRA adapter into its base model, run 'nemo customization automodel merge'.

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
            "a GPU execution profile: docker or kubernetes_job on a single node, "
            "volcano_job or kubernetes_job for multi-node."
        ),
        job_json="model, dataset, training, schedule, batch, optimizer, parallelism.",
        use_when="you need SFT, LoRA, or knowledge distillation on one or more GPUs.",
        command="nemo customization automodel --job-json job.json",
    )
    jobs_router_description: ClassVar[str] = "Automodel training jobs."

    generate_job_name = staticmethod(generate_automodel_id)

    def _get_config(self) -> AutomodelPluginConfig:
        return get_config()

    def get_routers(self) -> list[RouterSpec]:
        routers = super().get_routers()
        merge_router = add_job_routes(
            AutomodelMergeJob,
            service_name=CUSTOMIZATION_JOB_SOURCE,
            generate_job_name=lambda: generate_job_id("automodel-merge"),
            route_options=[JobRouteOption.CORE, JobRouteOption.RERUN, JobRouteOption.DRY_RUN],
            default_profile=self._get_config().default_training_execution_profile,
            authz=AuthzScope("customization").child(self.name, "merge-jobs"),
        )
        routers.append(
            RouterSpec(
                router=merge_router,
                prefix="/v2/workspaces/{workspace}",
                tag="Automodel Merge Jobs",
                description="Automodel LoRA merge jobs.",
            )
        )
        return routers

    def apply_cli_overrides(self, app: typer.Typer) -> None:
        from nemo_automodel_plugin.cli.inputs import apply_automodel_job_cli_overrides

        apply_automodel_job_cli_overrides(app)
        _add_submit_command(
            app,
            AutomodelMergeJob,
            NemoJobScheduler(),
            renderer_cls=CustomizationSubmitRenderer,
            command_name="merge",
        )

    def get_sdk_resources(self) -> CustomizationContributorSDKResources:
        from nemo_automodel_plugin.sdk.resources import AsyncAutomodelCustomization, AutomodelCustomization

        return CustomizationContributorSDKResources(
            sync_resource=AutomodelCustomization,
            async_resource=AsyncAutomodelCustomization,
        )
