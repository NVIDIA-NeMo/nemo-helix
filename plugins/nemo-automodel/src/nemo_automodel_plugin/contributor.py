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
from nemo_helix_plugin.customization_contributor import CustomizationContributorSDKResources
from nhx.customization_common.contributor.base import BaseContributor

from nemo_automodel_plugin.config import AutomodelPluginConfig, generate_automodel_id, get_config
from nemo_automodel_plugin.jobs.jobs import AutomodelJob


class AutomodelContributor(BaseContributor):
    """Registers Automodel routes/CLI under the customization router."""

    name: ClassVar[str] = "automodel"
    job_cls: ClassVar[type] = AutomodelJob
    cli_help: ClassVar[str] = "Automodel training jobs (SFT, distillation)."
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
