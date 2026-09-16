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

The shared shape lives in :class:`nhx.customization_common.contributor.base.BaseContributor`.
"""

from __future__ import annotations

from typing import ClassVar

import typer
from nemo_helix_plugin.customization_contributor import CustomizationContributorSDKResources
from nhx.customization_common.contributor.base import BaseContributor

from nemo_unsloth_plugin.config import UnslothPluginConfig, generate_unsloth_id, get_config
from nemo_unsloth_plugin.jobs.jobs import UnslothJob


class UnslothContributor(BaseContributor):
    """Registers Unsloth routes/CLI under the customization router (SFT only, container submit)."""

    name: ClassVar[str] = "unsloth"
    job_cls: ClassVar[type] = UnslothJob
    cli_help: ClassVar[str] = "Unsloth GPU fine-tuning (container submit). SFT only."
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
