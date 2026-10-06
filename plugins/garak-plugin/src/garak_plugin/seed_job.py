# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed job for the garak_plugin plugin."""

import logging
from typing import ClassVar

from garak_plugin._legacy import migrate_legacy_entities, migrate_legacy_job_sources
from garak_plugin.entities import (
    ScanClassConfig,
    ScanConfig,
    ScanPluginsData,
    ScanReportData,
    ScanRunData,
    ScanSystemData,
)
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.seed import NemoSeedJob

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "default"
SYSTEM_WORKSPACE = "system"

_DEFAULT_PROBE_SPEC = (
    "ansiescape,atkgen,continuation,dan.Ablation_Dan_11_0,dan.AutoDANCached,"
    "dan.DanInTheWild,divergence,encoding,exploitation,goodside,grandma,"
    "latentinjection,leakreplay,lmrc.Bullying,lmrc.Deadnaming,lmrc.QuackMedicine,"
    "lmrc.SexualContent,lmrc.Sexualisation,lmrc.SlurUsage,malwaregen,misleading,"
    "packagehallucination,phrasing,promptinject,realtoxicityprompts.RTPBlank,"
    "snowball.GraphConnectivity,suffix.GCGCached,tap.TAPCached,topic,web_injection"
)


class GarakPluginSeedJob(NemoSeedJob):
    """Ensure the garak_plugin plugin has a default scan config in the system workspace."""

    name: ClassVar[str] = "garak-plugin"
    description: ClassVar[str] = "Create the default garak_plugin config."
    dependencies: ClassVar[list[str]] = ["entities"]

    async def run(self) -> None:
        await self._migrate_legacy_data()
        config = ScanConfig(
            name=DEFAULT_CONFIG_NAME,
            workspace=SYSTEM_WORKSPACE,
            description="Default Garak Plugin configuration",
            system=ScanSystemData(lite=False, parallel_attempts=32),
            run=ScanRunData(generations=3),
            plugins=ScanPluginsData(
                probe_spec=_DEFAULT_PROBE_SPEC,
                probes={"encoding": ScanClassConfig({"payloads": ["default", "xss"]})},
            ),
            reporting=ScanReportData(),
        )
        try:
            await self.entities_client.create(config)
            logger.info("Created default garak_plugin config %r", DEFAULT_CONFIG_NAME)
        except NemoEntityConflictError:
            logger.debug("Default garak_plugin config %r already exists", DEFAULT_CONFIG_NAME)

    async def _migrate_legacy_data(self) -> None:
        """Carry data stored before the plugin rename over to the new names; never blocks seeding."""
        for migrate in (migrate_legacy_entities, migrate_legacy_job_sources):
            try:
                await migrate(self.sdk)
            except Exception:
                logger.warning(
                    "Legacy data migration %s failed; it will be retried on the next start",
                    migrate.__name__,
                    exc_info=True,
                )
