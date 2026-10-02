# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed job for the auditor plugin."""

import logging
from typing import ClassVar

from nemo_auditor.entities import (
    AuditClassConfig,
    AuditConfig,
    AuditPluginsData,
    AuditReportData,
    AuditRunData,
    AuditSystemData,
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


class AuditorSeedJob(NemoSeedJob):
    """Ensure the auditor plugin has a default audit config in the system workspace."""

    name: ClassVar[str] = "auditor"
    description: ClassVar[str] = "Create the default auditor config."
    dependencies: ClassVar[list[str]] = ["entities"]

    async def run(self) -> None:
        config = AuditConfig(
            name=DEFAULT_CONFIG_NAME,
            workspace=SYSTEM_WORKSPACE,
            description="Default Auditor configuration",
            system=AuditSystemData(lite=False, parallel_attempts=32),
            run=AuditRunData(generations=3),
            plugins=AuditPluginsData(
                probe_spec=_DEFAULT_PROBE_SPEC,
                probes={"encoding": AuditClassConfig({"payloads": ["default", "xss"]})},
            ),
            reporting=AuditReportData(),
        )
        try:
            await self.entities_client.create(config)
            logger.info("Created default auditor config %r", DEFAULT_CONFIG_NAME)
        except NemoEntityConflictError:
            logger.debug("Default auditor config %r already exists", DEFAULT_CONFIG_NAME)
