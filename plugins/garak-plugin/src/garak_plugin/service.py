# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the garak_plugin plugin."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from garak_plugin.authz import scope
from garak_plugin.jobs.scan import ScanJob
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec


class GarakPluginService(NemoService):
    """Garak Plugin service. Exposes healthz and CRUD over scan configs/targets."""

    name: ClassVar[str] = "garak-plugin"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        from garak_plugin.api.v2 import artifacts, configs, targets

        healthz_router = APIRouter()

        @healthz_router.get("/healthz")
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "mode": "sdk-backed-job-scaffold",
                "jobs": ["garak-plugin.scan"],
                "entities": ["garak_plugin_scan_config", "garak_plugin_scan_target"],
            }

        crud_prefix = "/v2/workspaces/{workspace}"
        return [
            RouterSpec(
                router=healthz_router,
                tag="Garak Plugin",
                description="Garak Plugin scaffold routes.",
                prefix="/v1",
            ),
            RouterSpec(
                router=configs.router,
                tag="Garak Plugin Configs",
                description="Scan configuration CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=targets.router,
                tag="Garak Plugin Targets",
                description="Scan target CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=artifacts.router,
                tag="Garak Plugin Jobs",
                description="Scan job aggregate artifact download.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                add_job_routes(ScanJob, authz=scope.child("scan")),
                tag="Garak Plugin Jobs",
                description="Scan job submission and retrieval.",
                prefix=crud_prefix,
            ),
        ]
