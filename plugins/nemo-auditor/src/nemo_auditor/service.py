# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the auditor plugin."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_auditor.authz import scope
from nemo_auditor.jobs.audit import AuditJob
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec


class AuditorPluginService(NemoService):
    """Auditor plugin service. Exposes healthz and CRUD over audit configs/targets."""

    name: ClassVar[str] = "auditor"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_auditor.api.v2 import artifacts, configs, targets

        healthz_router = APIRouter()

        @healthz_router.get("/healthz")
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "mode": "sdk-backed-job-scaffold",
                "jobs": ["auditor.audit"],
                "entities": ["auditor_audit_config", "auditor_audit_target"],
            }

        crud_prefix = "/v2/workspaces/{workspace}"
        return [
            RouterSpec(
                router=healthz_router,
                tag="Auditor Plugin",
                description="Auditor plugin scaffold routes.",
                prefix="/v1",
            ),
            RouterSpec(
                router=configs.router,
                tag="Auditor Configs",
                description="Audit configuration CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=targets.router,
                tag="Auditor Targets",
                description="Audit target CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=artifacts.router,
                tag="Auditor Jobs",
                description="Audit job aggregate artifact download.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                add_job_routes(AuditJob, authz=scope.child("audit")),
                tag="Auditor Jobs",
                description="Audit job submission and retrieval.",
                prefix=crud_prefix,
            ),
        ]
