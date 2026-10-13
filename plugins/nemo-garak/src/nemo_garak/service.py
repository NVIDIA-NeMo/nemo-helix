# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the garak plugin."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_garak.authz import scope
from nemo_garak.jobs.audit import AuditJob
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

# Job records stamp ``source`` with this value and the list route filters on it. Jobs created
# before the rename carry ``nemo-auditor`` (derived from the old ``nemo_auditor`` package), so
# pin it to keep them listed alongside new jobs. Drop it only with a ``source`` backfill.
_LEGACY_JOB_SOURCE = "nemo-auditor"


class GarakPluginService(NemoService):
    """Garak plugin service. Exposes healthz and CRUD over audit configs/targets."""

    name: ClassVar[str] = "garak"
    # Permissions keep their legacy ``auditor.*`` ids until they migrate to ``garak.*``.
    permission_namespace: ClassVar[str | None] = "auditor"
    dependencies: ClassVar[list[str]] = ["entities"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_garak.api.v2 import artifacts, configs, targets

        healthz_router = APIRouter()

        @healthz_router.get("/healthz")
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "mode": "sdk-backed-job-scaffold",
                "jobs": ["garak.audit"],
                "entities": ["auditor_audit_config", "auditor_audit_target"],
            }

        crud_prefix = "/v2/workspaces/{workspace}"
        return [
            RouterSpec(
                router=healthz_router,
                tag="Garak Plugin",
                description="Garak plugin scaffold routes.",
                prefix="/v1",
            ),
            RouterSpec(
                router=configs.router,
                tag="Garak Configs",
                description="Audit configuration CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=targets.router,
                tag="Garak Targets",
                description="Audit target CRUD.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                router=artifacts.router,
                tag="Garak Jobs",
                description="Audit job aggregate artifact download.",
                prefix=crud_prefix,
            ),
            RouterSpec(
                add_job_routes(AuditJob, service_name=_LEGACY_JOB_SOURCE, authz=scope.child("audit")),
                tag="Garak Jobs",
                description="Audit job submission and retrieval.",
                prefix=crud_prefix,
            ),
        ]
