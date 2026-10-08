# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the Agent Hardener plugin.

Mounts, under ``/apis/agent-hardener``: ``/v1/healthz``; read routes over ``AgentHardenerRun`` plus
``apply-mitigation`` (write the hardened workflow onto the agent) and ``compose-defense``
(preview a selected defense subset); ``AgentHardenerManifest`` CRUD/init/inspect; the live event
relay (ingest from the run + SSE stream to Studio); and the war-game job collection. Runs and
manifests are created by the war-game job and the manifests API via the entities SDK.
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_agent_hardener_plugin.authz import scope
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.service import NemoService, RouterSpec


class AgentHardenerPluginService(NemoService):
    """Agent Hardener plugin service. Exposes healthz and read-only access to war-game runs.

    Route authz lives on the handlers themselves (``@scope.read``/``@scope.write`` + ``@path_rule``);
    permission ids come from :mod:`nemo_agent_hardener_plugin._perms`, and the war-game job collection is
    ruled via ``add_job_routes``/``job_route_factory(authz=...)`` in :mod:`~.api.v2.jobs`.
    """

    name: ClassVar[str] = "agent-hardener"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_agent_hardener_plugin.api.v2 import events, jobs, manifests, runs

        healthz_router = APIRouter()

        @healthz_router.get("/healthz")
        @scope.read
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])  # authenticated, no permission required
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "jobs": ["agent-hardener.war-game", "agent-hardener.synth"],
                "entities": ["agent_hardener_run", "agent_hardener_manifest"],
            }

        return [
            RouterSpec(
                router=healthz_router,
                tag="Agent Hardener Plugin",
                description="Agent Hardener plugin health.",
                prefix="/v1",
            ),
            RouterSpec(
                router=runs.router,
                tag="Agent Hardener Runs",
                description="Read-only access to war-game runs.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                router=manifests.router,
                tag="Agent Hardener Manifests",
                description="Named war-game targets: init (create), list, get, delete.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                router=events.router,
                tag="Agent Hardener Events",
                description="Live run-event ingest (from the run) + SSE stream (to Studio).",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                router=jobs.router,
                tag="Agent Hardener Jobs",
                description="Submit and manage the war-game job.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                router=jobs.synth_router,
                tag="Agent Hardener Synth Jobs",
                description="Submit and manage the benign-suite synthesis job.",
                prefix="/v2/workspaces/{workspace}/synth-benign",
            ),
        ]
