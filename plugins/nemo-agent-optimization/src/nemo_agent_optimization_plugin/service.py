# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP surface for agent optimization runs.

Mounts the router job's submit / list / get / delete routes under
``/apis/agent-optimization/v2/workspaces/{workspace}/jobs/run-strategy``, plus a
listing of the strategies this deployment has installed at
``/apis/agent-optimization/v2/strategies``.

This plugin owns its own API segment rather than borrowing the agents one: the
agents plugin knows nothing about optimization, and a route it does not declare
is not a route it can mount.  Installing this distribution is what makes the
endpoint — and its ``agent-optimization.*`` permissions — exist.
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_agent_optimization_plugin.discovery import discover_strategies
from nemo_agent_optimization_plugin.jobs.run_strategy import RunStrategyJob
from nemo_agent_optimization_plugin.schemas.strategies import (
    STRATEGIES_ROUTE,
    OptimizationStrategyList,
)
from nemo_helix_plugin.authz import AuthzScope, CallerKind, path_rule
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

#: The scope every route below is minted under.  ``add_job_routes`` is inert
#: without one — it emits unauthz'd routes — so this is what gives the
#: collection its ``agent-optimization.*`` permissions and
#: ``agent-optimization:read`` / ``:write`` scopes.  Sibling plugins keep this
#: in an ``authz`` module because several route modules share it; here the
#: service is the only caller.
scope = AuthzScope("agent-optimization")

STRATEGIES_LIST_PERMISSION = scope.child("strategies").permission(
    "list",
    description="List the optimization strategies installed on the platform",
)


def _strategies_router() -> APIRouter:
    """A listing of what ``--strategy`` accepts, answered by the platform.

    Deliberately not workspace-scoped: which strategies exist is a property of
    what this deployment has installed, identical in every workspace, so
    scoping it to one would imply a per-workspace answer that can never differ.
    """
    router = APIRouter()

    @router.get(
        STRATEGIES_ROUTE,
        response_model=OptimizationStrategyList,
        summary="List installed optimization strategies",
        tags=["Agent Optimization"],
    )
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[STRATEGIES_LIST_PERMISSION])
    async def list_strategies() -> OptimizationStrategyList:
        """Every installed strategy, sorted by the name ``--strategy`` takes.

        Each entry is the object its own plugin declared, forwarded as-is: the
        listing describes strategies, and only the plugin shipping one knows
        what it optimizes.
        """
        return OptimizationStrategyList(data=discover_strategies())

    return router


class AgentOptimizationService(NemoService):
    """Plugin service contributing the agent optimization job collection."""

    name: ClassVar[str] = "agent-optimization"
    dependencies: ClassVar[list[str]] = ["entities", "auth", "jobs", "files"]

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                add_job_routes(
                    RunStrategyJob,
                    service_name="nemo-agent-optimization-plugin-run-strategy",
                    authz=scope.child("run-strategy"),
                ),
                tag="Agent Optimization",
                description="Submit and track agent optimization runs (strategy-dispatched).",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                _strategies_router(),
                tag="Agent Optimization",
                description="Discover the optimization strategies this deployment has installed.",
                prefix="/v2",
            ),
        ]
