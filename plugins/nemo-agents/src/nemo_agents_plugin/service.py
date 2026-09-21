# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agents plugin service — registers agent lifecycle management on the NeMo Platform."""

from __future__ import annotations

import logging
from typing import ClassVar, NamedTuple

from nemo_agents_plugin.api.v2._perms import GatewayPerms
from nemo_agents_plugin.authz import scope
from nemo_platform_plugin.authz import Permission
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.jobs.routes import add_job_routes
from nemo_platform_plugin.service import NemoService, RouterSpec

logger = logging.getLogger(__name__)


class _JobCollection(NamedTuple):
    """One agents job collection — the single source of truth for both its permission
    sub-namespace and its mounted router, so the two can't drift."""

    job_cls: type[NemoJob]
    subname: str  # permission sub-namespace suffix -> agents.<subname>.{create,...}
    service_name: str | None  # distinct jobs source (None ⇒ add_job_routes default)
    description: str


# Sub-names are concise and stable and need not match the job's URL path segment:
#   EvaluateAgentJob   /jobs/evaluate        -> agents.evaluate
#   EvaluateSuiteJob   /jobs/evaluate-suite  -> agents.suite
#   OptimizeJob        /jobs/optimize        -> agents.optimize
#   OptimizeSkillsJob  /jobs/optimize-skills -> agents.optimize-skills
#   AnalyzeBatchJob    /jobs/analyze         -> agents.analyze
#   ExecuteAgentJob /jobs/execute          -> agents.execute
#   PackageAgentJob    /jobs/package         -> agents.package
# Distinct service_name per job type so each list endpoint filters to rows of its own type only
# (add_job_routes filters source=service_name); sharing the default would let /jobs/<x> pull in
# sibling-type rows and 500 on the wrong schema.
def _job_collections() -> list[_JobCollection]:
    from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchJob
    from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob
    from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteJob
    from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
    from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
    from nemo_agents_plugin.jobs.package_agent import PackageAgentJob
    from nemo_optimization.jobs.optimize import OptimizeJob

    return [
        _JobCollection(EvaluateAgentJob, "evaluate", None, "Submit and track agent evaluation jobs"),
        _JobCollection(
            ExecuteAgentJob,
            "execute",
            "nemo-agents-plugin-execute",
            "Submit and track agent execution jobs.",
        ),
        _JobCollection(
            EvaluateSuiteJob,
            "suite",
            "nemo-agents-plugin-evaluate-suite",
            "Submit and track evaluate-suite jobs (Harbor / NAT eval runner).",
        ),
        _JobCollection(
            OptimizeJob,
            "optimize",
            "nemo-agents-plugin-optimize",
            "Submit and track numeric optimize jobs (Fabric-backed Optuna HPO).",
        ),
        _JobCollection(
            OptimizeSkillsJob,
            "optimize-skills",
            "nemo-agents-plugin-optimize-skills",
            "Submit and track optimize-skills jobs (skills-improvement loop).",
        ),
        _JobCollection(
            AnalyzeBatchJob,
            "analyze",
            "nemo-agents-plugin-analyze",
            "Submit and track analyze jobs (eval-suite batch analysis).",
        ),
        _JobCollection(
            PackageAgentJob,
            "package",
            "nemo-agents-plugin-package",
            "Submit and track agent container-image packaging jobs.",
        ),
    ]


class AgentsService(NemoService):
    """Plugin service that contributes agent CRUD, deployment lifecycle, and gateway proxy routes.

    Registered under the ``nemo.services`` entry-point group.  The platform
    wraps this in a ``NemoServiceAdapter`` at startup and mounts all routes
    under ``/apis/agents``.

    The :class:`~nemo_agents_plugin.runner.controller.AgentDeploymentController`
    reconcile loop is registered separately under the ``nemo.controllers``
    entry-point group and managed by the platform runner — this service does
    not own the controller lifecycle.
    """

    name: ClassVar[str] = "agents"
    dependencies: ClassVar[list[str]] = ["entities", "auth", "secrets", "jobs", "files", "inference-gateway"]

    def get_routers(self) -> list[RouterSpec]:
        from nemo_agents_plugin.api.v2 import (
            agents,
            deployment_logs,
            deployments,
            environments,
            gateway,
            sessions,
        )

        _prefix = "/v2/workspaces/{workspace}"
        specs: list[RouterSpec] = [
            RouterSpec(agents.router, tag="Agents", description="Agent CRUD", prefix=_prefix),
            RouterSpec(deployments.router, tag="Agent Deployments", description="Deployment lifecycle", prefix=_prefix),
            RouterSpec(sessions.router, tag="Agent Sessions", description="Session lifecycle", prefix=_prefix),
            RouterSpec(
                environments.router,
                tag="Agent Environments",
                description="AgentEnvironment, EnvironmentSpec, and ComputeSpec CRUD",
                prefix=_prefix,
            ),
            RouterSpec(
                deployment_logs.router,
                tag="Agent Deployments",
                description="Per-deployment log retrieval",
                prefix=_prefix,
            ),
            RouterSpec(
                gateway.router, tag="Agent Gateway", description="Proxy to running agent deployments", prefix=_prefix
            ),
        ]
        # Job-collection routers, derived from the single _job_collections() source so a new job
        # can't be wired here but missed in the permission map (or vice versa).
        for collection in _job_collections():
            specs.append(
                RouterSpec(
                    add_job_routes(
                        collection.job_cls,
                        service_name=collection.service_name,
                        authz=scope.child(collection.subname),
                    ),
                    tag="Agents",
                    description=collection.description,
                    prefix=_prefix,
                )
            )
        return specs

    def extra_role_permissions(self) -> dict[str, list[Permission]]:
        # A Viewer must be able to invoke a deployed agent through the gateway proxy — the grant
        # the pre-derivation authz gave Viewer explicitly. The permission's suffix (`invoke`)
        # isn't `read`/`list`, so the default heuristic assigns it to Editor only; grant Viewer
        # here. (Editor still gets it via that same default suffix heuristic.)
        return {"Viewer": [GatewayPerms.INVOKE]}
