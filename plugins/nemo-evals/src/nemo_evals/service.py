# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the evals plugin scaffold."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_evals.api.v2 import live as live_routes
from nemo_evals.api.v2 import metrics as metrics_routes
from nemo_evals.api.v2 import results as results_routes
from nemo_evals.api.v2 import tasks as tasks_routes
from nemo_evals.api.v2 import tasksets as tasksets_routes
from nemo_evals.authz import scope
from nemo_evals.core import say_hello
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.evaluate import EvaluateJob
from nemo_evals.jobs.retrieve_eval import RetrieveEvalJob
from nemo_evals.schema import HelloResponse
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

#: The ``source`` tag for agent-evaluate job records, passed as ``add_job_routes(..., service_name=)``.
#: Distinct from ``EvaluateJob``'s derived ``nemo-evals`` source: the evals plugin owns two job
#: types (row ``evaluate`` + ``agent-evaluate``), and a shared source makes each collection's list
#: endpoint 500 when it renders the other type's spec. ``EvaluateJob`` keeps the module-derived default.
AGENT_EVAL_JOB_SOURCE = "nemo-evals.agent-evaluate"

#: The ``source`` tag for retrieve-eval job records. Same rationale as
#: :data:`AGENT_EVAL_JOB_SOURCE`: BEIR retrieval specs share none of the row evaluate shape, so a
#: distinct source keeps each collection's list endpoint from rendering the other type's spec.
RETRIEVE_EVAL_JOB_SOURCE = "nemo-evals.retrieve-eval"


class EvaluatorPerms(PermissionSet, namespace="evaluator"):
    """Permissions owned by the evals plugin's hand-written routes.

    The ``EvaluateJob`` collection's permissions (``evaluator.create`` etc.) are stamped
    onto the factory routes and derived from there; only the bespoke ``hello`` route's
    permission is declared here.
    """

    HELLO_READ = perm("Read the evaluator hello greeting", suffix="hello.read")


class EvaluatorPluginService(NemoService):
    """Service surface for the evals plugin."""

    name: ClassVar[str] = "evals"
    dependencies: ClassVar[list[str]] = ["nhx-evals-sdk", "entities", "files"]

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()
        jobs_router = add_job_routes(EvaluateJob, authz=scope)
        # A distinct ``service_name`` gives agent-evaluate jobs their own ``source`` tag so this
        # collection's list endpoint doesn't return (and 500 rendering) row EvaluateJob specs, and
        # vice versa. EvaluateJob keeps its derived ``nemo-evals`` source. See AGENT_EVAL_JOB_SOURCE.
        agent_jobs_router = add_job_routes(AgentEvalJob, service_name=AGENT_EVAL_JOB_SOURCE, authz=scope)
        retrieve_jobs_router = add_job_routes(
            RetrieveEvalJob,
            service_name=RETRIEVE_EVAL_JOB_SOURCE,
            authz=scope,
        )

        @router.get("/healthz")
        @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
        async def healthz() -> dict[str, object]:
            return {
                "plugin": self.name,
                "status": "ok",
                "mode": "sdk-backed-job-scaffold",
                "jobs": ["evals.evaluate", "evals.agent-evaluate", "evals.retrieve-eval"],
            }

        return [
            RouterSpec(
                router=router,
                tag="Evals Plugin",
                description="Evals plugin scaffold routes.",
                prefix="/v1",
            ),
            RouterSpec(
                # curl localhost:8080/apis/evals/v1/hello/friend
                router=_build_hello_router(),
                tag="Evals Plugin Hello Routes",
                description="Evaluator hello endpoint.",
                prefix="/v1",
            ),
            RouterSpec(
                # POST /apis/evals/v2/workspaces/{workspace}/evaluate/jobs.
                router=jobs_router,
                tag="Evals Plugin Jobs Routes",
                description="Evals plugin jobs routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # POST /apis/evals/v2/workspaces/{workspace}/agent-evaluate/jobs.
                router=agent_jobs_router,
                tag="Evals Plugin Agent Eval Jobs Routes",
                description="Evals plugin agent-evaluation job routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # POST /apis/evals/v2/workspaces/{workspace}/retrieve-eval/jobs.
                router=retrieve_jobs_router,
                tag="Evals Plugin Retrieve Eval Jobs Routes",
                description="Evals plugin BEIR retrieval evaluation job routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evals/v2/workspaces/{workspace}/metrics.
                router=metrics_routes.router,
                tag="Evals Plugin Metrics Routes",
                description="Stored metric (metric bundle) CRUD routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # list/get/delete /apis/evals/v2/workspaces/{workspace}/agent-eval-results.
                router=results_routes.agent_eval_results_router,
                tag="Evals Plugin Agent Eval Results Routes",
                description="Queryable agent-evaluation result records.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # list/get/delete /apis/evals/v2/workspaces/{workspace}/eval-results.
                router=results_routes.evaluate_results_router,
                tag="Evals Plugin Eval Results Routes",
                description="Queryable (row) evaluation result records.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evals/v2/workspaces/{workspace}/tasks.
                router=tasks_routes.router,
                tag="Evals Plugin Tasks Routes",
                description="Stored agent-eval task CRUD routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evals/v2/workspaces/{workspace}/tasksets.
                router=tasksets_routes.router,
                tag="Evals Plugin Tasksets Routes",
                description="Stored taskset CRUD routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # POST /apis/evals/v2/workspaces/{workspace}/evaluate/live.
                router=live_routes.router,
                tag="Evals Plugin Live Evaluation Route",
                description="Single-row evaluation run in-process, without creating a job.",
                prefix="/v2/workspaces/{workspace}",
            ),
        ]


def _build_hello_router() -> APIRouter:
    router = APIRouter()

    @router.get("/hello/{name}", response_model=HelloResponse)
    @scope.read
    @path_rule(
        callers=[CallerKind.PRINCIPAL],
        permissions=[EvaluatorPerms.HELLO_READ],
    )
    async def hello(name: str) -> HelloResponse:
        """Greet a name."""
        return HelloResponse(message=say_hello(name))

    return router
