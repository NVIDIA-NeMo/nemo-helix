# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service surface for the evaluator plugin scaffold."""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_evaluator.api.v2 import metrics as metrics_routes
from nemo_evaluator.api.v2 import results as results_routes
from nemo_evaluator.api.v2 import tasks as tasks_routes
from nemo_evaluator.api.v2 import tasksets as tasksets_routes
from nemo_evaluator.authz import scope
from nemo_evaluator.core import say_hello
from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
from nemo_evaluator.jobs.evaluate import EvaluateJob
from nemo_evaluator.jobs.retrieve_eval import RetrieveEvalJob
from nemo_evaluator.schema import HelloResponse
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.jobs.routes import add_job_routes
from nemo_helix_plugin.service import NemoService, RouterSpec

#: The ``source`` tag for agent-evaluate job records, passed as ``add_job_routes(..., service_name=)``.
#: Distinct from ``EvaluateJob``'s derived ``nemo-evaluator`` source: the evaluator plugin owns two job
#: types (row ``evaluate`` + ``agent-evaluate``), and a shared source makes each collection's list
#: endpoint 500 when it renders the other type's spec. ``EvaluateJob`` keeps the derived default so its
#: existing records still match its list.
AGENT_EVAL_JOB_SOURCE = "nemo-evaluator.agent-evaluate"

#: The ``source`` tag for retrieve-eval job records. Same rationale as
#: :data:`AGENT_EVAL_JOB_SOURCE`: BEIR retrieval specs share none of the row evaluate shape, so a
#: distinct source keeps each collection's list endpoint from rendering the other type's spec.
RETRIEVE_EVAL_JOB_SOURCE = "nemo-evaluator.retrieve-eval"


class EvaluatorPerms(PermissionSet, namespace="evaluator"):
    """Permissions owned by the evaluator plugin's hand-written routes.

    The ``EvaluateJob`` collection's permissions (``evaluator.create`` etc.) are stamped
    onto the factory routes and derived from there; only the bespoke ``hello`` route's
    permission is declared here.
    """

    HELLO_READ = perm("Read the evaluator hello greeting", suffix="hello.read")


class EvaluatorPluginService(NemoService):
    """Service surface for the evaluator plugin."""

    name: ClassVar[str] = "evaluator"
    dependencies: ClassVar[list[str]] = ["nemo-evaluator-sdk", "entities", "files"]

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()
        jobs_router = add_job_routes(EvaluateJob, authz=scope)
        # A distinct ``service_name`` gives agent-evaluate jobs their own ``source`` tag so this
        # collection's list endpoint doesn't return (and 500 rendering) row EvaluateJob specs, and
        # vice versa. EvaluateJob keeps its derived ``nemo-evaluator`` source. See AGENT_EVAL_JOB_SOURCE.
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
                "jobs": ["evaluator.evaluate", "evaluator.agent-evaluate", "evaluator.retrieve-eval"],
            }

        return [
            RouterSpec(
                router=router,
                tag="Evaluator Plugin",
                description="Evaluator plugin scaffold routes.",
                prefix="/v1",
            ),
            RouterSpec(
                # curl localhost:8080/apis/evaluator/v1/hello/friend
                router=_build_hello_router(),
                tag="Evaluator Plugin Hello Routes",
                description="Evaluator hello endpoint.",
                prefix="/v1",
            ),
            RouterSpec(
                # POST /apis/evaluator/v2/workspaces/{workspace}/evaluate/jobs.
                router=jobs_router,
                tag="Evaluator Plugin Jobs Routes",
                description="Evaluator plugin jobs routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # POST /apis/evaluator/v2/workspaces/{workspace}/agent-evaluate/jobs.
                router=agent_jobs_router,
                tag="Evaluator Plugin Agent Eval Jobs Routes",
                description="Evaluator plugin agent-evaluation job routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # POST /apis/evaluator/v2/workspaces/{workspace}/retrieve-eval/jobs.
                router=retrieve_jobs_router,
                tag="Evaluator Plugin Retrieve Eval Jobs Routes",
                description="Evaluator plugin BEIR retrieval evaluation job routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evaluator/v2/workspaces/{workspace}/metrics.
                router=metrics_routes.router,
                tag="Evaluator Plugin Metrics Routes",
                description="Stored metric (metric bundle) CRUD routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # list/get/delete /apis/evaluator/v2/workspaces/{workspace}/agent-eval-results.
                router=results_routes.agent_eval_results_router,
                tag="Evaluator Plugin Agent Eval Results Routes",
                description="Queryable agent-evaluation result records.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # list/get/delete /apis/evaluator/v2/workspaces/{workspace}/eval-results.
                router=results_routes.evaluate_results_router,
                tag="Evaluator Plugin Eval Results Routes",
                description="Queryable (row) evaluation result records.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evaluator/v2/workspaces/{workspace}/tasks.
                router=tasks_routes.router,
                tag="Evaluator Plugin Tasks Routes",
                description="Stored agent-eval task CRUD routes.",
                prefix="/v2/workspaces/{workspace}",
            ),
            RouterSpec(
                # CRUD /apis/evaluator/v2/workspaces/{workspace}/tasksets.
                router=tasksets_routes.router,
                tag="Evaluator Plugin Tasksets Routes",
                description="Stored taskset CRUD routes.",
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
