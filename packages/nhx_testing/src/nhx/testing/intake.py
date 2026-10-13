# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An in-memory Intake service for unit tests.

Answers the Intake calls evaluation publishing makes — ATIF and OTLP ingest, evaluator results,
trace lookup by session, and evaluation get/patch — and records each request, behind a real
:class:`AsyncIntakeClient`::

    intake = InMemoryIntakeService()
    await publish_to_intake(result, client=intake.client(), experiment_id="exp-1")
    assert intake.atif_requests[0]["session_id"] == "run-1:t-1"

Code that builds its own HTTP clients can be pointed at the service with ``respx`` instead::

    with respx.mock:
        respx.route(host="intake.test").mock(side_effect=intake)
"""

import asyncio
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import unquote

import httpx
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nemo_helix_plugin.intake.types import EvaluationResponse, EvaluatorResult, IngestResponse, Trace
from nhx.testing.client import mock_async_nemo_client

#: Base URL of :meth:`InMemoryIntakeService.client`; route this host to the service under ``respx``.
INTAKE_TEST_BASE_URL = "http://intake.test"

_BASE = r"^/apis/intake/v2/workspaces/(?P<workspace>[^/]+)"
_ATIF = re.compile(_BASE + r"/ingest/atif$")
_OTLP = re.compile(_BASE + r"/ingest/otlp/v1/traces$")
_EVALUATOR_RESULTS = re.compile(_BASE + r"/evaluator-results$")
_TRACES = re.compile(_BASE + r"/traces$")
_EVALUATION = re.compile(_BASE + r"/evaluations/(?P<name>[^/]+)$")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)

#: How a configured call fails: an HTTP status to answer with, or an exception to raise.
Failure = int | Exception


class InMemoryIntakeService:
    """Records Intake requests and answers them from configurable in-memory state."""

    def __init__(self) -> None:
        #: ATIF ingest bodies, each with the request's ``workspace`` added.
        self.atif_requests: list[dict[str, object]] = []
        #: The event loop each ATIF ingest request ran on.
        self.atif_loops: list[asyncio.AbstractEventLoop] = []
        #: OTLP ingest requests as ``{"workspace": ..., "body": <protobuf bytes>}``.
        self.otlp_requests: list[dict[str, object]] = []
        #: Evaluator-result bodies, each with the request's ``workspace`` added.
        self.evaluator_result_requests: list[dict[str, object]] = []
        #: Session ids whose traces were looked up, in order.
        self.trace_lookups: list[str] = []
        #: Evaluations fetched, as ``{"name": ..., "workspace": ...}``.
        self.evaluation_lookups: list[dict[str, str]] = []
        #: Evaluation patches, as ``{"name": ..., **patch_body}``.
        self.evaluation_patches: list[dict[str, object]] = []

        #: Root span id prefix for trace lookups; each session resolves to ``"<prefix>:<session id>"``.
        #: ``None`` makes every trace lookup come back empty.
        self.root_span_prefix: str | None = "span"
        #: When set, ATIF ingest fails instead of recording the request: an ``int`` answers with that
        #: HTTP status, an exception is raised from the transport (e.g. ``httpx.ConnectError``).
        self.atif_failure: Failure | None = None
        #: When this returns True for an ATIF body, ingest answers 422 instead of recording it.
        self.atif_rejects: Callable[[dict[str, object]], bool] | None = None
        #: Seconds each ATIF ingest takes before answering.
        self.atif_delay_sec: float = 0.0
        #: Ingest errors reported back for every OTLP request.
        self.otlp_errors: list[str] = []
        #: Session ids whose evaluator results fail with a 500.
        self.failing_evaluator_result_sessions: set[str] = set()
        #: Whether the evaluation exists; ``False`` answers evaluation lookups with 404.
        self.evaluation_exists: bool = True
        #: When set, evaluation lookups fail this way (see :attr:`atif_failure`).
        self.evaluation_failure: Failure | None = None
        #: When set, evaluation patches fail this way (see :attr:`atif_failure`).
        self.patch_failure: Failure | None = None
        #: Metadata returned on the evaluation.
        self.evaluation_metadata: dict[str, str] = {}

    def client(self, *, workspace: str | None = "default") -> AsyncIntakeClient:
        """Return a real async Intake client backed by this service."""
        return mock_async_nemo_client(self, AsyncIntakeClient, base_url=INTAKE_TEST_BASE_URL, workspace=workspace)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        path = unquote(request.url.path)
        if match := _ATIF.match(path):
            return await self._atif(request, match["workspace"])
        if match := _OTLP.match(path):
            self.otlp_requests.append({"workspace": match["workspace"], "body": request.content})
            return httpx.Response(200, json=IngestResponse(errors=list(self.otlp_errors)).model_dump(mode="json"))
        if match := _EVALUATOR_RESULTS.match(path):
            return self._evaluator_result(request, match["workspace"])
        if match := _TRACES.match(path):
            return self._traces(request, match["workspace"])
        if match := _EVALUATION.match(path):
            return self._evaluation(request, match["workspace"], match["name"])
        return httpx.Response(404, json={"detail": f"no in-memory Intake route for {request.method} {path}"})

    async def _atif(self, request: httpx.Request, workspace: str) -> httpx.Response:
        self.atif_loops.append(asyncio.get_running_loop())
        if self.atif_delay_sec:
            await asyncio.sleep(self.atif_delay_sec)
        if self.atif_failure is not None:
            return _fail(self.atif_failure, "ATIF ingest failed")
        body = {**json.loads(request.content), "workspace": workspace}
        if self.atif_rejects is not None and self.atif_rejects(body):
            return httpx.Response(422, json={"detail": "ATIF trajectory rejected"})
        self.atif_requests.append(body)
        return httpx.Response(204)

    def _evaluator_result(self, request: httpx.Request, workspace: str) -> httpx.Response:
        body = {**json.loads(request.content), "workspace": workspace}
        if body["session_id"] in self.failing_evaluator_result_sessions:
            return httpx.Response(500, json={"detail": f"evaluator result failed for {body['session_id']}"})
        self.evaluator_result_requests.append(body)
        result = EvaluatorResult.model_validate(
            {
                **body,
                "evaluator_result_id": f"evaluator-result-{len(self.evaluator_result_requests)}",
                "created_at": _NOW,
                "ingested_at": _NOW,
            }
        )
        return httpx.Response(200, json=result.model_dump(mode="json"))

    def _traces(self, request: httpx.Request, workspace: str) -> httpx.Response:
        session_id = json.loads(request.url.params["filter"])["session_id"]
        self.trace_lookups.append(session_id)
        traces = []
        if self.root_span_prefix is not None:
            trace = Trace.model_validate(
                {
                    "id": f"trace-{session_id}",
                    "root_span_id": f"{self.root_span_prefix}:{session_id}",
                    "session_id": session_id,
                    "workspace": workspace,
                    "started_at": _NOW,
                    "status": "OK",
                }
            )
            traces.append(trace.model_dump(mode="json"))
        pagination = {
            "page": 1,
            "page_size": len(traces) or 1,
            "current_page_size": len(traces),
            "total_pages": 1,
            "total_results": len(traces),
        }
        return httpx.Response(200, json={"data": traces, "pagination": pagination})

    def _evaluation(self, request: httpx.Request, workspace: str, name: str) -> httpx.Response:
        if request.method == "GET":
            self.evaluation_lookups.append({"name": name, "workspace": workspace})
            if self.evaluation_failure is not None:
                return _fail(self.evaluation_failure, "evaluation lookup failed")
        elif request.method == "PATCH":
            if self.patch_failure is not None:
                return _fail(self.patch_failure, "evaluation patch failed")
            self.evaluation_patches.append({"name": name, **json.loads(request.content)})
        else:
            return httpx.Response(405, json={"detail": f"{request.method} is not supported here"})
        if not self.evaluation_exists:
            return httpx.Response(404, json={"detail": f"evaluation {workspace}/{name} not found"})
        evaluation = EvaluationResponse(
            id=f"evaluation-{name}",
            name=name,
            workspace=workspace,
            experiment_ids=[],
            dataset_name="",
            metadata=dict(self.evaluation_metadata),
        )
        return httpx.Response(200, json=evaluation.model_dump(mode="json"))


def _fail(failure: Failure, detail: str) -> httpx.Response:
    if isinstance(failure, Exception):
        raise failure
    return httpx.Response(failure, json={"detail": detail})
