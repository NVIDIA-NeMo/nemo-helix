# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model reference is resolved only when its submitter may read what it routes to.

Drives the real evaluate job routes and the real ``/live`` route, with the request-scoped client
pointed at a fake platform that grants or denies each model and VirtualModel read. Every model is
served through a VirtualModel of the same name, so reading either is enough for the model entity;
a user-managed VirtualModel, which serves something else under that name, must itself be readable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_evals.api.v2 import live as live_routes
from nemo_evals.jobs.evaluate import EvaluateInputSpec, EvaluateJob
from nemo_evals.jobs.metric_resolution import caller_model_read_authorizer, resolve_metrics_to_inline
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_entity_client, get_nemo_client
from nemo_helix_plugin.entity_client import get_entity_client as get_live_entity_client
from nemo_helix_plugin.jobs.routes import add_job_routes
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.values import ModelRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore

_PLATFORM = "http://platform.test"
_JOBS = "/apis/evals/v2/workspaces/default/evaluate/jobs"
_LIVE = "/v2/workspaces/default/evaluate/live"
_MODEL_PATH = "/apis/models/v2/workspaces/default/models/judge"
_VIRTUAL_MODEL_PATH = "/apis/inference-gateway/v2/workspaces/default/virtual-models/judge"

VirtualModelKind = Literal["passthrough", "user", "missing"]


@dataclass
class _Platform:
    """What the caller may read of model ``default/judge`` and the VirtualModel of the same name."""

    model_readable: bool = True
    model_exists: bool = True
    virtual_model: VirtualModelKind = "passthrough"
    virtual_model_readable: bool = True
    reads: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.reads.append(path)
        if path == _VIRTUAL_MODEL_PATH:
            if self.virtual_model == "missing":
                return httpx.Response(404, json={"detail": "not found"})
            if not self.virtual_model_readable:
                return httpx.Response(403, json={"detail": "Forbidden"})
            default_entity = "default/judge" if self.virtual_model == "passthrough" else "default/other-model"
            return httpx.Response(
                200,
                json={
                    "name": "judge",
                    "workspace": "default",
                    "autoprovisioned": self.virtual_model == "passthrough",
                    "default_model_entity": default_entity,
                },
            )
        if path == _MODEL_PATH:
            if not self.model_exists:
                return httpx.Response(404, json={"detail": "not found"})
            if not self.model_readable:
                return httpx.Response(403, json={"detail": "Forbidden"})
            return httpx.Response(
                200,
                json={
                    "id": "model-judge",
                    "name": "judge",
                    "workspace": "default",
                    "created_at": "2020-01-01T00:00:00Z",
                    "updated_at": "2020-01-01T00:00:00Z",
                },
            )
        return httpx.Response(404, json={"detail": f"unexpected path {path}"})

    def client(self) -> AsyncNemoClient:
        """The caller's request-scoped client, as the platform injects it."""
        return AsyncNemoClient(
            base_url=_PLATFORM,
            workspace="default",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )


def _job_response(spec: dict[str, Any]) -> MagicMock:
    job = SimpleNamespace(
        id="job-1",
        name="eval-job",
        description=None,
        project=None,
        workspace="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        spec=spec,
        status="created",
        status_details=None,
        error_details=None,
        ownership=None,
        custom_fields=None,
        control=None,
        attempt_id="attempt-1",
    )
    response = MagicMock()
    response.data.return_value = job
    return response


class _RecordingJobs:
    """Stands in for core Jobs; records whether a job was created or changed."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    async def create_job(self, *, workspace: str, body: Any) -> MagicMock:
        del workspace
        self.created.append(body.spec)
        return _job_response(body.spec)

    async def get_job(self, *, name: str, workspace: str) -> MagicMock:
        del name, workspace
        return _job_response({})

    async def update_job(self, *, name: str, workspace: str, body: Any) -> MagicMock:
        del name, workspace
        self.updated.append(body.spec)
        return _job_response(body.spec or {})


def _judge_ref_metric() -> dict[str, Any]:
    judge = LLMJudgeMetric(
        model=ModelRef("default/judge"),
        scores=[RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))],
    )
    return bundle_metric(judge, InlineMetricBundlePackager()).model_dump(mode="json")


def _exact_match_metric() -> dict[str, Any]:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{sample.output_text}}")
    return bundle_metric(metric, InlineMetricBundlePackager()).model_dump(mode="json")


def _judge_ref_spec() -> dict[str, Any]:
    return {"metrics": [_judge_ref_metric()], "dataset": [{"output_text": "hello"}]}


def _target_ref_spec() -> dict[str, Any]:
    return {
        "metrics": [_exact_match_metric()],
        "dataset": [{"question": "Capital of France?", "expected": "Paris"}],
        "target": "default/judge",
        # Studio's shape: an online-only key selects the online params member.
        "params": {"ignore_request_failure": False, "parallelism": 1},
        "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
    }


_SPECS = {"judge-ref": _judge_ref_spec, "target-ref": _target_ref_spec}


def _jobs_app(platform: _Platform) -> FastAPI:
    app = FastAPI()
    app.include_router(add_job_routes(EvaluateJob), prefix="/apis/evals/v2/workspaces/{workspace}")
    app.dependency_overrides[get_nemo_client] = platform.client
    app.dependency_overrides[get_entity_client] = lambda: MagicMock()
    return app


def _submit(
    platform: _Platform, spec: dict[str, Any], *, method: Literal["POST", "PATCH"]
) -> tuple[int, str, _RecordingJobs]:
    jobs = _RecordingJobs()
    client = TestClient(_jobs_app(platform))
    with patch("nemo_helix_plugin.jobs.api_factory.AsyncJobsClient.from_client", return_value=jobs):
        if method == "POST":
            response = client.post(_JOBS, json={"spec": spec})
        else:
            response = client.patch(f"{_JOBS}/eval-job", json={"spec": spec})
    return response.status_code, response.text, jobs


_DENIED = {
    "neither-readable": _Platform(model_readable=False, virtual_model_readable=False),
    "model-readable-user-virtual-model-not": _Platform(virtual_model="user", virtual_model_readable=False),
}

_ALLOWED = {
    "passthrough-and-model-readable": _Platform(),
    "user-virtual-model-readable-model-not": _Platform(model_readable=False, virtual_model="user"),
    "user-virtual-model-with-no-model-entity": _Platform(model_exists=False, virtual_model="user"),
}


@pytest.mark.parametrize("method", ["POST", "PATCH"])
@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize("platform", _DENIED)
def test_a_ref_the_caller_cannot_read_is_refused_with_403_and_nothing_is_stored(
    platform: str, spec: str, method: Literal["POST", "PATCH"]
) -> None:
    status, body, jobs = _submit(_DENIED[platform], _SPECS[spec](), method=method)

    assert status == 403, body
    assert jobs.created == [] and jobs.updated == []


@pytest.mark.parametrize("spec", _SPECS)
@pytest.mark.parametrize("platform", _ALLOWED)
def test_a_ref_readable_through_the_model_or_its_virtual_model_is_accepted(platform: str, spec: str) -> None:
    status, body, jobs = _submit(_ALLOWED[platform], _SPECS[spec](), method="POST")

    assert status == 201, body
    assert len(jobs.created) == 1


def _live_app(platform: _Platform) -> FastAPI:
    app = FastAPI()
    app.include_router(live_routes.router, prefix="/v2/workspaces/{workspace}")
    app.dependency_overrides[get_live_entity_client] = lambda: None
    app.dependency_overrides[get_nemo_client] = platform.client
    semaphore = asyncio.Semaphore(4)
    app.dependency_overrides[live_routes.get_live_semaphore] = lambda: semaphore
    return app


@pytest.mark.parametrize("platform", _DENIED)
def test_live_refuses_a_judge_ref_the_caller_cannot_read(platform: str) -> None:
    response = TestClient(_live_app(_DENIED[platform])).post(
        _LIVE,
        json={"dataset": [{"output_text": "hello"}], "metrics": [_judge_ref_metric()]},
    )

    assert response.status_code == 403, response.text


@pytest.mark.parametrize("platform", _DENIED)
def test_live_refuses_a_target_ref_the_caller_cannot_read(platform: str) -> None:
    spec = _target_ref_spec()
    response = TestClient(_live_app(_DENIED[platform])).post(
        _LIVE,
        json={
            "dataset": spec["dataset"],
            "metrics": spec["metrics"],
            "target": spec["target"],
            "prompt_template": spec["prompt_template"],
        },
    )

    assert response.status_code == 403, response.text


async def test_to_spec_without_a_platform_client_still_takes_ref_free_specs() -> None:
    """Offline and inline evaluations never needed a platform client; checking refs must not start needing one."""
    inline_target = _target_ref_spec() | {"target": {"url": "https://api.example.com/v1", "name": "external"}}
    for spec in (_judge_ref_spec() | {"metrics": [_exact_match_metric()]}, inline_target):
        await EvaluateJob.to_spec(
            EvaluateInputSpec.model_validate(spec),
            workspace="default",
            entity_client=None,
            async_sdk=None,
            is_local=True,
        )


@pytest.mark.parametrize("spec", _SPECS)
async def test_to_spec_without_a_platform_client_refuses_a_ref_clearly(spec: str) -> None:
    with pytest.raises(ValueError, match="requires a platform client"):
        await EvaluateJob.to_spec(
            EvaluateInputSpec.model_validate(_SPECS[spec]()),
            workspace="default",
            entity_client=None,
            async_sdk=None,
            is_local=True,
        )


async def test_an_authorizer_given_only_as_authorize_still_guards_model_refs() -> None:
    """Taskset submission passes one ``authorize`` for stored metrics and models alike; it must keep covering models."""
    checked: list[str] = []

    async def deny(method: str, path: str) -> None:
        checked.append(path)
        raise HTTPException(status_code=403, detail="denied")

    with pytest.raises(HTTPException) as denied:
        await resolve_metrics_to_inline(
            [EvaluateInputSpec.model_validate(_judge_ref_spec()).metrics[0]],
            workspace="default",
            entity_client=None,
            async_client=_Platform().client(),
            authorize=deny,
        )

    assert denied.value.status_code == 403
    assert _MODEL_PATH in checked


async def test_caller_authorizer_checks_only_model_and_virtual_model_reads() -> None:
    authorize = caller_model_read_authorizer(_Platform().client())

    with pytest.raises(ValueError, match="model or VirtualModel"):
        await authorize("GET", "/apis/evals/v2/workspaces/default/metrics/judge")


async def test_caller_authorizer_reuses_an_allowed_read_within_one_submission() -> None:
    platform = _Platform()
    authorize = caller_model_read_authorizer(platform.client())

    await authorize("GET", _MODEL_PATH)
    await authorize("GET", _MODEL_PATH)

    assert platform.reads == [_MODEL_PATH]


@pytest.mark.parametrize(
    "target",
    ["default/judge", {"url": "https://api.example.com/v1", "name": "external"}],
    ids=["ref", "inline-external"],
)
def test_params_that_do_not_fit_a_model_target_are_a_422_not_a_500(target: Any) -> None:
    spec = _target_ref_spec() | {"target": target, "params": {"parallelism": 1}}

    status, body, jobs = _submit(_Platform(), spec, method="POST")

    assert status == 422, body
    assert "model target requires RunConfigOnlineModel" in body
    assert jobs.created == []


def test_a_target_ref_is_stored_as_the_model_it_resolves_to() -> None:
    """Studio's params widen to the model params, and the stored target is the resolved gateway route."""
    status, body, jobs = _submit(_Platform(), _target_ref_spec(), method="POST")

    assert status == 201, body
    [stored] = jobs.created
    assert stored["target"]["url"] == f"{_PLATFORM}/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    assert stored["target"]["name"] == "judge"
    assert stored["params"]["ignore_request_failure"] is False


async def test_a_target_ref_resolved_over_loopback_is_stored_with_the_service_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded auth resolves over loopback; the job must get an address it can reach."""
    monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
    monkeypatch.setenv("NHX_INTERNAL_BASE_URL", "http://nemo-api:8080")
    platform = _Platform()
    loopback_client = AsyncNemoClient(
        base_url="http://localhost:8080",
        workspace="default",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(platform.handle)),
    )

    spec = await EvaluateJob.to_spec(
        EvaluateInputSpec.model_validate(_target_ref_spec()),
        workspace="default",
        entity_client=None,
        async_sdk=loopback_client,
        is_local=False,
    )

    assert spec.model_dump()["target"]["url"] == (
        "http://nemo-api:8080/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("default/judge", "not found"),
        ("judge", "should match pattern"),
    ],
    ids=["unknown", "malformed"],
)
def test_a_target_ref_that_names_no_model_is_a_422(target: str, message: str) -> None:
    platform = _Platform(model_exists=False, virtual_model="missing")

    status, body, jobs = _submit(platform, _target_ref_spec() | {"target": target}, method="POST")

    assert status == 422, body
    assert message in body
    assert jobs.created == []


@pytest.mark.parametrize("field", ["target", "judge"])
def test_an_inline_keyless_gateway_model_is_refused_at_submit(field: str) -> None:
    gateway = {"url": f"{_PLATFORM}/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1", "name": "judge"}
    if field == "target":
        spec = _target_ref_spec() | {"target": gateway}
    else:
        judge = LLMJudgeMetric(
            model=gateway,
            scores=[RangeScore(name="quality", minimum=0, maximum=1, parser=JSONScoreParser(json_path="quality"))],
        )
        spec = _judge_ref_spec() | {
            "metrics": [bundle_metric(judge, InlineMetricBundlePackager()).model_dump(mode="json")]
        }

    status, body, jobs = _submit(_Platform(), spec, method="POST")

    assert status == 422, body
    assert "workspace/name" in body and "api_key_secret" in body
    assert jobs.created == []


def test_an_inline_gateway_target_with_its_own_key_is_kept_as_written() -> None:
    url = f"{_PLATFORM}/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    spec = _target_ref_spec() | {"target": {"url": url, "name": "judge", "api_key_secret": "judge-key"}}

    status, body, jobs = _submit(_Platform(), spec, method="POST")

    assert status == 201, body
    assert jobs.created[0]["target"]["url"] == url
