# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP route-level tests for the /live single-row evaluation endpoint.

Drives the real router and the real SDK evaluator, with only the outbound inference call replaced.
The event-loop test is the load-bearing one: it is what catches a regression to ``run_sync``, which
would block the worker for the whole of generation plus judging.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evals.api.schemas import MetricInline
from nemo_evals.api.v2 import live as live_routes
from nemo_evals.entities import MetricBundleEntity
from nemo_evals.jobs import metric_resolution
from nemo_evals.shared.metric_bundles.bundles import bundle_metric
from nemo_evals.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evals.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx_evals_sdk import inference as sdk_inference
from nhx_evals_sdk.execution import benchmark_execution
from nhx_evals_sdk.metrics.exact_match import ExactMatchMetric
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.tool_calling import ToolCallingMetric
from nhx_evals_sdk.values.common import SecretRef, SupportedJobTypes
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore

_BASE = "/v2/workspaces/default/evaluate/live"


def _exact_match_metric() -> dict[str, Any]:
    metric = ExactMatchMetric(reference="{{item.expected}}", candidate="{{sample.output_text}}")
    bundle = bundle_metric(metric, InlineMetricBundlePackager())
    return MetricInline.model_validate_json(bundle.model_dump_json()).model_dump(mode="json")


def _build_app(concurrency: int = 64) -> FastAPI:
    """Build the route's app with its own concurrency cap.

    Each app gets a fresh semaphore, so tests neither contend over one shared cap nor have to
    rebind a module attribute. The default is wide enough that tests which are not about the cap
    never trip it.
    """
    app = FastAPI()
    app.include_router(live_routes.router, prefix="/v2/workspaces/{workspace}")

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        """An unrelated cheap endpoint, used to observe whether the loop is still turning.

        Every metric in these tests is inline with no model references, so reference resolution
        never reaches the entity or platform clients the app overrides to ``None``.
        """
        return {"status": "ok"}

    app.dependency_overrides[get_entity_client] = lambda: None
    # A real client, pointed nowhere: no metric here carries a secret or a model reference, so it is
    # constructed and never called. Faking it would hide a resolution the route really does perform.
    app.dependency_overrides[get_nemo_client] = lambda: AsyncSecretsClient(
        base_url="http://secrets.invalid", workspace="default"
    )
    sem = asyncio.Semaphore(concurrency)
    app.dependency_overrides[live_routes.get_live_semaphore] = lambda: sem
    return app


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(_build_app())


def _chat_response(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_offline_row_scores_without_creating_a_job(client: TestClient) -> None:
    """A row that already carries its output is scored inline and returned synchronously."""
    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "output": "Paris"}],
            "metrics": [_exact_match_metric()],
            "field_mapping": {"output": "output"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output"] == "Paris"
    assert [m["metric"] for m in body["metrics"]] == ["exact-match"]
    assert body["metrics"][0]["error"] is None
    assert body["metrics"][0]["scores"][0]["mean"] == pytest.approx(1.0)


def test_model_target_generates_server_side(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a target the server generates the candidate and returns it alongside the scores."""

    async def fake_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        return _chat_response("Paris")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", fake_inference)

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
            "metrics": [_exact_match_metric()],
            "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
            "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["output"] == "Paris"
    assert body["metrics"][0]["scores"][0]["mean"] == pytest.approx(1.0)


def test_generation_applies_prompt_template_and_system_prompt(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tuned prompt reaches the model. This is the fidelity gap client-side generation had."""
    seen: list[dict[str, Any]] = []

    async def fake_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, max_retries, kwargs
        seen.append(request)
        return _chat_response("Paris")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", fake_inference)

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
            "metrics": [_exact_match_metric()],
            "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
            "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
            "params": {"system_prompt": "Answer with one word.", "inference": {"temperature": 0.0}},
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(seen) == 1
    request = seen[0]
    assert request["messages"][-1]["content"] == "Capital of France?"
    assert any(
        message["role"] == "system" and message["content"] == "Answer with one word." for message in request["messages"]
    )
    assert request["temperature"] == pytest.approx(0.0)


def test_unreachable_target_fails_immediately_without_retrying(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed generation surfaces the upstream status, with no retry budget to burn first.

    The retry loop lives inside ``make_inference_request``, which these tests replace, so counting
    calls would prove nothing. What is asserted instead is the settings handed to it: ``max_retries``
    is the pin that removes the original reason Studio generated client-side.
    """
    seen: list[tuple[int, float | None]] = []

    async def failing_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request
        seen.append((max_retries, kwargs.get("timeout")))
        raise RuntimeError("upstream returned 502 Bad Gateway")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", failing_inference)

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
            "metrics": [_exact_match_metric()],
            "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
            "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
        },
    )
    assert resp.status_code == 502, resp.text
    assert "502 Bad Gateway" in resp.text
    assert seen == [(0, live_routes.LIVE_REQUEST_TIMEOUT_S)]


def test_timeout_returns_an_error_rather_than_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that never returns bounds out at the route's own timeout."""
    monkeypatch.setattr(live_routes, "LIVE_TIMEOUT_S", 0.1)

    async def never_returns(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", never_returns)

    with TestClient(_build_app()) as client:
        resp = client.post(
            _BASE,
            json={
                "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
                "metrics": [_exact_match_metric()],
                "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
                "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
            },
        )
    assert resp.status_code == 504


@pytest.mark.asyncio
async def test_slow_scoring_does_not_block_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrelated cheap endpoint stays responsive for the whole of a live evaluation.

    This is the check that catches ``Evaluator().run_sync(...)``: run_sync spawns a worker thread
    and joins it, pinning this loop for the entire call. Patching the inference function rather
    than the evaluator keeps the real execution path -- and therefore the real sync bridge -- in play.

    What is asserted is the *gap between* consecutive pings, not the latency of any one of them. A
    blocked loop cannot measure itself: while it is pinned, the sampler does not run either, so no
    individual request ever records a long duration. The stall shows up only as a hole between two
    timestamps that bracket it. For the same reason a test that waits for a "generation started"
    signal before probing passes even while the loop is being blocked -- run_sync pins the loop from
    the moment the handler calls it, so that signal can never arrive mid-stall.
    """
    inference_delay = 0.6

    async def slow_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        await asyncio.sleep(inference_delay)
        return _chat_response("Paris")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", slow_inference)

    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    loop = asyncio.get_running_loop()
    ticks: list[float] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def sample_ping() -> None:
            while True:
                ping = await client.get("/ping")
                assert ping.status_code == 200
                ticks.append(loop.time())
                await asyncio.sleep(0.01)

        sampler = asyncio.create_task(sample_ping())
        scored = await asyncio.wait_for(
            client.post(
                _BASE,
                json={
                    "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
                    "metrics": [_exact_match_metric()],
                    "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
                    "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
                },
                timeout=30.0,
            ),
            timeout=30.0,
        )
        sampler.cancel()

    assert scored.status_code == 200, scored.text
    assert len(ticks) > 2, f"the event loop was blocked: only {len(ticks)} ping(s) completed"
    gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False)]
    assert max(gaps) < inference_delay / 2


@pytest.mark.asyncio
async def test_excess_concurrency_is_rejected_not_fanned_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the cap, extra live tests are refused rather than queued into more inference calls."""
    in_flight = 0
    peak = 0

    async def slow_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        nonlocal in_flight, peak
        del model, request, max_retries, kwargs
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.3)
            return _chat_response("Paris")
        finally:
            in_flight -= 1

    monkeypatch.setattr(benchmark_execution, "make_inference_request", slow_inference)

    body = {
        "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
        "metrics": [_exact_match_metric()],
        "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
        "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
    }
    transport = httpx.ASGITransport(app=_build_app(concurrency=1))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(*(client.post(_BASE, json=body, timeout=30.0) for _ in range(3)))

    statuses = sorted(resp.status_code for resp in responses)
    assert 429 in statuses
    assert peak == 1


# ---- judge secret wiring ---------------------------------------------------


def _judge_metric(secret: str | None) -> dict[str, Any]:
    """An inline llm-judge metric whose model authenticates with a platform secret reference."""
    model = Model(
        url="http://judge.test/v1/chat/completions",
        name="judge-model",
        api_key_secret=SecretRef(root=secret) if secret else None,
    )
    metric = LLMJudgeMetric(
        model=model,
        scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="helpfulness"))],
        job_type=SupportedJobTypes.OFFLINE,
    )
    bundle = bundle_metric(metric, CloudpickleMetricBundlePackager())
    return MetricInline.model_validate_json(bundle.model_dump_json()).model_dump(mode="json")


def _app_with_secrets(make_secrets_client, secrets: dict[tuple[str, str], str]) -> tuple[FastAPI, object]:
    app = _build_app()
    secrets_client = make_secrets_client(secrets)
    app.dependency_overrides[get_nemo_client] = lambda: secrets_client
    return app, secrets_client


def test_judge_secret_reaches_the_judge_model(monkeypatch: pytest.MonkeyPatch, make_secrets_client) -> None:
    """End to end: the route's resolver, not the SDK's env-reading default, supplies the judge key.

    The unit tests above exercise the resolver directly; only this one proves it is actually the
    resolver the backend uses, which is what the per-request `LocalBackend` swap is for.
    """
    seen_keys: list[str | None] = []

    async def fake_judge_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries
        # The judge carries its key two ways: preflight passes `api_key=`, while the scoring call
        # (llm_judge.py:362) passes only `client=`, whose AsyncOpenAI was built with the key.
        judge_client = kwargs.get("client")
        seen_keys.append(kwargs.get("api_key") if judge_client is None else judge_client.api_key)
        return _chat_response('{"helpfulness": 4}')

    monkeypatch.setattr(sdk_inference, "make_inference_request", fake_judge_inference)

    app, secrets_client = _app_with_secrets(make_secrets_client, {("default", "judge-key"): "sk-judge-secret"})
    with TestClient(app) as client:
        resp = client.post(
            _BASE,
            json={
                "dataset": [{"question": "Capital of France?", "output": "Paris"}],
                "metrics": [_judge_metric("judge-key")],
                "field_mapping": {"output": "output"},
            },
        )

    assert resp.status_code == 200, resp.text
    assert secrets_client.lookups == [("default", "judge-key")]
    assert seen_keys and set(seen_keys) == {"sk-judge-secret"}


def test_missing_judge_secret_fails_instead_of_scoring(monkeypatch: pytest.MonkeyPatch, make_secrets_client) -> None:
    """An unresolvable judge key must not come back as a score on a 200."""

    async def fake_judge_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        raise AssertionError("inference must not run without a resolved judge key")

    monkeypatch.setattr(sdk_inference, "make_inference_request", fake_judge_inference)

    app, secrets_client = _app_with_secrets(make_secrets_client, {})
    with TestClient(app) as client:
        resp = client.post(
            _BASE,
            json={
                "dataset": [{"question": "Capital of France?", "output": "Paris"}],
                "metrics": [_judge_metric("judge-key")],
                "field_mapping": {"output": "output"},
            },
        )

    # The judge is the only metric here, so nothing scored at all -- a total failure, not a
    # partial one, and it must not come back as a 200.
    assert resp.status_code == 502, resp.text
    assert "judge-key" in resp.text
    assert secrets_client.lookups == [("default", "judge-key")]


def test_one_failing_metric_does_not_hide_the_others(client: TestClient) -> None:
    """A broken metric reports its own error; its siblings still return scores.

    This is the troubleshooting property: with eight metrics in a live test, the one that failed
    has to be identifiable instead of collapsing the whole response into a single error.
    """
    # tool-calling needs its reference to render to a list of OpenAI-style tool calls; a bare
    # string does not, so this metric fails while the exact-match beside it succeeds.
    broken = ToolCallingMetric(reference="{{item.not_tool_calls}}")
    broken_inline = MetricInline.model_validate_json(
        bundle_metric(broken, CloudpickleMetricBundlePackager()).model_dump_json()
    ).model_dump(mode="json")

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "output": "Paris", "not_tool_calls": "nonsense"}],
            "metrics": [_exact_match_metric(), broken_inline],
            "field_mapping": {"output": "output"},
        },
    )

    assert resp.status_code == 200, resp.text
    by_metric = {m["metric"]: m for m in resp.json()["metrics"]}
    assert by_metric["exact-match"]["error"] is None
    assert by_metric["exact-match"]["scores"][0]["mean"] == pytest.approx(1.0)
    assert by_metric["tool-calling"]["scores"] == []
    assert "tool call" in by_metric["tool-calling"]["error"].lower()


def test_too_many_model_backed_metrics_is_rejected(client: TestClient) -> None:
    """The cap that matters is on metrics that cost an upstream call, not on metric count."""
    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "output": "Paris"}],
            "metrics": [_judge_metric("judge-key") for _ in range(live_routes.MAX_MODEL_BACKED_METRICS + 1)],
            "field_mapping": {"output": "output"},
        },
    )

    assert resp.status_code == 422
    assert "model-backed" in resp.text


# ---- stored metric references ----------------------------------------------


class _StoredBundleResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class _FakeBundleFiles(AsyncFilesClient):
    """Serves one stored metric bundle, keyed the way `load_bundle` addresses it."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def download_file(self, *, path: str, name: str, workspace: str | None = None) -> _StoredBundleResponse:
        del path, name, workspace
        return _StoredBundleResponse(self._payload)


class _FakeBundleEntities(EntityClient):
    """Returns one stored metric entity, whatever name is asked for.

    Subclasses the real client because the route only forwards an entity client that passes an
    isinstance(EntityClient) check, and silently resolves references without one otherwise.
    """

    def __init__(self, entity: MetricBundleEntity) -> None:
        self._entity = entity
        self.lookups: list[tuple[str, str]] = []

    async def get(self, entity_type, *, name: str, workspace: str, parent: str | None = None):
        del entity_type, parent
        self.lookups.append((workspace, name))
        return self._entity


def test_stored_metric_reference_is_resolved_and_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `workspace/metric-name` ref is loaded from storage and scored like an inline metric.

    Studio stores reusable metrics; without this a caller would have to inline a stored metric
    client-side just to live-test it, which defeats the point of having stored it.
    """
    bundle = bundle_metric(
        ExactMatchMetric(reference="{{item.expected}}", candidate="{{sample.output_text}}"),
        CloudpickleMetricBundlePackager(),
    )
    payload = bundle.model_dump_json().encode()
    entity = MetricBundleEntity(
        name="stored-exact",
        workspace="default",
        metric_type="exact-match",
        outputs=bundle.outputs,
        payload_kind=bundle.payload.kind,
        bundle_ref="default/metric-bundles#stored-exact.json",
        payload_digest=bundle.payload.digest,
    )
    entities = _FakeBundleEntities(entity)
    # The Files client is built inside metric_resolution, not in the route module; exact-match
    # carries no model refs, so this stands in for the only client that call actually needs.
    monkeypatch.setattr(
        metric_resolution, "client_from_platform", lambda *_args, **_kw: _FakeBundleFiles(payload), raising=True
    )

    app = _build_app()
    app.dependency_overrides[get_entity_client] = lambda: entities

    with TestClient(app) as client:
        resp = client.post(
            _BASE,
            json={
                "dataset": [{"expected": "Paris", "output": "Paris"}],
                "metrics": ["default/stored-exact"],
                "field_mapping": {"output": "output"},
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [m["metric"] for m in body["metrics"]] == ["exact-match"]
    assert body["metrics"][0]["scores"][0]["mean"] == pytest.approx(1.0)
    assert entities.lookups == [("default", "stored-exact")]


def test_targeted_run_keeps_sibling_scores_when_one_metric_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a target, a failing metric reports its own error and the others still score."""

    async def fake_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        return _chat_response("Paris")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", fake_inference)

    broken = MetricInline.model_validate_json(
        bundle_metric(
            ToolCallingMetric(reference="{{item.expected}}"), CloudpickleMetricBundlePackager()
        ).model_dump_json()
    ).model_dump(mode="json")

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
            "metrics": [_exact_match_metric(), broken],
            "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
            "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
        },
    )

    assert resp.status_code == 200, resp.text
    by_metric = {m["metric"]: m for m in resp.json()["metrics"]}
    assert by_metric["exact-match"]["error"] is None
    assert by_metric["exact-match"]["scores"][0]["mean"] == pytest.approx(1.0)
    assert by_metric["tool-calling"]["scores"] == []
    assert "tool call" in by_metric["tool-calling"]["error"].lower()


def test_failed_target_generation_is_not_reported_as_metric_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row the target never answered for fails the request, not each metric separately."""

    async def failing_inference(model, request, max_retries=3, **kwargs) -> dict[str, Any]:
        del model, request, max_retries, kwargs
        raise RuntimeError("upstream returned 503 Service Unavailable")

    monkeypatch.setattr(benchmark_execution, "make_inference_request", failing_inference)

    resp = client.post(
        _BASE,
        json={
            "dataset": [{"expected": "Paris", "question": "Capital of France?"}],
            "metrics": [_exact_match_metric()],
            "target": {"url": "http://models.test/v1/chat/completions", "name": "test-model"},
            "prompt_template": {"messages": [{"role": "user", "content": "{{item.question}}"}]},
        },
    )

    assert resp.status_code == 502, resp.text
    assert "target generation failed" in resp.text
    assert "503 Service Unavailable" in resp.text
