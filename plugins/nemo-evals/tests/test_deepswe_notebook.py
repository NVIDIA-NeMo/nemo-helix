# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the notebook's model preflight without starting services or evaluations."""

import ast
import json
from collections.abc import Callable
from pathlib import Path
from types import CodeType, SimpleNamespace

import httpx
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples/deepswe_k8s_demo/deepswe_k8s_demo.ipynb"
MISSING_ROUTE = {
    "detail": (
        "No VirtualModel named 'routing/luna-closed-book' was found. Inference requests "
        "must resolve to a VirtualModel; one is auto-created for every served model entity. "
        "Check that the model entity exists and is currently served by a provider, or create a VirtualModel explicitly."
    ),
    "path": "/apis/inference-gateway/v2/workspaces/routing/model/luna-closed-book/-/v1/responses",
    "method": "POST",
}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(scope="module")
def preflight_code() -> CodeType:
    notebook = json.loads(NOTEBOOK.read_text())
    source = next(
        "".join(cell["source"]) for cell in notebook["cells"] if "def ensure_provider_model(" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    helper = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preflight_model"]
    preflight = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Call)
            and (
                isinstance(child.func, ast.Attribute)
                and child.func.attr == "post"
                or isinstance(child.func, ast.Name)
                and child.func.id == "preflight_model"
            )
            for child in ast.walk(node)
        )
    )
    return compile(ast.Module(body=[*helper, preflight], type_ignores=[]), NOTEBOOK, "exec")


def run_preflight(
    code: CodeType,
    handler: Callable[[httpx.Request], httpx.Response],
    clock: Clock,
    requests: list[httpx.Request],
    *,
    agent: str = "codex",
    run_id: str | None = None,
) -> httpx.Response | None:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer notebook-context-token"
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = SimpleNamespace(
            http_client=http_client,
            request_headers=lambda *, url: {"Authorization": "Bearer notebook-context-token"},
        )
        namespace = {
            "time": clock,
            "NemoClient": SimpleNamespace(from_config=lambda *, context: client),
            "CONTEXT": "notebook-context",
            "BASE_URL": "https://platform.example",
            "MODEL": "models/luna",
            "AGENT_MODEL": "routing/luna-closed-book",
            "CLOSED_BOOK": True,
            "AGENT": agent,
            "RUN_ID": run_id,
            "nemo": lambda *args, **kwargs: {"model_providers": ["models/provider"]},
        }
        exec(code, namespace)
        answer = namespace.get("answer")
        assert answer is None or isinstance(answer, httpx.Response)
        return answer


def test_preflight_waits_for_new_gateway_route_before_calling_model(preflight_code: CodeType) -> None:
    clock, requests, model_calls = Clock(), [], []

    def gateway(request: httpx.Request) -> httpx.Response:
        if clock.now < 3:
            return httpx.Response(404, json=MISSING_ROUTE)
        model_calls.append(request)
        return httpx.Response(200, json={"status": "completed"})

    answer = run_preflight(preflight_code, gateway, clock, requests)

    assert answer is not None
    assert answer.status_code == 200
    assert len(model_calls) == 1
    assert len(requests) == 3
    assert clock.sleeps == [2, 2]


@pytest.mark.parametrize(
    ("agent", "api", "prompt"),
    [
        ("codex", "responses", {"input": "Reply with OK"}),
        ("mini-swe-agent", "chat/completions", {"messages": [{"role": "user", "content": "Reply with OK"}]}),
    ],
)
def test_ready_route_keeps_agent_api_model_workspace_and_context(
    preflight_code: CodeType, agent: str, api: str, prompt: dict
) -> None:
    clock, requests = Clock(), []
    answer = run_preflight(
        preflight_code, lambda request: httpx.Response(200, json={"status": "completed"}), clock, requests, agent=agent
    )

    assert answer is not None
    assert answer.status_code == 200
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == (
        f"https://platform.example/apis/inference-gateway/v2/workspaces/routing/model/luna-closed-book/-/v1/{api}"
    )
    assert json.loads(requests[0].content) == {"model": "luna-closed-book", **prompt}
    assert requests[0].extensions["timeout"]["read"] == 120
    assert clock.sleeps == []


def test_preflight_retry_budget_includes_slow_cache_misses(preflight_code: CodeType) -> None:
    clock, requests, timeouts = Clock(), [], []

    def gateway(request: httpx.Request) -> httpx.Response:
        assert clock.now < 120, "no request may start after the retry deadline"
        timeouts.append(request.extensions["timeout"]["read"])
        clock.now += 57 if len(timeouts) < 3 else 1
        return httpx.Response(404, json=MISSING_ROUTE)

    with pytest.raises(RuntimeError, match="routing/luna-closed-book.*120"):
        run_preflight(preflight_code, gateway, clock, requests)

    assert timeouts == [120, 61, 2]
    assert clock.now == 120
    assert clock.sleeps == [2, 2, 1]


@pytest.mark.parametrize("status", [401, 403, 422, 429, 500, 503])
def test_preflight_does_not_retry_other_http_errors(preflight_code: CodeType, status: int) -> None:
    clock, requests = Clock(), []
    with pytest.raises(RuntimeError, match=str(status)):
        run_preflight(
            preflight_code, lambda request: httpx.Response(status, json={"detail": "request failed"}), clock, requests
        )

    assert len(requests) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "upstream model not found", "type": "invalid_request_error"}},
        {"detail": "No VirtualModel named 'another/model' was found."},
        {"detail": {"message": "upstream model not found"}},
        [],
        b"not JSON",
    ],
)
def test_preflight_does_not_replay_unrelated_404(preflight_code: CodeType, body: object) -> None:
    clock, requests = Clock(), []

    def gateway(request: httpx.Request) -> httpx.Response:
        if isinstance(body, bytes):
            return httpx.Response(404, content=body)
        return httpx.Response(404, json=body)

    with pytest.raises(RuntimeError, match="404"):
        run_preflight(preflight_code, gateway, clock, requests)

    assert len(requests) == 1
    assert clock.sleeps == []


def test_preflight_propagates_transport_failure(preflight_code: CodeType) -> None:
    clock, requests = Clock(), []

    def gateway(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(httpx.ConnectError, match="connection failed"):
        run_preflight(preflight_code, gateway, clock, requests)

    assert len(requests) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("agent", "run_id"), [("oracle", None), ("codex", "existing-run"), ("mini-swe-agent", "existing-run")]
)
def test_oracle_and_analysis_skip_preflight(preflight_code: CodeType, agent: str, run_id: str | None) -> None:
    clock, requests = Clock(), []

    def gateway(request: httpx.Request) -> httpx.Response:
        pytest.fail("oracle and analysis mode must not contact a model")

    assert run_preflight(preflight_code, gateway, clock, requests, agent=agent, run_id=run_id) is None
    assert requests == []
