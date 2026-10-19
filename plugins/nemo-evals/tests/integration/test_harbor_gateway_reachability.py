# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component integration reproducer for Harbor's missing container URL resolution.

Only the registry HTTP response is stubbed. Typed-client decoding, gateway binding,
Fabric translation, canonical-spec serialization, and Harbor runtime setup are real.
No platform, Docker daemon, model credentials, or listening sockets are required.
This checks the config sent to Harbor, not Docker DNS or network connectivity.

Run the component integration test::

    uv run --frozen pytest plugins/nemo-evals/tests/integration/test_harbor_gateway_reachability.py -v
"""

from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from nemo_evals.jobs.agent_evaluate import AgentEvalJob
from nemo_evals.jobs.agent_spec import AgentEvalSpec, HarborRunnerTarget
from nemo_evals.jobs.registered_agent_resolution import resolve_registered_agent
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborRuntimeConfig

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("http://127.0.0.1:8080", id="ipv4-loopback"),
        pytest.param("http://localhost:8080", id="localhost"),
        pytest.param("http://[::1]:8080", id="ipv6-loopback"),
        pytest.param("http://0.0.0.0:8080", id="wildcard-bind"),
        pytest.param("http://192.0.2.10:8080", id="routable-host-control"),
    ],
)
async def test_registered_agent_gateway_is_not_loopback_in_harbor(
    base_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    monkeypatch.setenv("NHX_BASE_URL", base_url)
    requests: list[httpx.Request] = []

    def registry(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/apis/agents/v2/workspaces/default/agents/gateway-probe"
        return httpx.Response(
            200,
            json={
                "name": "gateway-probe",
                "workspace": "default",
                "config_format": "nemo-agents-spec-v1",
                "config": {
                    "config_format": "nemo-agents-spec-v1",
                    "name": "gateway-probe",
                    "instructions": {"system": {"content": "Reply DONE."}},
                    "default_harness": "deepagents",
                    "harnesses": {"deepagents": {"kind": "deepagents"}},
                    "models": {"default": {"provider": "nvidia", "model": "test-model"}},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(registry)) as http_client:
        client = AsyncNemoClient(base_url=base_url, http_client=http_client)
        target = await resolve_registered_agent(
            HarborRunnerTarget.model_validate({"source": {"agent": "gateway-probe"}}),
            workspace="default",
            async_sdk=client,
        )
    assert len(requests) == 1
    assert isinstance(target, HarborRunnerTarget)

    # Cross the persisted job boundary before building the SDK runtime that launches Harbor.
    spec = AgentEvalSpec.model_validate(
        {
            "tasks": [
                {"id": "probe", "spec": {"kind": "evaluator", "intent": "Reply DONE", "inputs": {}, "metrics": []}}
            ],
            "target": target.model_dump(mode="json"),
        }
    )
    spec = AgentEvalSpec.model_validate_json(spec.model_dump_json())
    storage = StoragePaths(ephemeral=tmp_path / "ephemeral", persistent=tmp_path / "persistent")
    storage.ephemeral.mkdir()
    storage.persistent.mkdir()
    ctx = JobContext(workspace="default", storage=storage, results=LocalJobResults(root=storage.persistent / "results"))
    assert isinstance(spec.target, HarborRunnerTarget)
    runtime, _, _ = AgentEvalJob._resolve_target(spec.target, ctx)
    config = getattr(runtime, "_config")
    assert isinstance(config, HarborRuntimeConfig)
    model_url = config.agent_kwargs["fabric_config"]["models"]["default"]["base_url"]
    parsed = urlsplit(model_url)
    assert parsed.path == "/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert parsed.port == 8080
    expected_host = "192.0.2.10" if base_url == "http://192.0.2.10:8080" else "host.docker.internal"
    assert parsed.hostname == expected_host
    assert parsed.hostname not in {"localhost", "127.0.0.1", "::1"}, (
        f"Harbor would send model requests to {model_url}, which points at the test container itself"
    )
