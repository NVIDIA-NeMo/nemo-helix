# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for agent deployment e2e tests.

The subprocess, Docker, and Kubernetes modules deploy a real agent through the
agents plugin and invoke it through the agents gateway. The end-to-end chain is
identical apart from the deployment backend and endpoint projection::

    agents.invoke (gateway proxy, mode-specific endpoint resolution)
      -> NAT or Fabric agent process on subprocess | docker | kubernetes
      -> Inference Gateway /openai (base_url injected at deploy time)
      -> mock provider short-circuit (no real upstream / no API key)
      -> response back through the gateway

This module holds the shared core: the mock-provider-backed agent config, the
create -> wait-running -> assert-endpoint-shape -> invoke -> assert flow, and
cleanup. The per-backend modules own only what genuinely differs (pytest
markers, image resolution, timeouts, and best-effort backend cleanup).
"""

import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

import httpx
import pytest
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_agents_plugin.sdk import AgentsResource
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.intake.client import IntakeClient
from nhx.testing import MockProviderResponse, add_mock_provider

from e2e.utils import collect_sse_chunks

# The mocked completion the deployed agent must round-trip back to the caller.
TEST_AGENT_RESPONSE = "The answer to your question is 42."

AgentInvocationMode = Literal["non_streaming", "streaming", "session"]


def unique_name(prefix: str) -> str:
    return f"e2e-{prefix}-{uuid.uuid4().hex[:8]}"


def agents_resource(client: NemoClient) -> AgentsResource:
    """Plugin ``agents`` resource driven by the typed platform client."""
    return AgentsResource(client)  # ty: ignore[invalid-argument-type]


def wait_for_openai_model(
    client: NemoClient,
    *,
    workspace: str,
    name: str,
    timeout: float = 60,
    poll_interval: float = 0.5,
) -> None:
    """Wait until the gateway's OpenAI model route resolves ``workspace/name``."""
    gateway = InferenceGatewayClient.from_client(client)
    expected_model_id = f"{workspace}/{name}"
    deadline = time.monotonic() + timeout
    while True:
        try:
            model = gateway.get_openai_model(workspace=workspace, name=expected_model_id).data()
            if model.id == expected_model_id:
                return
        except NotFoundError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"OpenAI model {expected_model_id} not available in inference gateway after {timeout}s")
        time.sleep(poll_interval)


def wait_for_agent_spans(
    client: NemoClient,
    *,
    workspace: str,
    agent_name: str,
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> list[Any]:
    """Poll Intake for the agent's trajectory.

    Ingest is asynchronous: Relay posts the trajectory as the run finishes, and
    Intake writes it behind the API, so an invocation that has already returned
    does not mean the spans are queryable yet.
    """
    intake = IntakeClient.from_client(client)
    deadline = time.monotonic() + timeout
    while True:
        page = intake.list_spans(
            workspace=workspace, query_params={"filter": {"agent_name": agent_name}, "page_size": 50}
        ).page()
        spans = list(page.items)
        if spans or time.monotonic() >= deadline:
            return spans
        time.sleep(poll_interval)


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-agents-container-e2e",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _mock_backed_nat_agent_config(model_name: str) -> dict[str, Any]:
    """A deterministic single-LLM NAT workflow pointed at the mock model.

    ``base_url``/``api_key`` are intentionally omitted so the deployment injects
    the Inference Gateway URL (and the mock provider short-circuits the call).
    """
    return {
        "llms": {
            "main": {
                "_type": "openai",
                "model_name": model_name,
            }
        },
        "workflow": {
            "_type": "chat_completion",
            "llm_name": "main",
        },
    }


def mock_backed_fabric_agent_config(agent_name: str, model_name: str) -> dict[str, Any]:
    """A deterministic DeepAgents-backed Fabric agent pointed at the mock model."""
    return {
        "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        "name": agent_name,
        "default_harness": "deepagents",
        "harnesses": {
            "deepagents": {
                "kind": "deepagents",
                "settings": {
                    "deepagents": {},
                },
            }
        },
        "models": {
            "default": {
                "provider": "openai",
                "model": model_name,
                "temperature": 0.0,
            }
        },
    }


def _mock_backed_agent_config(config_format: str, *, agent_name: str, model_name: str) -> dict[str, Any]:
    """Return the runtime-valid mock-backed config for ``config_format``."""
    if config_format == NAT_WORKFLOW_CONFIG_FORMAT:
        return _mock_backed_nat_agent_config(model_name)
    if config_format == NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        return mock_backed_fabric_agent_config(agent_name, model_name)
    raise ValueError(f"Unsupported agent config format: {config_format!r}")


def _page_data(page: Any) -> list[dict[str, Any]]:
    if isinstance(page, dict):
        data = page.get("data", [])
    else:
        data = getattr(page, "data", [])
    assert isinstance(data, list)
    return data


def delete_agent_if_exists(client: NemoClient, *, workspace: str, name: str) -> None:
    try:
        agents_resource(client).delete(name, workspace=workspace)
    except NotFoundError:
        pass


def delete_deployment_if_exists(client: NemoClient, *, workspace: str, name: str) -> None:
    try:
        agents_resource(client).deployments.delete(name, workspace=workspace)
    except NotFoundError:
        pass


def get_deployment_log_text(client: NemoClient, *, workspace: str, name: str) -> str:
    try:
        response = client._client.get(
            f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}/logs",
            params={"tail": 100},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc.response.text

    payload = response.json()
    lines = _page_data(payload)
    return "\n".join(str(line.get("message", line)) for line in lines)


def wait_for_deployment_deleted(
    client: NemoClient,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 120,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None
    while time.monotonic() < deadline:
        try:
            deployment = agents_resource(client).deployments.get(name, workspace=workspace)
            last_status = deployment.get("status")
        except NotFoundError:
            return
        time.sleep(2)
    pytest.fail(f"Deployment {name!r} was not deleted within {timeout_seconds}s; last status={last_status!r}")


def wait_for_deployment_running(
    client: NemoClient,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_deployment: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        deployment = agents_resource(client).deployments.get(name, workspace=workspace)
        last_deployment = deployment
        status = deployment["status"]
        if status == "running":
            return deployment
        if status == "failed":
            logs = get_deployment_log_text(client, workspace=workspace, name=name)
            pytest.fail(f"Deployment {name!r} failed: {deployment.get('error', '')}\n{logs}")
        time.sleep(2)
    pytest.fail(f"Deployment {name!r} did not reach running within {timeout_seconds}s: {last_deployment}")


def _assert_non_streaming_invocation(
    client: NemoClient,
    *,
    workspace: str,
    agent_name: str,
) -> None:
    response = agents_resource(client).invoke(
        workspace=workspace,
        agent=agent_name,
        input="What is 12 multiplied by 8?",
    )
    content = response["choices"][0]["message"]["content"]
    assert TEST_AGENT_RESPONSE in content, response


def _assert_streaming_invocation(
    client: NemoClient,
    *,
    workspace: str,
    deployment_name: str,
) -> None:
    """Stream one Fabric turn through the deployed-agent gateway."""
    path = f"/apis/agents/v2/workspaces/{workspace}/deployments/{deployment_name}/-/v1/chat/completions"
    with client._client.stream(
        "POST",
        path,
        json={
            "messages": [{"role": "user", "content": "What is 12 multiplied by 8?"}],
            "stream": True,
        },
    ) as response:
        if response.status_code != 200:
            pytest.fail(f"Streaming agent invocation returned {response.status_code}: {response.read().decode()}")
        assert "text/event-stream" in response.headers.get("content-type", "")
        chunks = collect_sse_chunks(response)

    content = "".join(
        choice.get("delta", {}).get("content", "") for chunk in chunks for choice in chunk.get("choices", [])
    )
    assert TEST_AGENT_RESPONSE in content, chunks


def _assert_persisted_session_invocation(
    client: NemoClient,
    *,
    workspace: str,
    deployment_name: str,
    deployment_id: str,
) -> None:
    """Create a persisted session and invoke its streaming-enabled runtime."""
    session_name = unique_name("calc-session")
    sessions_path = f"/apis/agents/v2/workspaces/{workspace}/sessions"
    create_response = client._client.post(
        sessions_path,
        json={"deployment_id": deployment_id, "name": session_name},
    )
    create_response.raise_for_status()
    session = create_response.json()

    try:
        response = agents_resource(client).invoke(
            workspace=workspace,
            deployment=deployment_name,
            session_id=session["id"],
            input="What is 12 multiplied by 8?",
        )
        content = response["choices"][0]["message"]["content"]
        assert TEST_AGENT_RESPONSE in content, response
    finally:
        close_response = client._client.post(f"{sessions_path}/{session_name}/close")
        close_response.raise_for_status()


def run_agent_deploy_and_invoke(
    sdk: NeMoHelix,
    client: NemoClient,
    *,
    workspace: str,
    deployment_mode: str,
    config_format: str = NAT_WORKFLOW_CONFIG_FORMAT,
    image: str | None = None,
    running_timeout_seconds: float = 300,
    reap_backend_resources: Callable[[str], None] | None = None,
    after_invoke: Callable[[str], None] | None = None,
    invocation_modes: tuple[AgentInvocationMode, ...] = ("non_streaming",),
) -> None:
    """Deploy a mock-backed agent and invoke it through the gateway.

    Shared core for subprocess, Docker, and Kubernetes E2E modules:

    1. Register a mock inference provider + deterministic single-LLM agent in
       the requested ``config_format``.
    2. Deploy it using ``deployment_mode`` and the optional container ``image``.
    3. Wait for ``running`` and assert the mode-specific endpoint shape.
    4. Invoke through the gateway using each ``invocation_modes`` entry and assert the
       mocked completion round-trips.
    5. Clean up the deployment and agent (best-effort, isolated steps).

    ``after_invoke``, if given, is called with the agent name once the response
    has been asserted and while the deployment is still up, for checks a caller
    wants to make against a live deployment.

    ``streaming`` and ``session`` modes are Fabric-only. The latter sends a
    non-streaming chat request through a persisted session; opening that session
    still starts Fabric with streaming enabled and therefore exercises the
    collector dependency.

    ``reap_backend_resources``, if given, is called with the deployment name
    during teardown (after the deployment is deleted) so a backend module can
    best-effort reap leaked resources it uniquely knows about — e.g. a leftover
    docker container. It must not raise; failures are swallowed like the rest of
    teardown.
    """
    if not invocation_modes:
        raise ValueError("invocation_modes must contain at least one mode")

    agent_name = unique_name("calc-agent")
    deployment_name = unique_name("calc-deployment")
    model_name = unique_name("calc-model")

    add_mock_provider(
        sdk,
        workspace=workspace,
        name=unique_name("calc-provider"),
        mock_response_body_by_model={
            f"{workspace}/{model_name}": [
                MockProviderResponse(response_body=_chat_completion_response(TEST_AGENT_RESPONSE, model_name))
                for _ in invocation_modes
            ],
        },
        served_models={model_name: model_name},
    )

    agents = agents_resource(client)
    agents.create(
        workspace=workspace,
        name=agent_name,
        config=_mock_backed_agent_config(
            config_format,
            agent_name=agent_name,
            model_name=f"{workspace}/{model_name}",
        ),
        config_format=config_format,
    )

    try:
        created = agents.deployments.create(
            workspace=workspace,
            agent=agent_name,
            name=deployment_name,
            deployment_mode=deployment_mode,
            image=image,
        )
        assert created["deployment_mode"] == deployment_mode

        deployment = wait_for_deployment_running(
            client, workspace=workspace, name=deployment_name, timeout_seconds=running_timeout_seconds
        )
        assert deployment["agent"] == agent_name
        assert deployment["deployment_mode"] == deployment_mode

        if deployment_mode == "subprocess":
            assert image is None
            assert deployment["endpoint"]
            assert not deployment.get("endpoints")
            assert deployment["pid"] > 0
        else:
            # Container-mode addressing: the loopback scalar ``endpoint`` is
            # empty and the routable address lives in ``endpoints``. Guarding
            # this shape prevents a container test from passing via fallback.
            assert image
            assert deployment.get("endpoint", "") == ""
            endpoints = deployment.get("endpoints") or []
            assert endpoints and endpoints[0]["url"], deployment

        wait_for_openai_model(client, workspace=workspace, name=model_name)

        for invocation_mode in invocation_modes:
            if invocation_mode == "non_streaming":
                _assert_non_streaming_invocation(client, workspace=workspace, agent_name=agent_name)
            elif config_format != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
                raise ValueError(f"{invocation_mode!r} invocation mode requires a Fabric-backed agent")
            elif invocation_mode == "streaming":
                _assert_streaming_invocation(client, workspace=workspace, deployment_name=deployment_name)
            elif invocation_mode == "session":
                _assert_persisted_session_invocation(
                    client,
                    workspace=workspace,
                    deployment_name=deployment_name,
                    deployment_id=deployment["id"],
                )
            else:
                raise ValueError(f"Unsupported invocation mode: {invocation_mode!r}")

        if after_invoke is not None:
            after_invoke(agent_name)
    finally:
        # Each step is isolated so a failure (e.g. a deployment-delete timeout)
        # doesn't skip the remaining cleanup and leak resources.
        _safe(delete_deployment_if_exists, client, workspace=workspace, name=deployment_name)
        _safe(wait_for_deployment_deleted, client, workspace=workspace, name=deployment_name)
        if reap_backend_resources is not None:
            _safe(reap_backend_resources, deployment_name)
        _safe(delete_agent_if_exists, client, workspace=workspace, name=agent_name)


def run_container_agent_deploy_and_invoke(
    sdk: NeMoHelix,
    client: NemoClient,
    *,
    workspace: str,
    deployment_mode: str,
    image: str,
    config_format: str = NAT_WORKFLOW_CONFIG_FORMAT,
    running_timeout_seconds: float = 300,
    reap_backend_resources: Callable[[str], None] | None = None,
    invocation_modes: tuple[AgentInvocationMode, ...] = ("non_streaming",),
) -> None:
    """Deploy a mock-backed container agent and invoke it through the gateway."""
    run_agent_deploy_and_invoke(
        sdk,
        client,
        workspace=workspace,
        deployment_mode=deployment_mode,
        config_format=config_format,
        image=image,
        running_timeout_seconds=running_timeout_seconds,
        reap_backend_resources=reap_backend_resources,
        invocation_modes=invocation_modes,
    )


def _safe(fn: Any, *args: Any, **kwargs: Any) -> None:
    try:
        fn(*args, **kwargs)
    except Exception:
        pass
