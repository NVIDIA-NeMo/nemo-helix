# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from nemo_agents_plugin.entities import (
    AgentEnvironmentInline,
    ComputeResources,
    ComputeSpecInline,
    EnvironmentSpecInline,
)
from nemo_agents_plugin.sdk import AgentsResource, AsyncAgentsResource, agents_sdk_resources
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform

_Handler = Callable[[httpx.Request], httpx.Response]


def _read_json(req: httpx.Request) -> dict[str, Any]:
    body = json.loads(req.read())
    assert isinstance(body, dict)
    return body


def _platform(
    handler: _Handler,
    *,
    workspace: str | None = "team-a",
    default_headers: Mapping[str, str] | None = None,
) -> NeMoPlatform:
    return NeMoPlatform(
        base_url="http://test",
        workspace=workspace,
        default_headers=default_headers,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _async_platform(handler: _Handler, *, workspace: str | None = "team-a") -> AsyncNeMoPlatform:
    return AsyncNeMoPlatform(
        base_url="http://test",
        workspace=workspace,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def test_create_resolves_default_model_placeholder_before_post() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "calc"})

    client = AgentsResource(_platform(handler))
    config = {"llms": {"llm": {"_type": "openai", "model_name": "${NEMO_DEFAULT_MODEL}"}}}

    with patch("nemo_agents_plugin.utils.get_default_model", return_value="team-a/nemotron"):
        result = client.create(name="calc", config=config)

    assert result == {"name": "calc"}
    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/agents"
    assert captured["body"]["config"]["llms"]["llm"]["model_name"] == "team-a/nemotron"
    assert config["llms"]["llm"]["model_name"] == "${NEMO_DEFAULT_MODEL}"


def test_create_rejects_unresolved_default_model_placeholder() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST unresolved agent config")

    client = AgentsResource(_platform(handler))
    config = {"llms": {"llm": {"_type": "openai", "model_name": "$NEMO_DEFAULT_MODEL"}}}

    with (
        patch("nemo_agents_plugin.utils.get_default_model", return_value=None),
        pytest.raises(ValueError, match="NEMO_DEFAULT_MODEL"),
    ):
        client.create(name="calc", config=config)


def test_agents_resource_uses_default_workspace_when_platform_workspace_is_unset() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(201, json={"name": "calc"})

    resource = AgentsResource(_platform(handler, workspace=None))

    resource.create(name="calc", config={})
    resource.jobs.execute.create(spec={"agent": "calc", "input": "2+2"})

    assert paths == [
        "/apis/agents/v2/workspaces/default/agents",
        "/apis/agents/v2/workspaces/default/jobs/execute",
    ]


async def test_async_agents_resource_uses_default_workspace_when_platform_workspace_is_unset() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    await AsyncAgentsResource(_async_platform(handler, workspace=None)).jobs.execute.create(
        spec={"agent": "calc", "input": "2+2"}
    )

    assert paths == ["/apis/agents/v2/workspaces/default/jobs/execute"]


def test_deployments_create_uses_client_workspace_by_default() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "calc-dep"})

    client = AgentsResource(_platform(handler))

    result = client.deployments.create(agent="calc", deployment_mode="k8s", image="repo/calc:1.0")

    assert result == {"name": "calc-dep"}
    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/deployments"
    assert captured["body"] == {"agent": "calc", "deployment_mode": "k8s", "image": "repo/calc:1.0"}


def test_deployments_create_forwards_image_entrypoint_mode() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "calc-dep"})

    client = AgentsResource(_platform(handler))

    client.deployments.create(
        agent="calc",
        deployment_mode="docker",
        image="repo/calc:1.0",
        use_image_entrypoint=True,
    )

    assert captured["body"] == {
        "agent": "calc",
        "deployment_mode": "docker",
        "image": "repo/calc:1.0",
        "use_image_entrypoint": True,
    }


def test_deployments_create_rejects_image_entrypoint_for_subprocess() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not POST image entrypoint mode for subprocess")

    client = AgentsResource(_platform(handler))

    with pytest.raises(ValueError, match="use_image_entrypoint"):
        client.deployments.create(agent="calc", use_image_entrypoint=True)


def test_invoke_sends_session_id_as_header() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        captured["session_id"] = req.headers.get(SESSION_ID_HEADER)
        return httpx.Response(200, json={"id": "completion-id"})

    client = AgentsResource(_platform(handler))

    result = client.invoke(input="Continue", deployment="calc-dep", session_id="session-entity-id")

    assert result == {"id": "completion-id"}
    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/deployments/calc-dep/-/v1/chat/completions"
    assert captured["session_id"] == "session-entity-id"
    assert captured["body"] == {
        "messages": [{"role": "user", "content": "Continue"}],
        "stream": False,
    }


def test_invoke_without_session_id_omits_header() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["session_id"] = req.headers.get(SESSION_ID_HEADER)
        return httpx.Response(200, json={"id": "completion-id"})

    client = AgentsResource(_platform(handler))

    client.invoke(input="Hello", agent="calc")

    assert captured["session_id"] is None


def test_invoke_rejects_empty_session_id() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("should not invoke with an empty session ID")

    client = AgentsResource(_platform(handler))

    with pytest.raises(ValueError, match="session_id must not be empty"):
        client.invoke(input="Hello", agent="calc", session_id="")


# ---------------------------------------------------------------------------
# environment / environment-spec / compute-spec resources
# ---------------------------------------------------------------------------


def test_deployments_create_forwards_environment() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "d1"})

    client = AgentsResource(_platform(handler, workspace="default"))
    client.deployments.create(agent="calc", environment="default/env1")

    assert captured["body"]["environment"] == "default/env1"


def test_environment_specs_create_posts_inline_fields() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "ben"})

    client = AgentsResource(_platform(handler))
    result = client.environment_specs.create(
        name="ben", spec={"env": {"LOG_LEVEL": "debug"}, "secrets": {"TOK": "default/tok"}}
    )

    assert result == {"name": "ben"}
    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/environment-specs"
    assert captured["body"] == {"name": "ben", "env": {"LOG_LEVEL": "debug"}, "secrets": {"TOK": "default/tok"}}


def test_environments_create_with_refs() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "env1"})

    client = AgentsResource(_platform(handler, workspace="default"))
    client.environments.create(name="env1", environment_spec="default/ben", compute_spec="default/big")

    assert captured["path"] == "/apis/agents/v2/workspaces/default/environments"
    assert captured["body"]["environment_spec"] == "default/ben"
    assert captured["body"]["compute_spec"] == "default/big"


def test_environments_create_omits_unset_refs() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "env2"})

    client = AgentsResource(_platform(handler, workspace="default"))
    client.environments.create(name="env2", environment_spec="default/ben")

    assert "compute_spec" not in captured["body"]
    assert captured["body"]["environment_spec"] == "default/ben"


def test_compute_specs_get_and_delete() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.setdefault("calls", []).append((req.method, req.url.path))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"name": "big"})

    client = AgentsResource(_platform(handler))
    got = client.compute_specs.get("big")
    client.compute_specs.delete("big")

    assert got == {"name": "big"}
    assert ("GET", "/apis/agents/v2/workspaces/team-a/compute-specs/big") in captured["calls"]
    assert ("DELETE", "/apis/agents/v2/workspaces/team-a/compute-specs/big") in captured["calls"]


def test_execute_job_create_posts_spec_to_job_collection() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    resource = AgentsResource(_platform(handler))

    result = resource.jobs.execute.create(spec={"agent": "calc", "input": "2+2"}, workspace="team-a")

    assert result == {"name": "execute-a1b2"}
    assert captured == {
        "method": "POST",
        "path": "/apis/agents/v2/workspaces/team-a/jobs/execute",
        "body": {"spec": {"agent": "calc", "input": "2+2"}},
    }


def test_execute_job_create_omits_name_when_unset() -> None:
    """An omitted name lets the Jobs service generate a unique one."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    AgentsResource(_platform(handler)).jobs.execute.create(spec={"agent": "calc", "input": "hi"})

    assert isinstance(captured["body"], dict)
    assert "name" not in captured["body"]


def test_execute_job_create_includes_name_and_description_when_given() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    AgentsResource(_platform(handler)).jobs.execute.create(
        spec={"agent": "calc", "input": "hi"}, name="run-1", description="demo"
    )

    assert isinstance(captured["body"], dict)
    assert captured["body"]["name"] == "run-1"
    assert captured["body"]["description"] == "demo"


def test_execute_job_get_and_list_results_paths() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path.endswith("/results"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "result",
                            "job": "execute-a1b2",
                            "workspace": "team-a",
                            "artifact_url": "fileset://result",
                            "artifact_storage_type": "fileset",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"name": "execute-a1b2"})

    jobs = AgentsResource(_platform(handler)).jobs.execute

    jobs.get("execute-a1b2", workspace="team-a")
    jobs.list_results("execute-a1b2", workspace="team-a")

    assert paths == [
        "/apis/agents/v2/workspaces/team-a/jobs/execute/execute-a1b2",
        "/apis/agents/v2/workspaces/team-a/jobs/execute/execute-a1b2/results",
    ]


def test_execute_job_get_accepts_workspace_positionally() -> None:
    """Mirrors sibling ``get``/``delete`` methods, which take workspace positional-or-keyword."""
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(200, json={"name": "execute-a1b2"})

    jobs = AgentsResource(_platform(handler)).jobs.execute

    jobs.get("execute-a1b2", "team-a")

    assert paths[0] == "/apis/agents/v2/workspaces/team-a/jobs/execute/execute-a1b2"


def test_execute_job_create_uses_client_workspace_by_default() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    AgentsResource(_platform(handler, workspace="team-a")).jobs.execute.create(spec={"agent": "calc", "input": "2+2"})

    assert paths[0] == "/apis/agents/v2/workspaces/team-a/jobs/execute"


def test_execute_job_get_and_list_results_use_client_workspace_by_default() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path.endswith("/results"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "result",
                            "job": "execute-a1b2",
                            "workspace": "team-a",
                            "artifact_url": "fileset://result",
                            "artifact_storage_type": "fileset",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"name": "execute-a1b2"})

    jobs = AgentsResource(_platform(handler, workspace="team-a")).jobs.execute

    jobs.get("execute-a1b2")
    jobs.list_results("execute-a1b2")

    assert paths == [
        "/apis/agents/v2/workspaces/team-a/jobs/execute/execute-a1b2",
        "/apis/agents/v2/workspaces/team-a/jobs/execute/execute-a1b2/results",
    ]


async def test_async_execute_job_create_uses_client_workspace_by_default() -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    platform = _async_platform(handler, workspace="team-a")

    await AsyncAgentsResource(platform).jobs.execute.create(spec={"agent": "calc", "input": "2+2"})

    assert paths[0] == "/apis/agents/v2/workspaces/team-a/jobs/execute"


async def test_async_execute_job_create_mirrors_sync_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "execute-a1b2"})

    platform = _async_platform(handler, workspace=None)

    result = await AsyncAgentsResource(platform).jobs.execute.create(
        spec={"agent": "calc", "input": "2+2"}, workspace="team-a"
    )

    assert result == {"name": "execute-a1b2"}
    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/jobs/execute"
    assert captured["body"] == {"spec": {"agent": "calc", "input": "2+2"}}


def test_agents_sdk_resources_registers_both_sync_and_async() -> None:
    assert agents_sdk_resources.sync_resource is AgentsResource
    assert agents_sdk_resources.async_resource is AsyncAgentsResource


def test_environment_specs_create_accepts_typed_model() -> None:
    """A typed ``EnvironmentSpecInline`` sends only the fields that were set."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "ben"})

    client = AgentsResource(_platform(handler))
    spec = EnvironmentSpecInline(env={"LOG_LEVEL": "debug"}, secrets={"TOK": "default/tok"})
    client.environment_specs.create(name="ben", spec=spec)

    # exclude_unset: provider/description/etc. are NOT sent because the caller
    # never set them — matching the old **spec behavior.
    assert captured["body"] == {"name": "ben", "env": {"LOG_LEVEL": "debug"}, "secrets": {"TOK": "default/tok"}}


def test_environment_specs_create_accepts_dict_spec() -> None:
    """A plain dict passed as ``spec=`` is sent verbatim."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "ben"})

    client = AgentsResource(_platform(handler))
    client.environment_specs.create(name="ben", spec={"provider": "local"})

    assert captured["body"] == {"name": "ben", "provider": "local"}


def test_environment_specs_create_name_arg_is_authoritative() -> None:
    """An explicit ``name=`` wins over a ``name`` key smuggled in via ``spec``.

    Regression guard: the payload must apply ``name`` AFTER the spec/kwargs
    expansion so a dict spec carrying its own ``name`` cannot silently create the
    resource under a different name than the caller asked for.
    """
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "wanted"})

    client = AgentsResource(_platform(handler))
    client.environment_specs.create(name="wanted", spec={"name": "sneaky", "provider": "local"})

    assert captured["body"]["name"] == "wanted"


def test_compute_specs_create_accepts_typed_model() -> None:
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["path"] = req.url.path
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "big"})

    client = AgentsResource(_platform(handler))
    spec = ComputeSpecInline(resources=ComputeResources(limits={"cpu": "2"}))
    client.compute_specs.create(name="big", spec=spec)

    assert captured["path"] == "/apis/agents/v2/workspaces/team-a/compute-specs"
    assert captured["body"] == {"name": "big", "resources": {"limits": {"cpu": "2"}}}


def test_environments_create_accepts_typed_model() -> None:
    """A typed ``AgentEnvironmentInline`` sends its set fields; ref args override."""
    captured: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = _read_json(req)
        return httpx.Response(201, json={"name": "env1"})

    client = AgentsResource(_platform(handler, workspace="default"))
    spec = AgentEnvironmentInline(environment_spec="default/ben")
    client.environments.create(name="env1", spec=spec, compute_spec="default/big")

    assert captured["body"]["name"] == "env1"
    assert captured["body"]["environment_spec"] == "default/ben"
    # explicit compute_spec arg is applied on top of the typed spec
    assert captured["body"]["compute_spec"] == "default/big"


def test_default_headers_are_sent_on_every_request() -> None:
    """``platform.default_headers`` (how the CLI threads its token) reach the wire."""
    auth_headers: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        auth_headers.append(req.headers.get("authorization"))
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"name": "big"})

    client = AgentsResource(_platform(handler, default_headers={"Authorization": "Bearer tok-123"}))
    client.compute_specs.create(name="big", spec=ComputeSpecInline())
    client.compute_specs.get("big")
    client.compute_specs.delete("big")

    assert auth_headers == ["Bearer tok-123", "Bearer tok-123", "Bearer tok-123"]
