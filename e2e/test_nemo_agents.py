# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the nemo-agents plugin.

These tests cover the backend-agnostic API and SDK surface.
"""

import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.errors import NemoHTTPError
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest
from nmp.testing import MockProviderResponse, add_mock_provider

pytestmark = [pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml")]

_TEST_AGENT_RESPONSE = "The answer to your question is 42."
_NEMO_AGENTS_SPEC_CONFIG_FORMAT = "nemo-agents-spec-v1"


def _unique_name(prefix: str) -> str:
    return f"e2e-{prefix}-{uuid.uuid4().hex[:8]}"


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-agents-e2e",
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


def _mock_backed_agent_config(model_name: str) -> dict[str, Any]:
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


def _agent_config(label: str) -> dict[str, Any]:
    """Return a minimal config payload for API persistence tests.

    The create/list/get/delete API stores the config but does not execute it.
    Runtime-valid NAT configs are covered by the deployment tests.
    """
    return {
        "workflow": {
            "_type": "e2e-placeholder-agent",
            "label": label,
        }
    }


def _platform_agent_config(label: str) -> dict[str, Any]:
    """Return a minimal Platform-managed agent config for API persistence tests."""
    return {
        "config_format": _NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        "name": label,
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
            }
        },
    }


def _page_data(page: Any) -> list[dict[str, Any]]:
    if isinstance(page, dict):
        data = page.get("data", [])
    else:
        data = getattr(page, "data", [])
    assert isinstance(data, list)
    return data


def _pagination(page: Any) -> dict[str, Any]:
    if isinstance(page, dict):
        pagination = page.get("pagination")
    else:
        pagination = getattr(page, "pagination", None)
    assert isinstance(pagination, dict)
    return pagination


HTTPStatusException = httpx.HTTPStatusError | NemoHTTPError


def _http_status(exc: HTTPStatusException) -> int:
    return exc.status_code if isinstance(exc, NemoHTTPError) else exc.response.status_code


def _http_error_text(exc: HTTPStatusException) -> str:
    if isinstance(exc, NemoHTTPError):
        return str(exc.body) if exc.body is not None else str(exc)
    return exc.response.text


def _assert_http_status(exc_info: pytest.ExceptionInfo[HTTPStatusException], status_code: int) -> None:
    assert _http_status(exc_info.value) == status_code


def _assert_deployment_status(status: str) -> None:
    assert status in {"pending", "starting", "running", "failed"}


def _get_agents_page(
    sdk: NeMoPlatform,
    workspace: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = sdk._client.get(
        f"/apis/agents/v2/workspaces/{workspace}/agents",
        params=params,
    )
    response.raise_for_status()
    return response.json()


def _delete_agent_if_exists(sdk: NeMoPlatform, *, workspace: str, name: str) -> None:
    try:
        sdk.agents.delete(name, workspace=workspace)
    except (httpx.HTTPStatusError, NemoHTTPError) as exc:
        if _http_status(exc) != 404:
            raise


def _delete_deployment_if_exists(sdk: NeMoPlatform, *, workspace: str, name: str) -> None:
    try:
        sdk.agents.deployments.delete(name, workspace=workspace)
    except (httpx.HTTPStatusError, NemoHTTPError) as exc:
        if _http_status(exc) == 404:
            return
        if _http_status(exc) in {409, 500}:
            try:
                sdk.agents.deployments.get(name, workspace=workspace)
            except (httpx.HTTPStatusError, NemoHTTPError) as get_exc:
                if _http_status(get_exc) == 404:
                    return
        raise


def _get_deployment_log_text(sdk: NeMoPlatform, *, workspace: str, name: str) -> str:
    try:
        response = sdk._client.get(
            f"/apis/agents/v2/workspaces/{workspace}/deployments/{name}/logs",
            params={"tail": 50},
        )
        response.raise_for_status()
    except (httpx.HTTPStatusError, NemoHTTPError) as exc:
        return _http_error_text(exc)

    payload = response.json()
    lines = _page_data(payload)
    return "\n".join(str(line.get("message", line)) for line in lines)


def _wait_for_deployment_deleted(
    sdk: NeMoPlatform,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 60,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_status: str | None = None

    while time.monotonic() < deadline:
        try:
            deployment = sdk.agents.deployments.get(name, workspace=workspace)
            last_status = deployment.get("status")
        except (httpx.HTTPStatusError, NemoHTTPError) as exc:
            if _http_status(exc) == 404:
                return
            raise

        time.sleep(1)

    pytest.fail(f"Deployment {name!r} was not deleted within {timeout_seconds}s; last status={last_status!r}")


def _wait_for_deployment_running(
    sdk: NeMoPlatform,
    *,
    workspace: str,
    name: str,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_deployment: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        deployment = sdk.agents.deployments.get(name, workspace=workspace)
        last_deployment = deployment
        status = deployment["status"]

        if status == "running":
            assert deployment["endpoint"]
            return deployment
        if status == "failed":
            logs = _get_deployment_log_text(sdk, workspace=workspace, name=name)
            pytest.fail(f"Deployment {name!r} failed: {deployment.get('error', '')}\n{logs}")

        time.sleep(2)

    pytest.fail(f"Deployment {name!r} did not reach running within {timeout_seconds}s: {last_deployment}")


def test_agent_create_list_get_delete_lifecycle(sdk: NeMoPlatform, workspace: str) -> None:
    """Create, list, get, and delete an agent through the plugin SDK."""
    name = _unique_name("agent")
    config = _agent_config(name)

    created = sdk.agents.create(
        workspace=workspace,
        name=name,
        description="E2E agent lifecycle test",
        config=config,
    )
    assert created["name"] == name
    assert created["workspace"] == workspace
    assert created["description"] == "E2E agent lifecycle test"
    assert created["config"] == config
    assert created["config_format"] == "nat-workflow-v1"

    try:
        retrieved = sdk.agents.get(name, workspace=workspace)
        assert retrieved["name"] == name
        assert retrieved["config"] == config

        listed = sdk.agents.list(workspace=workspace)
        listed_names = {agent["name"] for agent in _page_data(listed)}
        assert name in listed_names
    finally:
        _delete_agent_if_exists(sdk, workspace=workspace, name=name)

    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
        sdk.agents.get(name, workspace=workspace)
    _assert_http_status(exc_info, 404)


def test_agent_duplicate_create_returns_conflict(sdk: NeMoPlatform, workspace: str) -> None:
    name = _unique_name("duplicate")
    sdk.agents.create(workspace=workspace, name=name, config=_agent_config(name))

    try:
        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            sdk.agents.create(workspace=workspace, name=name, config=_agent_config(f"{name}-again"))
        _assert_http_status(exc_info, 409)
    finally:
        _delete_agent_if_exists(sdk, workspace=workspace, name=name)


def test_agent_missing_get_and_delete_return_not_found(sdk: NeMoPlatform, workspace: str) -> None:
    missing_name = _unique_name("missing")

    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as get_exc_info:
        sdk.agents.get(missing_name, workspace=workspace)
    _assert_http_status(get_exc_info, 404)

    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as delete_exc_info:
        sdk.agents.delete(missing_name, workspace=workspace)
    _assert_http_status(delete_exc_info, 404)


def test_agent_list_pagination_sorting_and_filtering(sdk: NeMoPlatform, workspace: str) -> None:
    agent_names = [_unique_name(f"list-{i}") for i in range(3)]
    alternate_name = _unique_name("alternate-format")

    try:
        for name in agent_names:
            sdk.agents.create(workspace=workspace, name=name, config=_agent_config(name))

        sdk.agents.create(
            workspace=workspace,
            name=alternate_name,
            config=_platform_agent_config(alternate_name),
            config_format=_NEMO_AGENTS_SPEC_CONFIG_FORMAT,
        )

        first_page = _get_agents_page(sdk, workspace, params={"page": 1, "page_size": 2, "sort": "name"})
        assert len(_page_data(first_page)) == 2
        assert _pagination(first_page)["page"] == 1
        assert _pagination(first_page)["page_size"] == 2
        assert _pagination(first_page)["total_results"] >= 4

        all_created_page = _get_agents_page(sdk, workspace, params={"page_size": 100, "sort": "name"})
        listed_names = [agent["name"] for agent in _page_data(all_created_page) if agent["name"] in agent_names]
        assert listed_names == sorted(agent_names)

        filtered_page = _get_agents_page(
            sdk,
            workspace,
            params={"page_size": 100, "filter[config_format]": _NEMO_AGENTS_SPEC_CONFIG_FORMAT},
        )
        filtered_names = {agent["name"] for agent in _page_data(filtered_page)}
        assert alternate_name in filtered_names
        assert not filtered_names.intersection(agent_names)
    finally:
        for name in [*agent_names, alternate_name]:
            _delete_agent_if_exists(sdk, workspace=workspace, name=name)


def test_agents_are_isolated_by_workspace(sdk: NeMoPlatform, workspace: str) -> None:
    """Agents with the same name can exist independently in two workspaces."""
    workspaces = client_from_platform(sdk, WorkspacesClient)
    other_workspace = _unique_name("workspace")
    agent_name = _unique_name("shared")
    workspaces.create_workspace(body=CreateWorkspaceRequest(name=other_workspace)).data()

    try:
        sdk.agents.create(
            workspace=workspace,
            name=agent_name,
            description="Primary workspace agent",
            config=_agent_config(f"{agent_name}-primary"),
        )

        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            sdk.agents.get(agent_name, workspace=other_workspace)
        _assert_http_status(exc_info, 404)

        other_list = sdk.agents.list(workspace=other_workspace)
        assert agent_name not in {agent["name"] for agent in _page_data(other_list)}

        other_agent = sdk.agents.create(
            workspace=other_workspace,
            name=agent_name,
            description="Secondary workspace agent",
            config=_agent_config(f"{agent_name}-secondary"),
        )
        assert other_agent["workspace"] == other_workspace
        assert other_agent["name"] == agent_name

        primary_agent = sdk.agents.get(agent_name, workspace=workspace)
        assert primary_agent["workspace"] == workspace
        assert primary_agent["description"] == "Primary workspace agent"

        sdk.agents.delete(agent_name, workspace=workspace)

        other_agent_after_primary_delete = sdk.agents.get(agent_name, workspace=other_workspace)
        assert other_agent_after_primary_delete["workspace"] == other_workspace
        assert other_agent_after_primary_delete["description"] == "Secondary workspace agent"
    finally:
        _delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)
        _delete_agent_if_exists(sdk, workspace=other_workspace, name=agent_name)
        workspaces.delete_workspace(name=other_workspace).data()


def test_agents_sdk_resource_methods_are_available(sdk: NeMoPlatform, workspace: str) -> None:
    """Verify the agents SDK resource exposes the expected subprocess-safe methods."""
    agents = sdk.agents
    for method_name in ("create", "list", "get", "delete"):
        method = getattr(agents, method_name, None)
        assert isinstance(method, Callable), f"sdk.agents.{method_name} should be callable"

    for method_name in ("create", "list", "get", "delete"):
        method = getattr(agents.deployments, method_name, None)
        assert isinstance(method, Callable), f"sdk.agents.deployments.{method_name} should be callable"

    deployments = agents.deployments.list(workspace=workspace)
    assert isinstance(_page_data(deployments), list)


def test_agents_sdk_missing_get_raises_not_found(sdk: NeMoPlatform, workspace: str) -> None:
    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
        sdk.agents.get(_unique_name("sdk-missing"), workspace=workspace)
    _assert_http_status(exc_info, 404)


@pytest.mark.container_only
def test_agent_deployment_missing_agent_returns_not_found(sdk: NeMoPlatform, workspace: str) -> None:
    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
        sdk.agents.deployments.create(
            workspace=workspace,
            agent=_unique_name("missing-agent"),
            name=_unique_name("missing-agent-deployment"),
        )
    _assert_http_status(exc_info, 404)


@pytest.mark.container_only
def test_agent_deployment_create_list_get_delete_lifecycle(sdk: NeMoPlatform, workspace: str) -> None:
    agent_name = _unique_name("deployment-agent")
    deployment_name = _unique_name("deployment")
    sdk.agents.create(workspace=workspace, name=agent_name, config=_agent_config(agent_name))

    try:
        created = sdk.agents.deployments.create(
            workspace=workspace,
            agent=agent_name,
            name=deployment_name,
        )
        assert created["name"] == deployment_name
        assert created["workspace"] == workspace
        assert created["agent"] == agent_name
        _assert_deployment_status(created["status"])

        deployments = sdk.agents.deployments.list(workspace=workspace)
        deployment_names = {deployment["name"] for deployment in _page_data(deployments)}
        assert deployment_name in deployment_names

        retrieved = sdk.agents.deployments.get(deployment_name, workspace=workspace)
        assert retrieved["name"] == deployment_name
        assert retrieved["workspace"] == workspace
        assert retrieved["agent"] == agent_name
        assert isinstance(retrieved["config"], dict)
        _assert_deployment_status(retrieved["status"])
    finally:
        _delete_deployment_if_exists(sdk, workspace=workspace, name=deployment_name)
        _wait_for_deployment_deleted(sdk, workspace=workspace, name=deployment_name)
        _delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)

    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
        sdk.agents.deployments.get(deployment_name, workspace=workspace)
    _assert_http_status(exc_info, 404)


@pytest.mark.container_only
def test_agent_delete_is_blocked_while_deployment_is_active(sdk: NeMoPlatform, workspace: str) -> None:
    agent_name = _unique_name("blocked-delete-agent")
    deployment_name = _unique_name("blocked-delete-deployment")
    sdk.agents.create(workspace=workspace, name=agent_name, config=_agent_config(agent_name))

    try:
        deployment = sdk.agents.deployments.create(
            workspace=workspace,
            agent=agent_name,
            name=deployment_name,
        )
        _assert_deployment_status(deployment["status"])

        with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
            sdk.agents.delete(agent_name, workspace=workspace)
        _assert_http_status(exc_info, 409)

        agent = sdk.agents.get(agent_name, workspace=workspace)
        assert agent["name"] == agent_name
    finally:
        _delete_deployment_if_exists(sdk, workspace=workspace, name=deployment_name)
        _wait_for_deployment_deleted(sdk, workspace=workspace, name=deployment_name)
        _delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)


@pytest.mark.container_only
def test_agent_gateway_missing_deployment_returns_not_found(sdk: NeMoPlatform, workspace: str) -> None:
    with pytest.raises((httpx.HTTPStatusError, NemoHTTPError)) as exc_info:
        sdk.agents.invoke(
            workspace=workspace,
            deployment=_unique_name("missing-deployment"),
            input="hello",
        )
    _assert_http_status(exc_info, 404)


@pytest.mark.container_only
def test_agent_deployment_reaches_running(sdk: NeMoPlatform, workspace: str) -> None:
    agent_name = _unique_name("running-agent")
    deployment_name = _unique_name("running-deployment")
    model_name = _unique_name("agent-model")
    add_mock_provider(
        sdk,
        workspace=workspace,
        name=_unique_name("agent-provider"),
        mock_response_body_by_model={
            f"{workspace}/{model_name}": [
                MockProviderResponse(response_body=_chat_completion_response(_TEST_AGENT_RESPONSE, model_name)),
            ],
        },
        served_models={model_name: model_name},
    )
    sdk.agents.create(
        workspace=workspace,
        name=agent_name,
        config=_mock_backed_agent_config(f"{workspace}/{model_name}"),
    )

    try:
        sdk.agents.deployments.create(
            workspace=workspace,
            agent=agent_name,
            name=deployment_name,
        )
        deployment = _wait_for_deployment_running(sdk, workspace=workspace, name=deployment_name)
        assert deployment["agent"] == agent_name
        assert deployment["endpoint"]
    finally:
        _delete_deployment_if_exists(sdk, workspace=workspace, name=deployment_name)
        _wait_for_deployment_deleted(sdk, workspace=workspace, name=deployment_name)
        _delete_agent_if_exists(sdk, workspace=workspace, name=agent_name)
