# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the client.evaluator.tasks SDK resources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetricRef,
    Revision,
    Task,
    TaskInput,
    TaskInputs,
)
from nemo_evaluator.sdk.task_resources import AsyncEvaluatorTasksResource, EvaluatorTasksResource
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient

_BASE = "http://localhost:8080/apis/evaluator/v2/workspaces/default"


class _Recorder:
    def __init__(self, *payloads: dict[str, Any] | httpx.Response) -> None:
        self.payloads = list(payloads)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        payload = self.payloads.pop(0)
        if isinstance(payload, httpx.Response):
            return payload
        return httpx.Response(200, json=payload)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        return self(request)


def _task_payload(name: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return Task(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer the question.",
            inputs=TaskInputs(instruction="What is 2+2?"),
            metrics=[MetricRef("default/stored-metric")],
        ),
        id=f"task-{name}",
        name=name,
        workspace="default",
        revision=1,
        tags={"latest": 1},
        created_at=now,
        updated_at=now,
    ).model_dump(mode="json")


def _task_input() -> TaskInput:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Answer.",
            inputs=TaskInputs(instruction="x"),
            metrics=[MetricRef("default/stored-metric")],
        )
    )


def _revision_payload(ordinal: int, digest: str) -> dict[str, Any]:
    return Revision(
        revision=ordinal,
        content_hash=digest,
        tags=["latest"] if ordinal == 2 else [],
        created_at=datetime.now(timezone.utc),
    ).model_dump(mode="json")


def _page(items: list[dict[str, Any]], *, page: int = 1, page_size: int = 100) -> dict[str, Any]:
    return {
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "current_page_size": len(items),
            "total_pages": page,
            "total_results": len(items),
        },
    }


def _sync_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[EvaluatorTasksResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.Client(transport=httpx.MockTransport(recorder))
    client = EvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return EvaluatorTasksResource(client), recorder


def _async_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[AsyncEvaluatorTasksResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.async_handler))
    client = AsyncEvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return AsyncEvaluatorTasksResource(client), recorder


def _request_url(request: httpx.Request) -> str:
    return str(request.url).split("?", 1)[0]


def _request_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


def test_sync_create_posts_task_input_to_item_url() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))

    result = resource.create("task-1", task=_task_input())

    request = recorder.requests[0]
    assert isinstance(result, Task)
    assert result.name == "task-1"
    assert request.method == "POST"
    assert _request_url(request) == f"{_BASE}/tasks/task-1"
    assert _request_body(request)["spec"]["intent"] == "Answer."


def test_sync_retrieve_targets_item_url_and_parses_dto() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))

    result = resource.retrieve("task-1")

    request = recorder.requests[0]
    assert isinstance(result, Task)
    assert isinstance(result.spec, EvaluatorTaskDefinition)
    assert isinstance(result.spec.metrics[0], MetricRef)
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasks/task-1"


def test_sync_list_parses_page() -> None:
    resource, recorder = _sync_resource(_page([_task_payload("a"), _task_payload("b")]))

    page = resource.list(sort="-created_at")

    request = recorder.requests[0]
    assert {t.name for t in page.data} == {"a", "b"}
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasks"
    assert request.url.params["sort"] == "-created_at"


def test_sync_delete_issues_delete_request() -> None:
    resource, recorder = _sync_resource(httpx.Response(204))

    resource.delete("task-1")

    request = recorder.requests[0]
    assert request.method == "DELETE"
    assert _request_url(request) == f"{_BASE}/tasks/task-1"


async def test_async_retrieve_parses_dto() -> None:
    resource, recorder = _async_resource(_task_payload("task-9"))

    result = await resource.retrieve("task-9")

    request = recorder.requests[0]
    assert isinstance(result, Task)
    assert result.name == "task-9"
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasks/task-9"


# --- Revision-aware resources -------------------------------------------------


def test_sync_replace_puts_task_input_to_item_url() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))

    result = resource.replace("task-1", task=_task_input())

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasks/task-1"
    assert _request_body(request)["spec"]["intent"] == "Answer."
    assert isinstance(result, Task)


def test_sync_replace_passes_project_through() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))

    resource.replace("task-1", task=_task_input(), project="proj-a")

    assert recorder.requests[0].url.params["project"] == "proj-a"


def test_sync_retrieve_without_revision_targets_the_item_url() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))

    resource.retrieve("task-1")

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1"


def test_sync_retrieve_with_revision_targets_the_revision_sub_path() -> None:
    """The revision is a path segment, matching the tags route rather than a query parameter."""
    resource, recorder = _sync_resource(_task_payload("task-1"))
    digest = "a" * 64

    resource.retrieve("task-1", revision=digest)

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1/revisions/{digest}"


def test_sync_retrieve_with_tag_targets_the_same_sub_path() -> None:
    """``tag`` and ``revision`` are two names for one route segment, resolved server-side.

    Splitting them is a call-site readability change only, so a tag must reach exactly the URL a
    digest would; a separate query parameter or route would be a behaviour change nobody asked for.
    """
    resource, recorder = _sync_resource(_task_payload("task-1"))

    resource.retrieve("task-1", tag="blessed")

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1/revisions/blessed"


def test_sync_retrieve_rejects_both_selectors() -> None:
    """Two selectors is ambiguous intent, not a precedence question; refuse rather than pick one."""
    resource, _ = _sync_resource()

    with pytest.raises(ValueError, match="not both"):
        resource.retrieve("task-1", revision="a" * 64, tag="blessed")


def test_sync_retrieve_percent_encodes_a_tag() -> None:
    """Tags admit ``/`` (``release/v1``), which would otherwise open a path segment."""
    resource, recorder = _sync_resource(_task_payload("task-1"))

    resource.retrieve("task-1", tag="release/v1")

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1/revisions/release%2Fv1"


def test_sync_list_revisions_parses_the_page() -> None:
    resource, recorder = _sync_resource(_page([_revision_payload(2, "b" * 64), _revision_payload(1, "a" * 64)]))

    page = resource.list_revisions("task-1")

    request = recorder.requests[0]
    assert _request_url(request) == f"{_BASE}/tasks/task-1/revisions"
    assert [r.revision for r in page.data] == [2, 1]
    assert page.data[0].content_hash == "b" * 64


def test_sync_tag_puts_to_the_tag_url_with_the_revision() -> None:
    resource, recorder = _sync_resource(_task_payload("task-1"))
    digest = "a" * 64

    resource.tag("task-1", tag="blessed", revision=digest)

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasks/task-1/tags/blessed"
    assert request.url.params["revision"] == digest


def test_sync_tag_escapes_the_tag_name() -> None:
    """Tag names reach the URL as a path segment; anything needing escaping must be escaped."""
    resource, recorder = _sync_resource(_task_payload("task-1"))

    resource.tag("task-1", tag="release/v1", revision="a" * 64)

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1/tags/release%2Fv1"


async def test_async_replace_puts_task_input() -> None:
    resource, recorder = _async_resource(_task_payload("task-1"))

    result = await resource.replace("task-1", task=_task_input())

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasks/task-1"
    assert isinstance(result, Task)


async def test_async_retrieve_with_revision_targets_the_revision_sub_path() -> None:
    resource, recorder = _async_resource(_task_payload("task-1"))
    digest = "a" * 64

    await resource.retrieve("task-1", revision=digest)

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasks/task-1/revisions/{digest}"


async def test_async_tag_puts_to_the_tag_url() -> None:
    resource, recorder = _async_resource(_task_payload("task-1"))

    await resource.tag("task-1", tag="blessed", revision="a" * 64)

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasks/task-1/tags/blessed"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["create", "replace"])
@pytest.mark.parametrize("kind", ["ready", "sdk_harbor", "sdk_gym"])
@pytest.mark.parametrize("failed", [False, True])
async def test_preparation_dispatch_and_registration_errors(tmp_path, monkeypatch, asynchronous, method, kind, failed):
    from unittest.mock import AsyncMock, Mock

    from nemo_evaluator.sdk.task_preparation import TaskPublicationError
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HarborAgentEvalTask
    from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask

    prepared = _task_input()
    inputs = {
        "sdk_harbor": HarborAgentEvalTask(id="task", intent="test", inputs={}, source_dir=tmp_path),
        "sdk_gym": AgentEvalTask(id="task", intent="test", inputs={}),
        "ready": prepared,
    }
    prepare = AsyncMock(return_value=prepared) if asynchronous else Mock(return_value=prepared)
    preparation_name = "prepare_task_async" if asynchronous else "prepare_task"
    monkeypatch.setattr(f"nemo_evaluator.sdk.task_resources.{preparation_name}", prepare)
    factory = _async_resource if asynchronous else _sync_resource
    resource, recorder = factory(_task_payload("task-1"))
    failure = RuntimeError("registration failed")
    if failed:
        write = AsyncMock(side_effect=failure) if asynchronous else Mock(side_effect=failure)
        monkeypatch.setattr(resource._client, f"{method}_task", write)

    async def invoke():
        result = getattr(resource, method)("task-1", task=inputs[kind])
        return await result if asynchronous else result

    if failed:
        expected = RuntimeError if kind == "ready" else TaskPublicationError
        with pytest.raises(expected) as raised:
            await invoke()
        if kind == "ready":
            assert raised.value is failure
        else:
            assert isinstance(raised.value, TaskPublicationError)
            assert raised.value.prepared_task is prepared
            assert raised.value.__cause__ is failure
    else:
        assert (await invoke()).name == "task-1"
        assert len(recorder.requests) == 1
    assert prepare.call_count == (0 if kind == "ready" else 1)
    if asynchronous:
        assert prepare.await_count == (0 if kind == "ready" else 1)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("method", ["create", "replace"])
@pytest.mark.parametrize("option", ["fileset_ref", "path_prefix", "metric_bundle_packager"])
async def test_ready_input_rejects_preparation_options(monkeypatch, asynchronous, method, option):
    from unittest.mock import AsyncMock, Mock

    factory = _async_resource if asynchronous else _sync_resource
    resource, recorder = factory(_task_payload("task-1"))
    prepare = AsyncMock() if asynchronous else Mock()
    preparation_name = "prepare_task_async" if asynchronous else "prepare_task"
    monkeypatch.setattr(f"nemo_evaluator.sdk.task_resources.{preparation_name}", prepare)
    write = AsyncMock() if asynchronous else Mock()
    monkeypatch.setattr(resource._client, f"{method}_task", write)
    value = {
        "fileset_ref": "storage/custom",
        "path_prefix": "release-1",
        "metric_bundle_packager": Mock(),
    }[option]
    with pytest.raises(ValueError, match="Preparation options cannot be used with TaskInput"):
        result = getattr(resource, method)("task-1", task=_task_input(), **{option: value})
        if asynchronous:
            await result
    prepare.assert_not_called()
    write.assert_not_called()
    assert recorder.requests == []
