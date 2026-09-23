# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the client.evaluator.tasksets SDK resources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from nemo_evaluator.api.schemas import Revision, TaskRef, Taskset, TasksetInput
from nemo_evaluator.sdk.taskset_resources import AsyncEvaluatorTasksetsResource, EvaluatorTasksetsResource
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


def _taskset_payload(name: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return Taskset(
        id=f"taskset-{name}",
        name=name,
        workspace="default",
        description="A grouping.",
        tasks=[TaskRef("default/task-a")],
        revision=1,
        tags={"latest": 1},
        created_at=now,
        updated_at=now,
    ).model_dump(mode="json")


def _taskset_input() -> TasksetInput:
    return TasksetInput(description="A grouping.", tasks=[TaskRef("default/task-a")])


def _revision_payload(ordinal: int, digest: str) -> dict[str, Any]:
    return Revision(revision=ordinal, content_hash=digest, tags=[], created_at=datetime.now(timezone.utc)).model_dump(
        mode="json"
    )


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
) -> tuple[EvaluatorTasksetsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.Client(transport=httpx.MockTransport(recorder))
    client = EvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return EvaluatorTasksetsResource(client), recorder


def _async_resource(
    *payloads: dict[str, Any] | httpx.Response,
) -> tuple[AsyncEvaluatorTasksetsResource, _Recorder]:
    recorder = _Recorder(*payloads)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.async_handler))
    client = AsyncEvaluatorClient(base_url="http://localhost:8080", workspace="default", http_client=http_client)
    return AsyncEvaluatorTasksetsResource(client), recorder


def _request_url(request: httpx.Request) -> str:
    return str(request.url).split("?", 1)[0]


def _request_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


def test_sync_create_posts_taskset_input_to_item_url() -> None:
    resource, recorder = _sync_resource(_taskset_payload("ts-1"))

    result = resource.create("ts-1", taskset=_taskset_input())

    request = recorder.requests[0]
    assert isinstance(result, Taskset)
    assert result.name == "ts-1"
    assert request.method == "POST"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1"
    assert _request_body(request)["tasks"] == ["default/task-a"]


def test_sync_retrieve_targets_item_url_and_parses_dto() -> None:
    resource, recorder = _sync_resource(_taskset_payload("ts-1"))

    result = resource.retrieve("ts-1")

    request = recorder.requests[0]
    assert isinstance(result, Taskset)
    assert isinstance(result.tasks[0], TaskRef)
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1"


def test_sync_list_parses_page() -> None:
    resource, recorder = _sync_resource(_page([_taskset_payload("a"), _taskset_payload("b")]))

    page = resource.list(sort="-created_at")

    request = recorder.requests[0]
    assert {t.name for t in page.data} == {"a", "b"}
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasksets"
    assert request.url.params["sort"] == "-created_at"


def test_sync_delete_issues_delete_request() -> None:
    resource, recorder = _sync_resource(httpx.Response(204))

    resource.delete("ts-1")

    request = recorder.requests[0]
    assert request.method == "DELETE"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1"


async def test_async_retrieve_parses_dto() -> None:
    resource, recorder = _async_resource(_taskset_payload("ts-9"))

    result = await resource.retrieve("ts-9")

    request = recorder.requests[0]
    assert isinstance(result, Taskset)
    assert result.name == "ts-9"
    assert request.method == "GET"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-9"


# --- Revision-aware resources -------------------------------------------------


def test_sync_replace_puts_taskset_input_to_item_url() -> None:
    resource, recorder = _sync_resource(_taskset_payload("ts-1"))

    result = resource.replace("ts-1", taskset=_taskset_input())

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1"
    assert isinstance(result, Taskset)


def test_sync_retrieve_with_revision_targets_the_revision_sub_path() -> None:
    resource, recorder = _sync_resource(_taskset_payload("ts-1"))
    digest = "a" * 64

    resource.retrieve("ts-1", revision=digest)

    assert _request_url(recorder.requests[0]) == f"{_BASE}/tasksets/ts-1/revisions/{digest}"


def test_sync_list_revisions_requests_a_page() -> None:
    resource, recorder = _sync_resource(
        _page([_revision_payload(1, "a" * 64)], page=2, page_size=50),
    )

    page = resource.list_revisions("ts-1", page=2, page_size=50)

    request = recorder.requests[0]
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1/revisions"
    assert dict(request.url.params) == {"page": "2", "page_size": "50"}
    # The envelope is carried through so a caller can tell a truncated history from a complete one.
    assert page.pagination is not None and page.pagination.total_results == 1


def test_sync_tag_puts_to_the_tag_url_with_the_revision() -> None:
    resource, recorder = _sync_resource(_taskset_payload("ts-1"))
    digest = "a" * 64

    resource.tag("ts-1", tag="blessed", revision=digest)

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1/tags/blessed"
    assert request.url.params["revision"] == digest


async def test_async_replace_puts_taskset_input() -> None:
    resource, recorder = _async_resource(_taskset_payload("ts-1"))

    result = await resource.replace("ts-1", taskset=_taskset_input())

    request = recorder.requests[0]
    assert request.method == "PUT"
    assert _request_url(request) == f"{_BASE}/tasksets/ts-1"
    assert isinstance(result, Taskset)


async def test_async_list_revisions_parses_the_page() -> None:
    resource, _ = _async_resource(_page([_revision_payload(1, "a" * 64)]))

    page = await resource.list_revisions("ts-1")

    assert [r.revision for r in page.data] == [1]
