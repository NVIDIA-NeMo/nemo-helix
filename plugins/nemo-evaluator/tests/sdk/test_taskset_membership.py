# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from nemo_evaluator.api.schemas import TasksetInput
from nemo_evaluator.sdk.taskset_resources import AsyncEvaluatorTasksetsResource, EvaluatorTasksetsResource
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient


@pytest.mark.parametrize(
    "body", [{}, {"tasks": [], "task_ids": []}, {"tasks": None}, {"task_ids": None}, {"task_ids": ["a", "a"]}]
)
def test_invalid_membership(body):
    with pytest.raises(ValueError):
        TasksetInput.model_validate(body)


@pytest.mark.parametrize("field", ["tasks", "task_ids"])
@pytest.mark.parametrize("method", ["create", "replace"])
@pytest.mark.parametrize("empty", [True, False])
async def test_membership_wire(field, method, empty):
    values = [] if empty else ["member"]
    body = {field: values}
    payload = {
        "id": "suite-id",
        "name": "suite",
        "workspace": "default",
        "tasks": [],
        "revision": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }

    def handler(request):
        wire = json.loads(request.content)
        assert wire == body
        TasksetInput.model_validate(wire)
        return httpx.Response(200, json=payload)

    options = {"tasks": values} if field == "task_ids" else {"taskset": TasksetInput.model_validate({"tasks": values})}
    client = EvaluatorClient(
        base_url="http://test", workspace="default", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    getattr(EvaluatorTasksetsResource(client), method)("suite", **options)
    async_client = AsyncEvaluatorClient(
        base_url="http://test",
        workspace="default",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await getattr(AsyncEvaluatorTasksetsResource(async_client), method)("suite", **options)


@pytest.mark.parametrize(
    ("body", "valid"),
    [
        ({}, False),
        ({"tasks": [], "task_ids": []}, False),
        ({"tasks": []}, True),
        ({"task_ids": []}, True),
        ({"tasks": ["member"]}, True),
        ({"task_ids": ["id"]}, True),
    ],
)
def test_membership_json_schema(body, valid):
    from jsonschema import Draft202012Validator

    assert Draft202012Validator(TasksetInput.model_json_schema()).is_valid(body) is valid
