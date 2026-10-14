# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed client other services use to submit builds."""

from __future__ import annotations

import json

import httpx
from nemo_builder_plugin.client import AsyncBuilderClient
from nemo_builder_plugin.schema import BuildSet, BuildSpec, FileSetSource

ROW = {
    "name": "demo-1",
    "workspace": "ws-a",
    "status": "pending",
    "registry": "reg.example.com",
    "repository": "ws-a/demo",
    "provenance": {
        "build_set": "demo",
        "revision": 1,
        "job": "ws-a/builder-demo-1",
        "request_digest": "sha256:" + "a" * 64,
    },
}


async def test_submit_build_posts_the_build_set_and_returns_its_job_and_images() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"job": "builder-demo-1", "images": [ROW]})

    client = AsyncBuilderClient(
        base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    build_set = BuildSet(
        name="demo",
        revision=1,
        build_specs=[BuildSpec(source=FileSetSource(fileset="fs-a", archive="t/task_archive", context_path="env"))],
    )

    response = (await client.submit_build(workspace="ws-a", body=build_set)).data()

    assert (requests[0].method, requests[0].url.path) == ("POST", "/apis/builder/v2/workspaces/ws-a/builds")
    assert json.loads(requests[0].content)["build_specs"][0]["source"]["archive"] == "t/task_archive"
    assert response.job == "builder-demo-1"
    assert [(image.name, image.status) for image in response.images] == [("demo-1", "pending")]
