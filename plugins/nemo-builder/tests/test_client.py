# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed client other services use to submit builds."""

from __future__ import annotations

import json

import httpx
import pytest
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


async def _sent(*specs: BuildSpec) -> list[dict[str, object]]:
    """The build specs as the client sends them in a request for ``specs``."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"job": "builder-demo-1", "images": [ROW]})

    client = AsyncBuilderClient(
        base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await client.submit_build(workspace="ws-a", body=BuildSet(name="demo", revision=1, build_specs=list(specs)))
    return json.loads(requests[0].content)["build_specs"]


async def test_a_spec_built_without_a_role_is_sent_without_one() -> None:
    """The client sends the fields its caller set. An API pod from before roles forbids a field it doesn't
    know, so a request that names no role must not carry the key, or every caller would need the upgrade."""
    (spec,) = await _sent(BuildSpec(source=FileSetSource(fileset="fs-a")))
    assert spec == {"source": {"fileset": "fs-a"}}


@pytest.mark.parametrize(
    "spec",
    [
        BuildSpec(source=FileSetSource(fileset="fs-a"), role=None),
        BuildSpec.model_validate({"source": {"fileset": "fs-a"}, "role": None}),
        BuildSpec(source=FileSetSource(fileset="fs-a")).model_copy(update={"role": None}),
    ],
    ids=["passed-as-none", "read-from-a-request-that-says-null", "copied-with-none"],
)
async def test_a_role_that_was_set_to_none_is_not_sent_either(spec: BuildSpec) -> None:
    """The client sends the fields a spec has set, and a spec can have set its role without marking a helper:
    passed None, or read from a request that spells the role null. To this API a null role is no role. To an
    API pod from before roles it is a field it doesn't know, and a `422`, for a request that pod took before
    this client knew of roles. So a role with no value is never sent, set or not."""
    assert "role" in spec.model_fields_set
    (sent,) = await _sent(spec)
    assert sent == {"source": {"fileset": "fs-a"}}


async def test_a_helper_is_sent_as_one() -> None:
    main, sidecar = await _sent(
        BuildSpec(name="main", source=FileSetSource(fileset="fs-a")),
        BuildSpec(name="sidecar", source=FileSetSource(fileset="fs-a"), role="helper"),
    )
    assert "role" not in main
    assert sidecar["role"] == "helper"
