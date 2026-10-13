# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding or starting the Builder build of one image, and the HTTP calls behind it."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
from nemo_evals.image_builds.builder import (
    ArchiveSource,
    BuilderImage,
    HttpImageBuilder,
    SubmittedBuild,
    archive_source,
    builder_name,
    ensure_build,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError

_SOURCE = ArchiveSource(fileset="default/harbor-tasks", archive="sort/abc/task_archive", context_path="sort/tests")


class _Builder:
    def __init__(self, revisions: list[Literal["pending", "ready", "failed"]]) -> None:
        self.statuses = revisions
        self.submitted: list[int] = []

    async def get_image(self, name: str) -> BuilderImage | None:
        revision = int(name.rsplit("-", 1)[1])
        if revision > len(self.statuses):
            return None
        return BuilderImage(name=name, status=self.statuses[revision - 1])

    async def submit(self, *, name: str, revision: int, source: ArchiveSource, dockerfile: str) -> SubmittedBuild:
        self.submitted.append(revision)
        return SubmittedBuild(image=BuilderImage(name=f"{name}-{revision}", status="pending"), job=f"builder-{name}")


@pytest.mark.parametrize(
    ("existing", "submitted"),
    [
        ([], 1),
        (["pending"], 1),
        (["ready"], 1),
        (["failed"], 2),
        (["failed", "pending"], 2),
        (["failed", "failed"], 3),
    ],
)
async def test_ensure_build_resubmits_the_newest_revision_unless_it_failed(
    existing: list[Literal["pending", "ready", "failed"]], submitted: int
) -> None:
    builder = _Builder(existing)

    await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [submitted]


def test_builder_name_depends_on_every_build_input() -> None:
    base = {"archive_digest": "a" * 64, "context_path": "sort/environment", "dockerfile": "Dockerfile"}
    name = builder_name(**base)

    assert name == builder_name(**base)
    assert name.startswith("harbor-env-") and len(name) == len("harbor-env-") + 12
    for field, other in [("archive_digest", "b" * 64), ("context_path", "sort/tests"), ("dockerfile", "D2")]:
        assert builder_name(**{**base, field: other}) != name


def test_archive_source_splits_a_fileset_reference() -> None:
    assert archive_source("default/harbor-tasks#sort/abc/task_archive", "sort/tests") == _SOURCE


def _http_builder(handler) -> HttpImageBuilder:
    client = AsyncNemoClient(
        base_url="http://platform.test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return HttpImageBuilder(client, workspace="default")


async def test_http_builder_submits_one_unnamed_spec_per_image() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={
                "job": "builder-harbor-env-0123456789ab-2",
                "images": [{"name": "harbor-env-0123456789ab-2", "status": "pending", "registry": "r"}],
            },
        )

    submitted = await _http_builder(handler).submit(
        name="harbor-env-0123456789ab", revision=2, source=_SOURCE, dockerfile="Dockerfile"
    )

    assert requests[0].method == "POST"
    assert requests[0].url.path == "/apis/builder/v2/workspaces/default/builds"
    assert json.loads(requests[0].content) == {
        "name": "harbor-env-0123456789ab",
        "revision": 2,
        "build_specs": [
            {
                "source": {
                    "fileset": "default/harbor-tasks",
                    "archive": "sort/abc/task_archive",
                    "context_path": "sort/tests",
                },
                "dockerfile": "Dockerfile",
            }
        ],
    }
    assert submitted == SubmittedBuild(
        image=BuilderImage(name="harbor-env-0123456789ab-2", status="pending"), job="builder-harbor-env-0123456789ab-2"
    )


async def test_http_builder_reads_an_image_and_reports_a_missing_one_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/harbor-env-0123456789ab-1"):
            return httpx.Response(
                200, json={"name": "harbor-env-0123456789ab-1", "status": "ready", "image_ref": "r@x"}
            )
        return httpx.Response(404, json={"detail": "not found"})

    builder = _http_builder(handler)

    assert await builder.get_image("harbor-env-0123456789ab-1") == BuilderImage(
        name="harbor-env-0123456789ab-1", status="ready", image_ref="r@x"
    )
    assert await builder.get_image("harbor-env-0123456789ab-2") is None


async def test_http_builder_raises_when_builder_refuses_a_read() -> None:
    builder = _http_builder(lambda request: httpx.Response(403, json={"detail": "not yours"}))

    with pytest.raises(PermissionDeniedError):
        await builder.get_image("harbor-env-0123456789ab-1")
