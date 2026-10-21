# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding or starting the Builder build of one image, and the HTTP calls behind it."""

from __future__ import annotations

import json
from typing import Literal

import httpx
import pytest
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import FileSetSource, ImageSource, SubmitBuildResponse
from nemo_evals.image_builds.builder import (
    HttpImageBuilder,
    archive_source,
    builder_name,
    copy_builder_name,
    ensure_build,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError

_SOURCE = FileSetSource(fileset="default/harbor-tasks", archive="sort/abc/task_archive", context_path="sort/tests")


def _container_image(
    name: str, status: Literal["pending", "ready", "failed"] = "pending", *, detail: str | None = None
) -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace="default",
        status=status,
        status_detail=detail,
        registry="registry.test",
        repository=f"default/{name}",
        digest="sha256:" + "c" * 64 if status == "ready" else None,
        provenance=Provenance(
            build_set=name.rsplit("-", 1)[0],
            revision=int(name.rsplit("-", 1)[1]),
            job=f"default/builder-{name}",
            request_digest="sha256:" + "a" * 64,
        ),
    )


def _submitted(name: str, revision: int) -> SubmitBuildResponse:
    return SubmitBuildResponse(job=f"builder-{name}-{revision}", images=[_container_image(f"{name}-{revision}")])


class _Builder:
    def __init__(self, revisions: list[Literal["pending", "ready", "failed"]], *, job: str | None = "active") -> None:
        self.statuses = revisions
        self.job = job
        self.submitted: list[int] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        revision = int(name.rsplit("-", 1)[1])
        if revision > len(self.statuses):
            return None
        return _container_image(name, self.statuses[revision - 1])

    async def job_status(self, job: str) -> str | None:
        return self.job

    async def submit(
        self, *, name: str, revision: int, source: FileSetSource | ImageSource, dockerfile: str | None
    ) -> SubmitBuildResponse:
        self.submitted.append(revision)
        return _submitted(name, revision)


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
    base = {
        "archive": "default/harbor-tasks#sort/abc/task_archive",
        "archive_digest": "a" * 64,
        "context_path": "sort/environment",
        "dockerfile": "Dockerfile",
    }
    name = builder_name(**base)

    assert name == builder_name(**base)
    assert name.startswith("harbor-env-") and len(name) == len("harbor-env-") + 12
    for field, other in [
        # Builder fingerprints the source location too: one name for two locations would conflict there.
        ("archive", "team/harbor-tasks#sort/abc/task_archive"),
        ("archive_digest", "b" * 64),
        ("context_path", "sort/tests"),
        ("dockerfile", "D2"),
    ]:
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
        return httpx.Response(201, json=_submitted("harbor-env-0123456789ab", 2).model_dump(mode="json"))

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
    assert submitted.job == "builder-harbor-env-0123456789ab-2"
    assert [(image.name, image.status) for image in submitted.images] == [("harbor-env-0123456789ab-2", "pending")]


async def test_http_builder_reads_an_image_and_reports_a_missing_one_as_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/harbor-env-0123456789ab-1"):
            return httpx.Response(
                200, json=_container_image("harbor-env-0123456789ab-1", "ready").model_dump(mode="json")
            )
        return httpx.Response(404, json={"detail": "not found"})

    builder = _http_builder(handler)

    image = await builder.get_image("harbor-env-0123456789ab-1")
    assert image is not None
    assert (image.status, image.image_ref) == (
        "ready",
        f"registry.test/default/harbor-env-0123456789ab-1@sha256:{'c' * 64}",
    )
    assert await builder.get_image("harbor-env-0123456789ab-2") is None


async def test_http_builder_raises_when_builder_refuses_a_read() -> None:
    builder = _http_builder(lambda request: httpx.Response(403, json={"detail": "not yours"}))

    with pytest.raises(PermissionDeniedError):
        await builder.get_image("harbor-env-0123456789ab-1")


async def test_http_builder_submits_a_copy_without_a_dockerfile() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=_submitted("harbor-img-0123456789ab", 1).model_dump(mode="json"))

    await _http_builder(handler).submit(
        name="harbor-img-0123456789ab", revision=1, source=ImageSource(image="ubuntu:24.04"), dockerfile=None
    )

    spec = json.loads(requests[0].content)["build_specs"][0]
    assert spec["source"] == {"image": "ubuntu:24.04"}
    assert "dockerfile" not in spec


async def test_http_builder_reads_a_builder_jobs_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apis/jobs/v2/workspaces/default/jobs/builder-harbor-env-0123456789ab-1/status":
            return httpx.Response(
                200,
                json={
                    "id": "job-1",
                    "name": "builder-harbor-env-0123456789ab-1",
                    "status": "error",
                    "status_details": {},
                    "error_details": None,
                    "steps": [],
                    "created_at": "2026-10-15T00:00:00Z",
                    "updated_at": "2026-10-15T00:00:00Z",
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    builder = _http_builder(handler)

    assert await builder.job_status("builder-harbor-env-0123456789ab-1") == "error"
    assert await builder.job_status("builder-gone-1") is None


def test_copy_builder_name_depends_on_the_image() -> None:
    name = copy_builder_name("docker.io/library/ubuntu:24.04")

    assert name.startswith("harbor-img-") and len(name) == len("harbor-img-") + 12
    assert copy_builder_name("docker.io/library/ubuntu:22.04") != name


@pytest.mark.parametrize(("job", "submitted"), [("active", 1), ("completed", 2), ("error", 2), (None, 2)])
async def test_a_pending_revision_whose_job_ended_or_never_ran_is_replaced(job: str | None, submitted: int) -> None:
    """Builder leaves a dead build pending; rejoining it would fail every later request for that image."""
    builder = _Builder(["pending"], job=job)

    await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [submitted]
