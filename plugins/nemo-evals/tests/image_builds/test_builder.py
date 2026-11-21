# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding or starting the Builder build of one image, and the HTTP calls behind it."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
import pytest
from nemo_builder_plugin.entities import IN_CLUSTER_BACKEND, ContainerImage, Provenance
from nemo_builder_plugin.schema import BuildMount, FileSetSource, ImageSource, SubmitBuildResponse
from nemo_evals.image_builds.builder import (
    BuildRefused,
    HttpImageBuilder,
    adopt_or_build,
    archive_source,
    builder_name,
    copy_builder_name,
    ensure_build,
    usable_build,
)
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, PermissionDeniedError

_SOURCE = FileSetSource(fileset="default/harbor-tasks", archive="sort/abc/task_archive", context_path="sort/tests")
# The backend of a deployment that builds through a build service outside the cluster.
_REMOTE_BACKEND = "remote"


def _container_image(
    name: str,
    status: Literal["pending", "ready", "failed"] = "pending",
    *,
    detail: str | None = None,
    backend: str = IN_CLUSTER_BACKEND,
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
            backend=backend,
        ),
    )


def _submitted(name: str, revision: int) -> SubmitBuildResponse:
    return SubmitBuildResponse(job=f"builder-{name}-{revision}", images=[_container_image(f"{name}-{revision}")])


class _Builder:
    def __init__(
        self,
        revisions: list[Literal["pending", "ready", "failed"]],
        *,
        job: str | None = "active",
        conflicting: frozenset[int] = frozenset(),
        code: str | None = None,
        built_by: str = IN_CLUSTER_BACKEND,
    ) -> None:
        self.statuses = revisions
        #: The backend every revision there is was submitted under.
        self.built_by = built_by
        self.job = job
        #: Revisions Builder holds for another request.
        self.conflicting = conflicting
        #: What kind of 409 it answers a conflicting revision with, when it says: a refusal has a code.
        self.code = code
        #: The 409 itself, when it is not Builder's own: what a proxy in front of it answers, say.
        self.response: httpx.Response | None = None
        self.submitted: list[int] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        revision = int(name.rsplit("-", 1)[1])
        if revision > len(self.statuses):
            return None
        return _container_image(name, self.statuses[revision - 1], backend=self.built_by)

    async def job_status(self, job: str) -> str | None:
        return self.job

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
        mounts: Sequence[BuildMount] = (),
    ) -> SubmitBuildResponse:
        self.submitted.append(revision)
        if revision in self.conflicting:
            body = {"detail": "a revision names one request"} | ({"code": self.code} if self.code else {})
            raise ConflictError(self.response or httpx.Response(409, json=body))
        return _submitted(name, revision)

    async def image_platforms(self, image: str) -> list[str]:
        return []


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


async def test_ensure_build_moves_past_a_revision_builder_holds_for_another_request() -> None:
    """Such as one built for the platform the deployment used to default to."""
    builder = _Builder(["ready"], conflicting=frozenset({1}))

    submitted = await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [1, 2]
    assert submitted.job == "builder-harbor-env-0123456789ab-2"


@pytest.mark.parametrize("code", ["backend_rejected", "missing_secrets"])
async def test_ensure_build_raises_what_builder_refuses_outright(code: str) -> None:
    """A refusal says so in its code: the next revision would be refused the same way, so it isn't tried."""
    builder = _Builder(["ready"], conflicting=frozenset({1, 2}), code=code)

    with pytest.raises(BuildRefused, match="Builder refused build harbor-env-0123456789ab-1: a revision") as refused:
        await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert refused.value.detail == "a revision names one request"
    assert builder.submitted == [1]


@pytest.mark.parametrize("existing", [[], ["failed"]], ids=["the-first-revision", "after-a-failed-one"])
async def test_ensure_build_raises_a_refusal_of_a_revision_nothing_holds_yet(
    existing: list[Literal["pending", "ready", "failed"]],
) -> None:
    """A deployment that can't take the request refuses its first submit too, where there is no revision to rejoin."""
    revision = len(existing) + 1
    builder = _Builder(existing, conflicting=frozenset({revision}), code="backend_rejected")

    with pytest.raises(BuildRefused) as refused:
        await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert refused.value.detail == "a revision names one request"
    assert builder.submitted == [revision]


async def test_ensure_build_raises_a_409_that_names_no_code_as_builder_answered_it() -> None:
    """Only the code is read here: no row is looked for at the revision, as :func:`adopt_or_build` looks for one.

    So whether another request took the revision meanwhile or an older Builder refused the request, the 409 is the
    client's error it always was, not a refusal.
    """
    builder = _Builder([], conflicting=frozenset({1}))

    with pytest.raises(ConflictError):
        await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [1]


async def test_ensure_build_moves_past_a_revision_another_backend_built() -> None:
    """The deployment's backend changed: that revision's image is where the old backend published it."""
    builder = _Builder(["ready"], conflicting=frozenset({1}), code="backend_changed")

    submitted = await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [1, 2]
    assert submitted.job == "builder-harbor-env-0123456789ab-2"


def test_builder_name_depends_on_the_contexts_contents_and_dockerfile() -> None:
    base = {"context_digest": "a" * 64, "dockerfile": "Dockerfile"}
    name = builder_name(**base)

    assert name == builder_name(**base)
    assert name.startswith("harbor-env-") and len(name) == len("harbor-env-") + 12
    for field, other in [("context_digest", "b" * 64), ("dockerfile", "D2")]:
        assert builder_name(**{**base, field: other}) != name


@pytest.mark.parametrize(
    ("existing", "submitted", "revision"),
    [
        ([], [1], 1),
        (["pending"], [], 1),
        (["ready"], [], 1),
        (["failed"], [2], 2),
        (["failed", "ready"], [], 2),
        (["ready", "failed"], [3], 3),
    ],
    ids=["none", "pending", "ready", "failed", "failed-then-ready", "newest-failed"],
)
async def test_adopt_or_build_uses_a_usable_build_and_submits_only_without_one(
    existing: list[Literal["pending", "ready", "failed"]], submitted: list[int], revision: int
) -> None:
    builder = _Builder(existing)

    image = await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == submitted
    assert image.name == f"harbor-env-0123456789ab-{revision}"


@pytest.mark.parametrize("existing", ["ready", "pending"])
@pytest.mark.parametrize(
    ("built_by", "wanted"),
    [(IN_CLUSTER_BACKEND, _REMOTE_BACKEND), (_REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["the-cluster-built-it", "a-remote-service-built-it"],
)
async def test_adopt_or_build_submits_the_next_revision_past_a_build_another_backend_made(
    existing: Literal["pending", "ready"], built_by: str, wanted: str
) -> None:
    """The deployment's backend changed: the build is where the old backend published it, ready or still building."""
    builder = _Builder([existing], built_by=built_by)

    assert await usable_build(builder, "harbor-env-0123456789ab", backend=wanted) is None
    image = await adopt_or_build(
        builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=wanted
    )

    assert builder.submitted == [2]
    assert image.name == "harbor-env-0123456789ab-2"


@pytest.mark.parametrize("existing", ["ready", "pending"])
@pytest.mark.parametrize("backend", [IN_CLUSTER_BACKEND, _REMOTE_BACKEND])
async def test_adopt_or_build_uses_a_build_the_backend_it_names_made(
    existing: Literal["pending", "ready"], backend: str
) -> None:
    builder = _Builder([existing], built_by=backend)

    image = await adopt_or_build(
        builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=backend
    )

    assert builder.submitted == []
    assert image.name == "harbor-env-0123456789ab-1"
    usable = await usable_build(builder, "harbor-env-0123456789ab", backend=backend)
    assert usable is not None and usable.name == "harbor-env-0123456789ab-1"


@pytest.mark.parametrize("built_by", [IN_CLUSTER_BACKEND, _REMOTE_BACKEND])
async def test_adopt_or_build_naming_no_backend_uses_a_build_whichever_backend_made_it(built_by: str) -> None:
    """The default. An agent layer's Job asks this way, and then refuses a build this cluster didn't make itself."""
    builder = _Builder(["ready"], built_by=built_by)

    image = await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == []
    assert image.provenance.backend == built_by
    usable = await usable_build(builder, "harbor-env-0123456789ab")
    assert usable is not None and usable.name == "harbor-env-0123456789ab-1"


async def test_adopt_or_build_still_submits_past_a_failed_build_of_the_backend_it_names() -> None:
    builder = _Builder(["failed"], built_by=_REMOTE_BACKEND)

    assert await usable_build(builder, "harbor-env-0123456789ab", backend=_REMOTE_BACKEND) is None
    await adopt_or_build(
        builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=_REMOTE_BACKEND
    )

    assert builder.submitted == [2]


async def test_adopt_or_build_resubmits_a_pending_build_whose_job_ended() -> None:
    builder = _Builder(["pending"], job="error")

    await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == [2]


@pytest.mark.parametrize("backend", [None, IN_CLUSTER_BACKEND], ids=["none-named", "builders-named"])
async def test_adopt_or_build_uses_the_build_of_a_request_that_won_the_race(backend: str | None) -> None:
    """Another task submitted the same content first: Builder refuses ours, and theirs is read and used.

    A task image's Job names the backend its build expects, and an agent layer's Job names none. Where Builder
    builds with the backend named, a lost race is settled as it is with none named.
    """
    builder = _Builder([], conflicting=frozenset({1}))

    async def race(name: str) -> ContainerImage | None:
        # The winner's revision appears once our submit has lost to it.
        revision = int(name.rsplit("-", 1)[1])
        return _container_image(name, "pending") if revision == 1 and builder.submitted else None

    builder.get_image = race  # ty: ignore[invalid-assignment]
    image = await adopt_or_build(
        builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=backend
    )

    assert builder.submitted == [1]
    assert image.name == "harbor-env-0123456789ab-1"


@pytest.mark.parametrize(
    ("builders", "wanted"),
    [(IN_CLUSTER_BACKEND, _REMOTE_BACKEND), (_REMOTE_BACKEND, IN_CLUSTER_BACKEND)],
    ids=["builder-builds-in-cluster", "builder-builds-remotely"],
)
async def test_adopt_or_build_returns_the_build_that_won_the_race_under_another_backend(
    builders: str, wanted: str
) -> None:
    """Builder builds with another backend than the caller names, and another request took the next revision first.

    Tasks that share a build context share its name, and their Jobs start together. The winner's build is under
    Builder's backend, as the loser's own would have been, so it is returned for the caller to see that. Passed
    over, it would cost each Job that loses one more Builder build, or after three losses an error that names no
    backend.
    """
    builder = _Builder(["ready"], conflicting=frozenset({2}), built_by=builders)

    async def race(name: str) -> ContainerImage | None:
        # The winner's revision appears once our submit has lost to it.
        revision = int(name.rsplit("-", 1)[1])
        if revision == 1 or (revision == 2 and builder.submitted):
            return _container_image(name, "ready" if revision == 1 else "pending", backend=builders)
        return None

    builder.get_image = race  # ty: ignore[invalid-assignment]
    image = await adopt_or_build(
        builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=wanted
    )

    assert builder.submitted == [2]
    assert (image.name, image.provenance.backend) == ("harbor-env-0123456789ab-2", builders)


async def test_adopt_or_build_reports_a_refusal_that_wrote_no_build() -> None:
    """Builder answers 409 for a deployment it can't build on too; with no winner's build to read, that is not a race."""
    builder = _Builder([], conflicting=frozenset({1}))

    with pytest.raises(BuildRefused, match="a revision names one request") as refused:
        await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert refused.value.detail == "a revision names one request"
    # A caller that took it for the RuntimeError it was before it had a type of its own still does.
    assert isinstance(refused.value, RuntimeError)
    assert builder.submitted == [1]


@pytest.mark.parametrize("code", ["missing_secrets", "backend_rejected"])
async def test_adopt_or_build_raises_a_refusal_even_when_another_request_wrote_the_row_meanwhile(code: str) -> None:
    """What Builder says is read first. A row that appears after the refusal is another request's, and says
    nothing about whether this one can be built."""
    builder = _Builder([], conflicting=frozenset({1}), code=code)

    async def race(name: str) -> ContainerImage | None:
        revision = int(name.rsplit("-", 1)[1])
        return _container_image(name, "ready") if revision == 1 and builder.submitted else None

    builder.get_image = race  # ty: ignore[invalid-assignment]
    with pytest.raises(BuildRefused, match="Builder refused build harbor-env-0123456789ab-1: a revision") as refused:
        await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")
    assert refused.value.detail == "a revision names one request"
    assert builder.submitted == [1]


@pytest.mark.parametrize("backend", [None, IN_CLUSTER_BACKEND], ids=["none-named", "builders-named"])
async def test_adopt_or_build_giving_up_on_a_build_others_keep_submitting_is_not_a_refusal(
    backend: str | None,
) -> None:
    """Every submit lost its revision to another request, whose build then failed. Builder refused nothing.

    A winner's build under the backend the caller names is read again like any other: failed, it gets the next
    revision. Only one under another backend is returned as it is.
    """
    builder = _Builder(["failed"], conflicting=frozenset({2, 3, 4}))

    async def lost(name: str) -> ContainerImage | None:
        # Each submit of ours is answered 409, and the revision it asked for is there, failed, when read.
        revision = int(name.rsplit("-", 1)[1])
        return _container_image(name, "failed") if revision <= 1 + len(builder.submitted) else None

    builder.get_image = lost  # ty: ignore[invalid-assignment]
    with pytest.raises(RuntimeError, match="kept being submitted by other requests") as gave_up:
        await adopt_or_build(
            builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile", backend=backend
        )
    assert not isinstance(gave_up.value, BuildRefused)
    assert builder.submitted == [2, 3, 4]


@pytest.mark.parametrize(
    "code", [["backend_rejected"], {"is": "missing_secrets"}, 409], ids=["list", "mapping", "number"]
)
async def test_a_code_that_is_no_code_is_not_taken_for_a_refusal(code: object) -> None:
    """Whatever a proxy or a later Builder puts there: it is read, not looked up."""
    builder = _Builder(["ready"], conflicting=frozenset({1}))
    builder.code = code  # ty: ignore[invalid-assignment]
    submitted = await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")
    # A revision another request holds, as a 409 with no code is: the next one is submitted.
    assert (builder.submitted, submitted.images[0].name) == ([1, 2], "harbor-env-0123456789ab-2")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(409, text="<html>conflict</html>"),
        httpx.Response(409, json=["backend_rejected"]),
        httpx.Response(409, json="backend_rejected"),
    ],
    ids=["not-json", "a-list", "a-string"],
)
async def test_a_409_whose_body_is_no_object_is_not_taken_for_a_refusal(response: httpx.Response) -> None:
    """An ingress in front of Builder answers with a page of its own. It is no refusal of the request."""
    builder = _Builder(["ready"], conflicting=frozenset({1}))
    builder.response = response
    submitted = await ensure_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")
    assert (builder.submitted, submitted.images[0].name) == ([1, 2], "harbor-env-0123456789ab-2")


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
        name="harbor-env-0123456789ab", revision=2, source=_SOURCE, dockerfile="Dockerfile", platform=None
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


async def test_http_builder_submits_a_copy_without_a_dockerfile_for_the_platform_given() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=_submitted("harbor-img-0123456789ab", 1).model_dump(mode="json"))

    await _http_builder(handler).submit(
        name="harbor-img-0123456789ab",
        revision=1,
        source=ImageSource(image="ubuntu:24.04"),
        dockerfile=None,
        platform="linux/arm64",
    )

    spec = json.loads(requests[0].content)["build_specs"][0]
    assert spec["source"] == {"image": "ubuntu:24.04"}
    assert spec["platform"] == "linux/arm64"
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


def test_a_platform_is_its_own_builder_build_set() -> None:
    """Builder refuses one revision of a set for two requests, so each platform needs its own name."""
    built = {"context_digest": "a" * 64, "dockerfile": "Dockerfile"}

    assert builder_name(**built, platform="linux/arm64") != builder_name(**built)
    assert builder_name(**built, platform="linux/arm64") != builder_name(**built, platform="linux/amd64")
    assert copy_builder_name("docker.io/library/ubuntu:24.04", "linux/arm64") != copy_builder_name(
        "docker.io/library/ubuntu:24.04"
    )


class _RacingBuilder:
    """Builder's submit as it runs: the row is created atomically, then the job, with a pause in between.

    A request for an existing ``(name, revision)`` with another source is refused (409), as Builder refuses another
    request's revision; the same request rejoins it. A job exists once its submit has created it.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, ContainerImage]] = {}
        self.jobs: set[str] = set()
        self.builds: list[str] = []

    async def get_image(self, name: str) -> ContainerImage | None:
        row = self.rows.get(name)
        return row[1] if row is not None else None

    async def job_status(self, job: str) -> str | None:
        return "active" if job in self.jobs else None

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
        mounts: Sequence[BuildMount] = (),
    ) -> SubmitBuildResponse:
        image_name, request = f"{name}-{revision}", source.model_dump_json()
        if image_name in self.rows:
            if self.rows[image_name][0] != request:
                raise ConflictError(httpx.Response(409, json={"detail": "belongs to a different request"}))
        else:
            row = _container_image(image_name, "pending")
            row._created_at = datetime.now(timezone.utc)
            self.rows[image_name] = (request, row)
            self.builds.append(image_name)
            await asyncio.sleep(0.01)  # the job is created after the row: concurrent readers see no job yet
        self.jobs.add(f"builder-{image_name}")
        return SubmitBuildResponse(job=f"builder-{image_name}", images=[self.rows[image_name][1]])

    async def image_platforms(self, image: str) -> list[str]:
        return []


@pytest.mark.parametrize("backend", [None, IN_CLUSTER_BACKEND], ids=["none-named", "builders-named"])
async def test_a_hundred_tasks_sharing_one_context_submitted_at_once_make_one_build(backend: str | None) -> None:
    """Every task's archive is a different source, so all but one submit are refused; each uses the one build."""
    builder = _RacingBuilder()
    sources = [
        FileSetSource(fileset="default/harbor-tasks", archive=f"task-{i}/abc/task_archive", context_path="env")
        for i in range(100)
    ]

    images = await asyncio.gather(
        *(
            adopt_or_build(builder, name="harbor-env-0123456789ab", source=s, dockerfile="Dockerfile", backend=backend)
            for s in sources
        )
    )

    assert builder.builds == ["harbor-env-0123456789ab-1"]
    assert {image.name for image in images} == {"harbor-env-0123456789ab-1"}


@pytest.mark.parametrize(("age", "submitted"), [(timedelta(seconds=5), []), (timedelta(minutes=10), [2])])
async def test_a_pending_build_without_a_job_yet_is_used_until_its_job_is_overdue(
    age: timedelta, submitted: list[int]
) -> None:
    """Builder creates the job after the row: a young row without one is another submit in flight, not abandoned."""
    builder = _Builder(["pending"], job=None)
    created = datetime.now(timezone.utc) - age

    async def row(name: str) -> ContainerImage | None:
        if name.endswith("-1"):
            image = _container_image(name, "pending")
            image._created_at = created
            return image
        return None

    builder.get_image = row  # ty: ignore[invalid-assignment]
    await adopt_or_build(builder, name="harbor-env-0123456789ab", source=_SOURCE, dockerfile="Dockerfile")

    assert builder.submitted == submitted
