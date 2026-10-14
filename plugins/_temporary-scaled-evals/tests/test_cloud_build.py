# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.build import cloud_build
from scaled_evals.api.settings import settings


class FakeClient:
    posts: list[tuple[str, dict]] = []
    gets: list[str] = []
    responses: list[httpx.Response] = []

    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, *, headers: dict, json: dict) -> httpx.Response:
        self.posts.append((url, json))
        return self.responses.pop(0)

    def get(self, url: str, *, headers: dict) -> httpx.Response:
        self.gets.append(url)
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def reset_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeClient.posts = []
    FakeClient.gets = []
    FakeClient.responses = []
    cloud_build._TOKEN_CACHE = None
    monkeypatch.setattr(cloud_build.httpx, "Client", FakeClient)
    monkeypatch.setattr(settings, "object_store_backend", "gcs")
    monkeypatch.setattr(settings, "resolved_object_store_bucket", lambda: "task-packs")
    monkeypatch.setattr(settings, "gcs_access_token", "token")
    monkeypatch.setattr(settings, "cloud_build_enabled", True)
    monkeypatch.setattr(settings, "cloud_build_project", "project-1")
    monkeypatch.setattr(settings, "cloud_build_location", "us-central1")
    monkeypatch.setattr(settings, "cloud_build_timeout_seconds", 30.0)
    monkeypatch.setattr(settings, "cloud_build_poll_interval_seconds", 0.0)
    monkeypatch.setattr(
        settings,
        "image_registry",
        "us-central1-docker.pkg.dev/project-1/repo/tasks",
    )
    monkeypatch.setattr(settings, "image_build_platform", "linux/amd64")


def test_cloud_build_uses_gcs_source_and_returns_gar_digest() -> None:
    image_ref = "us-central1-docker.pkg.dev/project-1/repo/tasks/task_1:rev2"
    FakeClient.responses = [
        httpx.Response(
            200,
            json={
                "id": "build-1",
                "status": "SUCCESS",
                "results": {"images": [{"name": image_ref, "digest": "sha256:" + "a" * 64}]},
            },
        )
    ]

    assert cloud_build.build_revision_image("task_1", 2, "task_1/rev/2/tarball.tar.gz") == (
        image_ref,
        "sha256:" + "a" * 64,
    )

    url, payload = FakeClient.posts[0]
    assert url == "https://cloudbuild.googleapis.com/v1/projects/project-1/locations/us-central1/builds"
    assert payload["source"]["storageSource"] == {
        "bucket": "task-packs",
        "object": "task_1/rev/2/tarball.tar.gz",
    }
    assert payload["steps"][0]["args"] == [
        "build",
        "-t",
        image_ref,
        "--platform",
        "linux/amd64",
        ".",
    ]
    assert "env" not in payload["steps"][0]
    assert payload["steps"][1]["args"] == ["push", image_ref]
    assert payload["images"] == [image_ref]


def test_cloud_build_accepts_operation_response(monkeypatch: pytest.MonkeyPatch) -> None:
    image_ref = "us-central1-docker.pkg.dev/project-1/repo/tasks/task_1:rev2"
    monkeypatch.setattr(cloud_build.time, "sleep", Mock())
    FakeClient.responses = [
        httpx.Response(200, json={"name": "operations/build-op"}),
        httpx.Response(
            200,
            json={
                "done": True,
                "response": {
                    "id": "build-1",
                    "status": "SUCCESS",
                    "results": {"images": [{"name": image_ref, "digest": "sha256:" + "b" * 64}]},
                },
            },
        ),
    ]

    assert cloud_build.build_revision_image("task_1", 2, "task_1/rev/2/tarball.tar.gz") == (
        image_ref,
        "sha256:" + "b" * 64,
    )
    assert FakeClient.gets == ["https://cloudbuild.googleapis.com/v1/operations/build-op"]


def test_submit_cloud_build_returns_without_polling_and_keeps_metadata() -> None:
    image_ref = "us-central1-docker.pkg.dev/project-1/repo/server:git-abc"
    FakeClient.responses = [
        httpx.Response(
            200,
            json={
                "name": "operations/build-async",
                "metadata": {"build": {"id": "build-async", "status": "QUEUED"}},
            },
        )
    ]

    build = cloud_build.submit_image_build_from_gcs(
        "server/context.tar.gz",
        image_ref,
        substitutions={"_SCALED_EVALS_PURPOSE": "publish"},
    )

    assert build == {"id": "build-async", "status": "QUEUED"}
    assert FakeClient.gets == []
    payload = FakeClient.posts[0][1]
    assert payload["substitutions"] == {"_SCALED_EVALS_PURPOSE": "publish"}
    assert payload["options"]["substitutionOption"] == "ALLOW_LOOSE"


def test_get_cloud_build_reads_configured_location() -> None:
    build_id = "123e4567-e89b-12d3-a456-426614174000"
    FakeClient.responses = [httpx.Response(200, json={"id": build_id, "status": "WORKING"})]

    assert cloud_build.get_build(build_id)["status"] == "WORKING"
    assert FakeClient.gets == [
        f"https://cloudbuild.googleapis.com/v1/projects/project-1/locations/us-central1/builds/{build_id}"
    ]


def test_cloud_build_requires_gcs_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "object_store_backend", "s3")

    with pytest.raises(cloud_build.BuildError, match="OBJECT_STORE_BACKEND=gcs"):
        cloud_build.build_revision_image("task_1", 2, "task_1/rev/2/tarball.tar.gz")
