# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MVP preparer: ready builds or pinned prebuilt images in one call, never pending."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_evals.harbor.remote import preparation
from nemo_evals.harbor.remote.preparation import PreparationClients, prepare_task_images
from nemo_evals.image_builds.planning import TaskArchive
from nemo_helix_plugin.client.client import AsyncNemoClient

SHA = "sha256:" + "c" * 64
SEPARATE = {"environment": {}, "verifier": {"environment_mode": "separate"}}
PREBUILT = {"environment": {"docker_image": "public.ecr.aws/team/swe:abc"}}


def archive(*files: str, compose: str | None = None) -> TaskArchive:
    return TaskArchive(root="t", files=frozenset(f"t/{f}" for f in files), compose=compose)


def build(status: str, *roles: str) -> dict:
    images = [
        {
            "role": role,
            "source": "dockerfile",
            "status": "ready",
            "builder_image": f"harbor-env-{role}-1",
            "build_job": f"builder-{role}",
            "image_ref": f"registry.invalid/{role}@sha256:{(role[0] * 64)[:64].replace('m', 'd').replace('v', 'e')}",
        }
        for role in roles
    ]
    return {
        "id": "1",
        "task": "default/x",
        "revision": "r",
        "status": status,
        "job": "j",
        "allowed_registries": [],
        "images": images,
    }


@pytest.fixture
def builds() -> list[dict]:
    return []


@pytest.fixture
async def clients(builds):
    def api(request: httpx.Request) -> httpx.Response:
        if request.url.host == "platform.test" and request.url.path.endswith("/image-builds"):
            return httpx.Response(
                200,
                json={
                    "data": builds,
                    "pagination": {
                        "page": 1,
                        "page_size": 100,
                        "current_page_size": len(builds),
                        "total_pages": 1,
                        "total_results": len(builds),
                    },
                },
            )
        if request.url.host == "public.ecr.aws" and request.url.path.startswith("/v2/"):
            if request.headers.get("authorization") != "Bearer anon":
                challenge = 'Bearer realm="https://public.ecr.aws/token/",service="public.ecr.aws"'
                return httpx.Response(401, headers={"www-authenticate": challenge})
            if request.url.path.endswith("/manifests/abc"):
                return httpx.Response(200, headers={"docker-content-digest": SHA})
            return httpx.Response(404)
        if request.url.path == "/token/":
            return httpx.Response(200, json={"token": "anon"})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    transport = httpx.MockTransport(api)
    async with httpx.AsyncClient(transport=transport) as http, httpx.AsyncClient(transport=transport) as registry:
        async with AsyncNemoClient(base_url="http://platform.test", http_client=http) as client:
            yield PreparationClients(client=client, files=MagicMock(), registry=registry)


@pytest.fixture
def task_archive(monkeypatch):
    holder = {"archive": archive("environment/Dockerfile", "tests/Dockerfile")}

    async def read(files, source):
        return holder["archive"]

    monkeypatch.setattr(preparation, "read_task_archive", read)
    return holder


@pytest.fixture(autouse=True)
def allowed_registries(monkeypatch, harbor_config):
    config = harbor_config.model_copy(update={"allowed_registries": ["public.ecr.aws"]})
    monkeypatch.setattr(preparation, "get_config", lambda: SimpleNamespace(runs=SimpleNamespace(harbor=config)))


async def prepare(task, clients, policy):
    return await prepare_task_images(
        task,
        None,
        policy=policy,
        missing_prebuilt_image="fail",
        caller="user:alice",
        clients=clients,
        deadline=datetime.now(timezone.utc),
    )


async def test_ready_build_gives_agent_and_verifier(
    clients, builds, task_archive, sandbox_policy, harbor_resolved_task
):
    builds.extend([build("failed"), build("ready", "main", "verifier")])
    result = await prepare(harbor_resolved_task(config=SEPARATE), clients, sandbox_policy)
    assert result.status == "ready" and result.environment is not None
    assert result.environment.agent.image_ref.startswith("registry.invalid/main@")
    assert (
        result.environment.verifier is not None and result.environment.verifier.builder_image == "harbor-env-verifier-1"
    )
    assert result.images["agent"].source == "built"
    assert result.environment.agent.preparation_key.startswith("harbor-env-")


async def test_no_ready_build_is_image_unavailable(clients, builds, task_archive, sandbox_policy, harbor_resolved_task):
    builds.append(build("running"))
    result = await prepare(harbor_resolved_task(config=SEPARATE), clients, sandbox_policy)
    assert result.status == "failed" and result.error is not None
    assert result.error.code == "image_unavailable" and "no ready image build" in result.error.detail


async def test_prebuilt_tag_is_pinned_anonymously(clients, task_archive, sandbox_policy, harbor_resolved_task):
    task_archive["archive"] = archive()
    result = await prepare(harbor_resolved_task(config=PREBUILT), clients, sandbox_policy)
    assert result.status == "ready" and result.environment is not None
    assert result.environment.agent.image_ref == f"public.ecr.aws/team/swe@{SHA}"
    assert result.environment.verifier is None and result.images["agent"].source == "prebuilt"


async def test_prebuilt_tag_that_does_not_resolve_fails(clients, task_archive, sandbox_policy, harbor_resolved_task):
    task_archive["archive"] = archive()
    config = {"environment": {"docker_image": "public.ecr.aws/team/swe:gone"}}
    result = await prepare(harbor_resolved_task(config=config), clients, sandbox_policy)
    assert result.error is not None and result.error.code == "image_unavailable" and result.error.role == "agent"


@pytest.mark.parametrize(
    ("config", "files", "compose", "code"),
    [
        (
            {  # pinned agent, verifier built from tests/Dockerfile (deep-swe-1-1's shape)
                "environment": {"docker_image": "public.ecr.aws/team/swe:abc"},
                "verifier": {"environment_mode": "separate", "environment": {"cpus": 1}},
            },
            ("tests/Dockerfile",),
            None,
            "image_unavailable",
        ),
        ({"environment": {}}, ("environment/Dockerfile",), "services: {main: {}}", "multi_container_not_yet_supported"),
        ({"environment": {"gpus": 1}}, ("environment/Dockerfile",), None, "gpu_unsupported"),
        ({"environment": {"docker_image": "docker.io/library/ubuntu:24.04"}}, (), None, "registry_not_allowed"),
        ({"environment": {}}, (), None, "main_dockerfile_missing"),
    ],
)
async def test_unsupported_tasks_fail_with_stable_codes(
    clients, task_archive, sandbox_policy, config, files, compose, code, harbor_resolved_task
):
    task_archive["archive"] = archive(*files, compose=compose)
    result = await prepare(harbor_resolved_task(config=config), clients, sandbox_policy)
    assert result.status == "failed" and result.error is not None and result.error.code == code


async def test_empty_allowlist_allows_no_prebuilt_registry(
    monkeypatch, clients, task_archive, sandbox_policy, harbor_config, harbor_resolved_task
):
    """Unlike Builder's narrowing list, an empty evals.runs.harbor.allowed_registries denies every registry."""
    config = harbor_config.model_copy(update={"allowed_registries": []})
    monkeypatch.setattr(preparation, "get_config", lambda: SimpleNamespace(runs=SimpleNamespace(harbor=config)))
    result = await prepare(harbor_resolved_task(config=PREBUILT), clients, sandbox_policy)
    assert result.status == "failed" and result.error is not None
    assert (result.error.code, result.error.role) == ("registry_not_allowed", "agent")


async def test_settled_preparation_is_returned_unchanged(clients, sandbox_policy, harbor_resolved_task):
    from nemo_evals.harbor.remote.preparation import PreparationErrorInfo, TaskPreparation

    failed = TaskPreparation(status="failed", error=PreparationErrorInfo(code="build_failed", detail="x"))
    result = await prepare_task_images(
        harbor_resolved_task(),
        failed,
        policy=sandbox_policy,
        missing_prebuilt_image="fail",
        caller="user:alice",
        clients=clients,
        deadline=datetime.now(timezone.utc),
    )
    assert result is failed
