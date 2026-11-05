# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-built agent images: usage, cleanup, and the agent-images routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import images as images_api
from nemo_agents_plugin.entities import AgentDeployment, DeploymentStatus
from nemo_agents_plugin.images import (
    build_labels,
    list_agent_images,
    remove_unused_agent_images,
    remove_unused_image,
)
from nemo_deployments_plugin.backends.base import ImageInfo, ImageInUseError, ImagesUnsupportedError

_TAG = "nemo-agents/default/calc-abc:0.1"
_PUSHED = "nvcr.io/my-org/nemo-agents/default/calc-abc:0.1"


def _info(
    image_id: str = "sha256:a",
    tags: tuple[str, ...] = (_TAG,),
    labels: dict[str, str] | None = None,
    containers: int = 0,
    created: str = "2026-11-01",
) -> ImageInfo:
    return ImageInfo(
        id=image_id,
        tags=tags,
        labels={"com.nemo.agent.workspace": "default", "com.nemo.agent.name": "calc"} if labels is None else labels,
        size=10,
        created=created,
        containers=containers,
    )


def _dep(
    name: str = "calc-1",
    image: str = _TAG,
    workspace: str = "default",
    status: DeploymentStatus = "running",
    remove_image_on_delete: bool = False,
) -> AgentDeployment:
    return AgentDeployment(
        name=name,
        workspace=workspace,
        agent="calc",
        image=image,
        status=status,
        remove_image_on_delete=remove_image_on_delete,
    )


def _entities(deployments: list[AgentDeployment]) -> AsyncMock:
    entities = AsyncMock()
    entities.list = AsyncMock(return_value=MagicMock(data=deployments, pagination=None))
    return entities


def _store(images: list[ImageInfo] | None = None, tags: tuple[str, ...] = (_TAG,)) -> AsyncMock:
    store = AsyncMock()
    store.list_images = AsyncMock(return_value=images or [])
    store.remove_image = AsyncMock(return_value=True)
    store.image_tags = AsyncMock(return_value=list(tags))
    return store


class TestListAgentImages:
    async def test_marks_usage_and_attributes_agents(self) -> None:
        store = _store(
            [
                _info("sha256:a"),
                _info(
                    "sha256:b", tags=("nemo-agents/default/old:1",), labels={"org.opencontainers.image.title": "old"}
                ),
                _info("sha256:c", tags=("nemo-agents/default/busy:1",), labels={}, containers=1),
            ]
        )
        entities = _entities([_dep(), _dep(name="elsewhere", workspace="other")])

        images = {i.id: i for i in await list_agent_images(store, entities, workspace="default")}

        store.list_images.assert_awaited_once_with(repository_prefix="nemo-agents/default/")
        assert images["sha256:a"].agent == "calc"
        assert images["sha256:a"].in_use is True
        # Usage counts across workspaces, but only this workspace's deployment names are shown.
        assert images["sha256:a"].deployments == ["calc-1"]
        assert images["sha256:b"].agent == "old"
        assert images["sha256:b"].in_use is False
        assert images["sha256:c"].agent is None
        assert images["sha256:c"].in_use is True

    async def test_reports_the_spec_revision_an_image_was_built_from(self) -> None:
        labels = build_labels(workspace="default", agent="calc", spec_revision="abc123")
        store = _store([_info("sha256:a", labels=labels), _info("sha256:b", tags=("nemo-agents/default/b:1",))])

        images = {i.id: i for i in await list_agent_images(store, _entities([]), workspace="default")}

        assert images["sha256:a"].spec_revision == "abc123"
        assert images["sha256:b"].spec_revision == ""

    async def test_a_deployment_deleting_with_its_image_does_not_hold_it(self) -> None:
        entities = _entities([_dep(status="deleting", remove_image_on_delete=True)])

        [image] = await list_agent_images(_store([_info()]), entities, workspace="default")

        assert image.in_use is False

    async def test_a_deployment_deleting_without_its_image_still_holds_it(self) -> None:
        entities = _entities([_dep(status="deleting", remove_image_on_delete=False)])

        [image] = await list_agent_images(_store([_info()]), entities, workspace="default")

        assert image.in_use is True

    async def test_counts_push_tags_but_not_tags_the_platform_did_not_add(self) -> None:
        store = _store([_info(tags=("mine:latest", _TAG, _PUSHED))])

        [image] = await list_agent_images(store, _entities([_dep(image=_PUSHED)]), workspace="default")

        assert image.tags == [_TAG, _PUSHED]
        assert image.in_use is True


class TestRemoveUnusedImage:
    async def test_removes_an_unused_platform_image(self) -> None:
        store = _store()

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is True
        store.remove_image.assert_awaited_once_with(_TAG)

    @pytest.mark.parametrize(
        "reference",
        ["alpine:latest", "nvcr.io/nim/meta/llama:1", "my-agent-abc:0.1", "nemo-agents/other/calc-abc:0.1"],
    )
    async def test_never_touches_images_the_platform_did_not_build(self, reference: str) -> None:
        store = _store()

        assert await remove_unused_image(store, _entities([]), reference, "default") is False
        store.remove_image.assert_not_awaited()

    async def test_keeps_an_image_another_deployment_uses(self) -> None:
        store = _store()

        assert await remove_unused_image(store, _entities([_dep(workspace="other")]), _TAG, "default") is False
        store.remove_image.assert_not_awaited()

    async def test_a_deployment_deleting_with_its_image_does_not_hold_it(self) -> None:
        store = _store()
        entities = _entities([_dep(status="deleting", remove_image_on_delete=True)])

        assert await remove_unused_image(store, entities, _TAG, "default") is True

    async def test_a_deployment_deleting_without_its_image_still_holds_it(self) -> None:
        store = _store()
        entities = _entities([_dep(status="deleting", remove_image_on_delete=False)])

        assert await remove_unused_image(store, entities, _TAG, "default") is False
        store.remove_image.assert_not_awaited()

    async def test_keeps_an_image_a_container_uses(self) -> None:
        store = _store()
        store.remove_image.side_effect = ImageInUseError("conflict")

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is False

    async def test_failures_are_swallowed(self) -> None:
        store = _store()
        store.remove_image.side_effect = RuntimeError("daemon gone")

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is False

    async def test_a_gone_image_is_not_removed(self) -> None:
        store = _store()
        store.image_tags.return_value = None

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is False
        store.remove_image.assert_not_awaited()


class TestPushedImages:
    @pytest.mark.parametrize("reference", [_TAG, _PUSHED])
    async def test_removes_the_build_and_push_tags_together(self, reference: str) -> None:
        store = _store(tags=(_TAG, _PUSHED))

        assert await remove_unused_image(store, _entities([]), reference, "default") is True
        assert [c.args[0] for c in store.remove_image.await_args_list] == [_TAG, _PUSHED]

    async def test_keeps_the_image_when_a_deployment_uses_its_push_tag(self) -> None:
        store = _store(tags=(_TAG, _PUSHED))

        assert (
            await remove_unused_image(store, _entities([_dep(name="other", image=_PUSHED)]), _TAG, "default") is False
        )
        store.remove_image.assert_not_awaited()

    async def test_leaves_tags_the_platform_did_not_add(self) -> None:
        store = _store(tags=(_TAG, "mine:latest"))

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is True
        store.remove_image.assert_awaited_once_with(_TAG)

    async def test_needs_the_local_build_tag_to_prove_ownership(self) -> None:
        store = _store(tags=(_PUSHED,))

        assert await remove_unused_image(store, _entities([]), _PUSHED, "default") is False
        store.remove_image.assert_not_awaited()

    async def test_ignores_another_workspaces_push_tag(self) -> None:
        store = _store()

        assert await remove_unused_image(store, _entities([]), "nvcr.io/org/nemo-agents/other/a:1", "default") is False
        store.image_tags.assert_not_awaited()


async def test_remove_unused_agent_images_only_removes_that_agents_unused_images() -> None:
    store = _store(
        [
            _info("sha256:a", tags=(_TAG, "nemo-agents/default/calc-abc:latest")),
            _info("sha256:b", tags=("nemo-agents/default/calc-def:0.2",), containers=1),
            _info(
                "sha256:c",
                tags=("nemo-agents/default/other:1",),
                labels={"com.nemo.agent.workspace": "default", "com.nemo.agent.name": "other"},
            ),
        ]
    )

    removed = await remove_unused_agent_images(store, _entities([]), workspace="default", agent="calc")

    assert removed == 1
    assert [c.args[0] for c in store.remove_image.await_args_list] == [_TAG, "nemo-agents/default/calc-abc:latest"]


class TestImagesApi:
    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(images_api.router, prefix="/apis/agents/v2/workspaces/{workspace}")
        return TestClient(app, raise_server_exceptions=False)

    def _access(self, store: AsyncMock, deployments: list[AgentDeployment] | None = None):  # noqa: ANN202
        return patch.object(images_api, "image_access", AsyncMock(return_value=(store, _entities(deployments or []))))

    def test_list_is_empty_without_a_docker_executor(self, client: TestClient) -> None:
        with patch.object(images_api, "image_access", AsyncMock(return_value=None)):
            resp = client.get("/apis/agents/v2/workspaces/default/agent-images")

        assert resp.status_code == 200
        assert resp.json() == {"data": []}

    def test_list_is_empty_when_the_backend_has_no_image_store(self, client: TestClient) -> None:
        store = _store()
        store.list_images.side_effect = ImagesUnsupportedError("k8s")
        with self._access(store):
            resp = client.get("/apis/agents/v2/workspaces/default/agent-images")

        assert resp.json() == {"data": []}

    def test_list_filters_by_agent_newest_first(self, client: TestClient) -> None:
        store = _store(
            [
                _info("sha256:old", created="2026-10-01"),
                _info("sha256:new", created="2026-11-01"),
                _info("sha256:x", labels={"org.opencontainers.image.title": "other"}),
            ]
        )
        with self._access(store):
            resp = client.get("/apis/agents/v2/workspaces/default/agent-images?agent=calc")

        assert resp.status_code == 200
        assert [i["id"] for i in resp.json()["data"]] == ["sha256:new", "sha256:old"]

    def test_delete_removes_each_workspace_tag(self, client: TestClient) -> None:
        store = _store([_info(tags=(_TAG, "nemo-agents/default/calc-abc:latest"))])
        with self._access(store):
            resp = client.delete("/apis/agents/v2/workspaces/default/agent-images/sha256:a")

        assert resp.status_code == 204
        assert store.remove_image.await_count == 2

    def test_delete_unknown_image_is_404(self, client: TestClient) -> None:
        with self._access(_store()):
            resp = client.delete("/apis/agents/v2/workspaces/default/agent-images/sha256:nope")

        assert resp.status_code == 404

    def test_delete_without_a_docker_executor_is_404(self, client: TestClient) -> None:
        with patch.object(images_api, "image_access", AsyncMock(return_value=None)):
            resp = client.delete("/apis/agents/v2/workspaces/default/agent-images/sha256:a")

        assert resp.status_code == 404

    def test_delete_refuses_an_image_a_deployment_uses(self, client: TestClient) -> None:
        store = _store([_info()])
        with self._access(store, [_dep()]):
            resp = client.delete("/apis/agents/v2/workspaces/default/agent-images/sha256:a")

        assert resp.status_code == 409
        store.remove_image.assert_not_awaited()

    def test_delete_reports_a_container_race_as_conflict(self, client: TestClient) -> None:
        store = _store([_info()])
        store.remove_image.side_effect = ImageInUseError("conflict")
        with self._access(store):
            resp = client.delete("/apis/agents/v2/workspaces/default/agent-images/sha256:a")

        assert resp.status_code == 409
