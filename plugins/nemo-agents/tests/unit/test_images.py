# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Removing a platform-built agent image once nothing uses it."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from nemo_agents_plugin.entities import AgentDeployment, DeploymentStatus
from nemo_agents_plugin.images import remove_unused_image
from nemo_deployments_plugin.backends.base import ImageInUseError

_TAG = "nemo-agents/default/calc-abc:0.1"
_PUSHED = "nvcr.io/my-org/nemo-agents/default/calc-abc:0.1"


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


def _store(*tags: str) -> AsyncMock:
    store = AsyncMock()
    store.remove_image = AsyncMock(return_value=True)
    store.image_tags = AsyncMock(return_value=list(tags or [_TAG]))
    return store


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
        store = _store(_TAG, _PUSHED)

        assert await remove_unused_image(store, _entities([]), reference, "default") is True
        assert [c.args[0] for c in store.remove_image.await_args_list] == [_TAG, _PUSHED]

    async def test_keeps_the_image_when_a_deployment_uses_its_push_tag(self) -> None:
        store = _store(_TAG, _PUSHED)

        assert (
            await remove_unused_image(store, _entities([_dep(name="other", image=_PUSHED)]), _TAG, "default") is False
        )
        store.remove_image.assert_not_awaited()

    async def test_leaves_tags_the_platform_did_not_add(self) -> None:
        store = _store(_TAG, "mine:latest")

        assert await remove_unused_image(store, _entities([]), _TAG, "default") is True
        store.remove_image.assert_awaited_once_with(_TAG)

    async def test_needs_the_local_build_tag_to_prove_ownership(self) -> None:
        store = _store(_PUSHED)

        assert await remove_unused_image(store, _entities([]), _PUSHED, "default") is False
        store.remove_image.assert_not_awaited()

    async def test_ignores_another_workspaces_push_tag(self) -> None:
        store = _store()

        assert await remove_unused_image(store, _entities([]), "nvcr.io/org/nemo-agents/other/a:1", "default") is False
        store.image_tags.assert_not_awaited()
