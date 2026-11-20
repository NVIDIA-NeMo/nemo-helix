# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An agent layer's record, whose image is recorded as a task build's images are."""

from __future__ import annotations

from nemo_evals.agent_layers.models import AgentLayer, AgentLayerEntity, AgentLayerRequest
from nemo_evals.image_builds.models import BuildImage

# What a build image says of a remote build service's build. A layer has neither: only this cluster builds one.
_REMOTE_BUILD_KEYS = {"external_image_id", "builder_backend"}


def _layer() -> AgentLayerEntity:
    """A ready layer, built in this cluster."""
    return AgentLayerEntity(
        name="agent-layer.abc.def.1",
        workspace="default",
        task="default/sort-task",
        revision_digest="a" * 64,
        agent_key="k",
        agent=AgentLayerRequest(name="codex"),
        attempt=1,
        status="ready",
        job="agent-layer-build-abc-1",
        environment_digest="sha256:" + "e" * 64,
        image=BuildImage(
            type="main",
            source="dockerfile",
            status="ready",
            image_ref="registry.test/default/harbor-layer-abc-1@sha256:" + "d" * 64,
            builder_image="harbor-layer-abc-1",
            build_job="builder-harbor-layer-abc-1",
        ),
    )


def test_a_layer_is_recorded_and_returned_without_the_keys_of_a_remote_build() -> None:
    """A layer's record shares the build image's shape, and an older reader refuses a key it doesn't know."""
    layer = _layer()

    recorded = layer._get_data_fields()["image"]
    returned = AgentLayer.from_entity(layer).model_dump(mode="json")["image"]

    assert not _REMOTE_BUILD_KEYS & set(recorded) and not _REMOTE_BUILD_KEYS & set(returned)
    assert not _REMOTE_BUILD_KEYS & set(layer.model_dump()["image"])
