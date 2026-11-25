# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.models.metadata import (
    MAX_DESCRIPTION_LENGTH,
    MAX_DISPLAY_NAME_LENGTH,
    MetadataSourceContext,
    MetadataTarget,
    NemoModelMetadataSource,
    SourceUnavailableError,
)
from nemo_helix_plugin.models.types import ModelMetadata, ModelProvider


class CountingSource(NemoModelMetadataSource):
    name = "counting"

    def handles(self, provider: ModelProvider) -> bool:
        return True

    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        return None


async def test_cached_for_cycle_loads_once_per_key_until_next_cycle():
    source = CountingSource()
    calls: list[str] = []

    async def load() -> str:
        calls.append("load")
        return "listing"

    assert await source.cached_for_cycle("provider-a", load) == "listing"
    assert await source.cached_for_cycle("provider-a", load) == "listing"
    source.begin_cycle()
    assert await source.cached_for_cycle("provider-a", load) == "listing"

    assert len(calls) == 2


async def test_cached_for_cycle_remembers_unavailability_for_the_cycle():
    source = CountingSource()
    calls: list[str] = []

    async def load() -> str:
        calls.append("load")
        raise SourceUnavailableError("upstream down")

    for _ in range(2):
        with pytest.raises(SourceUnavailableError):
            await source.cached_for_cycle("provider-a", load)

    assert len(calls) == 1


async def test_cached_for_cycle_does_not_cache_unexpected_errors():
    source = CountingSource()
    calls: list[str] = []

    async def load() -> str:
        calls.append("load")
        raise RuntimeError("bug")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await source.cached_for_cycle("provider-a", load)

    assert len(calls) == 2


def test_length_limits_follow_model_metadata_schema():
    assert (
        MAX_DISPLAY_NAME_LENGTH
        == ModelMetadata.model_json_schema()["properties"]["display_name"]["anyOf"][0]["maxLength"]
    )
    assert (
        MAX_DESCRIPTION_LENGTH
        == ModelMetadata.model_json_schema()["properties"]["description"]["anyOf"][0]["maxLength"]
    )
