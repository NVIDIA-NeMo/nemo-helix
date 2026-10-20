# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agent_hardener_plugin.api.v2.jobs import _collection_filter
from nemo_helix_plugin.agent_hardener.types import SynthBenignSpec, WarGameSpec
from nemo_helix_plugin.in_memory_filter import InMemoryFilterRepository


@pytest.mark.parametrize(
    "spec,expected",
    [
        ({"manifest_id": "saved", "stop_after_synth": False}, "war"),
        ({"config": "/tmp/target.yaml", "stop_after_synth": True}, "war"),
        ({"manifest_id": "saved", "interview": "interactive"}, "synth"),
        ({"manifest_id": "saved", "interview": ""}, "synth"),
        ({"manifest_id": "saved"}, None),
        ({"manifest_id": "saved", "stop_after_synth": False, "interview": "interactive"}, None),
    ],
)
def test_historical_specs_partition_without_markers(spec: dict[str, Any], expected: str | None) -> None:
    repo = InMemoryFilterRepository(SimpleNamespace(spec=spec))
    assert _collection_filter("stop_after_synth", "interview").apply(repo) == (expected == "war")
    assert _collection_filter("interview", "stop_after_synth").apply(repo) == (expected == "synth")


def test_new_specs_keep_canonical_membership_despite_sibling_input() -> None:
    war = WarGameSpec.model_validate({"manifest_id": "saved", "interview": "interactive"}).model_dump(mode="json")
    synth = SynthBenignSpec.model_validate({"manifest_id": "saved", "stop_after_synth": False}).model_dump(mode="json")
    assert _collection_filter("stop_after_synth", "interview").apply(InMemoryFilterRepository({"spec": war}))
    assert _collection_filter("interview", "stop_after_synth").apply(InMemoryFilterRepository({"spec": synth}))
