# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-boundary invariants on the Agent Hardener job specs, enforced before any Job is scheduled."""

from __future__ import annotations

import pytest
from nemo_helix_plugin.agent_hardener.types import SynthBenignSpec, WarGameSpec
from pydantic import ValidationError


@pytest.mark.parametrize("replay", [None, "", "   "])
def test_validate_only_without_a_replay_hitlog_is_rejected(replay: str | None) -> None:
    # Without recorded attacks to replay, a sanity check would fall through to a live attacker.
    with pytest.raises(ValidationError, match="replay_hitlog_fileset"):
        WarGameSpec(manifest_id="m1", validate_only=True, replay_hitlog_fileset=replay)


def test_validate_only_with_a_replay_hitlog_is_accepted() -> None:
    spec = WarGameSpec(manifest_id="m1", validate_only=True, replay_hitlog_fileset="default/hits")
    assert spec.replay_hitlog_fileset == "default/hits"


def test_a_normal_war_game_does_not_need_a_replay_hitlog() -> None:
    assert WarGameSpec(manifest_id="m1").replay_hitlog_fileset is None


@pytest.mark.parametrize("manifest_id", ["", "   ", "\t\n"])
def test_blank_synth_manifest_id_is_rejected(manifest_id: str) -> None:
    with pytest.raises(ValidationError, match="manifest_id"):
        SynthBenignSpec(manifest_id=manifest_id)


def test_synth_manifest_id_is_stripped() -> None:
    assert SynthBenignSpec(manifest_id="  clockbot  ").manifest_id == "clockbot"
