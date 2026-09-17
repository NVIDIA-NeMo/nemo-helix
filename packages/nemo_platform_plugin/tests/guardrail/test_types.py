# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.guardrail.types import ActivatedRail, Rails, RailsConfig


def test_guardrail_value_assignment_validates_nested_models() -> None:
    config = RailsConfig()

    config["rails"] = {"input": {"flows": ["self check input"]}}

    assert isinstance(config.rails, Rails)
    assert config.rails.input is not None
    assert config.rails.input.flows == ["self check input"]


def test_guardrail_mapping_access_preserves_explicit_none() -> None:
    rail = ActivatedRail(stop=None)

    assert rail["stop"] is None
    assert rail.get("stop") is None
    assert list(rail.keys()) == ["stop"]
    assert dict(rail.items()) == {"stop": None}
    assert rail == {"stop": None}
