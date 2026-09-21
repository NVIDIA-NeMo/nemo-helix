# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.guardrail.types import ActivatedRail, GuardrailConfig, Rails, RailsConfig


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


def test_guardrail_config_accepts_null_data() -> None:
    # The guardrails entity stores ``data`` as Optional; configs created without a
    # body come back from the API as ``"data": null``.
    payload = {
        "id": "guardrail-1",
        "name": "empty",
        "namespace": "default",
        "workspace": "default",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "data": None,
    }

    assert GuardrailConfig.model_validate(payload).data is None
    assert GuardrailConfig.model_validate({**payload, "data": {}}).data == RailsConfig()
