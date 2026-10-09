# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Any

import pytest
from nemo_helix_plugin.errors import LocalRunError
from nemo_switchyard._native_config import (
    map_random_routing_config,
    validate_llm_classifier_config,
    validate_stage_router_config,
)
from nemo_switchyard.routing import IGW_API_KEY_ENV, build_combinations, rewrite_agent_config
from nemo_switchyard.schemas.optimize import SwitchyardOptimizeSpec
from pydantic import ValidationError

ALL_STRATEGIES = ["random_routing", "stage_router", "llm_classifier"]


def spec(**overrides: Any) -> SwitchyardOptimizeSpec:
    return SwitchyardOptimizeSpec.model_validate({"agent": "calc", "models": ["a", "team/b"], **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"models": ["a"]},
        {"models": ["a", "a"]},
        {"models": ["a", "https://x/b"]},
        {"routing_strategies": []},
        {"routing_strategies": ["random_routing", "random_routing"]},
        {"routing_strategies": ["weighted"]},
        {"strong_probability": 1.5},
        {"confidence_threshold": -0.1},
        {"base_threshold": 2},
        {"output": "default/results"},
        {"optimize_config": "x.yaml"},
    ],
)
def test_spec_rejects(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        spec(**overrides)


def test_combinations_cover_every_pair_and_strategy() -> None:
    combos = build_combinations(
        spec(models=["a", "b", "c"], routing_strategies=["random_routing", "stage_router"]), agent_name="calc"
    )

    assert [(c.capable, c.efficient, c.config_type) for c in combos] == [
        ("default/a", "default/b", "random_routing"),
        ("default/a", "default/b", "stage_router"),
        ("default/a", "default/c", "random_routing"),
        ("default/a", "default/c", "stage_router"),
        ("default/b", "default/c", "random_routing"),
        ("default/b", "default/c", "stage_router"),
    ]
    assert [c.virtual_model.rsplit("-", 1)[0] for c in combos[:3]] == [
        "calc-random-routing",
        "calc-stage-router",
        "calc-random-routing",
    ]
    assert len({c.virtual_model for c in combos}) == len(combos)


def test_virtual_model_names_are_stable_per_routing_and_differ_when_it_changes() -> None:
    (first,) = build_combinations(spec(models=["a", "b"]), agent_name="calc")
    (again,) = build_combinations(spec(models=["a", "b"]), agent_name="calc")
    (other_pair,) = build_combinations(spec(models=["a", "c"]), agent_name="calc")
    (other_setting,) = build_combinations(spec(models=["a", "b"], strong_probability=0.2), agent_name="calc")

    assert first.virtual_model == again.virtual_model
    assert len({first.virtual_model, other_pair.virtual_model, other_setting.virtual_model}) == 3


def test_combinations_refuse_a_virtual_model_name_over_the_entity_limit() -> None:
    with pytest.raises(LocalRunError, match="exceeds 63 characters"):
        build_combinations(spec(), agent_name="a" * 40)


def test_combination_configs_satisfy_the_middleware_validators() -> None:
    random, stage, classifier = build_combinations(
        spec(routing_strategies=ALL_STRATEGIES, strong_probability=0.2, confidence_threshold=0.7, base_threshold=0.9),
        agent_name="calc",
    )

    assert map_random_routing_config(random.config) == ([0.2, 0.8], None, {"any": ["default/a", "team/b"]})
    assert validate_stage_router_config(stage.config)["confidence_threshold"] == 0.7
    assert validate_llm_classifier_config(classifier.config)["base_threshold"] == 0.9
    assert classifier.judge == "default/a"
    assert classifier.models == ["default/a", "team/b"]
    assert random.judge is None


def test_explicit_judge_is_included_in_the_virtual_model_models() -> None:
    (combo,) = build_combinations(spec(routing_strategies=["llm_classifier"], judge_model="judge"), agent_name="calc")

    assert combo.config["models"]["judge"] == ["default/judge"]
    assert combo.models == ["default/judge", "default/a", "team/b"]


def test_rewrite_points_every_model_block_at_the_virtual_model() -> None:
    source = {
        "config_format": "nemo-agents-spec-v1",
        "models": {"default": {"provider": "anthropic", "model": "m", "base_url": "https://x", "api_key_env": "K"}},
        "harnesses": {
            "deepagents": {
                "kind": "deepagents",
                "model": {"provider": "nvidia", "model": "m", "settings": {"base_url": "https://y", "keep": 1}},
            },
            "codex": {"kind": "codex"},
        },
    }
    before = copy.deepcopy(source)

    rewritten = rewrite_agent_config(source, "default/calc-random-routing-1")

    assert source == before
    assert rewritten["models"]["default"] == {
        "provider": "openai",
        "model": "default/calc-random-routing-1",
        "api_key_env": IGW_API_KEY_ENV,
    }
    assert rewritten["harnesses"]["deepagents"]["model"] == {
        "provider": "nvidia",
        "model": "default/calc-random-routing-1",
        "settings": {"keep": 1},
        "api_key_env": IGW_API_KEY_ENV,
    }
    assert rewritten["harnesses"]["codex"] == {"kind": "codex"}


def test_rewrite_refuses_a_config_without_model_blocks() -> None:
    with pytest.raises(LocalRunError, match="nothing to rewrite"):
        rewrite_agent_config({"config_format": "nemo-agents-spec-v1", "harnesses": {"codex": {"kind": "codex"}}}, "x")
