# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
from nhx.guardrails.app.services.configs.sources import _normalize
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import (
    PatronusEvaluationSuccessStrategy,
    Rails,
    RailsConfig,
)


def test_model_dump_with_enum():
    """Test that the _normalize function correctly converts Enum values to primitives."""
    data = {"success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS, "param": "test"}

    normalized_data = _normalize(data)

    assert normalized_data.get("success_strategy") == "all_pass"
    assert isinstance(normalized_data["success_strategy"], str)

    yaml_content = yaml.safe_dump(normalized_data)
    assert "success_strategy: all_pass" in yaml_content


def test_config_data_with_enums():
    """Test that a RailsConfig object with Enum values is properly normalized."""
    rails_config = {
        "patronus": {"input": {"evaluate_config": {"success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS}}}
    }

    config_data = RailsConfig(models=[], rails=Rails(config=rails_config))
    dumped_data = config_data.model_dump()

    normalized_data = _normalize(dumped_data)

    success_strategy = normalized_data["rails"]["config"]["patronus"]["input"]["evaluate_config"]["success_strategy"]
    assert success_strategy == "all_pass"
    assert isinstance(success_strategy, str)

    yaml_content = yaml.safe_dump(normalized_data)
    assert "success_strategy: all_pass" in yaml_content


def test_guardrail_config_with_enums():
    """Test that a GuardrailConfig with Enum values is properly serialized."""
    rails_config = {
        "patronus": {"input": {"evaluate_config": {"success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS}}}
    }

    config_data = RailsConfig(models=[], rails=Rails(config=rails_config))

    config = GuardrailConfig(name="test", workspace="default", data=config_data)

    dumped_data = config.model_dump()

    normalized_data = _normalize(dumped_data)

    yaml_content = yaml.safe_dump(normalized_data)

    assert "success_strategy: all_pass" in yaml_content
