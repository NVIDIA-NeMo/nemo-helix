# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest
from nhx.guardrails.app.services.configs.sources import (
    FileSystemConfigSource,
    YamlConfigSource,
    _enum_to_primitive,
    _normalize,
)
from nhx.guardrails.entities.values._private import RailsConfig


class ConfigOptionEnum(Enum):
    OPTION_ONE = "option_one"
    OPTION_TWO = "option_two"
    ALL_PASS = "all_pass"


def test_get_config_valid_path():
    mock_config = MagicMock(spec=RailsConfig)
    mock_config.passthrough = True
    with patch.object(RailsConfig, "from_path", return_value=mock_config):
        config = FileSystemConfigSource.get_config("file:///valid/path")
        assert config == mock_config


def test_get_config_invalid_path():
    with patch.object(RailsConfig, "from_path", side_effect=ValueError("Invalid path")):
        with pytest.raises(ValueError, match="Invalid config files at /invalid/path"):
            FileSystemConfigSource.get_config("file:///invalid/path")


def test_get_config_passthrough_none():
    mock_config = MagicMock(spec=RailsConfig)
    mock_config.passthrough = None
    with patch.object(RailsConfig, "from_path", return_value=mock_config):
        config = FileSystemConfigSource.get_config("file:///valid/path")
        assert config.passthrough is True


def test_get_config_passthrough_not_none():
    mock_config = MagicMock(spec=RailsConfig)
    mock_config.passthrough = False
    with patch.object(RailsConfig, "from_path", return_value=mock_config):
        config = FileSystemConfigSource.get_config("file:///valid/path")
        assert config.passthrough is False


def test_enum_to_primitive():
    """Test the _enum_to_primitive function converts Enum to primitive value."""
    enum_value = ConfigOptionEnum.OPTION_ONE
    assert _enum_to_primitive(enum_value) == "option_one"
    assert _enum_to_primitive("string") == "string"
    assert _enum_to_primitive(123) == 123
    assert _enum_to_primitive(None) is None


def test_normalize_with_simple_values():
    """Test the _normalize function with simple values."""
    assert _normalize(ConfigOptionEnum.OPTION_ONE) == "option_one"
    assert _normalize("string") == "string"
    assert _normalize(123) == 123


def test_normalize_with_dict():
    """Test the _normalize function with dictionaries containing Enum values."""

    test_dict = {
        "key1": ConfigOptionEnum.OPTION_ONE,
        "key2": "string",
        "key3": 123,
        "nested": {"enum_key": ConfigOptionEnum.OPTION_TWO},
    }

    expected = {"key1": "option_one", "key2": "string", "key3": 123, "nested": {"enum_key": "option_two"}}

    assert _normalize(test_dict) == expected


def test_normalize_with_list():
    """Test the _normalize function with lists containing Enum values."""
    test_list = [
        ConfigOptionEnum.OPTION_ONE,
        "string",
        123,
        [ConfigOptionEnum.OPTION_TWO],
        {"key": ConfigOptionEnum.ALL_PASS},
    ]

    expected = ["option_one", "string", 123, ["option_two"], {"key": "all_pass"}]

    assert _normalize(test_list) == expected


def test_normalize_with_flat_list_of_enums():
    """Test _normalize with a flat list of Enums."""
    test_list = [ConfigOptionEnum.OPTION_ONE, ConfigOptionEnum.OPTION_TWO, ConfigOptionEnum.ALL_PASS]
    expected = ["option_one", "option_two", "all_pass"]
    assert _normalize(test_list) == expected


def test_normalize_with_list_of_lists_of_enums():
    """Test _normalize with a list of lists containing Enums."""
    test_list = [[ConfigOptionEnum.OPTION_ONE, ConfigOptionEnum.OPTION_TWO], [ConfigOptionEnum.ALL_PASS]]
    expected = [["option_one", "option_two"], ["all_pass"]]
    assert _normalize(test_list) == expected


def test_normalize_with_list_of_dicts_of_enums():
    """Test _normalize with a list of dictionaries containing Enums."""
    test_list = [
        {"key1": ConfigOptionEnum.OPTION_ONE, "key2": "string"},
        {"key3": ConfigOptionEnum.ALL_PASS, "nested_list": [ConfigOptionEnum.OPTION_TWO]},
    ]
    expected = [
        {"key1": "option_one", "key2": "string"},
        {"key3": "all_pass", "nested_list": ["option_two"]},
    ]
    assert _normalize(test_list) == expected


def test_normalize_with_empty_dict():
    """Test _normalize with an empty dictionary."""
    assert _normalize({}) == {}


def test_normalize_with_list_containing_empty_dict():
    """Test _normalize with a list containing an empty dictionary."""
    test_list = [{}]
    expected = [{}]
    assert _normalize(test_list) == expected
    test_list_with_enum = [{"key": ConfigOptionEnum.OPTION_ONE}, {}]
    expected_with_enum = [{"key": "option_one"}, {}]
    assert _normalize(test_list_with_enum) == expected_with_enum


def test_normalize_with_dict_containing_empty_structures():
    """Test _normalize with a dictionary containing empty dict and list."""
    test_dict = {"empty_dict": {}, "empty_list": [], "data": ConfigOptionEnum.OPTION_ONE}
    expected = {"empty_dict": {}, "empty_list": [], "data": "option_one"}
    assert _normalize(test_dict) == expected


def test_yaml_config_source_with_enums():
    """Test that YamlConfigSource correctly handles Enum values in config data."""
    config_data = {
        "schema_version": "1.0",
        "id": "guardrail-test",
        "description": "Test config for enum serialization",
        "type_prefix": "guardrail",
        "namespace": "test",
        "created_at": datetime(2025, 5, 8, 13, 34, 3, 923751),
        "updated_at": datetime(2025, 5, 8, 13, 34, 3, 923753),
        "custom_fields": {},
        "name": "test_config",
        "version_id": "main",
        "version_tags": [],
        "data": {
            "rails": {
                "config": {
                    "patronus": {
                        "input": {"evaluate_config": {"success_strategy": ConfigOptionEnum.ALL_PASS, "params": {}}},
                        "output": {"evaluate_config": {"success_strategy": ConfigOptionEnum.ALL_PASS, "params": {}}},
                    }
                }
            }
        },
    }

    mock_config = MagicMock(spec=RailsConfig)
    mock_config.passthrough = True

    with patch.object(RailsConfig, "from_content", return_value=mock_config) as mock_from_content:
        config = YamlConfigSource.get_config(config_data=config_data)

        mock_from_content.assert_called_once()

        yaml_content = mock_from_content.call_args[1]["yaml_content"]

        assert "success_strategy: all_pass" in yaml_content
        assert "TestEnum.ALL_PASS" not in yaml_content

    assert config == mock_config


@pytest.mark.xfail(reason="This test should fail without _normalize converting Enums to primitives before YAML dump.")
def test_yaml_config_source_with_enums_fails_without_normalize():
    """Test that YamlConfigSource would fail to serialize Enums correctly without _normalize."""

    config_data = {
        "name": "test_config_fail",
        "data": {
            "rails": {
                "config": {
                    "patronus": {
                        "input": {"evaluate_config": {"success_strategy": ConfigOptionEnum.ALL_PASS}},
                    }
                }
            }
        },
    }

    import yaml

    unnormalized_yaml_content = yaml.safe_dump(config_data["data"]["rails"]["config"])

    assert "success_strategy: all_pass" not in unnormalized_yaml_content
    assert "!!python" in unnormalized_yaml_content


class PatronusEvaluationSuccessStrategy(Enum):
    """Enum used in the real-world example."""

    ALL_PASS = "all_pass"
    MAJORITY_PASS = "majority_pass"


def test_real_example_serialization():
    """Test that the real-world example can be serialized correctly."""
    config_data = {
        "schema_version": "1.0",
        "id": "guardrail-PR4ANgDHLDTwQvvcaNW2oH",
        "description": "Test config for injection detection",
        "type_prefix": "guardrail",
        "namespace": "test",
        "created_at": datetime(2025, 5, 8, 13, 34, 3, 923751),
        "updated_at": datetime(2025, 5, 8, 13, 34, 3, 923753),
        "custom_fields": {},
        "name": "injection_detection",
        "version_id": "main",
        "version_tags": [],
        "data": {
            "models": [],
            "instructions": [
                {"type": "general", "content": "Below is a conversation between a helpful AI assistant and a user."}
            ],
            "sample_conversation": 'user "Hello there!"\n  express greeting\nbot express greeting\n  "Hello! How can I assist you today?"',
            "prompting_mode": "standard",
            "lowest_temperature": 0.001,
            "enable_multi_step_generation": False,
            "colang_version": "1.0",
            "custom_data": {},
            "rails": {
                "config": {
                    "fact_checking": {"parameters": {}, "fallback_to_self_check": False},
                    "autoalign": {
                        "parameters": {},
                        "input": {"guardrails_config": {}},
                        "output": {"guardrails_config": {}},
                    },
                    "patronus": {
                        "input": {
                            "evaluate_config": {
                                "success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS,
                                "params": {},
                            }
                        },
                        "output": {
                            "evaluate_config": {
                                "success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS,
                                "params": {},
                            }
                        },
                    },
                    "sensitive_data_detection": {
                        "recognizers": [],
                        "input": {"entities": [], "mask_token": "*", "score_threshold": 0.2},
                        "output": {"entities": [], "mask_token": "*", "score_threshold": 0.2},
                        "retrieval": {"entities": [], "mask_token": "*", "score_threshold": 0.2},
                    },
                    "jailbreak_detection": {
                        "length_per_perplexity_threshold": 89.79,
                        "prefix_suffix_perplexity_threshold": 1845.65,
                        "embedding": "nvidia/nv-embedqa-e5-v5",
                    },
                },
                "input": {"flows": []},
                "output": {
                    "flows": ["injection detection"],
                    "streaming": {"enabled": True, "chunk_size": 200, "context_size": 50, "stream_first": True},
                },
                "retrieval": {"flows": []},
                "dialog": {
                    "single_call": {"enabled": False, "fallback_to_multiple_calls": True},
                    "user_messages": {"embeddings_only": False},
                },
                "actions": {},
            },
            "enable_rails_exceptions": False,
        },
    }

    mock_config = MagicMock(spec=RailsConfig)
    mock_config.passthrough = True

    with patch.object(RailsConfig, "from_content", return_value=mock_config) as mock_from_content:
        _config = YamlConfigSource.get_config(config_data=config_data)

        yaml_content = mock_from_content.call_args[1]["yaml_content"]

        assert "success_strategy: all_pass" in yaml_content
        assert "PatronusEvaluationSuccessStrategy.ALL_PASS" not in yaml_content

        assert "created_at:" in yaml_content
        assert "updated_at:" in yaml_content


@pytest.mark.xfail(
    reason="This test demonstrates failure if enums in complex config are not normalized before YAML dump."
)
def test_real_example_serialization_fails_without_normalize():
    """Test that real world example serialization would fail without _normalize."""

    config_data = {
        "name": "injection_detection_fail",
        "data": {
            "rails": {
                "config": {
                    "patronus": {
                        "input": {
                            "evaluate_config": {
                                "success_strategy": PatronusEvaluationSuccessStrategy.ALL_PASS,
                            }
                        }
                    }
                }
            }
        },
        "created_at": datetime(2025, 5, 8, 13, 34, 3, 923751),
    }

    import yaml

    problematic_part = config_data["data"]["rails"]["config"]["patronus"]["input"]["evaluate_config"]
    yaml_content_if_not_normalized = yaml.safe_dump(problematic_part)

    assert "success_strategy: all_pass" not in yaml_content_if_not_normalized
    assert "!!python" in yaml_content_if_not_normalized

    raw_dump_output = yaml.safe_dump(config_data)
    assert "success_strategy: all_pass" not in raw_dump_output
    assert "PatronusEvaluationSuccessStrategy" in raw_dump_output
    assert "!!python" in raw_dump_output
