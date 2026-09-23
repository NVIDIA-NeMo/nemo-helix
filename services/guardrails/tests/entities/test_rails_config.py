# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for RailsConfig and Model entity behavior."""

import pytest
from nhx.guardrails.entities.values._private import _UNSUPPORTED_MODEL_TASKS, Model, RailsConfig
from pydantic import ValidationError


def _expected_error(model_type: str) -> str:
    detail = _UNSUPPORTED_MODEL_TASKS[model_type]
    return f"Model for task '{model_type}' cannot be manually set. {detail}"


class TestModelTypeValidation:
    """Tests for the Model validator that blocks unsupported model types."""

    def test_generate_user_intent_model_type_is_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            Model(type="generate_user_intent", engine="nim", model="workspace/some-model")
        assert _expected_error("generate_user_intent") in str(exc_info.value)

    def test_generate_user_intent_rejected_in_rails_config(self):
        with pytest.raises(ValidationError) as exc_info:
            RailsConfig.model_validate(
                {
                    "models": [
                        {"type": "generate_user_intent", "engine": "nim", "model": "workspace/some-model"},
                    ]
                }
            )
        assert _expected_error("generate_user_intent") in str(exc_info.value)


class TestRailsConfigExtraFields:
    """Tests verifying that RailsConfig correctly ignores extra fields.

    IMPORTANT: RailsConfig is configured with `extra = "ignore"` which means:
    - Extra fields in input data are silently ignored during validation
    - Extra fields are NOT stored on the model
    - model_dump() only returns defined fields

    This behavior is critical for compatibility with nemoguardrails configs
    that may contain fields we don't model in our schema.
    """

    def test_extra_fields_ignored_during_validation(self):
        """Test that extra fields in input data don't cause validation errors."""
        # This should NOT raise a validation error
        config = RailsConfig.model_validate(
            {
                "unknown_field": "should be ignored",
                "another_extra": {"nested": "data"},
                "passthrough": True,
            }
        )

        # Valid field should be set
        assert config.passthrough is True

    def test_extra_fields_not_stored_on_model(self):
        """Test that extra fields are not stored as attributes on the model."""
        config = RailsConfig.model_validate(
            {
                "some_nemo_specific_field": "value",
                "passthrough": True,
            }
        )

        # Extra field should not be accessible as an attribute
        assert not hasattr(config, "some_nemo_specific_field")

    def test_extra_fields_not_in_model_dump(self):
        """Test that model_dump() only returns defined schema fields."""
        config = RailsConfig.model_validate(
            {
                "extra_field_1": "ignored",
                "extra_field_2": 123,
                "passthrough": True,
                "models": [{"model": "test/model", "type": "main", "engine": "nim"}],
            }
        )

        dumped = config.model_dump()

        # Extra fields should not appear in dump
        assert "extra_field_1" not in dumped
        assert "extra_field_2" not in dumped

        # Valid fields should appear
        assert dumped["passthrough"] is True
        assert len(dumped["models"]) == 1

    def test_model_dump_keys_match_model_fields(self):
        """Test that all keys in model_dump are defined in model_fields."""
        config = RailsConfig.model_validate(
            {
                "random_extra_key": "should not appear",
                "passthrough": True,
            }
        )

        dumped = config.model_dump()
        model_field_keys = set(RailsConfig.model_fields.keys())

        # Every key in the dump should be a defined model field
        for key in dumped.keys():
            assert key in model_field_keys, f"Unexpected key '{key}' in model_dump"

    def test_extra_ignore_config_is_set(self):
        """Test that RailsConfig has extra='ignore' configured.

        This is a safeguard test - if someone changes the extra setting,
        this test will fail and alert them to the implications.
        """
        # Access the model config to verify extra setting
        # In Pydantic v2, this is in model_config or the Config class
        config_extra = getattr(RailsConfig, "model_config", {}).get("extra")
        if config_extra is None:
            # Check legacy Config class
            config_class = getattr(RailsConfig, "Config", None)
            if config_class:
                config_extra = getattr(config_class, "extra", None)

        assert config_extra == "ignore", (
            "RailsConfig.extra must be 'ignore' to maintain compatibility with "
            "nemoguardrails configs that may contain fields we don't model. "
            "Changing this will break config loading."
        )
