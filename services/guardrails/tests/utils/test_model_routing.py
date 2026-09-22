# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from nhx.guardrails.app.utils.model_routing import (
    build_openai_gateway_url,
    parse_model_entity_reference,
    resolve_model_entity_references,
)
from nhx.guardrails.entities.values._private import Model, RailsConfig


class TestParseModelEntityReference:
    """Tests for parse_model_entity_reference()."""

    def test_valid_references(self):
        """Test valid Model Entity references return (workspace, model_name) tuple."""
        assert parse_model_entity_reference("default/my-model") == ("default", "my-model")
        assert parse_model_entity_reference("workspace/meta-llama-3-1-8b") == ("workspace", "meta-llama-3-1-8b")
        assert parse_model_entity_reference("my-workspace/model-name") == ("my-workspace", "model-name")
        assert parse_model_entity_reference("prod/llama-3-3-70b-instruct") == ("prod", "llama-3-3-70b-instruct")

    def test_simple_model_names_return_none(self):
        """Test simple model names without workspace return None."""
        assert parse_model_entity_reference("gpt-4") is None
        assert parse_model_entity_reference("llama-3-1-8b") is None
        assert parse_model_entity_reference("meta-llama-3-1-8b-instruct") is None

    def test_empty_and_none_return_none(self):
        """Test empty strings and None return None."""
        assert parse_model_entity_reference(None) is None
        assert parse_model_entity_reference("") is None

    def test_too_many_slashes_return_none(self):
        """Test strings with more than one slash return None."""
        assert parse_model_entity_reference("a/b/c") is None
        assert parse_model_entity_reference("workspace/subdir/model") is None

    def test_empty_parts_return_none(self):
        """Test strings with empty parts return None."""
        assert parse_model_entity_reference("/model") is None
        assert parse_model_entity_reference("workspace/") is None
        assert parse_model_entity_reference("/") is None


class TestBuildOpenAIGatewayUrl:
    """Tests for build_openai_gateway_url()."""

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_url_construction(self, mock_get_sdk):
        """Test URL construction for Model Entity reference."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        url = build_openai_gateway_url("default/my-model")
        assert url == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"

        # Test non-default workspace
        url = build_openai_gateway_url("custom-workspace/my-model")
        assert url == "http://localhost:8000/apis/inference-gateway/v2/workspaces/custom-workspace/openai/-/v1"

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_v1_suffix_preserved(self, mock_get_sdk):
        """Test /v1 suffix is preserved from typed client URL."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        url = build_openai_gateway_url("default/model")
        assert url == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        assert url.endswith("/v1")

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_typed_client_adds_v1(self, mock_get_sdk):
        """Test the typed Models client helper adds the OpenAI /v1 suffix."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        url = build_openai_gateway_url("default/model")
        assert url == "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"

    def test_invalid_reference_raises(self):
        """Test invalid reference raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model entity reference"):
            build_openai_gateway_url("no-slash-here")


class TestResolveModelEntityReferences:
    """Tests for resolve_model_entity_references()."""

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_resolve_single_model(self, mock_get_sdk):
        """Test resolving a single model with Model Entity reference."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="default/llama-3-3-70b-instruct"),
            ]
        )

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models[0].parameters["base_url"] == (
            "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_resolve_all_models(self, mock_get_sdk):
        """Test that ALL models in config get resolved (multiple models use case)."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="default/llama-3-3-70b-instruct"),
                Model(type="content_safety", engine="nim", model="default/llama-3-1-nemoguard-8b-content-safety"),
                Model(type="topic_control", engine="nim", model="default/llama-3-1-nemoguard-8b-topic-control"),
            ],
        )

        resolved = resolve_model_entity_references(rails_config)

        expected_url = "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        assert resolved.models[0].parameters["base_url"] == expected_url
        assert resolved.models[1].parameters["base_url"] == expected_url
        assert resolved.models[2].parameters["base_url"] == expected_url

    def test_explicit_base_url_preserved_for_non_entity(self):
        """Test that explicit base_url is preserved for non-entity models."""
        rails_config = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="nim",
                    model="gpt-4",
                    parameters={"base_url": "http://custom-endpoint/v1"},
                ),
            ]
        )

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models[0].parameters["base_url"] == "http://custom-endpoint/v1"

    def test_explicit_base_url_preserved_for_entity_reference(self):
        """Test that explicit base_url is preserved even for Model Entity references."""
        rails_config = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="nim",
                    model="default/my-model",
                    parameters={"base_url": "http://custom-override/v1"},
                ),
            ]
        )

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models[0].parameters["base_url"] == "http://custom-override/v1"

    @patch("nhx.guardrails.app.utils.model_routing.get_platform_sdk")
    def test_mixed_config(self, mock_get_sdk):
        """Test config with one Model Entity ref, one explicit URLs."""
        mock_sdk = MagicMock()
        mock_sdk.base_url = "http://localhost:8000"
        mock_get_sdk.return_value = mock_sdk

        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="default/my-model"),
                Model(
                    type="content_safety",
                    engine="nim",
                    model="default/nemoguard-content-safety",
                    parameters={"base_url": "https://direct-nim:8000/v1"},
                ),
            ]
        )

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models[0].parameters["base_url"] == (
            "http://localhost:8000/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
        )
        assert resolved.models[1].parameters["base_url"] == "https://direct-nim:8000/v1"

    def test_empty_models_returns_unchanged(self):
        """Test that config with empty models list is returned unchanged."""
        rails_config = RailsConfig(models=[])

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models == []

    def test_non_entity_model_without_base_url_unchanged(self):
        """Test that non-entity models without base_url are not modified."""
        rails_config = RailsConfig(
            models=[
                Model(type="main", engine="nim", model="gpt-4"),
            ]
        )

        resolved = resolve_model_entity_references(rails_config)

        assert resolved.models[0].parameters == {}
