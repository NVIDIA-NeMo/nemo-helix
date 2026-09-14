# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.values.models import (
    Model,
    filter_auth_headers,
    is_auth_header_name,
    normalize_header_name,
)
from pydantic_core import ValidationError


class TestHeaderNameHelpers:
    def test_normalize_header_name_strips_lowercases_and_replaces_underscores(self):
        assert normalize_header_name("  X_AUTH_Token  ") == "x-auth-token"

    @pytest.mark.parametrize(
        ("header_name", "expected"),
        [
            ("Authorization", True),
            ("Cookie", True),
            ("Set-Cookie", True),
            ("x-auth-token", True),
            ("openai-api-key", True),
            ("my_secret_header", True),
            ("X-Trace-Id", False),
            ("X-NMP-Principal-Id", False),
        ],
    )
    def test_is_auth_header_name(self, header_name: str, expected: bool):
        assert is_auth_header_name(header_name) is expected

    def test_filter_auth_headers_removes_only_auth_style_headers(self):
        assert filter_auth_headers(
            {
                "Authorization": "Bearer secret-token",
                "X-Trace-Id": "trace-123",
                "X-NMP-Principal-Id": "service:evaluator",
            }
        ) == {
            "X-Trace-Id": "trace-123",
            "X-NMP-Principal-Id": "service:evaluator",
        }

    def test_filter_auth_headers_returns_none_when_all_headers_are_filtered(self):
        assert filter_auth_headers({"Authorization": "Bearer secret-token", "x-auth-token": "secret-token"}) is None


class TestModelDefaultHeaders:
    def test_with_default_headers_returns_copy_with_merged_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-Existing": "model"},
        )

        updated = model.with_default_headers({"X-NMP-Principal-Id": "service:evaluator"})

        assert updated is not model
        assert model.default_headers == {"X-Existing": "model"}
        assert updated.default_headers == {
            "X-Existing": "model",
            "X-NMP-Principal-Id": "service:evaluator",
        }

    def test_model_dump_excludes_default_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-NMP-Principal-Id": "service:evaluator"},
        )

        assert "default_headers" not in model.model_dump(mode="python")

    @pytest.mark.parametrize(
        "header_name",
        [
            "Authorization",
            "Cookie",
            "Set-Cookie",
            "Proxy-Authorization",
            "X-API-Key",
            "x-auth-token",
            "openai-api-key",
            "my_secret_header",
        ],
    )
    def test_rejects_auth_style_default_headers(self, header_name: str):
        with pytest.raises(
            ValidationError,
            match="default_headers cannot include authentication headers .*model.api_key_secret",
        ):
            Model(
                url="https://judge.example.test/v1/chat/completions",
                name="judge-model",
                default_headers={header_name: "blocked"},
            )

    @pytest.mark.parametrize(
        "header_name",
        [
            "X-NMP-Principal-Id",
            "X-Trace-Id",
        ],
    )
    def test_allows_non_auth_default_headers(self, header_name: str):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={header_name: "value"},
        )

        assert model.default_headers == {header_name: "value"}


class TestDeprecatedFormatField:
    """`format` is deprecated and ignored; these pin the contract API consumers see."""

    def test_format_defaults_to_openai(self):
        model = Model(url="https://judge.example.test/v1/chat/completions", name="judge-model")

        assert model.format == ModelFormat.OPEN_AI

    def test_format_is_marked_deprecated_in_the_json_schema(self):
        # The published OpenAPI spec is generated from this schema, so the deprecation signal
        # reaching API consumers depends on this flag rather than on the docstring.
        assert Model.model_json_schema()["properties"]["format"]["deprecated"] is True

    def test_format_is_still_accepted_so_existing_callers_keep_working(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            format=ModelFormat.NVIDIA_NIM,
        )

        assert model.format == ModelFormat.NVIDIA_NIM


class TestModelInferenceConfig:
    def test_inference_discriminator_is_required_in_ranking_schema(self) -> None:
        schema = Model.model_json_schema()
        ranking_schema = schema["$defs"]["RankingInference"]
        assert ranking_schema["properties"]["type"]["const"] == "ranking" or ranking_schema["properties"]["type"][
            "enum"
        ] == ["ranking"]
        assert "type" in ranking_schema["required"]

    def test_flat_ranking_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Model.model_validate(
                {
                    "url": "https://igw.example.test/v1",
                    "name": "reranker",
                    "ranking_contract": "hosted-rerank-v1",
                    "ranking_path": "/rerank",
                }
            )

    def test_unknown_ranking_keys_are_rejected_without_inference(self) -> None:
        with pytest.raises(ValidationError):
            Model.model_validate(
                {
                    "url": "https://igw.example.test/v1",
                    "name": "reranker",
                    "ranking_endpoint": "/v1/ranking",
                }
            )

    def test_unknown_inference_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Model.model_validate(
                {
                    "url": "https://igw.example.test/v1",
                    "name": "embedder",
                    "inference": {"type": "embeddings", "path": "/embeddings"},
                }
            )
