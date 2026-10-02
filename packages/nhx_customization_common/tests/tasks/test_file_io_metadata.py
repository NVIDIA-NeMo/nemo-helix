# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output fileset metadata helpers."""

from types import SimpleNamespace

import pytest
from nemo_helix_plugin.files.metadata import (
    FilesetMetadata,
    ModelMetadataContent,
    ToolCallingMetadataContent,
)
from nhx.customization_common.tasks.file_io_metadata import (
    build_model_fileset_metadata,
    build_output_fileset_metadata_from_model_entity,
    extract_tool_calling_metadata,
)
from pydantic import ValidationError


def assert_conforms_to_fileset_metadata(meta: dict) -> FilesetMetadata:
    """Validate a produced metadata dict against the real ``FilesetMetadata`` schema.

    Asserting on raw dicts (as this suite originally did) cannot catch structural
    drift: a payload with a misplaced or misspelled key still "looks right" but is
    silently dropped by the schema the platform actually enforces. Round-tripping
    through ``FilesetMetadata`` (validate -> dump) guarantees every key the helper
    emits lands in a real schema field, which is the guarantee the upload path needs.
    """
    validated = FilesetMetadata.model_validate(meta)
    assert validated.model_dump(exclude_none=True, by_alias=True) == meta
    return validated


class TestBuildModelFilesetMetadata:
    def test_wraps_tool_calling_under_model(self) -> None:
        meta = build_model_fileset_metadata(tool_calling={"tool_call_parser": "llama3_json"})
        assert meta == {"model": {"tool_calling": {"tool_call_parser": "llama3_json"}}}

        validated = assert_conforms_to_fileset_metadata(meta)
        assert validated.model is not None
        assert validated.model.tool_calling is not None
        assert validated.model.tool_calling.tool_call_parser == "llama3_json"

    def test_returns_none_when_empty(self) -> None:
        assert build_model_fileset_metadata(tool_calling=None) is None


class TestExtractToolCallingMetadata:
    def test_extracts_from_model_entity_spec(self) -> None:
        me = SimpleNamespace(
            spec=SimpleNamespace(
                chat_template="{% for m in messages %}{{ m }}{% endfor %}",
                tool_call_config=SimpleNamespace(
                    tool_call_parser="llama3_json",
                    tool_call_plugin="default/plugin-fs",
                    auto_tool_choice=True,
                ),
            ),
        )
        assert extract_tool_calling_metadata(me) == {
            "chat_template": "{% for m in messages %}{{ m }}{% endfor %}",
            "tool_call_parser": "llama3_json",
            "tool_call_plugin": "default/plugin-fs",
            "auto_tool_choice": True,
        }

    def test_returns_none_without_spec(self) -> None:
        assert extract_tool_calling_metadata(SimpleNamespace(spec=None)) is None


class TestBuildOutputFilesetMetadataFromModelEntity:
    def test_builds_nested_model_metadata(self) -> None:
        me = SimpleNamespace(
            spec=SimpleNamespace(
                chat_template=None,
                tool_call_config=SimpleNamespace(
                    tool_call_parser="hermes",
                    tool_call_plugin=None,
                    auto_tool_choice=None,
                ),
            ),
        )
        meta = build_output_fileset_metadata_from_model_entity(me)
        assert meta == {"model": {"tool_calling": {"tool_call_parser": "hermes"}}}

        validated = assert_conforms_to_fileset_metadata(meta)
        assert validated.model is not None
        assert validated.model.tool_calling is not None
        assert validated.model.tool_calling.tool_call_parser == "hermes"

    def test_all_tool_calling_fields_map_onto_schema(self) -> None:
        me = SimpleNamespace(
            spec=SimpleNamespace(
                chat_template="{% for m in messages %}{{ m }}{% endfor %}",
                tool_call_config=SimpleNamespace(
                    tool_call_parser="hermes",
                    tool_call_plugin="default/plugin-fs",
                    auto_tool_choice=True,
                ),
            ),
        )
        meta = build_output_fileset_metadata_from_model_entity(me)

        validated = assert_conforms_to_fileset_metadata(meta)
        assert validated == FilesetMetadata(
            model=ModelMetadataContent(
                tool_calling=ToolCallingMetadataContent(
                    chat_template="{% for m in messages %}{{ m }}{% endfor %}",
                    tool_call_parser="hermes",
                    tool_call_plugin="default/plugin-fs",
                    auto_tool_choice=True,
                ),
            ),
        )


class TestFilesetMetadataSchemaConformance:
    """Regression coverage for the metadata-drift bug.

    The original failure stamped an output fileset with metadata that did not match
    the ``FilesetMetadata`` schema the platform validates against. These tests pin
    the schema as the source of truth rather than trusting hand-written dicts.
    """

    def test_schema_rejects_string_metadata(self) -> None:
        # Root cause of the original failure: metadata was set to a bare string
        # instead of the tagged {model: {tool_calling: {...}}} structure.
        with pytest.raises(ValidationError):
            FilesetMetadata.model_validate("llama3_json")

    def test_schema_rejects_string_tool_calling(self) -> None:
        with pytest.raises(ValidationError):
            FilesetMetadata.model_validate({"model": {"tool_calling": "llama3_json"}})

    def test_round_trip_detects_misplaced_tool_calling(self) -> None:
        # The pre-fix shape omitted the `model` wrapper. It validates (extra keys
        # are ignored) but the tool_calling payload is silently dropped -- exactly
        # the drift the round-trip assertion in assert_conforms_to_fileset_metadata
        # is designed to catch.
        drifted = {"tool_calling": {"tool_call_parser": "hermes"}}
        validated = FilesetMetadata.model_validate(drifted)
        assert validated.model is None
        assert validated.model_dump(exclude_none=True, by_alias=True) != drifted
