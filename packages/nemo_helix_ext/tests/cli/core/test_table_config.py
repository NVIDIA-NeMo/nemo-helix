# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for table configuration and column handling."""

from unittest.mock import Mock

from nemo_helix_ext.cli.core.formatters import Column
from nemo_helix_ext.cli.core.table_config import (
    get_available_nested_fields,
    get_nested_value,
    is_timestamp_field,
    resolve_and_validate_columns,
    should_truncate_field,
    validate_output_columns,
)


def test_validate_output_columns_all_keyword():
    result = validate_output_columns("all")
    assert result == "all"


def test_validate_output_columns_comma_separated():
    result = validate_output_columns("name,status")
    assert len(result) == 2
    assert result[0].field == "name"
    assert result[1].field == "status"


def test_get_nested_value_simple():
    obj = {"name": "test-model", "status": "active"}
    assert get_nested_value(obj, "name") == "test-model"
    assert get_nested_value(obj, "status") == "active"


def test_get_nested_value_nested():
    obj = {"name": "test", "config": {"type": "llm", "hyperparameters": {"temperature": "0.7"}}}
    assert get_nested_value(obj, "config.type") == "llm"
    assert get_nested_value(obj, "config.hyperparameters.temperature") == "0.7"


def test_get_nested_value_computed_field():
    obj = {"namespace": "default", "name": "my-model"}
    assert get_nested_value(obj, "{namespace}/{name}") == "default/my-model"


def test_get_nested_value_missing_field():
    obj = {"name": "test"}
    assert get_nested_value(obj, "missing") == ""
    assert get_nested_value(obj, "nested.missing") == ""


def test_get_nested_value_none_value():
    obj = {"name": None}
    assert get_nested_value(obj, "name") == ""


def test_get_nested_value_list_handling():
    obj = {"tags": ["tag1", "tag2", "tag3"]}
    assert get_nested_value(obj, "tags") == "tag1"


def test_get_nested_value_empty_list():
    obj = {"tags": []}
    assert get_nested_value(obj, "tags") == ""


def test_get_available_nested_fields_simple():
    obj = {"name": "test", "status": "active"}
    fields = get_available_nested_fields(obj)
    assert "name" in fields
    assert "status" in fields


def test_get_available_nested_fields_nested():
    obj = {"name": "test", "config": {"type": "llm", "params": {"temperature": 0.7}}}
    fields = get_available_nested_fields(obj)
    assert all(f in fields for f in ["name", "config", "config.type", "config.params", "config.params.temperature"])


def test_get_available_nested_fields_with_list():
    obj = [{"name": "item1", "status": "active"}, {"name": "item2", "status": "inactive"}]
    fields = get_available_nested_fields(obj)
    assert "name" in fields
    assert "status" in fields


def test_get_available_nested_fields_max_depth():
    obj = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
    fields_default = get_available_nested_fields(obj)
    assert "a.b.c.d.e" in fields_default

    fields_limited = get_available_nested_fields(obj, max_depth=2)
    assert "a.b" in fields_limited
    assert "a.b.c" not in fields_limited


def test_resolve_and_validate_columns_all():
    mock_item = Mock()
    mock_item.model_dump.return_value = {"name": "test", "namespace": "default", "status": "active"}
    mock_response = Mock()
    mock_response.data = [mock_item]

    result = resolve_and_validate_columns("all", mock_response)
    field_paths = [col.field for col in result]
    # Order must match response (same as JSON output)
    assert field_paths == ["name", "namespace", "status"]


def test_resolve_and_validate_columns_valid():
    mock_item = Mock()
    mock_item.model_dump.return_value = {"name": "test", "namespace": "default"}
    mock_response = Mock()
    mock_response.data = [mock_item]

    columns = [Column("name", "Name"), Column("namespace", "Namespace")]
    result = resolve_and_validate_columns(columns, mock_response)
    assert result == columns


def test_resolve_and_validate_columns_computed_field():
    mock_item = Mock()
    mock_item.model_dump.return_value = {"name": "test", "namespace": "default"}
    mock_response = Mock()
    mock_response.data = [mock_item]

    columns = [Column("{namespace}/{name}", "Resource ID")]
    result = resolve_and_validate_columns(columns, mock_response)
    assert result == columns


def test_resolve_and_validate_columns_invalid():
    """All invalid columns returns all valid fields as fallback."""
    mock_item = Mock()
    mock_item.model_dump.return_value = {"name": "test", "namespace": "default"}
    mock_response = Mock()
    mock_response.data = [mock_item]

    # All invalid columns returns all valid fields as fallback
    result = resolve_and_validate_columns([Column("nonexistent_field", "Invalid")], mock_response)
    field_paths = {col.field for col in result}
    assert field_paths == {"name", "namespace"}


def test_resolve_and_validate_columns_mixed_valid_invalid():
    """Mixed valid/invalid columns returns only valid ones."""
    mock_item = Mock()
    mock_item.model_dump.return_value = {"name": "test", "namespace": "default"}
    mock_response = Mock()
    mock_response.data = [mock_item]

    columns = [Column("name", "Name"), Column("nonexistent", "Bad"), Column("namespace", "Namespace")]
    result = resolve_and_validate_columns(columns, mock_response)
    assert len(result) == 2
    assert result[0].field == "name"
    assert result[1].field == "namespace"


def test_resolve_and_validate_columns_empty_data():
    mock_response = Mock()
    mock_response.data = []
    result = resolve_and_validate_columns("all", mock_response)
    assert result == []


def test_is_timestamp_field():
    timestamp_fields = ["created_at", "updated_at", "started_at", "completed_at", "finished_at", "metadata.created_at"]
    assert all(is_timestamp_field(f) for f in timestamp_fields)
    assert not is_timestamp_field("name")
    assert not is_timestamp_field("status")


def test_should_truncate_field():
    assert not should_truncate_field("name")
    assert not should_truncate_field("{namespace}/{name}")
    assert not should_truncate_field("metadata.name")
    assert should_truncate_field("description")
    assert should_truncate_field("files_url")


def test_get_nested_value_with_quoted_field_names():
    test_data = {
        "dataset_schemas": {
            "$defs": {"SourceTypeEnum": {"description": "Source type enum", "enum": ["type1", "type2"]}}
        }
    }

    value1 = get_nested_value(test_data, "dataset_schemas.$defs.SourceTypeEnum.description")
    assert value1 == "Source type enum"

    value2 = get_nested_value(test_data, "dataset_schemas.'$defs'.SourceTypeEnum.description")
    assert value2 == "Source type enum"

    value3 = get_nested_value(test_data, 'dataset_schemas."$defs".SourceTypeEnum.description')
    assert value3 == "Source type enum"

    assert value1 == value2 == value3
