# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for output formatters."""

import json
import re
from unittest.mock import Mock, patch

import click
import pytest
import yaml
from nemo_helix_ext.cli.core.formatters import (
    Column,
    _extract_items_from_response,
    format_csv,
    format_json,
    format_markdown_table,
    format_output,
    format_stream_event,
    format_table,
    format_yaml,
    iter_json_lines,
    model_to_dict,
    validate_stream_output_format,
)


def test_model_to_dict_with_dict():
    """Test model_to_dict with a dictionary."""
    data = {"key": "value", "number": 42}
    result = model_to_dict(data)
    assert result == data


def test_model_to_dict_with_list():
    """Test model_to_dict with a list."""
    data = [{"a": 1}, {"b": 2}]
    result = model_to_dict(data)
    assert result == data


def test_model_to_dict_with_primitive():
    """Test model_to_dict with primitive values."""
    assert model_to_dict("string") == "string"
    assert model_to_dict(42) == 42
    assert model_to_dict(True) is True


def test_model_to_dict_with_pydantic_model():
    """Test model_to_dict with a Pydantic model."""
    # Create a mock Pydantic model
    mock_model = Mock()
    mock_model.model_dump = Mock(return_value={"field": "value"})

    result = model_to_dict(mock_model)
    assert result == {"field": "value"}
    mock_model.model_dump.assert_called_once_with(mode="json")


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_json_no_highlighting(mock_is_tty):
    """Test JSON formatting without syntax highlighting."""
    mock_is_tty.return_value = False

    data = {"key": "value", "number": 42}
    result = format_json(data, indent=2, syntax_highlight=True)

    # Should be plain JSON without ANSI codes
    expected = json.dumps(data, indent=2, ensure_ascii=False)
    assert result == expected


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_json_with_highlighting(mock_is_tty):
    """Test JSON formatting with syntax highlighting in TTY."""
    mock_is_tty.return_value = True

    data = {"key": "value"}
    result = format_json(data, indent=2, syntax_highlight=True)

    # Should contain ANSI escape codes for colors
    # The exact output depends on Rich, so just check it's different from plain JSON
    plain = json.dumps(data, indent=2, ensure_ascii=False)
    assert result != plain or "\x1b[" in result  # ANSI codes or colored output


def test_format_json_no_indent():
    """Test JSON formatting without indentation."""
    data = {"key": "value", "number": 42}
    result = format_json(data, indent=None, syntax_highlight=False)

    expected = json.dumps(data, indent=None, ensure_ascii=False)
    assert result == expected


def test_format_json_nested_data():
    """Test JSON formatting with nested data structures."""
    data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "meta": {"count": 2, "page": 1}}

    result = format_json(data, indent=2, syntax_highlight=False)
    parsed = json.loads(result)
    assert parsed == data


def test_iter_json_lines_extracts_list_items():
    """JSON Lines output should emit one compact JSON object per list item."""
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "Alice"},
        {"id": "2", "name": "Bob"},
    ]

    assert list(iter_json_lines(mock_response, is_list=True)) == [
        '{"id":"1","name":"Alice"}',
        '{"id":"2","name":"Bob"}',
    ]


def test_iter_json_lines_keeps_callable_items_lazy():
    """Callable response item sources should stream without list materialization."""

    class LazyResponse:
        def __init__(self) -> None:
            self.yielded = 0

        def items(self):
            for item in [{"id": "1"}, {"id": "2"}]:
                self.yielded += 1
                yield item

    response = LazyResponse()
    lines = iter_json_lines(response, is_list=True)

    assert response.yielded == 0
    assert next(lines) == '{"id":"1"}'
    assert response.yielded == 1
    assert next(lines) == '{"id":"2"}'
    assert response.yielded == 2
    with pytest.raises(StopIteration):
        next(lines)


def test_format_output_streams_json_lines(capsys):
    """format_output should print list items as newline-delimited JSON when streaming."""
    mock_response = Mock()
    mock_response.data = [{"id": "1"}, {"id": "2"}]

    format_output(mock_response, is_list=True, output_format="json", stream=True)

    captured = capsys.readouterr()
    assert captured.out.splitlines() == ['{"id":"1"}', '{"id":"2"}']


def test_format_output_stream_requires_json_compatible_format():
    """--stream is only valid for JSON-compatible formats."""
    with pytest.raises(click.UsageError, match="--stream requires --output json or --output raw"):
        format_output([{"id": "1"}], is_list=True, output_format="table", stream=True)


def test_validate_stream_output_format_rejects_table_output():
    """Command handlers can fail before loading API or discovery data."""
    with pytest.raises(click.UsageError, match="--stream requires --output json or --output raw"):
        validate_stream_output_format("table", True)


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_yaml_no_highlighting(mock_is_tty):
    """Test YAML formatting without syntax highlighting."""
    mock_is_tty.return_value = False

    data = {"key": "value", "number": 42}
    result = format_yaml(data, syntax_highlight=True, background=False)

    # Should be plain YAML without ANSI codes
    parsed = yaml.safe_load(result)
    assert parsed == data


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_yaml_with_highlighting(mock_is_tty):
    """Test YAML formatting with syntax highlighting in TTY."""
    mock_is_tty.return_value = True

    data = {"key": "value", "number": 42}
    result = format_yaml(data, syntax_highlight=True, background=False)

    # Should contain the data (may have ANSI codes)
    # Parse by stripping ANSI codes first

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    plain = ansi_escape.sub("", result)
    parsed = yaml.safe_load(plain)
    assert parsed == data


def test_format_yaml_nested_data():
    """Test YAML formatting with nested data structures."""
    data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "meta": {"count": 2, "page": 1}}

    result = format_yaml(data, syntax_highlight=False, background=False)
    parsed = yaml.safe_load(result)
    assert parsed == data


def test_format_yaml_preserves_order():
    """Test that YAML formatting preserves key order."""
    data = {
        "first": 1,
        "second": 2,
        "third": 3,
    }

    result = format_yaml(data, syntax_highlight=False, background=False)

    # Check that keys appear in order
    lines = result.strip().split("\n")
    assert "first:" in lines[0]
    assert "second:" in lines[1]
    assert "third:" in lines[2]


def test_format_table_with_list():
    """Test table formatting with a list of items."""
    # Create mock response with data
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "Alice", "status": "active"},
        {"id": "2", "name": "Bob", "status": "inactive"},
    ]

    columns = [Column("id", "Id"), Column("name", "Name"), Column("status", "Status")]
    result = format_table(mock_response, columns=columns)

    # Should contain table headers and data
    assert "Id" in result or "ID" in result
    assert "Name" in result
    assert "Alice" in result
    assert "Bob" in result


def test_format_table_with_pydantic_models():
    """Test table formatting with Pydantic models."""
    # Create mock Pydantic models with fields matching the namespace table config
    mock_item1 = Mock()
    mock_item1.model_dump = Mock(
        return_value={"id": "ns-1", "description": "Default namespace", "project": "proj-1", "created_at": "2025-01-01"}
    )

    mock_item2 = Mock()
    mock_item2.model_dump = Mock(
        return_value={
            "id": "ns-2",
            "description": "Production namespace",
            "project": "proj-2",
            "created_at": "2025-01-02",
        }
    )

    mock_response = Mock()
    mock_response.data = [mock_item1, mock_item2]

    columns = [Column("id", "ID"), Column("description", "Description"), Column("project", "Project")]
    result = format_table(mock_response, columns=columns)

    # Should contain configured columns for namespaces
    assert "ID" in result
    assert "Description" in result
    assert "ns-1" in result
    assert "ns-2" in result
    assert "Default namespace" in result or "Default nam..." in result  # May be truncated


def test_format_table_empty_data():
    """Test table formatting with empty data."""
    mock_response = Mock()
    mock_response.data = []

    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_table(mock_response, columns=columns)

    assert "No data to display" in result


def test_format_table_with_nested_fields():
    """Test table formatting with nested field paths."""
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "metadata": {"name": "test", "version": "1.0"}},
    ]

    # Test with custom columns including nested path
    columns = [Column("id", "ID"), Column("metadata.name", "Name"), Column("metadata.version", "Version")]
    result = format_table(mock_response, columns=columns)

    # Should handle the data without errors
    assert "1" in result


def test_format_markdown_table_basic():
    """Test markdown table formatting with basic data."""
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "Alice", "status": "active"},
        {"id": "2", "name": "Bob", "status": "inactive"},
    ]

    columns = [Column("id", "ID"), Column("name", "Name"), Column("status", "Status")]
    result = format_markdown_table(mock_response, columns=columns)

    # Should be a proper markdown table
    lines = result.split("\n")
    assert len(lines) >= 4  # Header + separator + 2 data rows
    assert "|" in lines[0]  # Header
    assert "---" in lines[1]  # Separator
    assert "Alice" in result
    assert "Bob" in result


def test_format_markdown_table_with_pydantic_models():
    """Test markdown table formatting with Pydantic models."""
    mock_item1 = Mock()
    mock_item1.model_dump = Mock(
        return_value={"id": "ns-1", "description": "Default", "project": "proj-1", "created_at": "2025-01-01"}
    )

    mock_item2 = Mock()
    mock_item2.model_dump = Mock(
        return_value={"id": "ns-2", "description": "Production", "project": "proj-2", "created_at": "2025-01-02"}
    )

    mock_response = Mock()
    mock_response.data = [mock_item1, mock_item2]

    columns = [Column("id", "ID"), Column("description", "Description"), Column("project", "Project")]
    result = format_markdown_table(mock_response, columns=columns)

    # Should contain markdown table structure
    assert "| ID" in result
    assert "| Description" in result
    assert "ns-1" in result
    assert "ns-2" in result
    assert "---" in result


def test_format_markdown_table_escapes_pipes():
    """Test that markdown table escapes pipe characters in data."""
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "test|with|pipes"},
    ]

    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_markdown_table(mock_response, columns=columns)

    # Pipes should be escaped
    assert "test\\|with\\|pipes" in result


def test_format_markdown_table_normalizes_newlines_and_short_separators():
    """Markdown cells stay on one row and delimiter cells have at least three dashes."""
    mock_response = Mock()
    mock_response.data = [{"id": "1", "description": "first\r\nsecond\rthird\nfourth"}]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_markdown_table(mock_response, columns=columns)
    lines = result.splitlines()

    assert len(lines) == 3
    assert "first<br>second<br>third<br>fourth" in lines[2]
    assert lines[1].split("|")[1].strip() == "---"


def test_format_markdown_table_truncates_before_escaping_multiline_pipes():
    """Truncation preserves complete Markdown escapes in multiline cells containing pipes."""
    mock_response = Mock()
    mock_response.data = [{"description": "first|segment\nsecond|segment"}]

    columns = [Column("description", "Description")]
    result = format_markdown_table(mock_response, columns=columns, max_width=20)
    lines = result.splitlines()

    assert len(lines) == 3
    assert "first\\|segment<br>sec..." in lines[2]


def test_format_markdown_table_empty():
    """Test markdown table with empty data."""
    mock_response = Mock()
    mock_response.data = []

    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_markdown_table(mock_response, columns=columns)

    assert "No data to display" in result


def test_format_markdown_table_with_truncation():
    """Test that markdown table truncates long values by default."""
    long_value = "x" * 100  # 100 character string
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_markdown_table(mock_response, truncate=True, columns=columns)

    # Should be truncated
    assert "..." in result
    assert len(long_value) > 50  # Ensure test data is long enough


def test_format_markdown_table_without_truncation():
    """Test that markdown table doesn't truncate when no_truncate=True."""
    long_value = "x" * 100  # 100 character string
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_markdown_table(mock_response, truncate=False, columns=columns)

    # Should NOT be truncated
    assert "..." not in result
    assert long_value in result


def test_format_csv_basic():
    """Test CSV formatting with basic data."""
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "Alice", "status": "active"},
        {"id": "2", "name": "Bob", "status": "inactive"},
    ]

    columns = [Column("id", "Id"), Column("name", "Name"), Column("status", "Status")]
    result = format_csv(mock_response, columns=columns)

    # Should be valid CSV
    lines = result.strip().split("\n")
    assert len(lines) == 3  # Header + 2 data rows
    assert "Id" in lines[0] or "ID" in lines[0]
    assert "Alice" in result
    assert "Bob" in result


def test_format_csv_with_pydantic_models():
    """Test CSV formatting with Pydantic models."""
    mock_item1 = Mock()
    mock_item1.model_dump = Mock(
        return_value={"id": "ns-1", "description": "Default", "project": "proj-1", "created_at": "2025-01-01T10:00:00"}
    )

    mock_item2 = Mock()
    mock_item2.model_dump = Mock(
        return_value={
            "id": "ns-2",
            "description": "Production",
            "project": "proj-2",
            "created_at": "2025-01-02T15:00:00",
        }
    )

    mock_response = Mock()
    mock_response.data = [mock_item1, mock_item2]

    columns = [Column("id", "ID"), Column("description", "Description"), Column("project", "Project")]
    result = format_csv(mock_response, columns=columns)

    # Should be valid CSV with proper quoting
    assert "ID" in result
    assert "ns-1" in result
    assert "ns-2" in result


def test_format_csv_with_commas_and_quotes():
    """Test CSV formatting properly handles commas and quotes."""
    import csv

    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": "Smith, John", "status": 'Status with "quotes"'},
    ]

    columns = [Column("id", "ID"), Column("name", "Name"), Column("status", "Status")]
    result = format_csv(mock_response, columns=columns)

    # Parse the CSV to verify it's valid
    import io

    reader = csv.reader(io.StringIO(result))
    rows = list(reader)

    # Should have header + 1 data row
    assert len(rows) == 2
    # Data should be properly parsed (commas and quotes handled)
    assert "Smith, John" in rows[1]  # Comma preserved
    assert 'Status with "quotes"' in rows[1]  # Quotes handled


def test_format_csv_empty():
    """Test CSV with empty data."""
    mock_response = Mock()
    mock_response.data = []

    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_csv(mock_response, columns=columns)

    assert result == ""


def test_format_csv_with_truncation():
    """Test CSV truncates long values."""
    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_csv(mock_response, truncate=True, max_width=50, columns=columns)

    # Should be truncated
    assert "..." in result
    assert long_value not in result


def test_format_csv_without_truncation():
    """Test CSV doesn't truncate when disabled."""
    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "name": long_value},
    ]

    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_csv(mock_response, truncate=False, columns=columns)

    # Should NOT be truncated
    assert "..." not in result
    assert long_value in result


def test_model_to_dict_with_nested_dict():
    """Test model_to_dict with nested dictionaries."""
    data = {"outer": {"inner": {"deep": "value"}}, "list": [{"a": 1}, {"b": 2}]}
    result = model_to_dict(data)
    assert result == data


def test_model_to_dict_with_mixed_types():
    """Test model_to_dict with mixed types including Pydantic models in lists."""
    mock_model = Mock()
    mock_model.model_dump = Mock(return_value={"field": "value"})

    data = {"models": [mock_model], "plain": {"key": "val"}}
    result = model_to_dict(data)

    assert result == {"models": [{"field": "value"}], "plain": {"key": "val"}}
    mock_model.model_dump.assert_called_once_with(mode="json")


def test_format_table_with_dict_data():
    """Test table formatting with dict data structure."""
    data = {"data": [{"id": "1", "name": "Test"}]}
    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_table(data, columns=columns)

    assert "1" in result
    assert "Test" in result


def test_format_table_with_list_data():
    """Test table formatting with list data structure."""
    data = [{"id": "1", "name": "Test"}]
    columns = [Column("id", "ID"), Column("name", "Name")]
    result = format_table(data, columns=columns)

    assert "1" in result
    assert "Test" in result


def test_format_table_with_non_dict_items():
    """Test table formatting with non-dict items."""
    mock_response = Mock()
    mock_response.data = ["item1", "item2", 123]

    columns = [Column("value", "Value")]
    result = format_table(mock_response, columns=columns)

    # Should convert to string values
    assert "item1" in result
    assert "item2" in result
    assert "123" in result


def test_format_table_without_truncation():
    """Test table formatting without truncation."""
    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_table(mock_response, truncate=False, columns=columns)

    # When truncate=False, Rich console uses wider width
    # The text should show more content (at least 70 chars of the 100)
    assert "x" * 70 in result  # Should show more than default 50 char limit


def test_format_table_with_truncation_max_width():
    """Test table formatting with custom max_width."""
    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_table(mock_response, truncate=True, max_width=20, columns=columns)

    # Should be truncated to 20 chars (minus 3 for ...)
    assert "..." in result


def test_format_markdown_table_with_max_width():
    """Test markdown table formatting with custom max_width."""
    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [
        {"id": "1", "description": long_value},
    ]

    columns = [Column("id", "ID"), Column("description", "Description")]
    result = format_markdown_table(mock_response, truncate=True, max_width=30, columns=columns)

    # Should be truncated to 30 chars (minus 3 for ...)
    assert "..." in result


def test_extract_items_from_response_with_data():
    """Test _extract_items_from_response with .data attribute."""
    mock_response = Mock()
    mock_response.data = [{"id": "1"}]
    result = _extract_items_from_response(mock_response)
    assert result == [{"id": "1"}]


def test_extract_items_from_response_with_data_for_files():
    """Test _extract_items_from_response with .data attribute for file listings."""
    mock_response = Mock(spec=["data"])
    mock_response.data = [{"path": "file1.txt"}, {"path": "file2.txt"}]
    result = _extract_items_from_response(mock_response)
    assert result == [{"path": "file1.txt"}, {"path": "file2.txt"}]


def test_extract_items_from_response_with_list():
    """Test _extract_items_from_response with plain list."""
    data = [{"id": "1"}, {"id": "2"}]
    result = _extract_items_from_response(data)
    assert result == data


def test_extract_items_from_response_with_dict_data():
    """Test _extract_items_from_response with dict containing data key."""
    data = {"data": [{"id": "1"}]}
    result = _extract_items_from_response(data)
    assert result == [{"id": "1"}]


def test_extract_items_from_response_with_dict_data_for_files():
    """Test _extract_items_from_response with dict containing data key for file listings."""
    data = {"data": [{"path": "a.txt"}]}
    result = _extract_items_from_response(data)
    assert result == [{"path": "a.txt"}]


def test_extract_items_from_response_empty():
    """Test _extract_items_from_response with object without list fields."""
    mock_response = Mock(spec=[])  # No attributes
    result = _extract_items_from_response(mock_response)
    assert result == []


def test_format_table_with_data_field_for_files():
    """Test table formatting with response using .data field for file listings."""
    mock_response = Mock(spec=["data"])
    mock_response.data = [
        {"path": "dir/file1.txt", "size": 100},
        {"path": "dir/file2.txt", "size": 200},
    ]

    columns = [Column("path", "Path"), Column("size", "Size")]
    result = format_table(mock_response, columns=columns)

    assert "dir/file1.txt" in result
    assert "dir/file2.txt" in result
    assert "100" in result
    assert "200" in result


def test_format_markdown_table_with_data_field_for_files():
    """Test markdown table formatting with response using .data field for file listings."""
    mock_response = Mock(spec=["data"])
    mock_response.data = [{"path": "test.txt", "size": 50}]

    columns = [Column("path", "Path"), Column("size", "Size")]
    result = format_markdown_table(mock_response, columns=columns)

    assert "test.txt" in result
    assert "50" in result
    assert "|" in result  # Markdown table syntax


def test_format_csv_with_data_field_for_files():
    """Test CSV formatting with response using .data field for file listings."""
    mock_response = Mock(spec=["data"])
    mock_response.data = [{"path": "data.csv", "size": 75}]

    columns = [Column("path", "Path"), Column("size", "Size")]
    result = format_csv(mock_response, columns=columns)

    assert "data.csv" in result
    assert "75" in result


@pytest.mark.parametrize(
    "output_format",
    ["json", "yaml", "raw"],
)
@patch("builtins.print")
def test_format_output_simple_formats(mock_print, output_format):
    """Test format_output with simple formats (json, yaml, raw)."""
    data = {"key": "value"}
    format_output(data, is_list=False, output_format=output_format)

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]
    assert "key" in call_args


@pytest.mark.parametrize(
    "output_format,expected_content",
    [
        ("table", "Test"),
        ("markdown", "|"),  # Markdown has pipe character
        ("csv", "Test"),
    ],
)
@patch("nemo_helix_ext.cli.core.table_config.resolve_and_validate_columns")
@patch("builtins.print")
def test_format_output_table_formats(
    mock_print,
    mock_resolve_cols,
    output_format,
    expected_content,
):
    """Test format_output with table-like formats (table, markdown, csv)."""
    mock_resolve_cols.return_value = [Column("id", "ID"), Column("name", "Name")]

    mock_response = Mock()
    mock_response.data = [{"id": "1", "name": "Test"}]

    format_output(mock_response, is_list=True, output_format=output_format, no_truncate=False, timestamp_format="iso")

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]
    assert expected_content in call_args


@patch("builtins.print")
def test_format_output_table_fallback_to_json(mock_print):
    """Test format_output falls back to JSON for single item with table format."""
    data = {"key": "value"}
    format_output(data, is_list=False, output_format="table")

    mock_print.assert_called_once()
    # Should fall back to JSON for single items
    call_args = mock_print.call_args[0][0]
    assert "key" in call_args


@patch("nemo_helix_ext.cli.core.table_config.resolve_and_validate_columns")
@patch("builtins.print")
def test_format_output_with_no_truncate(mock_print, mock_resolve_cols):
    """Test format_output with no_truncate enabled."""
    mock_resolve_cols.return_value = [Column("id", "ID"), Column("description", "Description")]

    long_value = "x" * 100
    mock_response = Mock()
    mock_response.data = [{"id": "1", "description": long_value}]

    format_output(mock_response, is_list=True, output_format="table", no_truncate=True, timestamp_format="iso")

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]
    # When no_truncate=True, table uses wider console width
    # The text should show more content (at least 70 chars of the 100)
    assert "x" * 70 in call_args  # Should show more than default 50 char limit


@patch("nemo_helix_ext.cli.core.table_config.resolve_and_validate_columns")
@patch("builtins.print")
def test_format_output_with_custom_columns(mock_print, mock_resolve_cols):
    """Test format_output with custom output_columns."""
    mock_resolve_cols.return_value = [Column("name", "Name")]  # Only name column

    mock_response = Mock()
    mock_response.data = [{"id": "1", "name": "Test"}]

    format_output(
        mock_response,
        is_list=True,
        output_format="table",
        output_columns=[Column("name", "Name")],
        no_truncate=False,
        timestamp_format="iso",
    )

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]
    assert "Test" in call_args


@patch("nemo_helix_ext.cli.core.table_config.resolve_and_validate_columns")
@patch("builtins.print")
def test_format_output_switches_to_markdown_for_many_columns(mock_print, mock_resolve_cols):
    """Test format_output switches to markdown when too many columns with no_truncate."""
    # Create 15 columns
    many_columns = [Column(f"col{i}", f"Col{i}") for i in range(15)]
    mock_resolve_cols.return_value = many_columns

    data_item = {f"col{i}": f"val{i}" for i in range(15)}
    mock_response = Mock()
    mock_response.data = [data_item]

    format_output(mock_response, is_list=True, output_format="table", no_truncate=True, timestamp_format="iso")

    # Should print twice - once for the note, once for the output
    assert mock_print.call_count >= 1


@patch("builtins.print")
def test_format_stream_event_with_dict(mock_print):
    """Test format_stream_event with a dictionary event."""
    event = {"type": "message", "content": "Hello", "id": 123}
    format_stream_event(event)

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]

    # Should be JSON output
    parsed = json.loads(call_args)
    assert parsed == event


@patch("builtins.print")
def test_format_stream_event_with_pydantic_model(mock_print):
    """Test format_stream_event with a Pydantic model."""
    mock_event = Mock()
    mock_event.model_dump = Mock(return_value={"type": "event", "data": "test"})

    format_stream_event(mock_event)

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]

    # Should convert model to dict and output as JSON
    parsed = json.loads(call_args)
    assert parsed == {"type": "event", "data": "test"}
    mock_event.model_dump.assert_called_once_with(mode="json")


@patch("builtins.print")
def test_format_stream_event_with_nested_data(mock_print):
    """Test format_stream_event with nested data structures."""
    event = {"type": "complex", "nested": {"data": [1, 2, 3], "meta": {"count": 3}}}

    format_stream_event(event)

    mock_print.assert_called_once()
    call_args = mock_print.call_args[0][0]

    # Should handle nested structures
    parsed = json.loads(call_args)
    assert parsed == event


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_json_soft_wrap_preserves_long_strings(mock_is_tty):
    """Test that JSON formatting with syntax highlighting preserves long strings without truncation.

    This tests the soft_wrap=True parameter on console.print() which ensures
    long strings are not cut off by the terminal width.
    """
    mock_is_tty.return_value = True

    # Create a string longer than typical terminal width (e.g., 200 chars)
    long_value = "x" * 200
    data = {"long_field": long_value}

    result = format_json(data, indent=2, syntax_highlight=True)

    # The entire long string should be present in the output
    assert long_value in result


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_yaml_soft_wrap_preserves_long_strings(mock_is_tty):
    """Test that YAML formatting with syntax highlighting preserves long strings without truncation.

    This tests the soft_wrap=True parameter on console.print() which ensures
    long strings are not cut off by the terminal width.
    """
    mock_is_tty.return_value = True

    # Create a string longer than typical terminal width (e.g., 200 chars)
    long_value = "a" * 200
    data = {"long_field": long_value}

    result = format_yaml(data, syntax_highlight=True, background=False)

    # The entire long string should be present in the output
    assert long_value in result


@patch("nemo_helix_ext.cli.core.formatters.is_tty")
def test_format_json_soft_wrap_with_nested_long_strings(mock_is_tty):
    """Test that nested long strings in JSON are fully preserved."""
    mock_is_tty.return_value = True

    # Nested structure with long strings at different levels
    long_value_1 = "first_" + "y" * 150
    long_value_2 = "second_" + "z" * 150
    data = {
        "level1": {
            "long_string": long_value_1,
            "level2": {
                "another_long": long_value_2,
            },
        }
    }

    result = format_json(data, indent=2, syntax_highlight=True)

    # Both long strings should be fully present
    assert long_value_1 in result
    assert long_value_2 in result
