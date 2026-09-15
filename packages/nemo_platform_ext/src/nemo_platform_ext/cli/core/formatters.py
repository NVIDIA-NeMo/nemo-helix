# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output formatting for the NeMo CLI."""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import click
import yaml
from nemo_platform_plugin.client.response import NemoResponse
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from nemo_platform_ext.cli.core.api import is_tty
from nemo_platform_ext.cli.core.help_formatter import add_warning
from nemo_platform_ext.cli.core.timestamp_formatter import format_timestamp
from nemo_platform_ext.cli.core.types import ListOutputFormat

# Maximum number of columns to display in table format with --no-truncate
# before automatically switching to markdown format for better readability
MAX_TABLE_COLUMNS_WITHOUT_TRUNCATE = 10


@dataclass
class Column:
    """Table column definition.

    Attributes:
        field: Field path to extract value (e.g., "name", "status", "{namespace}/{name}")
        header: Optional custom header. If None, field is used as header.
    """

    field: str
    header: str | None = None

    @property
    def display_name(self) -> str:
        """Get the display name for the column header."""
        return self.header if self.header is not None else self.field


# Timestamp fields that should be formatted
TIMESTAMP_FIELDS = {
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
    "finished_at",
}

# Fields that should never be truncated
NO_TRUNCATE_FIELDS = {
    "name",
    "id",
    "{namespace}/{name}",  # Computed OpenAI path field
}

# Common field names for list items in API responses.
# Checked in order via getattr. Callables (e.g. NemoPaginatedResponse.items)
# are invoked; plain attributes/properties are returned directly.
LIST_ITEM_FIELDS = ["data", "items"]


def _extract_items_from_response(data: Any) -> list[Any]:
    """Extract list items from an API response.

    Handles various response formats:
    - Paginated responses with .items() method (NemoPaginatedResponse)
    - Paginated responses with .data attribute (legacy Page)
    - Plain lists
    - Dict responses

    Args:
        data: API response object

    Returns:
        List of items from the response
    """
    if isinstance(data, list):
        return data

    return list(_iter_items_from_response(data))


def _iter_items_from_response(data: Any) -> Iterator[Any]:
    """Iterate list items from an API response without materializing callables."""
    if isinstance(data, list):
        yield from data
        return

    for field in LIST_ITEM_FIELDS:
        if isinstance(data, dict):
            if field in data:
                items = data[field]
            else:
                continue
        else:
            items = getattr(data, field, None)
            if items is None:
                continue
            items = items() if callable(items) else items

        yield from items
        return


def unwrap_response(data: Any) -> Any:
    """Return the parsed body of a typed entity response, or *data* unchanged."""
    if isinstance(data, NemoResponse):
        return data.data()
    return data


def _to_dict_items(items: list[Any]) -> list[dict[str, Any]]:
    """Convert items to dicts (Pydantic models, dicts, or fallback to {value: str})."""
    result = []
    for item in items:
        if hasattr(item, "model_dump"):
            result.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append({"value": str(item)})
    return result


def get_nested_value(obj: Any, path: str) -> str:
    """
    Get a nested value from an object using dot notation.

    Args:
        obj: Object to extract value from (dict or object with attributes)
        path: Dot-notation path (e.g., "metadata.name") or computed path (e.g., "{namespace}/{name}")

    Returns:
        String representation of the value, or empty string if not found
    """
    # Handle computed fields (e.g., "{namespace}/{name}")
    if "{" in path and "}" in path:
        # Extract field names from the pattern
        field_pattern = re.findall(r"\{([^}]+)\}", path)
        result = path
        for field_name in field_pattern:
            # Get the field value recursively
            field_value = get_nested_value(obj, field_name)
            # Replace the placeholder with the value
            result = result.replace(f"{{{field_name}}}", field_value)
        return result

    parts = path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return ""

        # Try dict access
        if isinstance(current, dict):
            current = current.get(part)
        # Try attribute access
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return ""

    # Convert to string, handling None
    if current is None:
        return ""

    if isinstance(current, (dict, list)):
        return json.dumps(current, ensure_ascii=False)

    return str(current)


def is_timestamp_field(field_path: str) -> bool:
    """
    Check if a field path represents a timestamp field.

    Args:
        field_path: Field path (e.g., "created_at", "metadata.updated_at")

    Returns:
        True if this is a timestamp field
    """
    # Get the last part of the path (e.g., "updated_at" from "metadata.updated_at")
    field_name = field_path.split(".")[-1]
    return field_name in TIMESTAMP_FIELDS


def should_truncate_field(field_path: str) -> bool:
    """
    Check if a field should be truncated in table output.

    Args:
        field_path: Field path (e.g., "name", "description", "{namespace}/{name}")

    Returns:
        False if this field should never be truncated, True otherwise
    """
    # Check exact match
    if field_path in NO_TRUNCATE_FIELDS:
        return False

    # Get the last part of the path (e.g., "name" from "metadata.name")
    field_name = field_path.split(".")[-1]
    return field_name not in NO_TRUNCATE_FIELDS


def model_to_dict(obj: Any) -> Any:
    """
    Convert a Pydantic model or other object to a JSON-serializable dict.

    Args:
        obj: Object to convert (can be a model, list, dict, etc.)

    Returns:
        JSON-serializable representation
    """
    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    # Handle lists
    elif isinstance(obj, list):
        return [model_to_dict(item) for item in obj]
    # Handle dicts
    elif isinstance(obj, dict):
        return {k: model_to_dict(v) for k, v in obj.items()}
    # Return primitives as-is
    else:
        return obj


def format_json(
    data: Any,
    indent: int | None = 2,
    syntax_highlight: bool = True,
    background: bool = False,
) -> str:
    """
    Format data as JSON with optional syntax highlighting.

    Args:
        data: Data to format
        indent: Number of spaces for indentation
        syntax_highlight: Whether to apply syntax highlighting (only in TTY)
        background: Whether to include background color in syntax highlighting

    Returns:
        Formatted JSON string
    """
    # Convert to dict if it's a Pydantic model
    json_data = model_to_dict(data)

    # Serialize to JSON
    json_str = json.dumps(json_data, indent=indent, ensure_ascii=False)

    # Apply syntax highlighting if requested and in TTY
    if syntax_highlight and is_tty():
        console = Console()
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai" if background else "ansi_dark",
            line_numbers=False,
            background_color="default" if not background else None,
        )

        # Capture the output instead of printing directly
        with console.capture() as capture:
            console.print(syntax, soft_wrap=True)

        return capture.get()
    else:
        return json_str


def iter_json_lines(data: Any, *, is_list: bool = False) -> Iterator[str]:
    """Return newline-delimited JSON records for streaming-style output."""
    records = _iter_items_from_response(data) if is_list else iter((data,))
    for record in records:
        yield json.dumps(model_to_dict(record), ensure_ascii=False, separators=(",", ":"))


def validate_stream_output_format(output_format: str | None, stream: bool) -> None:
    """Validate output format compatibility before streaming data acquisition starts."""
    if stream and output_format not in {"json", "raw"}:
        raise click.UsageError("--stream requires --output json or --output raw.")


def format_yaml(
    data: Any,
    syntax_highlight: bool = True,
    background: bool = False,
) -> str:
    """
    Format data as YAML with optional syntax highlighting.

    Args:
        data: Data to format
        syntax_highlight: Whether to apply syntax highlighting (only in TTY)
        background: Whether to include background color in syntax highlighting

    Returns:
        Formatted YAML string
    """
    # Convert to dict if it's a Pydantic model
    yaml_data = model_to_dict(data)

    # Serialize to YAML
    yaml_str = yaml.dump(
        yaml_data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )

    # Apply syntax highlighting if requested and in TTY
    if syntax_highlight and is_tty():
        console = Console()
        syntax = Syntax(
            yaml_str,
            "yaml",
            theme="monokai" if background else "ansi_dark",
            line_numbers=False,
            background_color="default" if not background else None,
        )

        # Capture the output instead of printing directly
        with console.capture() as capture:
            console.print(syntax, soft_wrap=True)

        return capture.get()
    else:
        return yaml_str


def _format_table_cell(
    value: str,
    col: Column,
    truncate: bool,
    max_width: int,
    timestamp_format: str,
) -> str:
    """Format a single table cell: timestamp, empty placeholder, padding, truncation."""
    if is_timestamp_field(col.field):
        value = format_timestamp(value, format_type=timestamp_format)
    if not truncate and (value == "" or value is None):
        value = "-"
    if not truncate and len(value) < len(col.display_name):
        value = value.ljust(len(col.display_name))
    if truncate and should_truncate_field(col.field) and len(value) > max_width:
        value = value[: max_width - 3] + "..."
    return value


def format_table(
    data: Any,
    columns: list[Column],
    truncate: bool = True,
    max_width: int = 50,
    timestamp_format: str = "iso",
    wrap: bool = False,
    wrap_max_width: int | None = 90,
) -> str:
    """
    Format data as a table.

    Args:
        data: Data to format (should have a 'data' attribute with list of items)
        columns: List of Column objects defining table columns
        truncate: Whether to truncate long values (default: True)
        max_width: Maximum width for truncated values (default: 50)
        timestamp_format: Format for timestamp fields ("iso", "relative", "datetime")
        wrap: When True, render long cells with word-wrap (Rich `overflow="fold"`)
            instead of truncating with "...". Implies `truncate=False` for the
            cell-formatting step (no string-level clipping), but still bounds the
            column visually via `wrap_max_width`. Use this for prose-ish columns
            (descriptions, summaries) so long entries fold across multiple lines
            instead of either being cut at 50 chars or stretching the table wide.
        wrap_max_width: Per-column character cap for wrapped columns. `None`
            removes the cap and lets Rich use the full terminal width — typically
            paired with `--no-truncate` for "show everything, wrap to terminal".
    Returns:
        Formatted table string
    """
    items = _extract_items_from_response(data)
    if not items:
        return "No data to display"

    # In wrap mode, never clip cell strings — Rich does the visual wrapping.
    cell_truncate = truncate and not wrap

    dict_items = _to_dict_items(items)
    formatted_rows = [
        [
            _format_table_cell(get_nested_value(item, col.field), col, cell_truncate, max_width, timestamp_format)
            for col in columns
        ]
        for item in dict_items
    ]

    if not truncate and not wrap:
        table = Table(show_header=True, header_style="bold cyan", padding=(0, 1), collapse_padding=False)
    else:
        table = Table(show_header=True, header_style="bold cyan")

    column_widths = []
    for i, col in enumerate(columns):
        truncate_col = should_truncate_field(col.field)
        is_wrap_col = wrap and truncate_col
        if is_wrap_col:
            # Width is determined by Rich at render time (it folds the cell to
            # fit whatever space is left after the non-wrap columns). We track
            # 0 here so the table_width pre-budget below accounts only for
            # fixed-size columns.
            width = 0
        elif truncate_col:
            width = max_width
        else:
            width = max(
                len(col.display_name),
                max((len(row[i]) for row in formatted_rows), default=0),
            )
        column_widths.append(width)
        if is_wrap_col:
            # Wrappable, prose-ish columns: fold long content. `max_width=None`
            # lets Rich use whatever room is left after the other columns;
            # otherwise we cap so the column doesn't push the table wide.
            table.add_column(col.display_name, overflow="fold", max_width=wrap_max_width)
        elif truncate_col:
            table.add_column(col.display_name)
        else:
            table.add_column(col.display_name, min_width=width, overflow="ignore")

    for row in formatted_rows:
        table.add_row(*row)

    try:
        term_width = Console().width
    except Exception:
        term_width = 80
    table_width = sum(column_widths) + 3 * max(0, len(columns) - 1) + 2
    if wrap:
        # In wrap mode, always render at the actual terminal width — Rich will
        # fold the wrap column to fit whatever room is left. Anything wider than
        # the terminal causes the terminal to hard-wrap our box-drawing chars.
        console_width = term_width
    elif not truncate:
        console_width = 10000
    else:
        console_width = max(term_width, table_width)
    console = Console(width=console_width, legacy_windows=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def format_markdown_table(
    data: Any,
    columns: list[Column],
    truncate: bool = True,
    max_width: int = 50,
    timestamp_format: str = "iso",
) -> str:
    """
    Format data as a markdown table with nice alignment.

    Args:
        data: Data to format (should have a 'data' attribute with list of items)
        columns: List of Column objects defining table columns
        truncate: Whether to truncate long values (default: True)
        max_width: Maximum width for truncated values (default: 50)
        timestamp_format: Format for timestamp fields ("iso", "relative", "datetime")
    Returns:
        Markdown table string with aligned columns
    """
    items = _extract_items_from_response(data)
    if not items:
        return "No data to display"

    dict_items = _to_dict_items(items)

    # Extract all values and calculate column widths
    all_rows = []
    for item in dict_items:
        row_values = []
        for col in columns:
            value = get_nested_value(item, col.field)
            # Format timestamps if this is a timestamp field
            if is_timestamp_field(col.field):
                value = format_timestamp(value, format_type=timestamp_format)
            # Normalize all newline forms before truncating so Markdown escape
            # sequences are always added in full.
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            # Truncate long values if requested (unless it's a protected field)
            if truncate and should_truncate_field(col.field) and len(value) > max_width:
                value = value[: max_width - 3] + "..."
            # Keep each item on a single Markdown table row and escape cell delimiters.
            value = value.replace("\n", "<br>").replace("|", "\\|")
            row_values.append(value)
        all_rows.append(row_values)

    # Calculate max width for each column
    col_widths = []
    for i, col in enumerate(columns):
        # Start with header width
        max_col_width = len(col.display_name)
        # Check all row values
        for row in all_rows:
            if i < len(row):
                max_col_width = max(max_col_width, len(row[i]))
        col_widths.append(max_col_width)

    # Build markdown table with aligned columns
    lines = []

    # Header row with padding
    header_parts = []
    for i, col in enumerate(columns):
        header_parts.append(col.display_name.ljust(col_widths[i]))
    header = "| " + " | ".join(header_parts) + " |"
    lines.append(header)

    # Separator row with proper width
    separator_parts = []
    for width in col_widths:
        separator_parts.append("-" * max(3, width))
    separator = "| " + " | ".join(separator_parts) + " |"
    lines.append(separator)

    # Data rows with padding
    for row in all_rows:
        row_parts = []
        for i, value in enumerate(row):
            row_parts.append(value.ljust(col_widths[i]))
        row_line = "| " + " | ".join(row_parts) + " |"
        lines.append(row_line)

    return "\n".join(lines)


def format_csv(
    data: Any,
    columns: list[Column],
    truncate: bool = True,
    max_width: int = 50,
    timestamp_format: str = "iso",
) -> str:
    """
    Format data as CSV.

    Args:
        data: Data to format (should have a 'data' attribute with list of items)
        columns: List of Column objects defining table columns
        truncate: Whether to truncate long values (default: True)
        max_width: Maximum width for truncated values (default: 50)
        timestamp_format: Format for timestamp fields ("iso", "relative", "datetime")
    Returns:
        CSV string
    """
    items = _extract_items_from_response(data)
    if not items:
        return ""

    dict_items = _to_dict_items(items)

    # Build CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header row
    header = [col.display_name for col in columns]
    writer.writerow(header)

    # Write data rows
    for item in dict_items:
        row = []
        for col in columns:
            value = get_nested_value(item, col.field)
            # Format timestamps if this is a timestamp field
            if is_timestamp_field(col.field):
                value = format_timestamp(value, format_type=timestamp_format)
            # Truncate long values if requested (unless it's a protected field)
            if truncate and should_truncate_field(col.field) and len(value) > max_width:
                value = value[: max_width - 3] + "..."
            row.append(value)
        writer.writerow(row)

    return output.getvalue()


# TODO: create separate methods for list outputs and other outputs
def format_output(
    data: Any,
    *,
    is_list: bool = False,
    output_format: str | None = None,
    output_columns: Literal["all"] | str | list[Column] | None = None,
    indent: int = 2,
    no_truncate: bool | None = None,
    timestamp_format: str | None = None,
    wrap: bool = False,
    wrap_max_width: int | None = 90,
    stream: bool = False,
) -> None:
    """
    Format and print output to stdout.

    Args:
        data: Data to output
        is_list: Whether the data is a list of items
        output_columns: List of Column objects for table formatting.
                Required when is_list=True and output_format is table/markdown/csv.
                Can be a comma-separated string. Special values: "all" - uses all columns.
        output_format: Output format ("json", "yaml", "table", "markdown", "csv", "raw").
                      If None, uses state or falls back to "table".
        indent: Number of spaces for indentation (for JSON)
        no_truncate: Whether to disable truncation in table/markdown/csv formats.
                    If None, uses state or falls back to False.
        timestamp_format: Format for timestamps in table/markdown/csv ("iso", "relative", "datetime").
                         If None, uses state or falls back to "iso".
        wrap: When True and `output_format == "table"`, render long cells with
                word-wrap (Rich `overflow="fold"`) instead of truncating with
                "...". Use this for prose-ish columns (descriptions, summaries).
                When `no_truncate` is also set, the wrap cap is removed and Rich
                uses the full terminal width.
        wrap_max_width: Per-column character cap for wrapped columns when
                `wrap=True` and `no_truncate=False`. Defaults to 90.
        stream: Emit newline-delimited JSON records. List responses emit one
                record per item; entity responses emit one record.
    """
    from nemo_platform_ext.cli.core.table_config import resolve_and_validate_columns, validate_output_columns

    data = unwrap_response(data)
    timestamp_format = timestamp_format or "iso"

    if stream:
        validate_stream_output_format(output_format, stream)
        for line in iter_json_lines(data, is_list=is_list):
            click.echo(line)
            sys.stdout.flush()
        return

    if not is_list and output_format in {"table", "markdown", "csv"}:
        output_format = "json"  # Fallback to JSON for single items in these formats

    # Automatically resolve and validate output columns for list outputs with table-like formats
    if is_list and output_format in {"table", "markdown", "csv"}:
        if output_format == "table" and output_columns == "all":
            add_warning("For all columns, use --output-format json for readable output.")
        # If it's a string, convert to list of tuples using validate_output_columns
        if isinstance(output_columns, str):
            output_columns = validate_output_columns(output_columns)

        if output_columns is None:
            output_columns = "all"  # Default to all columns if not specified

        # Now resolve and validate it (handles "all" keyword and validates field names)
        output_columns = resolve_and_validate_columns(output_columns, data)

    # Determine truncate setting (inverse of no_truncate)
    truncate = not no_truncate
    timestamp_format = timestamp_format or "iso"

    # The "use --no-truncate to see full values" hint only makes sense when
    # the table actually clips with "..."; in wrap mode nothing is hidden.
    if is_list and output_format == "table" and truncate and not wrap and _extract_items_from_response(data):
        add_warning("Use --no-truncate to see full values.")

    # Rich's Table has limitations with many columns and no_truncate
    # Automatically switch to markdown format for better readability
    if (
        output_format == "table"
        and not truncate
        and isinstance(output_columns, list)
        and len(output_columns) > MAX_TABLE_COLUMNS_WITHOUT_TRUNCATE
    ):
        print(
            f"Note: Switching to markdown format for better display of {len(output_columns)} columns with --no-truncate",
            file=sys.stderr,
        )
        print("      You can also use --output-format csv or --output-format json for wide tables.\n", file=sys.stderr)
        output_format = "markdown"

    if output_format == "json":
        # JSON with syntax highlighting (if TTY), no background
        output = format_json(data, indent=indent, syntax_highlight=True, background=False)
        print(output)
    elif output_format == "yaml":
        # YAML with syntax highlighting (if TTY), no background
        output = format_yaml(data, syntax_highlight=True, background=False)
        print(output)
    elif output_format == "table":
        # Table format. When wrapping is on and --no-truncate is set, drop the
        # per-column cap so wrapping uses the full terminal width.
        if not isinstance(output_columns, list):
            raise ValueError("output columns must resolve to a list before formatting table output")
        effective_wrap_max_width = None if (wrap and not truncate) else wrap_max_width
        output = format_table(
            data,
            columns=output_columns,
            truncate=truncate,
            timestamp_format=timestamp_format,
            wrap=wrap,
            wrap_max_width=effective_wrap_max_width,
        )
        print(output)
    elif output_format == "markdown":
        # Markdown table format
        if not isinstance(output_columns, list):
            raise ValueError("output columns must resolve to a list before formatting markdown output")
        output = format_markdown_table(
            data, columns=output_columns, truncate=truncate, timestamp_format=timestamp_format
        )
        print(output)
    elif output_format == "csv":
        # CSV format
        if not isinstance(output_columns, list):
            raise ValueError("output columns must resolve to a list before formatting CSV output")
        output = format_csv(data, columns=output_columns, truncate=truncate, timestamp_format=timestamp_format)
        print(output, end="")  # CSV already includes newlines
    elif output_format == "raw":
        # Raw JSON without highlighting
        output = format_json(data, indent=None, syntax_highlight=False, background=False)
        print(output)
    else:
        # Default to JSON
        output = format_json(data, indent=indent, syntax_highlight=True, background=False)
        print(output)


def format_stream_event(event: Any) -> None:
    """
    Format and print a single streaming event.

    Args:
        event: Streaming event to format
    """
    # Convert to dict if it's a Pydantic model
    event_data = model_to_dict(event)

    # For streaming, output each event as a line of JSON
    # with minimal formatting to maintain real-time feel
    json_str = json.dumps(event_data, ensure_ascii=False)

    # Use direct print for streaming to ensure immediate output
    print(json_str, flush=True)


def check_output_columns_with_format(
    output_columns: str | None,
    output_format: ListOutputFormat,
) -> None:
    """
    Check if --output-columns is used with formats that don't support it.

    Adds a warning if --output-columns is used with --output-format=code/json/yaml.

    Args:
        output_columns: The output columns option value
        output_format: The output format option value
    """
    if output_columns is not None:
        # Check if format is code, json, or yaml
        if output_format in ("code", "json", "yaml"):
            add_warning(
                f"Note: --output-columns is not used with `--output-format {output_format}`. "
                "This option only affects table/csv/markdown formats."
            )
