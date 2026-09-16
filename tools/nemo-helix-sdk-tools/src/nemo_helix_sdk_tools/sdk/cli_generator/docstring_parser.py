# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse Google-style docstrings for CLI generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from nhx.common.api.utils import parse_deep_object
from starlette.datastructures import QueryParams


@dataclass
class ParsedDocstring:
    """Parsed docstring with description and parameter descriptions."""

    description: str
    param_descriptions: dict[str, str]

    @classmethod
    def parse(cls, docstring: str | None) -> ParsedDocstring:
        """Parse a Google-style docstring.

        Extracts:
        - The main description (everything before 'Args:')
        - Parameter descriptions from the 'Args:' section

        Args:
            docstring: The raw docstring to parse

        Returns:
            ParsedDocstring with extracted information
        """
        if not docstring:
            return cls(description="", param_descriptions={})

        # Split on 'Args:' to separate description from parameters
        parts = re.split(r"\n\s*Args:\s*\n", docstring, maxsplit=1)

        # Extract description (first part, before Args:)
        description = parts[0].strip()

        param_descriptions: dict[str, str] = {}

        if len(parts) > 1:
            args_section = parts[1]
            # Parse parameter descriptions
            # Format: "  param_name: Description that can span\n      multiple lines."
            param_pattern = re.compile(
                r"^\s{2,4}(\w+):\s*(.+?)(?=\n\s{2,4}\w+:|\n\s{0,1}\S|\Z)",
                re.MULTILINE | re.DOTALL,
            )
            for match in param_pattern.finditer(args_section):
                param_name = match.group(1)
                param_desc = match.group(2).strip()
                # Normalize whitespace but preserve list item newlines
                param_desc = normalize_preserving_lists(param_desc)
                param_descriptions[param_name] = param_desc

        return cls(description=description, param_descriptions=param_descriptions)


def normalize_preserving_lists(text: str) -> str:
    """Normalize whitespace while preserving newlines before list items.

    Joins continuation lines (wrapped text) but keeps newlines before lines
    that start with '- ' (list items) or 'For example:'.
    """
    lines = text.split("\n")
    result_parts: list[str] = []
    current_part: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Check if this line should start a new paragraph
        if stripped.startswith("- ") or stripped.startswith("For example"):
            # Save current accumulated text
            if current_part:
                result_parts.append(" ".join(current_part))
                current_part = []
            # Add newline before this item
            if result_parts:
                result_parts.append("\n")
            current_part.append(stripped)
        elif stripped == "":
            # Blank line - might be paragraph break
            if current_part:
                result_parts.append(" ".join(current_part))
                current_part = []
        else:
            # Continuation line - join with previous
            current_part.append(stripped)

    # Don't forget the last part
    if current_part:
        result_parts.append(" ".join(current_part))

    return "".join(result_parts)


def transform_query_to_cli(description: str, param_name: str) -> str:
    """Transform HTTP query param examples to CLI syntax in a description.

    Transforms:
    - `?search[name]=imagenet` → `--search.name imagenet`
    - `?search[name]=imagenet&search[split]=train` → `--search.name imagenet --search.split train`
    - `?search[updated_at][start]=2024-01-01` → `--search '{"updated_at": {"start": "2024-01-01"}}'`

    Args:
        description: The help text to transform
        param_name: The parameter name (e.g., "search" or "filter")

    Returns:
        Transformed help text with CLI syntax examples
    """
    if not description:
        return description

    def transform_query_string(query: str) -> str | None:
        """Transform a full query string like ?search[name]=foo&search[split]=bar."""
        query = query.lstrip("?")

        query_params = QueryParams(query)
        options = parse_deep_object(param_name, query_params)
        cli_parts = []
        if not options:
            return None

        for key, val in options.items():
            if isinstance(val, dict):
                cli_parts.append(f"--{param_name} '{json.dumps(options, separators=(',', ':'))}'")
                # We dump all the options, so no need to continue processing
                break

            elif isinstance(val, list):
                cli_parts.append(f"--{param_name}.{key.replace('_', '-')} {','.join(val)}")
            else:
                cli_parts.append(f"--{param_name}.{key.replace('_', '-')} {val}")

        return " ".join(sorted(cli_parts)) if cli_parts else query

    # Find and replace query param patterns in the description
    result = description

    # Replace backtick-wrapped query strings
    def replace_backtick_query(match: re.Match) -> str:
        query = match.group(1)
        if f"{param_name}[" in query:
            transformed = transform_query_string(query)
            if transformed:
                return f"`{transformed}`"

        return match.group(0)

    # Match both `?query...` and `query...` patterns
    result = re.sub(
        r"`(\??[^`]*" + re.escape(param_name) + r"\[[^`]+)`",
        replace_backtick_query,
        result,
    )

    return result
