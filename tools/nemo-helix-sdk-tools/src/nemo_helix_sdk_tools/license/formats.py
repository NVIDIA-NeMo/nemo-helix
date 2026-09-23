# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multiple output format options for license reports.

This module provides different formatters for license data that can be:
- Machine-readable (JSON, JSONL, CSV)
- Human-readable (Table, Markdown)
- Or both
"""

import csv
import json
from io import StringIO


class LicenseFormatter:
    """Base class for license formatters."""

    def format(self, packages: list[dict[str, str]]) -> str:
        """Format package license data into the desired output format."""
        raise NotImplementedError


class JSONLFormatter(LicenseFormatter):
    """
    JSONL (JSON Lines) formatter - one JSON object per line.

    Pros:
    - Highly parsable
    - Can be streamed/processed line-by-line
    - Each line is a complete record
    - Easy to grep/search

    Example:
        {"name": "requests", "license": "APACHE-2.0", "compatible": true}
        {"name": "numpy", "license": "BSD-3-CLAUSE", "compatible": true}
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        lines = []
        for pkg in packages:
            obj = {
                "name": pkg["name"],
                "license": pkg["license"],
                "compatible": pkg["license"] not in ["UNKNOWN", "NON-STANDARD"],
            }
            lines.append(json.dumps(obj))
        return "\n".join(lines)


class CSVFormatter(LicenseFormatter):
    """
    CSV formatter for easy spreadsheet import.

    Pros:
    - Universal format
    - Easy Excel/Google Sheets import
    - Simple parsing
    - Good for sorting/filtering

    Example:
        Package,License,License URL
        requests,APACHE-2.0,https://github.com/psf/requests/blob/main/LICENSE
        numpy,BSD-3-CLAUSE,https://github.com/numpy/numpy/blob/main/LICENSE.txt
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=["Package", "License", "License URL"])
        writer.writeheader()

        for pkg in packages:
            writer.writerow(
                {
                    "Package": pkg["name"],
                    "License": pkg["license"],
                    "License URL": pkg.get("license_url", ""),
                }
            )

        return output.getvalue()


class CompactJSONFormatter(LicenseFormatter):
    """
    Compact JSON array format.

    Pros:
    - Standard JSON
    - Good for APIs
    - Compact

    Example:
        [
          {"name": "requests", "license": "APACHE-2.0", "compatible": true},
          {"name": "numpy", "license": "BSD-3-CLAUSE", "compatible": true}
        ]
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        output = []
        for pkg in packages:
            output.append(
                {
                    "name": pkg["name"],
                    "license": pkg["license"],
                    "compatible": pkg["license"] not in ["UNKNOWN", "NON-STANDARD"],
                }
            )
        return json.dumps(output, indent=2)


class MarkdownTableFormatter(LicenseFormatter):
    """
    Markdown table format.

    Pros:
    - Renders in GitHub/GitLab
    - Human-readable
    - Can be viewed in any markdown viewer
    - Still parsable with basic text tools

    Example:
        | Compatible | Package  | License      |
        |------------|----------|--------------|
        | ✔          | requests | APACHE-2.0   |
        | ✔          | numpy    | BSD-3-CLAUSE |
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        lines = []

        # Header
        lines.append("| Compatible | Package | License |")
        lines.append("|------------|---------|---------|")

        # Rows
        for pkg in packages:
            compat = "✔" if pkg["license"] not in ["UNKNOWN", "NON-STANDARD"] else "✘"
            name = pkg["name"]
            license_text = pkg["license"]

            lines.append(f"| {compat} | {name} | {license_text} |")

        return "\n".join(lines)


class CompactTextFormatter(LicenseFormatter):
    """
    Simple text format - one package per line.

    Pros:
    - Extremely simple
    - Easy to grep/search
    - Minimal size
    - Tab-separated for easy parsing

    Example:
        requests	APACHE-2.0	✔
        numpy	BSD-3-CLAUSE	✔
        problematic-pkg	UNKNOWN	✘
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        lines = []
        for pkg in packages:
            compat = "✔" if pkg["license"] not in ["UNKNOWN", "NON-STANDARD"] else "✘"
            name = pkg["name"]
            license_text = pkg["license"]

            lines.append(f"{name}\t{license_text}\t{compat}")

        return "\n".join(lines)


class RichTableFormatter(LicenseFormatter):
    """
    The current Rich table format (Unicode box drawing).

    Pros:
    - Beautiful in terminal
    - Clear visual structure
    - Easy to scan

    This is the existing format - kept for backward compatibility.
    """

    def format(self, packages: list[dict[str, str]]) -> str:
        output = []

        # Header
        output.append("┏━━━━━━━━━━━━┳" + "━" * 48 + "┳" + "━" * 40 + "┓")
        output.append("┃ Compatible ┃ Package" + " " * 40 + "┃ License(s)" + " " * 29 + "┃")
        output.append("┡━━━━━━━━━━━━╇" + "━" * 48 + "╇" + "━" * 40 + "┩")

        for pkg in packages:
            compat = "✔" if pkg["license"] not in ["UNKNOWN", "NON-STANDARD"] else "✘"
            name = pkg["name"][:50]
            license_text = pkg["license"]

            # Split license into chunks that fit in 40 chars
            if len(license_text) <= 40:
                output.append(f"│ {compat:<10} │ {name:<46} │ {license_text:<38} │")
            else:
                # First line with package name
                output.append(f"│ {compat:<10} │ {name:<46} │ {license_text[:40]:<38} │")
                # Additional lines for license overflow
                remaining = license_text[40:]
                while remaining:
                    chunk = remaining[:40]
                    remaining = remaining[40:]
                    output.append(f"│            │ {'':46} │ {chunk:<38} │")

        output.append(
            "┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩"
        )

        return "\n".join(output)


# Registry of available formatters
FORMATTERS = {
    "table": RichTableFormatter(),
    "jsonl": JSONLFormatter(),
    "json": CompactJSONFormatter(),
    "csv": CSVFormatter(),
    "markdown": MarkdownTableFormatter(),
    "text": CompactTextFormatter(),
}


def get_formatter(format_type: str) -> LicenseFormatter:
    """Get a formatter by name."""
    if format_type not in FORMATTERS:
        raise ValueError(f"Unknown format: {format_type}. Available: {', '.join(FORMATTERS.keys())}")
    return FORMATTERS[format_type]
