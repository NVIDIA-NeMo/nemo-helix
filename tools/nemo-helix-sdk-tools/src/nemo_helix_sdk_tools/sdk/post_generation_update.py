# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""License header maintenance for the Python SDK tree."""

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import typer
from nemo_helix_sdk_tools.sdk.core.common import get_sdk_dir

app = typer.Typer(name="post-generation", help="Post-generation update tool for NeMo Helix SDK.", no_args_is_help=True)


def get_license_header(file_type: Literal["python"] = "python") -> str:
    """
    Get the standard SPDX license header for NVIDIA files.
    """
    current_year = datetime.now().year
    license_header = f"""\
SPDX-FileCopyrightText: Copyright (c) {current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

    # Use a comment style based on the file type
    if file_type == "python":
        header = "\n".join("# " + line for line in license_header.splitlines()) + "\n\n"
    else:
        raise RuntimeError(f"Unsupported license style: {file_type}")

    # Remove trailing whitespace from all lines
    header = "\n".join(line.rstrip(" \t") for line in header.split("\n"))

    return header


def has_license_header(file_content: str) -> bool:
    """Check if file already has a license header."""
    lines = file_content.splitlines()
    if not lines:
        return False

    # Check first few lines for license header patterns
    first_lines = lines[:10]  # Check first 10 lines
    license_patterns = [
        r"SPDX-FileCopyrightText.*NVIDIA",
        r"Copyright.*NVIDIA",
        r"SPDX-License-Identifier",
    ]

    for line in first_lines:
        for pattern in license_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True

    return False


def should_add_license_header(file_path: Path) -> bool:
    """Determine if a file should have a license header added."""
    # Skip certain files
    skip_patterns = [
        "__pycache__",
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".egg-info",
        ".git",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
    ]

    # Skip if file path contains any skip patterns
    file_str = str(file_path)
    for pattern in skip_patterns:
        if pattern in file_str:
            return False

    # Only process Python files
    if file_path.suffix != ".py":
        return False

    # Skip certain specific files
    skip_files = []

    # Allow __init__.py files that are not in the root of the SDK
    if file_path.name in skip_files:
        return False

    return True


def add_license_header_to_file(file_path: Path, license_header: str) -> bool:
    """Add license header to a single file. Returns True if header was added."""
    try:
        # Read file content
        content = file_path.read_text(encoding="utf-8")

        # Check if license header already exists
        if has_license_header(content):
            return False

        # Handle shebang lines
        lines = content.splitlines(keepends=True)
        insert_pos = 0

        # If file starts with shebang, insert after it
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
            # Add empty line after shebang if there isn't one
            if len(lines) > 1 and not lines[1].strip() == "":
                license_header += "\n"

        # Insert license header
        if insert_pos < len(lines):
            lines.insert(insert_pos, license_header)
        else:
            lines.append(license_header)

        # Write back to file
        file_path.write_text("".join(lines), encoding="utf-8")
        return True

    except (UnicodeDecodeError, PermissionError) as e:
        typer.echo(f"  - Skipped {file_path} ({e})", err=True)
        return False


def process_license_headers(sdk_dir: Path) -> None:
    """Update license headers in all Python files in the SDK."""
    license_header = get_license_header()

    # File patterns to process
    patterns = ["**/*.py"]

    processed_files = 0
    updated_files = 0
    skipped_files = 0

    for pattern in patterns:
        for file_path in sdk_dir.glob(pattern):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if file shouldn't have license header
            if not should_add_license_header(file_path):
                continue

            processed_files += 1

            # Add license header
            if add_license_header_to_file(file_path, license_header):
                updated_files += 1
            else:
                skipped_files += 1

    typer.echo(f"  - Processed {processed_files} files")
    typer.echo(f"  - Updated {updated_files} files with license headers")
    typer.echo(f"  - Skipped {skipped_files} files (already had headers)")


@app.command()
def update_license_headers() -> None:
    """Update license headers in all Python files in the SDK."""
    typer.echo("Updating license headers in all Python files...")

    process_license_headers(get_sdk_dir())

    typer.echo("License headers update completed!")
