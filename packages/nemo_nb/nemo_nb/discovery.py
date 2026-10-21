# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and processing utilities for nemo-nb notebooks.

This module provides utilities for:
- Finding notebooks with the @nemo-nb: process marker
- Filtering out notebooks with the @nemo-nb: skip-test marker
- Expanding MyST literalinclude directives
- Detecting and filtering excluded files/directories
"""

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

# Directories to exclude from notebook discovery (build artifacts, generated files)
EXCLUDED_DIRS = {"_generated", "_build"}

# Files to exclude from notebook discovery (documentation metadata files)
EXCLUDED_FILENAMES = {"REDIRECTS.md", "README.md", "LINTING_FINDINGS.md"}

# Pattern to detect MyST literalinclude directives (multi-line with options)
# Captures: (leading whitespace), (path), (option lines and closing fence line)
# Option lines are :key: value; block ends at a line that is only whitespace + ```
RE_LITERALINCLUDE = re.compile(
    r"^(\s*)```\{literalinclude\}\s+([^\n]+)\n((?:(?!^\s*```\s*$).*\n)*?)^\s*```\s*$",
    re.MULTILINE,
)
# Option line: optional whitespace, :key: value
RE_LITERALINCLUDE_OPTION = re.compile(r"^\s*:(\w[-\w]*):\s*(.*)$", re.MULTILINE)


def is_excluded_file(file_path: Path) -> bool:
    """Check if a file should be excluded by name.

    Args:
        file_path: Path to check

    Returns:
        True if the file should be excluded
    """
    return file_path.name in EXCLUDED_FILENAMES


def is_in_excluded_dir(file_path: Path) -> bool:
    """Check if a file is inside an excluded directory.

    Args:
        file_path: Path to check

    Returns:
        True if the file is inside an excluded directory
    """
    return any(part in EXCLUDED_DIRS for part in file_path.parts)


def _parse_literalinclude_options(option_block: str) -> dict:
    """Parse :key: value options from a literalinclude directive body."""
    options = {}
    for line in option_block.splitlines():
        match = RE_LITERALINCLUDE_OPTION.match(line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            # Remove surrounding quotes from value if present
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            options[key] = value
    return options


def _apply_literalinclude_slice(content: str, start_after: str | None, end_before: str | None) -> str:
    """Extract a slice of content between start-after and end-before markers."""
    lines = content.splitlines(keepends=True)
    start_idx = 0
    end_idx = len(lines)
    if start_after:
        for i, line in enumerate(lines):
            if start_after in line:
                start_idx = i + 1
                break
    if end_before:
        for i, line in enumerate(lines):
            if i >= start_idx and end_before in line:
                end_idx = i
                break
    return "".join(lines[start_idx:end_idx])


def expand_literalincludes(content: str, base_path: Path) -> str:
    """Expand MyST literalinclude directives into fenced code blocks.

    Replaces ```{literalinclude} path ... ``` with the file content (optionally
    sliced by :start-after: / :end-before: and :dedent:) so that run_notebooks
    and the notebook converter see normal code blocks instead of directive text.

    Args:
        content: The markdown content to process
        base_path: Base directory for resolving relative include paths

    Returns:
        Content with literalinclude directives expanded to ```language\\n...\\n```
    """

    def replace_literalinclude(match):
        leading = match.group(1)
        include_path = match.group(2).strip()
        option_block = match.group(3)
        resolved_path = (base_path / include_path).resolve()
        try:
            if not resolved_path.exists():
                return match.group(0)
            raw = resolved_path.read_text(encoding="utf-8")
        except Exception:
            return match.group(0)
        options = _parse_literalinclude_options(option_block)
        start_after = options.get("start-after") or options.get("start_after")
        end_before = options.get("end-before") or options.get("end_before")
        dedent_value = options.get("dedent")
        language = (options.get("language") or "python").strip()
        # Sphinx/MyST use "output" for example output; use "text" so it doesn't run as code
        if language == "output":
            language = "text"
        body = _apply_literalinclude_slice(raw, start_after, end_before)
        if dedent_value is not None:
            if dedent_value == "" or dedent_value is True:
                body = textwrap.dedent(body)
            else:
                try:
                    n = int(dedent_value)
                    body = "\n".join(line[n:] if len(line) > n else line for line in body.split("\n"))
                except ValueError:
                    body = textwrap.dedent(body)
        body = body.rstrip()
        return f"{leading}```{language}\n{body}\n{leading}```"

    return RE_LITERALINCLUDE.sub(replace_literalinclude, content)


def _notebook_text(notebook_path: Path) -> str | None:
    """Read the combined markdown cell text from an .ipynb file."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_data = json.load(f)

        parts: list[str] = []
        for cell in notebook_data.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                parts.append("".join(source) if isinstance(source, list) else source)
        return "\n".join(parts)
    except Exception as e:
        print(f"Warning: Could not read {notebook_path}: {e}")
        return None


def _markdown_text(md_path: Path) -> str | None:
    """Read the full text of a markdown file."""
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: Could not read {md_path}: {e}")
        return None


def has_process_marker_notebook(notebook_path: Path) -> bool:
    """
    Check if a .ipynb notebook has the @nemo-nb: process marker.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if the marker is found, False otherwise
    """
    text = _notebook_text(notebook_path)
    return text is not None and "@nemo-nb: process" in text


def has_process_marker_markdown(md_path: Path) -> bool:
    """
    Check if a markdown file has the @nemo-nb: process marker.

    Args:
        md_path: Path to the markdown file

    Returns:
        True if the marker is found, False otherwise
    """
    text = _markdown_text(md_path)
    return text is not None and "@nemo-nb: process" in text


def has_skip_test_marker_notebook(notebook_path: Path) -> bool:
    """Check if a .ipynb notebook has the ``@nemo-nb: skip-test`` marker.

    Notebooks with this marker are still processed for docs builds but
    excluded from automated e2e test runs.
    """
    text = _notebook_text(notebook_path)
    return text is not None and "@nemo-nb: skip-test" in text


def has_skip_test_marker_markdown(md_path: Path) -> bool:
    """Check if a markdown file has the ``@nemo-nb: skip-test`` marker.

    Notebooks with this marker are still processed for docs builds but
    excluded from automated e2e test runs.
    """
    text = _markdown_text(md_path)
    return text is not None and "@nemo-nb: skip-test" in text


@dataclass
class NotebookDiscoveryResult:
    """Result of notebook discovery."""

    ipynb_files: List[Path]
    md_files: List[Path]
    conflicts: Set[Path]


def find_processable_notebooks(path: str) -> NotebookDiscoveryResult:
    """
    Find notebooks with @nemo-nb: process marker in a path.

    Supports both directories (recursive search) and single files.

    Args:
        path: Directory or file path to search

    Returns:
        NotebookDiscoveryResult with lists of ipynb/md files and any conflicts
    """
    path_obj = Path(path)
    ipynb_files: List[Path] = []
    md_files: List[Path] = []
    conflicts: Set[Path] = set()

    if path_obj.is_dir():
        # Find .ipynb files, excluding build artifacts and temp files
        all_ipynb = list(path_obj.rglob("*.ipynb"))
        all_ipynb = [
            nb
            for nb in all_ipynb
            if not nb.name.endswith((".executed.ipynb", ".tmp.ipynb")) and not is_in_excluded_dir(nb)
        ]

        # Find Markdown/MDX files, excluding build artifacts and excluded filenames.
        all_md = [*path_obj.rglob("*.md"), *path_obj.rglob("*.mdx")]
        all_md = [md for md in all_md if not is_in_excluded_dir(md) and not is_excluded_file(md)]

        # Filter notebooks with the process marker
        ipynb_files = [nb for nb in all_ipynb if has_process_marker_notebook(nb)]
        md_files = [md for md in all_md if has_process_marker_markdown(md)]

        # Check for conflicts: both .md and .ipynb with same base name
        md_basenames = {md.with_suffix("") for md in md_files}
        ipynb_basenames = {nb.with_suffix("") for nb in ipynb_files}
        conflicts = md_basenames & ipynb_basenames

    elif path_obj.suffix in {".md", ".mdx"}:
        md_files = [path_obj]
    elif path_obj.suffix == ".ipynb":
        ipynb_files = [path_obj]

    return NotebookDiscoveryResult(
        ipynb_files=ipynb_files,
        md_files=md_files,
        conflicts=conflicts,
    )


def find_testable_notebooks(path: str) -> NotebookDiscoveryResult:
    """Find processable notebooks that are **not** marked ``@nemo-nb: skip-test``.

    This is a convenience wrapper around :func:`find_processable_notebooks`
    that filters out notebooks authors have opted out of automated testing
    (e.g. conceptual "about" pages that aren't self-contained).

    Args:
        path: Directory or file path to search.

    Returns:
        NotebookDiscoveryResult excluding skip-test notebooks.
    """
    result = find_processable_notebooks(path)
    ipynb_files = [nb for nb in result.ipynb_files if not has_skip_test_marker_notebook(nb)]
    md_files = [md for md in result.md_files if not has_skip_test_marker_markdown(md)]
    return NotebookDiscoveryResult(
        ipynb_files=ipynb_files,
        md_files=md_files,
        conflicts=result.conflicts,
    )


def print_conflicts_error(conflicts: Set[Path]) -> None:
    """Print error message for notebook conflicts."""
    print("ERROR: Found both .md and .ipynb files with same name (both marked for processing):")
    for conflict in conflicts:
        print(f"  - {conflict}.md and {conflict}.ipynb")
    print("Please remove one of them.")
