#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flag doc snippets that bypass the active CLI context with ``NHX_BASE_URL``.

Docs should connect through the current CLI context: the ``nemo`` CLI reads it
directly and Python snippets use ``NemoClient.from_config()``. Two patterns
silently ignore a context that points at a remote deployment:

- Python code that reads ``NHX_BASE_URL`` (usually with a localhost fallback)
  to build a client.
- Shell code that assigns a literal URL to ``NHX_BASE_URL``, which overrides
  the saved context for every later ``nemo`` command in that shell.

Shell code may still default the variable for tools that can't read the CLI
configuration, such as ``curl``, with ``${NHX_BASE_URL:-...}`` or
``: "${NHX_BASE_URL:=...}"``. Mark a deliberate override with an inline
``nhx-base-url-allow`` comment, or with ``{/* @nemo-docs: allow-nhx-base-url */}``
on the line before the code fence to keep the marker out of the rendered page.

Run from the repository root::

    uv run --frozen python -m docs._scripts.check_base_url_usage docs
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docs._scripts.lint_python_snippets import (
    FENCE_RE,
    PYTHON_LANGUAGES,
    fence_closes,
    find_doc_files,
    get_language,
)

SHELL_LANGUAGES = {"bash", "sh", "shell", "zsh", "console"}
INLINE_ALLOW_MARKER = "nhx-base-url-allow"
ALLOW_NEXT_BLOCK_MARKERS = {
    "<!-- @nemo-docs: allow-nhx-base-url -->",
    "{/* @nemo-docs: allow-nhx-base-url */}",
}
CONTEXTS_DOC = "docs/cli/connect-to-deployments.mdx"
# Generated pages can't carry hand-added markers. The CLI reference renders
# ``nemo --help`` examples, which document NHX_BASE_URL overrides for CI runs.
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DOCS = {REPO_ROOT / "docs/cli/reference.mdx"}

# ``os.environ.get("NHX_BASE_URL", ...)``, ``os.getenv("NHX_BASE_URL")``, or a
# subscript read. A subscript assignment (``os.environ["NHX_BASE_URL"] = ...``)
# sets the variable for ``from_config()`` and is allowed.
PYTHON_READ_RE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["']NHX_BASE_URL["']"""
    r"""|os\.environ\[\s*["']NHX_BASE_URL["']\s*\](?!\s*=(?!=))"""
)
# A literal assignment, with or without ``export``. The parameter-expansion
# default forms (``${NHX_BASE_URL:-...}``, ``${NHX_BASE_URL:=...}``) don't match.
SHELL_ASSIGN_RE = re.compile(r"""(?<![\w$])NHX_BASE_URL=(?!"?\$\{NHX_BASE_URL:)""")


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    start_line: int
    language: str
    lines: tuple[str, ...]
    allowed: bool


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    message: str


def extract_code_blocks(text: str, path: Path, line_offset: int = 0) -> list[CodeBlock]:
    """Return fenced code blocks with their language and allow-marker state."""
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    allow_next_block = False
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped in ALLOW_NEXT_BLOCK_MARKERS:
            allow_next_block = True
            index += 1
            continue

        fence_match = FENCE_RE.match(lines[index])
        if not fence_match:
            if stripped:
                allow_next_block = False
            index += 1
            continue

        _, fence_marker, info_string = fence_match.groups()
        start_line = line_offset + index + 2
        index += 1
        code_lines: list[str] = []
        while index < len(lines) and not fence_closes(lines[index], fence_marker):
            code_lines.append(lines[index])
            index += 1
        if index < len(lines):
            index += 1

        blocks.append(
            CodeBlock(
                path=path,
                start_line=start_line,
                language=get_language(info_string),
                lines=tuple(code_lines),
                allowed=allow_next_block,
            )
        )
        allow_next_block = False

    return blocks


def extract_notebook_blocks(path: Path) -> list[CodeBlock]:
    """Return code cells as Python blocks plus fenced blocks inside markdown cells.

    Notebook line numbers aren't meaningful in the JSON file, so ``start_line``
    is the 1-based cell number.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[CodeBlock] = []
    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "code":
            blocks.append(
                CodeBlock(
                    path=path, start_line=cell_number, language="python", lines=tuple(text.splitlines()), allowed=False
                )
            )
        elif cell.get("cell_type") == "markdown":
            blocks.extend(
                CodeBlock(path=path, start_line=cell_number, language=b.language, lines=b.lines, allowed=b.allowed)
                for b in extract_code_blocks(text, path)
            )
    return blocks


def check_block(block: CodeBlock, *, notebook: bool = False) -> list[Violation]:
    if block.allowed:
        return []
    if block.language in PYTHON_LANGUAGES:
        pattern = PYTHON_READ_RE
        message = (
            "Python reads NHX_BASE_URL, which ignores the CLI context. "
            "Use NemoClient.from_config(), which honors NHX_BASE_URL when it is set."
        )
    elif block.language in SHELL_LANGUAGES:
        pattern = SHELL_ASSIGN_RE
        message = (
            "Shell assigns a literal NHX_BASE_URL, which overrides the CLI context. "
            'Leave it unset for nemo commands, or default it with "${NHX_BASE_URL:-http://localhost:8080}".'
        )
    else:
        return []

    violations: list[Violation] = []
    for offset, line in enumerate(block.lines):
        if INLINE_ALLOW_MARKER in line or not pattern.search(line):
            continue
        line_number = block.start_line if notebook else block.start_line + offset
        violations.append(Violation(path=block.path, line=line_number, message=message))
    return violations


def check_paths(paths: Iterable[Path]) -> list[Violation]:
    path_list = list(paths)
    files = find_doc_files(path_list)
    for path in path_list:
        if path.is_dir():
            files.extend(p for p in path.rglob("*.ipynb") if "node_modules" not in p.parts)
        elif path.suffix == ".ipynb":
            files.append(path)

    violations: list[Violation] = []
    for path in sorted(set(files)):
        if path.resolve() in GENERATED_DOCS:
            continue
        if path.suffix == ".ipynb":
            for block in extract_notebook_blocks(path):
                violations.extend(check_block(block, notebook=True))
        else:
            for block in extract_code_blocks(path.read_text(encoding="utf-8"), path):
                violations.extend(check_block(block))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag doc snippets that bypass the CLI context with NHX_BASE_URL.")
    parser.add_argument("paths", nargs="+", type=Path, help="Markdown/MDX/notebook files or directories to check.")
    args = parser.parse_args()

    violations = check_paths(args.paths)
    for violation in violations:
        location = f" cell {violation.line}:" if violation.path.suffix == ".ipynb" else f"{violation.line}:"
        print(f"{violation.path}:{location} {violation.message}")
    if violations:
        print(
            f"\n{len(violations)} NHX_BASE_URL usage(s) bypass the CLI context. See the "
            f'"When to use NHX_BASE_URL" section of {CONTEXTS_DOC}. Mark a deliberate override with an inline '
            f"`{INLINE_ALLOW_MARKER}: <reason>` comment or `{{/* @nemo-docs: allow-nhx-base-url */}}` before the fence.",
            file=sys.stderr,
        )
        return 1
    print("No NHX_BASE_URL usages bypass the CLI context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
