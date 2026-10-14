#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Python fenced code blocks in Markdown and MDX docs.

The checker is intentionally extraction-based instead of MDX-renderer-based:
Fern pages may contain JSX, imports, or custom components, but Python snippet
validation only needs fenced ``python``/``py`` blocks.

By default this script syntax-checks every Python snippet, checks module imports
in execution order, and combines snippets per page to run ``ty`` over extracted
source while mapping diagnostics back to the original doc line numbers. Module
import checks also cover type-check-exempt cells. Pass ``--no-type-check`` to
skip ``ty`` (syntax and import checks still run).
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DOC_SUFFIXES = {".md", ".mdx"}
PYTHON_LANGUAGES = {"python", "py"}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-docs",
    ".venv-mkdocs",
    "_build",
    "_generated",
    "node_modules",
    "site",
}

FENCE_RE = re.compile(r"^([ \t]*)(`{3,}|~{3,})(.*)$")
TY_OUTPUT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): (?P<message>.+)$")

# Both comment syntaxes are accepted. MDX v3 rejects HTML comments outright -- a page
# carrying one fails `validate-mdx` -- so an .mdx page must use the ``{/* ... */}`` form.
SKIP_NEXT_BLOCK_MARKERS = {
    "<!-- @nemo-docs: skip-python-snippet-check -->",
    "{/* @nemo-docs: skip-python-snippet-check */}",
}
SKIP_NEXT_TYPE_CHECK_MARKERS = {
    "<!-- @nemo-docs: skip-python-type-check -->",
    "<!-- @nemo-nb: skip-type-check -->",
    "{/* @nemo-docs: skip-python-type-check */}",
    "{/* @nemo-nb: skip-type-check */}",
}
# ``possibly-unbound-attribute`` was renamed upstream; passing the old name makes ty emit
# ``warning[unknown-rule]``, which fails this check for every doc regardless of its snippets.
DEFAULT_IGNORED_TY_RULES = ("possibly-missing-attribute",)


@dataclass(frozen=True)
class PythonSnippet:
    path: Path
    start_line: int
    source: str
    type_check: bool


@dataclass(frozen=True)
class SnippetDiagnostic:
    path: Path
    line: int
    column: int | None
    message: str

    def format(self) -> str:
        if self.column is None:
            return f"{self.path}:{self.line}: {self.message}"
        return f"{self.path}:{self.line}:{self.column}: {self.message}"


@dataclass(frozen=True)
class PageResult:
    path: Path
    snippets: int
    syntax_errors: tuple[SnippetDiagnostic, ...]
    type_errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.syntax_errors and not self.type_errors


@dataclass(frozen=True)
class PreparedTypeCheckFile:
    doc_path: Path
    temp_path: Path
    line_mapping: tuple[int, ...]


def find_doc_files(paths: Iterable[Path]) -> list[Path]:
    doc_files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        if path.is_file():
            if path.suffix in DOC_SUFFIXES:
                doc_files.append(path)
                continue
            expected = ", ".join(sorted(DOC_SUFFIXES))
            raise ValueError(f"Expected a Markdown/MDX file ({expected}) or directory: {path}")

        for root, dirs, files in os.walk(path):
            dirs[:] = [directory for directory in dirs if directory not in SKIP_DIRS]
            for filename in files:
                file_path = Path(root) / filename
                if file_path.suffix in DOC_SUFFIXES:
                    doc_files.append(file_path)

    return sorted(set(doc_files))


def fence_closes(line: str, fence_marker: str) -> bool:
    stripped = line.lstrip(" \t")
    marker_char = fence_marker[0]
    marker_len = len(fence_marker)
    if not stripped.startswith(marker_char * marker_len):
        return False
    return stripped.strip(marker_char).strip() == ""


def get_language(info_string: str) -> str:
    stripped = info_string.strip()
    if not stripped:
        return ""
    return stripped.split(maxsplit=1)[0].lower()


def strip_markdown_indent(line: str, indent: str) -> str:
    if indent and line.startswith(indent):
        return line[len(indent) :]
    return line


def source_for_static_check(source: str) -> str:
    """Return Python-parseable source for notebook-style cells.

    IPython line magics and shell escapes are valid in notebooks, but not in
    ``ast`` or ``ty``. Keep line numbers stable by replacing them with comments.
    """
    transformed_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip(" \t")
        if stripped.startswith("%") or stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            transformed_lines.append(f"{indent}# {stripped}")
            continue
        transformed_lines.append(line)
    return "\n".join(transformed_lines)


def extract_python_snippets(path: Path) -> list[PythonSnippet]:
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets: list[PythonSnippet] = []
    skip_next_block = False
    skip_next_type_check = False
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped in SKIP_NEXT_BLOCK_MARKERS:
            skip_next_block = True
            index += 1
            continue
        if stripped in SKIP_NEXT_TYPE_CHECK_MARKERS:
            skip_next_type_check = True
            index += 1
            continue

        fence_match = FENCE_RE.match(lines[index])
        if not fence_match:
            index += 1
            continue

        indent, fence_marker, info_string = fence_match.groups()
        language = get_language(info_string)
        code_start_line = index + 2
        index += 1

        code_lines: list[str] = []
        while index < len(lines) and not fence_closes(lines[index], fence_marker):
            code_lines.append(strip_markdown_indent(lines[index], indent))
            index += 1

        if index < len(lines):
            index += 1

        if language not in PYTHON_LANGUAGES:
            skip_next_block = False
            skip_next_type_check = False
            continue

        if not skip_next_block:
            source = "\n".join(code_lines)
            if source.strip():
                snippets.append(
                    PythonSnippet(
                        path=path,
                        start_line=code_start_line,
                        source=source,
                        type_check=not skip_next_type_check,
                    )
                )

        skip_next_block = False
        skip_next_type_check = False

    return snippets


def syntax_check(snippets: Iterable[PythonSnippet]) -> list[SnippetDiagnostic]:
    diagnostics: list[SnippetDiagnostic] = []
    for snippet in snippets:
        try:
            ast.parse(source_for_static_check(snippet.source))
        except SyntaxError as error:
            line_offset = error.lineno or 1
            diagnostics.append(
                SnippetDiagnostic(
                    path=snippet.path,
                    line=snippet.start_line + line_offset - 1,
                    column=error.offset,
                    message=error.msg,
                )
            )
    return diagnostics


def import_usage_check(snippets: Iterable[PythonSnippet]) -> list[SnippetDiagnostic]:
    """Check module references in execution order, even in type-check-exempt cells.

    Joining a page for type checking can hide an import that occurs only in a
    later (possibly optional) cell. Only inspect immediately executed code:
    function bodies may legitimately use modules imported before the call.
    Standard-library module names also catch imports missing from the entire
    page; third-party aliases are learned from the page's import statements.
    This is deliberately not a general undefined-variable checker.
    """
    parsed: list[tuple[PythonSnippet, ast.Module]] = []
    module_names = set(sys.stdlib_module_names)
    for snippet in snippets:
        try:
            tree = ast.parse(source_for_static_check(snippet.source))
        except SyntaxError:
            continue  # Reported separately by syntax_check.
        parsed.append((snippet, tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    available: set[str] = set()
    diagnostics: list[SnippetDiagnostic] = []

    class ModuleUsageVisitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            available.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            available.update(alias.asname or alias.name for alias in node.names)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                available.add(node.id)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name):
                name = node.value.id
                if name in module_names and name not in available:
                    diagnostics.append(
                        SnippetDiagnostic(
                            path=snippet.path,
                            line=snippet.start_line + node.lineno - 1,
                            column=node.col_offset + 1,
                            message=f"Module `{name}` is used before it is imported or defined",
                        )
                    )
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            self.visit(node.value)
            for target in node.targets:
                self.visit(target)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self.visit(node.value)
                self.visit(node.target)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            self.visit(node.value)
            self.visit(node.target)

        def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
            self.visit(node.iter)
            self.visit(node.target)
            for statement in [*node.body, *node.orelse]:
                self.visit(statement)

        visit_AsyncFor = visit_For

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.type is not None:
                self.visit(node.type)
            if node.name is not None:
                available.add(node.name)
            for statement in node.body:
                self.visit(statement)
            if node.name is not None:
                available.discard(node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            # Decorators and defaults execute now; the body executes at call time.
            for expression in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
                if expression is not None:
                    self.visit(expression)
            available.add(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for expression in [*node.args.defaults, *node.args.kw_defaults]:
                if expression is not None:
                    self.visit(expression)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for expression in [*node.decorator_list, *node.bases]:
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)
            # Class bodies execute immediately, but their imports/bindings do
            # not become globals available to subsequent cells.
            outer_names = available.copy()
            for statement in node.body:
                self.visit(statement)
            available.clear()
            available.update(outer_names)
            available.add(node.name)

    visitor = ModuleUsageVisitor()
    for snippet, tree in parsed:
        visitor.visit(tree)
    return diagnostics


def prepare_type_check_file(
    doc_path: Path,
    snippets: Iterable[PythonSnippet],
    temp_dir: Path,
) -> PreparedTypeCheckFile | None:
    source_lines: list[str] = []
    line_mapping: list[int] = []

    for snippet in snippets:
        if not snippet.type_check:
            continue

        snippet_lines = source_for_static_check(snippet.source).splitlines()
        for offset, line in enumerate(snippet_lines):
            source_lines.append(line)
            line_mapping.append(snippet.start_line + offset)

        source_lines.append("")
        if snippet_lines:
            line_mapping.append(snippet.start_line + len(snippet_lines) - 1)
        else:
            line_mapping.append(snippet.start_line)

    if not any(line.strip() for line in source_lines):
        return None

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=temp_dir,
        prefix="snippet-",
        suffix=".py",
        delete=False,
    ) as temp_file:
        temp_file.write("\n".join(source_lines))
        temp_path = Path(temp_file.name)

    return PreparedTypeCheckFile(doc_path=doc_path, temp_path=temp_path, line_mapping=tuple(line_mapping))


def translate_line_number(line_in_combined: int, line_mapping: tuple[int, ...]) -> int:
    index = line_in_combined - 1
    if 0 <= index < len(line_mapping):
        return line_mapping[index]
    return line_in_combined


def run_type_check(
    snippets_by_path: dict[Path, list[PythonSnippet]],
    project_root: Path,
    timeout_seconds: int,
) -> dict[Path, tuple[str, ...]]:
    results: dict[Path, list[str]] = {path: [] for path in snippets_by_path}
    if not snippets_by_path:
        return {}

    with tempfile.TemporaryDirectory(prefix="nemo-docs-snippets-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        prepared_files = [
            prepared
            for path, snippets in snippets_by_path.items()
            if (prepared := prepare_type_check_file(path, snippets, temp_dir)) is not None
        ]
        if not prepared_files:
            return {path: tuple(messages) for path, messages in results.items()}

        temp_to_prepared = {str(prepared.temp_path): prepared for prepared in prepared_files}
        command = [
            "uv",
            "run",
            "--frozen",
            "ty",
            "check",
            "--project",
            str(project_root),
            "--output-format",
            "concise",
            "--no-progress",
        ]
        for rule in DEFAULT_IGNORED_TY_RULES:
            command.extend(["--ignore", rule])
        command.extend(str(prepared.temp_path) for prepared in prepared_files)

        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return {path: ("Type checking failed: `uv` was not found on PATH",) for path in snippets_by_path}
        except subprocess.TimeoutExpired:
            return {path: (f"Type checking timed out after {timeout_seconds} seconds",) for path in snippets_by_path}

        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        unmatched_lines: list[str] = []

        for line in output.splitlines():
            match = TY_OUTPUT_RE.match(line)
            if match is None:
                if line.strip():
                    unmatched_lines.append(line)
                continue

            matched_prepared = temp_to_prepared.get(match.group("path"))
            if matched_prepared is None:
                if line.strip():
                    unmatched_lines.append(line)
                continue

            combined_line = int(match.group("line"))
            column = match.group("column")
            message = match.group("message")
            doc_line = translate_line_number(combined_line, matched_prepared.line_mapping)
            results[matched_prepared.doc_path].append(f"{matched_prepared.doc_path}:{doc_line}:{column}: {message}")

        if completed.returncode != 0 and not any(results.values()) and unmatched_lines:
            shared_output = tuple(unmatched_lines)
            return {path: shared_output for path in snippets_by_path}

        return {path: tuple(messages) for path, messages in results.items()}


def check_paths(
    paths: Iterable[Path],
    type_check: bool,
    project_root: Path,
    timeout_seconds: int,
) -> list[PageResult]:
    doc_files = find_doc_files(paths)
    snippets_by_path = {path: extract_python_snippets(path) for path in doc_files}
    snippets_by_path = {path: snippets for path, snippets in snippets_by_path.items() if snippets}

    type_results: dict[Path, tuple[str, ...]] = {}
    if type_check:
        type_results = run_type_check(snippets_by_path, project_root, timeout_seconds)

    return [
        PageResult(
            path=path,
            snippets=len(snippets),
            syntax_errors=tuple([*syntax_check(snippets), *import_usage_check(snippets)]),
            type_errors=type_results.get(path, ()),
        )
        for path, snippets in snippets_by_path.items()
    ]


def display_results(results: list[PageResult], type_check: bool) -> None:
    if not results:
        print("No Python snippets found.")
        return

    total_snippets = sum(result.snippets for result in results)
    checks = "syntax + imports + type" if type_check else "syntax + imports"
    print(f"Checked {total_snippets} Python snippet(s) across {len(results)} doc file(s) ({checks}).\n")

    for result in results:
        if result.passed:
            print(f"✓ {result.path} ({result.snippets} snippet(s))")
            continue

        print(f"✗ {result.path} ({result.snippets} snippet(s))")
        for diagnostic in result.syntax_errors:
            print(f"  SNIPPET: {diagnostic.format()}")
        for message in result.type_errors:
            print(f"  TYPE: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Python fenced snippets in Markdown/MDX docs.")
    parser.add_argument("paths", nargs="+", type=Path, help="Markdown/MDX files or directories to check.")
    type_check_group = parser.add_mutually_exclusive_group()
    type_check_group.add_argument(
        "--type-check",
        dest="type_check",
        action="store_true",
        help="Run `ty check` over extracted snippets. This is the default.",
    )
    type_check_group.add_argument(
        "--no-type-check",
        dest="type_check",
        action="store_false",
        help="Run syntax and module-import checks; skip `ty check`.",
    )
    parser.set_defaults(type_check=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root passed to `ty --project` when type checking. Defaults to cwd.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for the `ty check` subprocess when --type-check is enabled.",
    )
    args = parser.parse_args()

    try:
        results = check_paths(
            paths=args.paths,
            type_check=args.type_check,
            project_root=args.project_root.resolve(),
            timeout_seconds=args.timeout_seconds,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    display_results(results, args.type_check)
    return 1 if any(not result.passed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
