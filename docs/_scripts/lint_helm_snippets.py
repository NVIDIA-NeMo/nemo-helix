#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lint YAML fences in Helm docs, without rendering MDX.

Run: uv run --frozen python docs/_scripts/lint_helm_snippets.py [paths ...]

All yaml/yml fences get safe YAML parsing and duplicate-key checks. Documents
with a known chart root key also get structural checks against values.yaml.
Kubernetes manifests and unrelated chart/config examples are syntax-only.
Overrides are partial: missing keys are fine. Empty default maps/lists are
extensible; platformConfig, env and Kubernetes extension maps are not schemas.
Scalar types are deliberately not inferred from defaults (ports, resource
quantities and Helm template strings can have multiple valid types).
Implicit empty mapping values are rejected in chart snippets, including inside
free-form maps: use explicit null if intentional. This catches collapsed
indentation even where the chart provides no nested schema.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

# Support both direct script execution and namespace-package imports in tests.
if __package__:
    from docs._scripts.lint_python_snippets import (
        FENCE_RE,
        SnippetDiagnostic,
        fence_closes,
        find_doc_files,
        get_language,
        strip_markdown_indent,
    )
else:
    from lint_python_snippets import (
        FENCE_RE,
        SnippetDiagnostic,
        fence_closes,
        find_doc_files,
        get_language,
        strip_markdown_indent,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALUES_FILE = PROJECT_ROOT / "k8s/helm/values.yaml"
DEFAULT_DOCS = PROJECT_ROOT / "docs/set-up/helm"
# These Kubernetes/config objects accept fields absent from chart defaults.
OPEN_MAP_KEYS = {
    "platformConfig",
    "env",
    "annotations",
    "labels",
    "podAnnotations",
    "podLabels",
    "nodeSelector",
    "affinity",
    "resources",
    "podSecurityContext",
    "securityContext",
    "matchLabels",
    "telemetry",
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject explicit duplicate keys, while permitting YAML merge overrides."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        seen = set()
        merge_seen = False
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                if merge_seen:
                    raise ConstructorError(
                        "while constructing a mapping", node.start_mark, "duplicate key '<<'", key_node.start_mark
                    )
                merge_seen = True
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark, "unhashable key", key_node.start_mark
                ) from error
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class YamlSnippet:
    path: Path
    start_line: int
    source: str
    indent: int = 0
    closed: bool = True


@dataclass(frozen=True)
class PageResult:
    path: Path
    snippets: int
    diagnostics: tuple[SnippetDiagnostic, ...]

    @property
    def passed(self) -> bool:
        return not self.diagnostics


def extract_yaml_snippets(path: Path) -> list[YamlSnippet]:
    lines = path.read_text(encoding="utf-8").splitlines()
    snippets = []
    index = 0
    while index < len(lines):
        match = FENCE_RE.match(lines[index])
        if not match:
            index += 1
            continue
        indent, marker, info = match.groups()
        start_line = index + 2
        index += 1
        source = []
        while index < len(lines) and not fence_closes(lines[index], marker):
            source.append(strip_markdown_indent(lines[index], indent))
            index += 1
        closed = index < len(lines)
        index += 1
        if get_language(info) in {"yaml", "yml"}:
            snippets.append(YamlSnippet(path, start_line, "\n".join(source), len(indent), closed))
    return snippets


def mapping_entries(node: MappingNode) -> dict[str, Node]:
    return {key.value: value for key, value in node.value if isinstance(key, ScalarNode)}


def check_snippet(snippet: YamlSnippet, defaults: dict) -> list[SnippetDiagnostic]:
    diagnostics = []

    def report(node: Node, message: str) -> None:
        diagnostics.append(
            SnippetDiagnostic(
                snippet.path,
                snippet.start_line + node.start_mark.line,
                snippet.indent + node.start_mark.column + 1,
                message,
            )
        )

    def check_empty_values(node: Node, path: str, visited: set[int]) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, MappingNode):
            for key, value in node.value:
                child_path = f"{path}.{key.value}" if path else str(key.value)
                if isinstance(value, ScalarNode) and value.tag == "tag:yaml.org,2002:null" and value.value == "":
                    report(key, f"{child_path}: empty YAML value (possible collapsed indentation; use explicit null)")
                check_empty_values(value, child_path, visited)
        elif isinstance(node, SequenceNode):
            for index, value in enumerate(node.value):
                check_empty_values(value, f"{path}[{index}]", visited)

    def check_shape(node: Node, reference: object, path: str, visited: set[tuple[int, int]]) -> None:
        pair = (id(node), id(reference))
        if pair in visited:
            return
        visited.add(pair)
        if isinstance(reference, dict):
            if not isinstance(node, MappingNode):
                report(node, f"{path or '<root>'}: expected mapping")
                return
            if not reference or path.rsplit(".", 1)[-1] in OPEN_MAP_KEYS:
                return
            for key, value in node.value:
                if key.tag == "tag:yaml.org,2002:merge":
                    # Validate the mapping(s) behind a merge rather than treating << as a chart key.
                    merged = value.value if isinstance(value, SequenceNode) else [value]
                    for entry in merged:
                        check_shape(entry, reference, path, visited)
                    continue
                child_path = f"{path}.{key.value}" if path else str(key.value)
                if key.value not in reference:
                    report(key, f"{child_path}: unknown chart key (possible misplaced sibling)")
                else:
                    check_shape(value, reference[key.value], child_path, visited)
        elif isinstance(reference, list):
            if not isinstance(node, SequenceNode):
                report(node, f"{path}: expected sequence")
            elif reference:
                for index, value in enumerate(node.value):
                    check_shape(value, reference[0], f"{path}[{index}]", visited)
        elif isinstance(node, (MappingNode, SequenceNode)) and reference is not None:
            report(node, f"{path}: expected scalar")

    if not snippet.closed:
        diagnostics.append(SnippetDiagnostic(snippet.path, snippet.start_line - 1, None, "Unclosed YAML fence"))
    loader = UniqueKeyLoader(snippet.source)
    try:
        while loader.check_data():
            node = loader.get_node()
            # Construction enforces safe tags, valid keys, and duplicates at every depth.
            loader.construct_document(node)
            if isinstance(node, MappingNode):
                entries = mapping_entries(node)
                if "apiVersion" in entries and "kind" in entries:
                    continue
                if set(entries) & defaults.keys():
                    check_empty_values(node, "", set())
                    check_shape(node, defaults, "", set())
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        diagnostics.append(
            SnippetDiagnostic(
                snippet.path,
                snippet.start_line + (mark.line if mark else 0),
                snippet.indent + mark.column + 1 if mark else None,
                f"YAML: {getattr(error, 'problem', None) or str(error)}",
            )
        )
    finally:
        loader.dispose()
    return diagnostics


def check_paths(paths: Iterable[Path], values_file: Path = DEFAULT_VALUES_FILE) -> list[PageResult]:
    with values_file.open(encoding="utf-8") as stream:
        defaults = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(defaults, dict) or not defaults:
        raise ValueError(f"Expected a nonempty chart values mapping: {values_file}")
    results = []
    for path in find_doc_files(paths):
        snippets = extract_yaml_snippets(path)
        if snippets:
            diagnostics = tuple(diagnostic for snippet in snippets for diagnostic in check_snippet(snippet, defaults))
            results.append(PageResult(path, len(snippets), diagnostics))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown/MDX files or directories (default: Helm docs).")
    parser.add_argument("--values-file", type=Path, default=DEFAULT_VALUES_FILE, help="Chart values shape reference.")
    args = parser.parse_args(argv)
    try:
        results = check_paths(args.paths or [DEFAULT_DOCS], args.values_file)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    for result in results:
        for diagnostic in result.diagnostics:
            print(diagnostic.format())
    errors = sum(len(result.diagnostics) for result in results)
    print(f"Checked {sum(result.snippets for result in results)} YAML snippet(s); {errors} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
