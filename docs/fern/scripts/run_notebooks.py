#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Fern documentation notebooks using nemo-nb discovery semantics.

Fern pages generated from notebooks are ``.mdx`` files, but the executable
source remains the adjacent ``.ipynb``. This wrapper resolves Fern ``.mdx``
pages back to their source notebooks and then executes only notebooks marked
with ``@nemo-nb: process`` and not marked with ``@nemo-nb: skip-test``.
Hand-authored Fern pages without a source notebook are materialized as
temporary Markdown files and executed through the same snippet runner.
"""

import argparse
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from nemo_nb import (
    find_testable_notebooks,
    has_process_marker_markdown,
    has_process_marker_notebook,
    has_skip_test_marker_markdown,
    has_skip_test_marker_notebook,
    print_conflicts_error,
)

COLAB_NOTEBOOK_RE = re.compile(
    r"https://colab\.research\.google\.com/github/[^/]+/[^/]+/blob/(?:[^/]+/)+(?P<path>docs/[^)\"'\s]+\.ipynb)"
)
FERN_NOTEBOOK_RE = re.compile(r"colabUrl=[\"'].*?/blob/(?:[^/]+/)+(?P<path>docs/[^\"']+\.ipynb)[\"']")
FERN_MARKDOWN_RE = re.compile(r"^[ \t]*<Markdown\s+src=[\"'](?P<src>[^\"']+)[\"']\s*/>[ \t]*$", re.MULTILINE)
EXECUTABLE_FENCE_RE = re.compile(r"^```(?P<language>[\w+-]*)\s*$", re.MULTILINE)
EXECUTABLE_FENCE_LANGUAGES = {"python", "py", "sh", "bash", "shell"}
TIMEOUT_SECONDS = 3600

load_dotenv()


@dataclass(frozen=True)
class NotebookSelection:
    path: Path
    source: Path


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_mdx_notebook(mdx_path: Path, repo_root: Path) -> Path:
    sibling = mdx_path.with_suffix(".ipynb")
    if sibling.exists():
        return sibling

    text = mdx_path.read_text(encoding="utf-8")
    for pattern in (COLAB_NOTEBOOK_RE, FERN_NOTEBOOK_RE):
        match = pattern.search(text)
        if match:
            linked = repo_root / match.group("path")
            if linked.exists():
                return linked

    raise FileNotFoundError(f"Could not find source .ipynb for Fern page: {mdx_path}")


def resolve_fern_markdown_src(src: str, mdx_path: Path, repo_root: Path) -> Path:
    if src.startswith("/snippets/"):
        return repo_root / "docs" / "fern" / src.removeprefix("/")
    if src.startswith("/"):
        return repo_root / "docs" / src.removeprefix("/")
    return mdx_path.parent / src


def mdx_to_markdown_text(mdx_path: Path, repo_root: Path, seen: set[Path] | None = None) -> str:
    seen = seen or set()
    resolved_mdx_path = mdx_path.resolve()
    if resolved_mdx_path in seen:
        raise RuntimeError(f"Recursive Fern Markdown include detected: {mdx_path}")
    seen.add(resolved_mdx_path)

    try:
        text = mdx_path.read_text(encoding="utf-8")

        def replace_markdown(match: re.Match[str]) -> str:
            include_path = resolve_fern_markdown_src(match.group("src"), mdx_path, repo_root)
            if not include_path.exists():
                return match.group(0)
            return mdx_to_markdown_text(include_path, repo_root, seen)

        return FERN_MARKDOWN_RE.sub(replace_markdown, text)
    finally:
        seen.remove(resolved_mdx_path)


def has_executable_snippets(mdx_path: Path, repo_root: Path) -> bool:
    text = mdx_to_markdown_text(mdx_path, repo_root)
    return any(match.group("language") in EXECUTABLE_FENCE_LANGUAGES for match in EXECUTABLE_FENCE_RE.finditer(text))


def materialize_mdx_as_markdown(mdx_path: Path, repo_root: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=mdx_path.parent,
        prefix=f"{mdx_path.stem}-",
        suffix=".tmp.md",
        delete=False,
    ) as temp_file:
        temp_file.write(mdx_to_markdown_text(mdx_path, repo_root))
        temp_md_path = Path(temp_file.name)
    return temp_md_path


def is_processable(path: Path) -> bool:
    if path.suffix == ".ipynb":
        return has_process_marker_notebook(path)
    if path.suffix in {".md", ".mdx"}:
        return has_process_marker_markdown(path)
    return False


def is_skip_test(path: Path) -> bool:
    if path.suffix == ".ipynb":
        return has_skip_test_marker_notebook(path)
    if path.suffix in {".md", ".mdx"}:
        return has_skip_test_marker_markdown(path)
    return False


def select_single_file(path: Path, repo_root: Path) -> list[NotebookSelection]:
    if path.suffix == ".mdx":
        try:
            notebook_path = resolve_mdx_notebook(path, repo_root)
        except FileNotFoundError:
            if is_skip_test(path):
                print(f"Skipping {path}: has @nemo-nb: skip-test marker")
                return []
            if not is_processable(path) and not has_executable_snippets(path, repo_root):
                print(f"Skipping {path}: missing @nemo-nb: process marker and executable snippets")
                return []
            return [NotebookSelection(path=path, source=path)]
    else:
        notebook_path = path

    if notebook_path.suffix not in {".ipynb", ".md"}:
        raise ValueError(f"Expected .mdx, .ipynb, or .md file: {path}")
    if not is_processable(notebook_path):
        print(f"Skipping {notebook_path}: missing @nemo-nb: process marker")
        return []
    if is_skip_test(notebook_path):
        print(f"Skipping {notebook_path}: has @nemo-nb: skip-test marker")
        return []
    return [NotebookSelection(path=notebook_path, source=path)]


def select_notebooks(paths: list[Path], repo_root: Path) -> list[NotebookSelection]:
    selections: list[NotebookSelection] = []
    seen: set[Path] = set()

    for input_path in paths:
        path = input_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{input_path} does not exist")

        if path.is_file():
            candidates = select_single_file(path, repo_root)
        else:
            result = find_testable_notebooks(str(path))
            if result.conflicts:
                print_conflicts_error(result.conflicts)
                raise RuntimeError("Found conflicting .md and .ipynb notebook sources")
            candidates = [
                NotebookSelection(path=notebook, source=path) for notebook in [*result.ipynb_files, *result.md_files]
            ]

        for candidate in candidates:
            resolved = candidate.path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                selections.append(candidate)

    return selections


def cleanup_selection_temp_files(selection_path: Path) -> None:
    for suffix in (".tmp.ipynb", ".executed.ipynb"):
        stale = selection_path.with_suffix(suffix)
        if stale.exists():
            stale.unlink()
    for stale in (selection_path.with_suffix(".expanded.md"), selection_path.with_suffix(".tmp.md")):
        if stale.exists():
            stale.unlink()


def create_kernel(use_temporary_venv: bool, requirements_file: str | None) -> tuple[str, str | None, str | None]:
    if requirements_file and not use_temporary_venv:
        print("Warning: --requirements requires --use-temporary-venv and will be ignored.")
        requirements_file = None

    if not use_temporary_venv:
        return "python3", None, None

    from nhx.testing.notebooks import create_temp_venv_with_kernel

    kernel_name, temp_venv_dir, temp_kernel_spec_dir = create_temp_venv_with_kernel(requirements_file)
    os.environ["VIRTUAL_ENV"] = temp_venv_dir
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    os.environ["PATH"] = str(Path(temp_venv_dir) / bin_dir) + os.pathsep + os.environ["PATH"]
    return kernel_name, temp_venv_dir, temp_kernel_spec_dir


def cleanup_kernel(kernel_name: str, temp_venv_dir: str | None, temp_kernel_spec_dir: str | None) -> None:
    if temp_venv_dir and temp_kernel_spec_dir:
        from nhx.testing.notebooks import cleanup_temp_venv_and_kernel

        cleanup_temp_venv_and_kernel(kernel_name, temp_venv_dir, temp_kernel_spec_dir)


def run_selected_notebooks(
    selections: list[NotebookSelection],
    language: str,
    keep_temp_files: bool,
    use_temporary_venv: bool,
    requirements_file: str | None,
    execution_timeout: int | None,
) -> int:
    if not selections:
        print("No testable notebooks found.")
        return 0

    print(f"Found {len(selections)} testable notebook(s):")
    for selection in selections:
        print(f"  [{selection.path.suffix.removeprefix('.')}] {selection.path}")

    start_time = time.monotonic()
    kernel_name, temp_venv_dir, temp_kernel_spec_dir = create_kernel(use_temporary_venv, requirements_file)
    failures: list[Path] = []

    try:
        from nhx.testing.notebooks import execute_notebook

        for selection in selections:
            run_path = selection.path
            temp_md_path: Path | None = None
            elapsed = time.monotonic() - start_time
            if elapsed > TIMEOUT_SECONDS:
                raise TimeoutError(f"Timeout running notebooks after {TIMEOUT_SECONDS} seconds")

            print(f"\nRunning {selection.path}...")
            if selection.path.suffix == ".mdx":
                temp_md_path = materialize_mdx_as_markdown(selection.path, resolve_repo_root())
                run_path = temp_md_path
            output_path = run_path.with_suffix(".executed.ipynb")
            try:
                execute_notebook(
                    run_path,
                    language_filter=language,
                    kernel_name=kernel_name,
                    execution_timeout=execution_timeout,
                )
                print(f"SUCCESS: {selection.path}")
            except Exception as error:
                print(f"FAILURE: {selection.path}")
                print(f"Error: {error}")
                failures.append(selection.path)
            finally:
                if not keep_temp_files and output_path.exists():
                    output_path.unlink()
                if not keep_temp_files and temp_md_path:
                    cleanup_selection_temp_files(temp_md_path)
                    temp_md_path.unlink(missing_ok=True)
    finally:
        cleanup_kernel(kernel_name, temp_venv_dir, temp_kernel_spec_dir)

    if failures:
        print(f"\nFAILED: {len(failures)} notebook(s) failed.")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nSUCCESS: All notebook(s) ran successfully.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fern source notebooks with nemo-nb marker semantics.")
    parser.add_argument("paths", nargs="+", type=Path, help="Fern .mdx page, source .ipynb/.md, or directory.")
    parser.add_argument(
        "--language",
        choices=["all", "python", "shell"],
        default="python",
        help="Which cells to execute. Defaults to Python cells only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List selected notebooks without executing them.")
    parser.add_argument("--keep-temp-files", action="store_true", help="Keep generated .executed.ipynb files.")
    parser.add_argument("--use-temporary-venv", action="store_true", help="Run notebooks in a temporary venv.")
    parser.add_argument("--requirements", help="Requirements file to install when using --use-temporary-venv.")
    parser.add_argument("--execution-timeout", type=int, default=None, help="Per-cell execution timeout in seconds.")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    selections: list[NotebookSelection] = []
    try:
        selections = select_notebooks(args.paths, repo_root)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        if not selections:
            print("No testable notebooks found.")
            return 0
        for selection in selections:
            print(selection.path)
        return 0

    try:
        return run_selected_notebooks(
            selections=selections,
            language=args.language,
            keep_temp_files=args.keep_temp_files,
            use_temporary_venv=args.use_temporary_venv,
            requirements_file=args.requirements,
            execution_timeout=args.execution_timeout,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        if not args.keep_temp_files:
            for selection in selections:
                cleanup_selection_temp_files(selection.path)


if __name__ == "__main__":
    raise SystemExit(main())
