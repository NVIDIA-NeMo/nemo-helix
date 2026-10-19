# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nemo docs command - read NeMo Helix documentation from the CLI."""

from __future__ import annotations

import os
import posixpath
import sys
from pathlib import Path
from typing import Annotated

import typer

DOC_EXT = ".mdx"


def _find_docs_root(module_file: Path | None = None) -> Path | None:
    """Find the docs/ directory, checking env var then the CLI source tree."""
    env_root = os.environ.get("NHX_DOCS_ROOT")
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p.resolve()

    module_path = Path(__file__) if module_file is None else module_file
    parents = module_path.resolve().parents

    for parent in parents:
        docs_dir = parent / "docs"
        if docs_dir.is_dir() and any(docs_dir.rglob(f"*{DOC_EXT}")):
            return docs_dir.resolve()

    return None


def _is_visible_doc(rel_path: Path) -> bool:
    return not any(part.startswith("_") or part.startswith(".") for part in rel_path.parts)


def _list_docs(docs_root: Path) -> list[str]:
    """List doc topics under the docs root."""
    paths = []
    for doc_file in sorted(docs_root.rglob(f"*{DOC_EXT}")):
        rel = doc_file.relative_to(docs_root)
        if not _is_visible_doc(rel):
            continue
        paths.append(rel.with_suffix("").as_posix())
    return paths


def _topic_from_user_path(path: str) -> str | None:
    raw_path = path.strip().replace("\\", "/")
    if not raw_path or raw_path.startswith("/"):
        return None

    normalized = posixpath.normpath(raw_path)
    if normalized in {".", ".."} or normalized.startswith("../"):
        return None

    topic = normalized.removeprefix("./")
    for ext in (DOC_EXT, ".md"):
        if topic.endswith(ext):
            topic = topic[: -len(ext)]
            break

    return topic or None


def _resolve_doc_path(docs_root: Path, path: str) -> Path | None:
    topic = _topic_from_user_path(path)
    if topic is None:
        return None

    topics = set(_list_docs(docs_root))
    if topic not in topics:
        return None

    return (docs_root / f"{topic}{DOC_EXT}").resolve()


def docs_command(
    path: Annotated[
        str | None,
        typer.Argument(
            help="Path to a doc topic (e.g., get-started/setup or set-up/helm/install). Omit to see available topics.",
        ),
    ] = None,
    list_topics: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List available documentation topics.",
        ),
    ] = False,
) -> None:
    """Read NeMo Helix documentation.

    Examples:
    nemo docs get-started/setup
    nemo docs set-up/helm/install
    nemo docs --list
    nemo docs cli/configuration
    """
    docs_root = _find_docs_root()
    if docs_root is None:
        typer.echo(
            "Error: Could not find docs directory. Set NHX_DOCS_ROOT environment variable to the docs/ path.",
            err=True,
        )
        raise typer.Exit(code=1)

    if list_topics or path is None:
        topics = _list_docs(docs_root)
        if not topics:
            typer.echo("No documentation found.", err=True)
            raise typer.Exit(code=1)
        typer.echo("Available documentation topics:\n")
        for topic in topics:
            typer.echo(f"  {topic}")
        typer.echo("\nUsage: nemo docs <topic>")
        raise typer.Exit()

    doc_path = _resolve_doc_path(docs_root, path)
    if doc_path is None:
        if _topic_from_user_path(path) is None:
            typer.echo(f"Error: Invalid path: {path}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Error: Documentation not found: {path}", err=True)
        stem = Path(path).stem
        matches = [t for t in _list_docs(docs_root) if stem in t]
        if matches:
            typer.echo("\nDid you mean:", err=True)
            for m in matches[:5]:
                typer.echo(f"  nemo docs {m}", err=True)
        else:
            typer.echo("Run `nemo docs --list` to see available topics.", err=True)
        raise typer.Exit(code=1)

    try:
        doc_path.relative_to(docs_root.resolve())
    except ValueError:
        typer.echo(f"Error: Invalid path: {path}", err=True)
        raise typer.Exit(code=1) from None

    content = doc_path.read_text(encoding="utf-8")
    sys.stdout.write(content)
