# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from docs.fern.scripts.run_notebooks import materialize_mdx_as_markdown, resolve_mdx_notebook, select_notebooks


def _write_notebook(path: Path, marker: str = "<!-- @nemo-nb: process -->") -> None:
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [marker],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_resolve_mdx_notebook_prefers_sibling(tmp_path: Path) -> None:
    mdx = tmp_path / "tutorial.mdx"
    notebook = tmp_path / "tutorial.ipynb"
    mdx.write_text("# Tutorial\n", encoding="utf-8")
    _write_notebook(notebook)

    assert resolve_mdx_notebook(mdx, tmp_path) == notebook


def test_resolve_mdx_notebook_uses_colab_link(tmp_path: Path) -> None:
    repo_root = tmp_path
    notebook = repo_root / "docs" / "customizer" / "tutorials" / "tutorial.ipynb"
    notebook.parent.mkdir(parents=True)
    _write_notebook(notebook)
    mdx = repo_root / "page.mdx"
    mdx.write_text(
        "[Run in Google Colab](https://colab.research.google.com/github/NVIDIA-NeMo/nemo-helix/blob/main/docs/customizer/tutorials/tutorial.ipynb)\n",
        encoding="utf-8",
    )

    assert resolve_mdx_notebook(mdx, repo_root) == notebook


def test_resolve_mdx_notebook_accepts_slash_branch_refs(tmp_path: Path) -> None:
    repo_root = tmp_path
    notebook = repo_root / "docs" / "customizer" / "tutorials" / "tutorial.ipynb"
    notebook.parent.mkdir(parents=True)
    _write_notebook(notebook)

    colab_mdx = repo_root / "colab.mdx"
    colab_mdx.write_text(
        "[Run in Google Colab](https://colab.research.google.com/github/NVIDIA-NeMo/nemo-helix/blob/release/2026.06/docs/customizer/tutorials/tutorial.ipynb)\n",
        encoding="utf-8",
    )
    fern_mdx = repo_root / "fern.mdx"
    fern_mdx.write_text(
        '<Notebook colabUrl="https://github.com/NVIDIA-NeMo/nemo-helix/blob/feature/docs-update/docs/customizer/tutorials/tutorial.ipynb" />\n',
        encoding="utf-8",
    )

    assert resolve_mdx_notebook(colab_mdx, repo_root) == notebook
    assert resolve_mdx_notebook(fern_mdx, repo_root) == notebook


def test_select_notebooks_skips_skip_test_marker(tmp_path: Path) -> None:
    notebook = tmp_path / "skip.ipynb"
    _write_notebook(notebook, "<!-- @nemo-nb: process -->\n<!-- @nemo-nb: skip-test -->")
    mdx = tmp_path / "skip.mdx"
    mdx.write_text("# Skip\n", encoding="utf-8")

    assert select_notebooks([mdx], tmp_path) == []


def test_select_notebooks_resolves_fern_mdx_source(tmp_path: Path) -> None:
    notebook = tmp_path / "tutorial.ipynb"
    _write_notebook(notebook)
    mdx = tmp_path / "tutorial.mdx"
    mdx.write_text("# Tutorial\n", encoding="utf-8")

    selections = select_notebooks([mdx], tmp_path)

    assert len(selections) == 1
    assert selections[0].path == notebook
    assert selections[0].source == mdx


def test_select_notebooks_falls_back_to_mdx_snippets_without_notebook(tmp_path: Path) -> None:
    mdx = tmp_path / "tutorial.mdx"
    mdx.write_text(
        "# Tutorial\n\n```python\nprint('hello')\n```\n",
        encoding="utf-8",
    )

    selections = select_notebooks([mdx], tmp_path)

    assert len(selections) == 1
    assert selections[0].path == mdx
    assert selections[0].source == mdx


def test_select_notebooks_skips_mdx_without_notebook_or_snippets(tmp_path: Path) -> None:
    mdx = tmp_path / "tutorial.mdx"
    mdx.write_text("# Tutorial\n\nNo executable snippets.\n", encoding="utf-8")

    assert select_notebooks([mdx], tmp_path) == []


def test_materialize_mdx_as_markdown_expands_fern_markdown_snippets(tmp_path: Path) -> None:
    snippet = tmp_path / "docs" / "fern" / "snippets" / "_snippets" / "setup.mdx"
    snippet.parent.mkdir(parents=True)
    snippet.write_text("```python\nprint('from snippet')\n```\n", encoding="utf-8")
    mdx = tmp_path / "tutorial.mdx"
    mdx.write_text(
        '# Tutorial\n\n<Markdown src="/snippets/_snippets/setup.mdx" />\n',
        encoding="utf-8",
    )

    temp_md = materialize_mdx_as_markdown(mdx, tmp_path)

    try:
        assert temp_md.parent == mdx.parent
        assert temp_md != mdx.with_suffix(".tmp.md")
        assert temp_md.read_text(encoding="utf-8") == "# Tutorial\n\n```python\nprint('from snippet')\n```\n\n"
    finally:
        temp_md.unlink(missing_ok=True)
