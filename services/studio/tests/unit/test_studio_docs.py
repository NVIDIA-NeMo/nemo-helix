# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from pathlib import Path

import pytest
from nhx.studio import studio_docs


def _write_page(root: Path, relative: str, title: str | None, body: str = "Body text.") -> None:
    page = root / f"{relative}.mdx"
    page.parent.mkdir(parents=True, exist_ok=True)
    front_matter = ["---", "# SPDX-License-Identifier: Apache-2.0", ""]
    if title is not None:
        front_matter.append(f'title: "{title}"')
    front_matter.extend(['description: ""', "---", ""])
    page.write_text("\n".join(front_matter) + body, encoding="utf-8")


@pytest.fixture
def docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A bundled docs tree with the readable sections plus one section that must stay hidden."""
    root = tmp_path / "docs"
    _write_page(root, "studio/index", "About Studio", "Studio overview.")
    _write_page(root, "studio/agents", "Agents in Studio")
    _write_page(root, "agents/governance/policies", "Governance policies")
    _write_page(root, "cli/index", None, "CLI without a title.")
    _write_page(root, "evals/index", "Evals (not a readable section)")
    (tmp_path / "secret.mdx").write_text("outside the docs root", encoding="utf-8")
    monkeypatch.setattr(studio_docs, "_BUNDLED_DOCS_DIR", root)
    monkeypatch.setattr(studio_docs, "_REPO_DOCS_DIR", tmp_path / "no-repo-docs")
    studio_docs._index.cache_clear()
    yield root
    studio_docs._index.cache_clear()


def test_list_pages_covers_only_readable_sections_with_titles(docs_dir: Path):
    pages = {page.path: page.title for page in studio_docs.list_pages()}

    assert pages == {
        "studio/index": "About Studio",
        "studio/agents": "Agents in Studio",
        "agents/governance/policies": "Governance policies",
        # A page with no title falls back to its path.
        "cli/index": "cli/index",
    }


def test_format_index_lists_each_page_path_and_title(docs_dir: Path):
    index = studio_docs.format_index()

    assert "- studio/index: About Studio" in index
    assert "- agents/governance/policies: Governance policies" in index
    assert "evals/index" not in index


@pytest.mark.parametrize("path", ["studio/index", "studio/index.mdx", "/studio/index", "  studio/index  "])
def test_read_page_accepts_equivalent_spellings_of_a_path(docs_dir: Path, path: str):
    assert "Studio overview." in studio_docs.read_page(path)


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "studio/../../secret",
        "studio/../evals/index",
        "evals/index",
        "studio/missing",
        "",
        "/etc/passwd",
    ],
)
def test_read_page_rejects_paths_outside_the_index(docs_dir: Path, path: str):
    with pytest.raises(LookupError):
        studio_docs.read_page(path)


def test_read_page_pages_long_content_with_offset(docs_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _write_page(docs_dir, "studio/long", "Long page", "x" * 100)
    studio_docs._index.cache_clear()
    monkeypatch.setattr(studio_docs, "MAX_PAGE_CHARS", 40)
    full_length = len((docs_dir / "studio/long.mdx").read_text(encoding="utf-8"))

    first = studio_docs.read_page("studio/long")
    assert "Truncated at character 40" in first
    assert "offset=40" in first

    last = studio_docs.read_page("studio/long", offset=full_length - 10)
    assert "Truncated" not in last
    assert last == "x" * 10


def test_read_page_treats_negative_offset_as_zero(docs_dir: Path):
    assert studio_docs.read_page("studio/index", offset=-5) == studio_docs.read_page("studio/index", offset=0)


def test_no_bundled_docs_is_reported_not_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(studio_docs, "_BUNDLED_DOCS_DIR", tmp_path / "missing")
    monkeypatch.setattr(studio_docs, "_REPO_DOCS_DIR", tmp_path / "also-missing")
    studio_docs._index.cache_clear()

    assert studio_docs.list_pages() == []
    assert "No product docs are bundled" in studio_docs.format_index()
    with pytest.raises(LookupError):
        studio_docs.read_page("studio/index")


def test_source_checkout_falls_back_to_the_repo_docs():
    studio_docs._index.cache_clear()

    paths = {page.path for page in studio_docs.list_pages()}

    assert "studio/index" in paths
    assert any(path.startswith("cli/") for path in paths)
    assert any(path.startswith("agents/") for path in paths)
