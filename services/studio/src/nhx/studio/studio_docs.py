# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only access to the product docs bundled with the Studio service.

The Studio assistant (Felix) reads these pages instead of relying on hardcoded knowledge of the
product, so what he says tracks the docs shipped with the running Studio.

Only pages found by :func:`list_pages` can be read, so a caller-supplied path can never reach an
arbitrary file.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Docs sections the assistant may read. Each is a folder of ``.mdx`` pages under the docs root.
DOC_SECTIONS: tuple[str, ...] = ("studio", "agents", "cli")

# Characters returned per read. Long pages (e.g. the CLI reference) are paged with ``offset``.
MAX_PAGE_CHARS = 40_000

_PACKAGE_DIR = Path(__file__).resolve().parent
# Where the build copies the docs inside the package (see ``pyproject.toml``).
_BUNDLED_DOCS_DIR = _PACKAGE_DIR / "docs"
# In a source checkout the docs live at the repository root (services/studio/src/nhx/studio -> repo).
# An installed package sits in site-packages, where this path may not exist or may be shallow.
_REPO_DOCS_DIR = _PACKAGE_DIR.parents[4] / "docs" if len(_PACKAGE_DIR.parents) > 4 else _PACKAGE_DIR / "_no_docs"

_TITLE_RE = re.compile(r'^title:\s*["\']?(.*?)["\']?\s*$', re.MULTILINE)
_FRONT_MATTER_SCAN_CHARS = 2_000


@dataclass(frozen=True)
class DocPage:
    """One readable docs page."""

    path: str
    """Section-relative path without extension, e.g. ``studio/agents``."""
    title: str
    file: Path


def docs_root() -> Path | None:
    """Return the docs root, preferring the bundled copy, or ``None`` when no docs are available."""
    for candidate in (_BUNDLED_DOCS_DIR, _REPO_DOCS_DIR):
        if any((candidate / section).is_dir() for section in DOC_SECTIONS):
            return candidate
    return None


def _page_title(text: str, fallback: str) -> str:
    match = _TITLE_RE.search(text[:_FRONT_MATTER_SCAN_CHARS])
    return match.group(1).strip() if match and match.group(1).strip() else fallback


@lru_cache(maxsize=1)
def _index(root: Path) -> dict[str, DocPage]:
    pages: dict[str, DocPage] = {}
    for section in DOC_SECTIONS:
        section_dir = root / section
        if not section_dir.is_dir():
            continue
        for file in sorted(section_dir.rglob("*.mdx")):
            if not file.is_file():
                continue
            path = file.relative_to(root).with_suffix("").as_posix()
            title = _page_title(file.read_text(encoding="utf-8", errors="replace"), fallback=path)
            pages[path] = DocPage(path=path, title=title, file=file)
    return pages


def list_pages() -> list[DocPage]:
    """Return every readable docs page, or an empty list when no docs are bundled."""
    root = docs_root()
    return list(_index(root).values()) if root is not None else []


def format_index() -> str:
    """Return the docs index as text the assistant can scan to pick a page."""
    pages = list_pages()
    if not pages:
        return "No product docs are bundled with this Studio, so there is nothing to look up."
    lines = ["Available docs pages (call studio_docs again with a path to read one):"]
    lines.extend(f"- {page.path}: {page.title}" for page in pages)
    return "\n".join(lines)


def read_page(path: str, offset: int = 0) -> str:
    """Return one page's text, starting at ``offset`` and capped at ``MAX_PAGE_CHARS``.

    Raises ``LookupError`` for a path that is not an indexed docs page.
    """
    root = docs_root()
    normalized = path.strip().strip("/").removesuffix(".mdx")
    page = _index(root).get(normalized) if root is not None else None
    if page is None:
        raise LookupError(f"Unknown docs page '{path}'. Call studio_docs with no path to list the available pages.")

    text = page.file.read_text(encoding="utf-8", errors="replace")
    start = max(offset, 0)
    chunk = text[start : start + MAX_PAGE_CHARS]
    end = start + len(chunk)
    if end < len(text):
        chunk += (
            f"\n\n[Truncated at character {end} of {len(text)}. Call studio_docs again with offset={end} to continue.]"
        )
    return chunk
