# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keyword search over the NeMo Helix documentation bundled with the assistant."""

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DOCS_DIR_ENV = "NEMO_STUDIO_ASSISTANT_DOCS_DIR"
SNAPSHOT_DIR_NAME = "docs-snapshot"
DOC_SUFFIXES = (".md", ".mdx")
DEFAULT_LIMIT = 5
MAX_LIMIT = 10
SNIPPET_CHARS = 700
_TITLE_WEIGHT = 3.0
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or so that the this to use "
    "using what when where which why with you your".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)
_TITLE_RE = re.compile(r"^title:\s*(?P<title>.+)$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,3})\s+(?P<heading>.+?)\s*#*\s*$", re.MULTILINE)
_MARKUP_LINE_RE = re.compile(r"^\s*(?:import\s.+from\s.+|export\s.+|<[^>]*>\s*)$", re.MULTILINE)


@dataclass(frozen=True)
class DocSection:
    path: str
    title: str
    heading: str
    text: str


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1]


def _repo_docs_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs"
        if (candidate / "index.mdx").is_file() and (parent / "agents").is_dir():
            return candidate
    return None


def docs_dir() -> Path | None:
    """Return the documentation root: an explicit override, the bundled snapshot, or a repo checkout."""
    configured = os.environ.get(DOCS_DIR_ENV)
    if configured:
        return Path(configured)
    for candidate in (Path.cwd() / SNAPSHOT_DIR_NAME, Path("/workspace") / SNAPSHOT_DIR_NAME):
        if candidate.is_dir():
            return candidate
    return _repo_docs_dir()


def _split_sections(relative_path: str, content: str) -> list[DocSection]:
    frontmatter = _FRONTMATTER_RE.match(content)
    title = relative_path
    if frontmatter:
        title_match = _TITLE_RE.search(frontmatter.group("body"))
        if title_match:
            title = title_match.group("title").strip().strip("\"'") or relative_path
        content = content[frontmatter.end() :]
    content = _MARKUP_LINE_RE.sub("", content)

    sections: list[DocSection] = []
    heading = title
    start = 0
    for match in _HEADING_RE.finditer(content):
        text = content[start : match.start()].strip()
        if text:
            sections.append(DocSection(relative_path, title, heading, text))
        heading = match.group("heading").strip()
        start = match.end()
    text = content[start:].strip()
    if text:
        sections.append(DocSection(relative_path, title, heading, text))
    return sections


@cache
def _load_index(root: Path) -> tuple[tuple[DocSection, ...], tuple[Counter[str], ...], dict[str, float]]:
    sections: list[DocSection] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in DOC_SUFFIXES or not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        sections.extend(_split_sections(relative_path, path.read_text(encoding="utf-8", errors="replace")))

    term_counts = tuple(
        Counter(_tokens(section.text))
        + Counter({token: _TITLE_WEIGHT for token in _tokens(f"{section.title} {section.heading}")})
        for section in sections
    )
    document_frequency: Counter[str] = Counter()
    for counts in term_counts:
        document_frequency.update(counts.keys())
    total = max(len(sections), 1)
    idf = {term: math.log(1 + total / frequency) for term, frequency in document_frequency.items()}
    return tuple(sections), term_counts, idf


def _snippet(text: str, terms: set[str]) -> str:
    lowered = text.lower()
    positions = [index for term in terms if (index := lowered.find(term)) >= 0]
    start = max(min(positions) - SNIPPET_CHARS // 4, 0) if positions else 0
    snippet = text[start : start + SNIPPET_CHARS].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if start + SNIPPET_CHARS < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def search(query: str, limit: int = DEFAULT_LIMIT, root: Path | None = None) -> list[dict[str, str | float]]:
    """Return the best-matching documentation sections for ``query``."""
    docs_root = root or docs_dir()
    terms = set(_tokens(query))
    if docs_root is None or not docs_root.is_dir() or not terms:
        return []
    sections, term_counts, idf = _load_index(docs_root.resolve())

    scored: list[tuple[float, int]] = []
    for index, counts in enumerate(term_counts):
        matched = [term for term in terms if term in counts]
        if not matched:
            continue
        length_norm = 1 + math.log(1 + sum(counts.values()))
        score = sum(idf[term] * (1 + math.log(counts[term])) for term in matched) / length_norm
        score *= len(matched) / len(terms)
        scored.append((score, index))
    scored.sort(reverse=True)

    results: list[dict[str, str | float]] = []
    for score, index in scored[: max(1, min(limit, MAX_LIMIT))]:
        section = sections[index]
        results.append(
            {
                "path": section.path,
                "title": section.title,
                "section": section.heading,
                "score": round(score, 3),
                "excerpt": _snippet(section.text, terms),
            }
        )
    return results


def search_docs(query: str, limit: int = DEFAULT_LIMIT) -> str:
    """Search the NeMo Helix documentation for how a feature works.

    Use this before explaining a feature, workflow, or concept so the answer
    matches the documentation. ``query`` is a few keywords, for example
    ``guardrail virtual model`` or ``add inference provider``. Returns up to
    ``limit`` sections (at most 10) with the doc path, title, section heading,
    and a short excerpt. Read-only.
    """
    if not query.strip():
        return json.dumps({"status": "error", "message": "query is required"})
    if docs_dir() is None:
        return json.dumps({"status": "unavailable", "message": "No documentation is bundled with this assistant."})
    results = search(query, limit)
    if not results:
        return json.dumps({"status": "no_results", "message": f"No documentation matched {query!r}."})
    return json.dumps({"status": "ok", "results": results}, indent=2)
