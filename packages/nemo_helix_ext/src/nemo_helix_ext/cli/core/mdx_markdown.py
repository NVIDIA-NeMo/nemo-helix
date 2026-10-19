# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Fern MDX doc pages into plain Markdown for terminal readers.

The published docs site renders MDX components (``<Note>``, ``<Tabs>``, ``<Card>``...)
and shared snippets. ``nemo docs`` reads the raw source, so this module maps that
markup onto Markdown equivalents. Fenced code blocks and inline code are never touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DOCS_SITE_URL = "https://docs.nvidia.com/nemo-helix"

_MAX_INCLUDE_DEPTH = 5

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_TITLE_RE = re.compile(r"^title:\s*(?P<q>[\"']?)(?P<title>.*?)(?P=q)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)(?:.+?)\1")
_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}")
_ATTR_RE = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|\{\s*["']([^"']*)["']\s*\})""")
# A line holding exactly one JSX component tag, e.g. `<Note>`, `</Tabs>`, `<Card title="x" />`.
_COMPONENT_LINE_RE = re.compile(r"^\s*<(?P<close>/)?(?P<name>[A-Z][\w.]*)(?P<attrs>[^>]*?)(?P<self>/)?>\s*$")
_COMPONENT_TAG_RE = re.compile(r"</?[A-Z][\w.]*(?:\s[^<>]*)?/?>")
_BUTTON_RE = re.compile(r"<Button\b(?P<attrs>[^>]*)>(?P<text>.*?)</Button>")
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>")
_ANCHOR_OPEN_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>")
_MD_TAG_RE = re.compile(r"<span\s+class(?:Name)?=\"md-tag\">(?P<text>.*?)</span>")
_HIDDEN_SPAN_RE = re.compile(r"<span\s+aria-hidden=\"true\">.*?</span>")
_PRESENTATIONAL_TAG_RE = re.compile(r"</?(?:small|span|div)\b[^>]*>|</a>")
_SITE_LINK_RE = re.compile(r"\]\((/documentation/[^)\s]*)")

_CALLOUTS = {
    "Note": "Note",
    "Info": "Info",
    "Tip": "Tip",
    "Warning": "Warning",
    "Error": "Error",
    "Check": "Check",
    "Success": "Success",
    "Callout": "Note",
}
_TITLED_SECTIONS = {"Tab", "Accordion"}


def _attrs(raw: str) -> dict[str, str]:
    return {m.group(1): next(g for g in m.groups()[1:] if g is not None) for m in _ATTR_RE.finditer(raw)}


def _site_url(href: str) -> str:
    """Make site-relative ``/documentation/...`` links usable outside the docs site."""
    if href.startswith("/documentation/") or href == "/documentation":
        return f"{DOCS_SITE_URL}{href}"
    return href


def _link(text: str, href: str | None) -> str:
    return f"[{text}]({_site_url(href)})" if href else text


@dataclass
class _State:
    docs_root: Path | None
    depth: int = 0
    out: list[str] = field(default_factory=list)
    callouts: int = 0
    fence: str | None = None
    in_comment: bool = False
    pending_link: tuple[str, list[str]] | None = None
    steps: list[int] = field(default_factory=list)

    @property
    def prefix(self) -> str:
        return "> " * self.callouts

    def emit(self, text: str) -> None:
        line = (self.prefix + text).rstrip()
        # Dropped tag lines leave stray blank lines; collapse runs of them outside code.
        if self.fence is None and not line.strip("> ") and self.out and not self.out[-1].strip("> "):
            return
        self.out.append(line)


def _split_frontmatter(source: str) -> tuple[str | None, str]:
    match = _FRONTMATTER_RE.match(source)
    if match is None:
        return None, source
    title_match = _TITLE_RE.search(match.group(1))
    title = title_match.group("title").strip() if title_match else None
    return title or None, source[match.end() :]


def _resolve_snippet(docs_root: Path | None, src: str) -> Path | None:
    """Map a Fern ``<Markdown src="/snippets/...">`` path to a file under ``docs/fern/snippets``."""
    if docs_root is None or not src.startswith("/snippets/"):
        return None
    snippets_root = (docs_root / "fern" / "snippets").resolve()
    candidate = (snippets_root / src.removeprefix("/snippets/")).resolve()
    if not candidate.is_relative_to(snippets_root) or not candidate.is_file():
        return None
    return candidate


def _protect_inline_code(line: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    return _INLINE_CODE_RE.sub(_stash, line), spans


def _restore_inline_code(line: str, spans: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], line)


def _strip_comments(line: str, state: _State) -> str:
    if state.in_comment:
        end = line.find("*/}")
        if end == -1:
            return ""
        state.in_comment = False
        line = line[end + 3 :]
    line = _COMMENT_RE.sub("", line)
    start = line.find("{/*")
    if start != -1:
        state.in_comment = True
        line = line[:start]
    return line


def _convert_inline(line: str, state: _State) -> str | None:
    """Rewrite inline HTML/JSX on a prose line. Returns None when the line is consumed."""
    line = _BUTTON_RE.sub(lambda m: _link(m.group("text"), _attrs(m.group("attrs")).get("href")), line)
    line = _MD_TAG_RE.sub(lambda m: f"`{m.group('text')}`", line)
    line = _HIDDEN_SPAN_RE.sub("", line)
    line = _ANCHOR_RE.sub(lambda m: _link(m.group("text"), _attrs(m.group("attrs")).get("href")), line)

    # An <a href> whose text spans several lines: buffer the text until </a>.
    if state.pending_link is not None:
        href, parts = state.pending_link
        text, closed, rest = line.partition("</a>")
        parts.append(_PRESENTATIONAL_TAG_RE.sub("", text).strip())
        if not closed:
            return None
        state.pending_link = None
        line = _link(" ".join(p for p in parts if p), href) + rest
    elif (opening := _ANCHOR_OPEN_RE.search(line)) is not None:
        href = _attrs(opening.group("attrs")).get("href")
        before = line[: opening.start()]
        if href is None:
            line = before + line[opening.end() :]
        else:
            state.pending_link = (href, [_PRESENTATIONAL_TAG_RE.sub("", line[opening.end() :]).strip()])
            return before.rstrip() or None

    line = _PRESENTATIONAL_TAG_RE.sub("", line)
    line = _COMPONENT_TAG_RE.sub("", line)
    return _SITE_LINK_RE.sub(lambda m: f"]({_site_url(m.group(1))}", line)


def _handle_component(match: re.Match[str], state: _State) -> None:
    name = match.group("name")
    closing = match.group("close") is not None
    self_closing = match.group("self") is not None
    attrs = _attrs(match.group("attrs"))

    if name in _CALLOUTS:
        if closing:
            # Drop the quoted blank lines that preceded the closing tag.
            while state.out and state.out[-1].startswith(">") and not state.out[-1].strip("> "):
                state.out.pop()
            state.callouts = max(0, state.callouts - 1)
            state.emit("")
            return
        if not self_closing:
            label = _CALLOUTS[name]
            title = attrs.get("title")
            state.emit("")
            state.callouts += 1
            state.emit(f"**{label}: {title}**" if title else f"**{label}:**")
        return

    if closing:
        if name == "Steps" and state.steps:
            state.steps.pop()
        return

    if name == "Markdown":
        _include_snippet(attrs.get("src", ""), state)
    elif name == "Card":
        title = attrs.get("title", "")
        link = _link(title, attrs.get("href")) if title else ""
        if self_closing:
            if link:
                state.emit(f"- {link}")
        elif link:
            state.emit(f"**{link}**")
    elif name == "Steps":
        state.steps.append(0)
    elif name == "Step":
        if state.steps:
            state.steps[-1] += 1
        number = f"Step {state.steps[-1]}" if state.steps else "Step"
        title = attrs.get("title")
        state.emit(f"**{number}: {title}**" if title else f"**{number}**")
    elif name in _TITLED_SECTIONS and attrs.get("title"):
        state.emit(f"**{attrs['title']}**")
    # Anything else (wrappers such as <Cards>, <Tabs>, unknown components) is dropped,
    # leaving its children to render as ordinary Markdown.


def _include_snippet(src: str, state: _State) -> None:
    snippet = _resolve_snippet(state.docs_root, src)
    if snippet is None or state.depth >= _MAX_INCLUDE_DEPTH:
        return
    _, body = _split_frontmatter(snippet.read_text(encoding="utf-8"))
    nested = _State(docs_root=state.docs_root, depth=state.depth + 1)
    _convert_body(body, nested)
    for line in nested.out:
        state.emit(line)


def _convert_body(body: str, state: _State) -> None:
    for raw_line in body.splitlines():
        fence_match = _FENCE_RE.match(raw_line)
        if state.fence is not None:
            if fence_match and fence_match.group("fence").startswith(state.fence) and not raw_line.strip(" `~"):
                state.fence = None
            state.emit(raw_line)
            continue
        if fence_match and not state.in_comment:
            state.fence = fence_match.group("fence")
            state.emit(raw_line)
            continue

        line, spans = _protect_inline_code(raw_line)
        line = _strip_comments(line, state)
        if not line.strip() and raw_line.strip():
            continue  # the line held only a comment

        component = _COMPONENT_LINE_RE.match(line)
        if component is not None and state.pending_link is None:
            _handle_component(component, state)
            continue

        converted = _convert_inline(line, state)
        if converted is None:
            continue
        if not converted.strip() and line.strip():
            continue  # the line held only presentational markup
        state.emit(_restore_inline_code(converted, spans))


def mdx_to_markdown(source: str, docs_root: Path | None = None) -> str:
    """Convert an MDX doc page into plain Markdown.

    Args:
        source: Raw ``.mdx`` file contents.
        docs_root: The ``docs/`` directory, used to expand ``<Markdown src="/snippets/..."/>`` includes.

    Returns:
        Markdown with frontmatter, MDX comments, and JSX components removed or mapped to
        Markdown equivalents. The frontmatter ``title`` becomes an H1 unless the page has one.
    """
    title, body = _split_frontmatter(source)
    state = _State(docs_root=docs_root)
    _convert_body(body, state)

    lines = state.out
    while lines and not lines[0].strip():
        lines.pop(0)
    if title and not (lines and lines[0].startswith("# ")):
        lines = [f"# {title}", "", *lines]
    return "\n".join(lines).rstrip() + "\n"
