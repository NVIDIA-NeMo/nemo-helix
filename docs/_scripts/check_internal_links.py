#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that published Fern docs use routable, canonical internal links."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

DOCS_HOST = "docs.nvidia.com"
DOCS_HOST_PREFIX = "/nemo-platform"
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(\s*<?([^)\s>]+)>?(?:\s+['\"][^)]*['\"])?\s*\)")
HREF_RE = re.compile(r"\bhref\s*=\s*(['\"])([^'\"]+)\1")
ABSOLUTE_DOCS_URL_RE = re.compile(r"https?://docs\.nvidia\.com/nemo-platform(?:/[^<>\s)'\"`]*)?")
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})", re.MULTILINE)


@dataclass(frozen=True)
class Link:
    file: Path
    line: int
    target: str


@dataclass(frozen=True)
class Redirect:
    source: str
    destination: str

    def apply(self, path: str) -> str | None:
        marker = ":path*"
        if marker not in self.source:
            return self.destination if path == self.source else None
        prefix, suffix = self.source.split(marker, 1)
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        end = len(path) - len(suffix) if suffix else len(path)
        captured = path[len(prefix) : end]
        return self.destination.replace(marker, captured)


@dataclass
class RouteIndex:
    routes: set[str]
    files: dict[Path, str]
    redirects: list[Redirect]

    def resolve_redirect(self, path: str) -> str | None:
        current = path
        seen = {current}
        redirected = False
        for _ in range(20):
            match = None
            for redirect in self.redirects:
                match = redirect.apply(current)
                if match is not None:
                    break
            if match is None:
                return current if redirected else None
            current = normalize_path(match)
            redirected = True
            if current in seen:
                return None
            seen.add(current)
        return None


def slugify(value: str) -> str:
    """Approximate Fern's default nav slug generation."""
    value = value.replace("&", " and ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", value)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    return value.strip("-").lower()


def normalize_path(path: str) -> str:
    path = re.sub(r"/+", "/", path)
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def load_route_index(nav_path: Path, config_path: Path) -> RouteIndex:
    nav = yaml.safe_load(nav_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    routes: set[str] = set()
    files: dict[Path, str] = {}

    def visit(nodes: list[object], parent: tuple[str, ...]) -> None:
        for raw_node in nodes:
            if not isinstance(raw_node, dict):
                continue
            label = next((raw_node[key] for key in ("section", "page", "api") if key in raw_node), None)
            if not isinstance(label, str):
                continue
            segment = str(raw_node.get("slug") or slugify(label))
            segments = (*parent, segment)
            route = normalize_path("/" + "/".join(segments))
            routes.add(route)
            raw_path = raw_node.get("path")
            if isinstance(raw_path, str):
                source = (nav_path.parent / raw_path).resolve()
                # Fern permits the same source page at multiple routes (for example, a
                # section landing page repeated as its "Overview" child). Scan it once.
                files.setdefault(source, route)
            contents = raw_node.get("contents")
            if isinstance(contents, list):
                visit(contents, segments)

    navigation = nav.get("navigation")
    if not isinstance(navigation, list):
        raise ValueError(f"{nav_path}: navigation must be a list")
    visit(navigation, ())

    redirects = [
        Redirect(str(item["source"]), str(item["destination"]))
        for item in config.get("redirects", [])
        if isinstance(item, dict) and "source" in item and "destination" in item
    ]
    return RouteIndex(routes=routes, files=files, redirects=redirects)


def mask_code(text: str) -> str:
    """Blank code spans and fenced blocks while preserving line numbers."""
    chars = list(text)
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if match and (fence is None or match.group(1)[0] == fence[0]):
            fence = None if fence else match.group(1)
            for index in range(offset, offset + len(line)):
                if chars[index] != "\n":
                    chars[index] = " "
        elif fence:
            for index in range(offset, offset + len(line)):
                if chars[index] != "\n":
                    chars[index] = " "
        offset += len(line)
    masked = "".join(chars)
    return re.sub(r"`[^`\n]*`", lambda match: " " * len(match.group(0)), masked)


def extract_links(file: Path) -> list[Link]:
    text = mask_code(file.read_text(encoding="utf-8"))
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, target_group in ((MARKDOWN_LINK_RE, 1), (HREF_RE, 2)):
        for match in pattern.finditer(text):
            target = match.group(target_group)
            matches.append((match.start(), match.end(), target))
            occupied.append((match.start(), match.end()))
    for match in ABSOLUTE_DOCS_URL_RE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        matches.append((match.start(), match.end(), match.group(0)))
    return [Link(file=file, line=text.count("\n", 0, start) + 1, target=target) for start, _, target in sorted(matches)]


def internal_path(target: str) -> tuple[str, str] | None:
    split = urlsplit(target)
    if split.scheme or split.netloc:
        if split.netloc != DOCS_HOST or not split.path.startswith(DOCS_HOST_PREFIX):
            return None
        path = split.path[len(DOCS_HOST_PREFIX) :] or "/"
    else:
        path = split.path
    if (
        not path.startswith("/documentation/")
        and path != "/documentation"
        and not path.startswith("/latest/documentation")
    ):
        return None
    return normalize_path(path), split.fragment


def check_links(index: RouteIndex) -> tuple[list[tuple[Link, str]], list[tuple[Link, str]]]:
    redirected: list[tuple[Link, str]] = []
    broken: list[tuple[Link, str]] = []
    for file in sorted(index.files):
        if not file.exists():
            broken.append((Link(file=file, line=1, target=str(file)), "nav references a missing file"))
            continue
        for link in extract_links(file):
            parsed = internal_path(link.target)
            if parsed is None:
                continue
            path, fragment = parsed
            if path in index.routes:
                continue
            if path.startswith("/latest/"):
                unversioned = path[len("/latest") :]
                if unversioned in index.routes:
                    suggestion = unversioned
                    if fragment:
                        suggestion += f"#{fragment}"
                    redirected.append((link, suggestion))
                    continue
            destination = index.resolve_redirect(path)
            canonical_destination = destination
            if canonical_destination and canonical_destination.startswith("/latest/"):
                canonical_destination = canonical_destination[len("/latest") :]
            if canonical_destination in index.routes:
                suggestion = canonical_destination
                if fragment:
                    suggestion += f"#{fragment}"
                redirected.append((link, suggestion))
            else:
                broken.append((link, "no matching Fern route or redirect"))
    return redirected, broken


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    docs_dir = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav", type=Path, default=docs_dir / "fern/versions/latest.yml")
    parser.add_argument("--config", type=Path, default=docs_dir / "fern/docs.yml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        index = load_route_index(args.nav.resolve(), args.config.resolve())
        redirected, broken = check_links(index)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"docs-internal-links: configuration error: {exc}", file=sys.stderr)
        return 2

    docs_dir = Path(__file__).resolve().parents[1]
    if not redirected and not broken:
        print(f"docs-internal-links: checked {len(index.files)} published pages; all internal links are canonical.")
        return 0

    for link, detail in broken:
        print(
            f"{link.file.relative_to(docs_dir)}:{link.line}: unroutable internal link {link.target!r} ({detail})",
            file=sys.stderr,
        )
    for link, suggestion in redirected:
        print(
            f"{link.file.relative_to(docs_dir)}:{link.line}: noncanonical internal link {link.target!r}; use {suggestion!r}",
            file=sys.stderr,
        )
    print(
        f"docs-internal-links: {len(broken)} unroutable and {len(redirected)} redirected/noncanonical link(s).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
