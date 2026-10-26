#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail when an image ships a third-party component whose source its build did not collect.

Compares a syft JSON inventory of the image against the manifests the source collectors wrote
under ``source-distributions``. A coverage policy names first-party packages, which need no
collected source; vendored directories whose contents ship inside a collected parent's source; and
components embedded in a Python package whose sdist includes their source.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SOURCE_DISTRIBUTIONS = "source-distributions"
WORKSPACE_SDIST = re.compile(r"^(?P<name>.+)-(?P<version>[^-]+)\.tar\.gz$")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Policy:
    first_party: dict[str, frozenset[str]]
    vendored: tuple[tuple[str, str], ...]
    embedded: dict[tuple[str, str], str]

    @classmethod
    def load(cls, path: Path | None) -> Policy:
        if path is None:
            return cls(first_party={}, vendored=(), embedded={})
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            first_party={
                kind: frozenset(normalize(entry["name"]) for entry in entries)
                for kind, entries in data.get("first_party", {}).items()
            },
            vendored=tuple((entry["path"], normalize(entry["covered_by"])) for entry in data.get("vendored", ())),
            embedded={
                (entry["type"], entry["name"]): normalize(entry["covered_by"]) for entry in data.get("embedded", ())
            },
        )


@dataclass
class Collected:
    """Every component a source collector recorded, by syft component type."""

    python: set[tuple[str, str]]
    python_names: set[str]
    deb_sources: set[str]
    go_modules: set[tuple[str, str]]
    go_toolchains: set[str]
    crates: set[tuple[str, str]]
    npm: set[tuple[str, str]]
    cpython: set[str]
    covers: set[tuple[str, str, str]]

    @classmethod
    def read(cls, root: Path) -> Collected:
        collected = cls(set(), set(), set(), set(), set(), set(), set(), set(), set())

        def lines(pattern: str):
            for path in root.rglob(pattern):
                yield from (line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        for line in lines("downloaded-pypi-sdists*.txt"):
            name, _, version = line.split("\t", 1)[0].partition("==")
            collected.python.add((normalize(name), version))
        for path in root.rglob("workspace/*.tar.gz"):
            if match := WORKSPACE_SDIST.match(path.name):
                collected.python.add((normalize(match["name"]), match["version"]))
        for line in lines("downloaded-apt-sources.txt"):
            collected.deb_sources.add(line.split("=", 1)[0])
        for line in lines("downloaded-go-modules.txt"):
            path, _, version = line.split("\t", 1)[0].rpartition("@")
            collected.go_modules.add((path, version))
        for line in lines("downloaded-go-toolchain-sources.txt"):
            collected.go_toolchains.add(line.split("\t", 1)[0])
        for pattern in ("downloaded-cargo-crates.txt", "local-cargo-crates.txt"):
            for line in lines(pattern):
                name, _, version = line.split("\t", 1)[0].partition("==")
                collected.crates.add((name, version))
        for line in lines("downloaded-npm-packages*.txt"):
            name, _, version = line.split("\t", 1)[0].rpartition("@")
            collected.npm.add((name, version))
        for line in lines("downloaded-cpython-source.txt"):
            collected.cpython.add(line.split("\t", 1)[0].removeprefix("Python-").removesuffix(".tgz"))
        for line in lines("downloaded-pinned-sources.txt"):
            fields = line.split("\t")
            for cover in filter(None, fields[4].split(",") if len(fields) > 4 else ()):
                kind, _, rest = cover.partition(":")
                name, _, version = rest.rpartition("@")
                collected.covers.add((kind, normalize(name) if kind == "python" else name, version))
        collected.python |= {(name, version) for kind, name, version in collected.covers if kind == "python"}
        collected.python_names = {name for name, _ in collected.python}
        return collected


def component_paths(artifact: dict) -> list[str]:
    return [location.get("path", "") for location in artifact.get("locations", ())]


def is_covered(artifact: dict, collected: Collected, policy: Policy) -> bool:
    kind, name, version = artifact["type"], artifact["name"], artifact["version"]
    paths = component_paths(artifact)
    if any(fnmatch.fnmatchcase(normalize(name), pattern) for pattern in policy.first_party.get(kind, ())):
        return True
    for vendored_path, parent in policy.vendored:
        if any(vendored_path in path for path in paths) and parent in collected.python_names:
            return True
    if (kind, name, version) in collected.covers:
        return True
    if policy.embedded.get((kind, name)) in collected.python_names:
        return True
    if kind == "python":
        return (normalize(name), version) in collected.python
    if kind == "deb":
        source = (artifact.get("metadata") or {}).get("source") or name
        return source in collected.deb_sources or name in collected.deb_sources
    if kind == "go-module":
        return (name, version) in collected.go_modules or (name == "stdlib" and version in collected.go_toolchains)
    if kind == "rust-crate":
        return (name, version) in collected.crates
    if kind == "npm":
        return (name, version) in collected.npm
    if kind == "binary" and name == "python":
        return version in collected.cpython
    return False


def uncovered(sbom: dict, collected: Collected, policy: Policy) -> dict[str, list[str]]:
    gaps: dict[str, set[str]] = defaultdict(set)
    for artifact in sbom["artifacts"]:
        paths = component_paths(artifact)
        if paths and all(f"/{SOURCE_DISTRIBUTIONS}/" in path or "/.source-distributions/" in path for path in paths):
            continue
        if not is_covered(artifact, collected, policy):
            gaps[artifact["type"]].add(f"{artifact['name']} {artifact['version']} ({paths[0] if paths else 'no path'})")
    return {kind: sorted(items) for kind, items in sorted(gaps.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sbom", required=True, help="syft JSON inventory of the image")
    parser.add_argument("--sources", required=True, help="the image's source-distributions directory, copied out")
    parser.add_argument("--policy", help="coverage policy JSON")
    args = parser.parse_args()

    sources = Path(args.sources)
    if any(sources.rglob("source-collection-disabled.txt")):
        print(f"{sources}: source collection was disabled in this build", file=sys.stderr)
        return 1
    gaps = uncovered(
        json.loads(Path(args.sbom).read_text(encoding="utf-8")),
        Collected.read(sources),
        Policy.load(Path(args.policy) if args.policy else None),
    )
    for kind, items in gaps.items():
        print(f"{len(items)} {kind} component(s) without collected source:")
        print("".join(f"  {item}\n" for item in items), end="")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
