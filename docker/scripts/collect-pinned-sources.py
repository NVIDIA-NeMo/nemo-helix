#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download source archives pinned by URL and SHA-256 in a JSON manifest.

For components no package-manager collector covers: static binaries copied from other images,
release downloads, and Python packages published without a source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import sleep

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".zip", ".crate")

DOWNLOAD_ATTEMPTS = 5


@dataclass(frozen=True)
class PinnedSource:
    name: str
    version: str
    license: str
    url: str
    sha256: str
    covers: tuple[str, ...]
    #: Where the covered copy ships when it isn't the installed package, e.g. vendored inside another.
    bundled_in: str | None = None

    @property
    def filename(self) -> str:
        """``<name>-<version>`` plus the URL's archive suffix, so release archives named ``v1.0.tar.gz`` don't collide."""
        path = urllib.parse.urlparse(self.url).path
        suffix = next((s for s in ARCHIVE_SUFFIXES if path.endswith(s)), PurePosixPath(path).suffix)
        return f"{self.name}-{self.version}{suffix}"


def load_manifest(path: Path) -> list[PinnedSource]:
    entries = json.loads(path.read_text(encoding="utf-8"))["sources"]
    sources = []
    for entry in entries:
        source = PinnedSource(
            name=entry["name"],
            version=entry["version"],
            license=entry["license"],
            url=entry["url"],
            sha256=entry["sha256"].lower(),
            covers=tuple(entry.get("covers", ())),
            bundled_in=entry.get("bundled_in"),
        )
        if not SHA256_PATTERN.fullmatch(source.sha256):
            raise ValueError(f"{path}: {source.name} has an invalid sha256: {entry['sha256']!r}")
        if not source.url.startswith("https://"):
            raise ValueError(f"{path}: {source.name} must be downloaded over https: {source.url}")
        sources.append(source)
    return sources


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


INSTALLED_DISTRIBUTIONS = (
    "import importlib.metadata as m, json; "
    "print(json.dumps([[d.metadata['Name'], d.version] for d in m.distributions()]))"
)


def installed_versions(python: str) -> dict[str, set[str]]:
    """The version of each distribution installed in ``python``'s environment, by normalized name."""
    output = subprocess.run([python, "-c", INSTALLED_DISTRIBUTIONS], check=True, capture_output=True, text=True).stdout
    versions: dict[str, set[str]] = {}
    for name, version in json.loads(output):
        versions.setdefault(normalize(name), set()).add(version)
    return versions


def stale_python_pins(sources: list[PinnedSource], installed: dict[str, set[str]]) -> list[str]:
    """Python packages a manifest pins at a version other than the one installed, as when a lock moved on.

    A pin for a copy bundled elsewhere, such as inside another package, is independent of what's installed.
    """
    stale = []
    for source in sources:
        if source.bundled_in:
            continue
        for cover in source.covers:
            kind, _, rest = cover.partition(":")
            name, _, version = rest.rpartition("@")
            found = installed.get(normalize(name), set())
            if kind == "python" and found and version not in found:
                stale.append(f"{source.name} pins {name} {version}, but {', '.join(sorted(found))} is installed")
    return stale


def source_collection_enabled() -> bool:
    return os.environ.get("NHX_COLLECT_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def record_source_collection_disabled(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "source-collection-disabled.txt").write_text(
        "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable\n",
        encoding="utf-8",
    )


def verify_sha256(data: bytes, expected: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"sha256 mismatch: expected {expected}, got {actual}")


def download(source: PinnedSource, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        verify_sha256(destination.read_bytes(), source.sha256)
        return
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(source.url, timeout=120) as response:
                data = response.read()
            verify_sha256(data, source.sha256)
            destination.write_bytes(data)
            return
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            sleep(2**attempt)
    raise RuntimeError(f"could not download {source.name} {source.version} from {source.url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="JSON manifest of pinned sources")
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    parser.add_argument(
        "--installed-in",
        action="append",
        default=[],
        metavar="PYTHON",
        help="Python whose environment must have each pinned Python package at the pinned version, if at all",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    sources = [source for manifest in args.manifest for source in load_manifest(Path(manifest))]
    stale = [line for python in args.installed_in for line in stale_python_pins(sources, installed_versions(python))]
    if stale:
        raise RuntimeError("update the pinned sources: " + "; ".join(stale))
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    lines = []
    for source in sources:
        download(source, output_dir / "pinned" / source.filename)
        covers = ",".join(source.covers)
        lines.append(f"{source.name}=={source.version}\t{source.filename}\t{source.url}\t{source.license}\t{covers}\n")
    with (manifests / "downloaded-pinned-sources.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
