#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the published tarball of every production npm package a bundled web app is built from.

For a pnpm workspace, takes the package tree from ``pnpm list --prod --depth Infinity --json`` and
checks each tarball against the ``integrity`` hash ``pnpm-lock.yaml`` records, the same hash pnpm
checked at install. For an app shipped with an npm ``package-lock.json`` (v2 or v3), takes its
non-dev packages and their recorded hashes. Workspace packages and packages matching
``--exclude`` are skipped, though their dependencies are still collected.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

NPM_REGISTRY = "https://registry.npmjs.org"
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_WORKERS = 8
LOCK_KEY = re.compile(r"^  '?(?P<name>@?[^@'\s]+)@(?P<version>[^'(\s]+)'?:\s*$")
LOCK_INTEGRITY = re.compile(r"integrity: (?P<integrity>sha512-[A-Za-z0-9+/=]+)")


@dataclass(frozen=True)
class NpmPackage:
    name: str
    version: str
    resolved: str

    @property
    def filename(self) -> str:
        return f"{self.name.removeprefix('@').replace('/', '_')}-{self.version}.tgz"


def production_packages(pnpm_list: list[dict], exclude: list[str]) -> list[NpmPackage]:
    """Every registry package in ``pnpm list --json`` output, once each, including the dependencies of skipped ones.

    pnpm prints a repeated subtree in full in one place and elsewhere as a ``deduped`` stub, in no fixed
    order, and breaks cycles the same way, so every node of the (finite) tree is walked.
    """
    found: dict[tuple[str, str], NpmPackage] = {}

    def walk(dependencies: dict | None) -> None:
        for key, info in (dependencies or {}).items():
            # An aliased dependency ("typescript": "npm:@typescript/typescript6@6.0.2") is keyed by its alias.
            name = info.get("from") or key
            version = info.get("version", "")
            skipped = version.startswith(("link:", "file:", "workspace:")) or any(
                fnmatch.fnmatchcase(name, pattern) for pattern in exclude
            )
            if not skipped:
                tarball = f"{NPM_REGISTRY}/{name}/-/{name.rsplit('/', 1)[-1]}-{version}.tgz"
                found[(name, version)] = NpmPackage(name, version, info.get("resolved") or tarball)
            walk(info.get("dependencies"))

    for project in pnpm_list:
        walk(project.get("dependencies"))
    return [found[key] for key in sorted(found)]


def lock_integrities(pnpm_lock: str) -> dict[tuple[str, str], str]:
    """``(name, version) -> integrity`` from the ``packages:`` section of a v9 ``pnpm-lock.yaml``."""
    integrities: dict[tuple[str, str], str] = {}
    in_packages = False
    current: tuple[str, str] | None = None
    for line in pnpm_lock.splitlines():
        if line and not line.startswith(" "):
            in_packages = line.rstrip() == "packages:"
            current = None
            continue
        if not in_packages:
            continue
        if match := LOCK_KEY.match(line):
            current = (match["name"], match["version"])
        elif current is not None and (match := LOCK_INTEGRITY.search(line)):
            integrities[current] = match["integrity"]
            current = None
    return integrities


def package_lock_packages(
    package_lock: dict, exclude: list[str]
) -> tuple[list[NpmPackage], dict[tuple[str, str], str]]:
    """Non-dev registry packages of an npm v2/v3 ``package-lock.json``, and their integrity hashes."""
    packages: dict[tuple[str, str], NpmPackage] = {}
    integrities: dict[tuple[str, str], str] = {}
    for path, info in package_lock["packages"].items():
        if not path or info.get("dev") or info.get("devOptional") or info.get("link") or "resolved" not in info:
            continue
        name = info.get("name") or path.rsplit("node_modules/", 1)[-1]
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude):
            continue
        key = (name, info["version"])
        packages[key] = NpmPackage(name, info["version"], info["resolved"])
        if "integrity" in info:
            integrities[key] = info["integrity"]
    return [packages[key] for key in sorted(packages)], integrities


def sha512_integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def download(package: NpmPackage, integrity: str, destination: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(package.resolved, timeout=120) as response:
                data = response.read()
            actual = sha512_integrity(data)
            if actual != integrity:
                raise RuntimeError(f"{package.name}@{package.version}: expected {integrity}, downloaded {actual}")
            destination.write_bytes(data)
            return
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {package.resolved}: {last_error}")


def source_collection_enabled() -> bool:
    return os.environ.get("NHX_COLLECT_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def record_source_collection_disabled(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "source-collection-disabled.txt").write_text(
        "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnpm-list", help="output of `pnpm list --prod --depth Infinity --json`")
    parser.add_argument("--pnpm-lock", help="the workspace's pnpm-lock.yaml, with --pnpm-list")
    parser.add_argument("--package-lock", help="an npm package-lock.json, instead of --pnpm-list and --pnpm-lock")
    parser.add_argument("--exclude", action="append", default=[], help="package name pattern to skip, e.g. @nvidia/*")
    parser.add_argument("--label", required=True, help="names this app's manifest, downloaded-npm-packages-<label>.txt")
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    args = parser.parse_args()
    if bool(args.package_lock) == bool(args.pnpm_list and args.pnpm_lock):
        parser.error("pass either --package-lock, or --pnpm-list and --pnpm-lock")

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    if args.package_lock:
        packages, integrities = package_lock_packages(
            json.loads(Path(args.package_lock).read_text(encoding="utf-8")), args.exclude
        )
    else:
        packages = production_packages(json.loads(Path(args.pnpm_list).read_text(encoding="utf-8")), args.exclude)
        integrities = lock_integrities(Path(args.pnpm_lock).read_text(encoding="utf-8"))
    unlocked = [
        f"{package.name}@{package.version}"
        for package in packages
        if (package.name, package.version) not in integrities
    ]
    if unlocked:
        raise RuntimeError(f"{args.package_lock or args.pnpm_lock} has no integrity for: {', '.join(unlocked)}")

    npm_dir = output_dir / "npm"
    npm_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(
            pool.map(
                lambda package: download(
                    package, integrities[(package.name, package.version)], npm_dir / package.filename
                ),
                packages,
            )
        )
    with (manifests / f"downloaded-npm-packages-{args.label}.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(f"{package.name}@{package.version}\t{package.filename}\n" for package in packages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
