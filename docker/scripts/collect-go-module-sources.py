#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the source of every Go module, and of the Go standard library, compiled into Go binaries.

Reads each binary's embedded build information, downloads every module's source zip from the Go
module proxy and checks it against the ``h1:`` hash the binary records (or, when the binary
records none, the Go checksum database's), and downloads the Go toolchain source for each Go
version the binaries were built with. Needs no Go toolchain. A main module built from a checkout
reports version ``(devel)``, and one built from a modified checkout a ``+dirty`` version; neither is
on the proxy, so collect their source as pinned archives instead.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BUILDINFO_MAGIC = b"\xff Go buildinf:"
GO_PROXY = "https://proxy.golang.org"
GO_SUMDB = "https://sum.golang.org"
GO_DOWNLOADS = "https://go.dev/dl/"
DEVEL_VERSION = "(devel)"
DOWNLOAD_WORKERS = 8

DOWNLOAD_ATTEMPTS = 5


@dataclass(frozen=True)
class GoModule:
    path: str
    version: str
    sum: str

    @property
    def filename(self) -> str:
        return f"{self.path.replace('/', '_')}@{self.version}.zip"


@dataclass(frozen=True)
class BuildInfo:
    go_version: str
    modules: tuple[GoModule, ...]


def _uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def read_build_info(data: bytes) -> tuple[str, str]:
    """The Go version and module info strings a Go 1.18+ binary embeds after its buildinfo magic."""
    start = data.find(BUILDINFO_MAGIC)
    while start != -1 and start % 16:
        start = data.find(BUILDINFO_MAGIC, start + 1)
    if start == -1:
        raise RuntimeError("no Go build information found")
    if not data[start + 15] & 0x2:
        raise RuntimeError("Go build information predates Go 1.18 and is not supported")
    offset = start + 32
    fields = []
    for _ in range(2):
        length, offset = _uvarint(data, offset)
        fields.append(data[offset : offset + length])
        offset += length
    version, modinfo = fields
    # Module info is framed by 16-byte binary sentinels.
    return version.decode("utf-8"), modinfo[16:-16].decode("utf-8") if len(modinfo) >= 32 else ""


def parse_module_info(go_version: str, modinfo: str) -> BuildInfo:
    """Every module in ``go version -m`` style module info, with replacements applied."""
    modules: list[GoModule] = []
    for line in modinfo.splitlines():
        fields = line.split("\t")
        kind = fields[0]
        if kind in {"mod", "dep"}:
            modules.append(GoModule(fields[1], fields[2], fields[3] if len(fields) > 3 else ""))
        elif kind == "=>":
            if len(fields) < 3 or not fields[2].startswith("v"):
                raise RuntimeError(f"{modules[-1].path} is replaced by a local directory: {line}")
            modules[-1] = GoModule(fields[1], fields[2], fields[3] if len(fields) > 3 else "")
    return BuildInfo(go_version=go_version, modules=tuple(modules))


def not_on_proxy(module: GoModule) -> bool:
    """Modules built from a local checkout, which the proxy cannot serve; collect these as pinned archives."""
    return module.version == DEVEL_VERSION or module.version.endswith("+dirty")


def linked_modules(binaries: list[Path]) -> tuple[list[GoModule], dict[GoModule, list[str]], set[str]]:
    """Each module once across ``binaries``, with the hash any of them records; plus which use it, and Go versions."""
    by_version: dict[tuple[str, str], GoModule] = {}
    binaries_by_version: dict[tuple[str, str], list[str]] = {}
    go_versions: set[str] = set()
    for binary in binaries:
        info = parse_module_info(*read_build_info(binary.read_bytes()))
        go_versions.add(info.go_version)
        for module in info.modules:
            if not_on_proxy(module):
                continue
            key = (module.path, module.version)
            known = by_version.get(key)
            if known is not None and known.sum and module.sum and known.sum != module.sum:
                raise RuntimeError(f"{module.path}@{module.version} has different hashes: {known.sum}, {module.sum}")
            if known is None or not known.sum:
                by_version[key] = module
            binaries_by_version.setdefault(key, []).append(str(binary))
    modules = [by_version[key] for key in sorted(by_version)]
    used_by = {by_version[key]: binaries_by_version[key] for key in sorted(by_version)}
    return modules, used_by, go_versions


def escape_module_path(path: str) -> str:
    """Escape a module path or version for the proxy protocol: each upper-case letter becomes ``!`` and its lower case."""
    return "".join(f"!{char.lower()}" if char.isupper() else char for char in path)


def module_zip_hash(data: bytes) -> str:
    """Go's ``h1:`` hash of a module zip: SHA-256 over sorted ``<sha256>  <name>`` lines of its files."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        summary = "".join(f"{hashlib.sha256(archive.read(name)).hexdigest()}  {name}\n" for name in names)
    return "h1:" + base64.b64encode(hashlib.sha256(summary.encode("utf-8")).digest()).decode("ascii")


def fetch(url: str, timeout: int = 120) -> bytes:
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


def checksum_database_hash(module: GoModule) -> str:
    lookup = fetch(
        f"{GO_SUMDB}/lookup/{escape_module_path(module.path)}@{escape_module_path(module.version)}", timeout=60
    )
    for line in lookup.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == module.path and fields[1] == module.version:
            return fields[2]
    raise RuntimeError(f"the Go checksum database has no hash for {module.path}@{module.version}")


def download_module(module: GoModule, destination: Path) -> None:
    expected = module.sum or checksum_database_hash(module)
    data = fetch(f"{GO_PROXY}/{escape_module_path(module.path)}/@v/{escape_module_path(module.version)}.zip")
    actual = module_zip_hash(data)
    if actual != expected:
        raise RuntimeError(f"{module.path}@{module.version}: expected {expected}, downloaded {actual}")
    destination.write_bytes(data)


def go_source_release(version: str) -> tuple[str, str]:
    """URL and SHA-256 of the Go toolchain source archive for ``version`` (e.g. ``go1.26.8``)."""
    releases = json.loads(fetch(f"{GO_DOWNLOADS}?mode=json&include=all", timeout=60))
    filename = f"{version}.src.tar.gz"
    for release in releases:
        for file in release.get("files", ()):
            if file.get("filename") == filename:
                return f"{GO_DOWNLOADS}{filename}", file["sha256"]
    raise RuntimeError(f"no source archive for {version} on {GO_DOWNLOADS}")


def download_toolchain_source(version: str, destination: Path) -> str:
    url, sha256 = go_source_release(version)
    data = fetch(url, timeout=600)
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"sha256 mismatch for {url}: expected {sha256}, got {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return url


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
    parser.add_argument("--binary", action="append", required=True, help="Go binary whose modules to collect")
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    modules, used_by, go_versions = linked_modules([Path(binary) for binary in args.binary])

    module_dir = output_dir / "go-modules"
    module_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(lambda module: download_module(module, module_dir / module.filename), modules))

    toolchain_lines = []
    for version in sorted(go_versions):
        url = download_toolchain_source(version, output_dir / "go-toolchain" / f"{version}.src.tar.gz")
        toolchain_lines.append(f"{version}\t{version}.src.tar.gz\t{url}\n")

    with (manifests / "downloaded-go-modules.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(
            f"{module.path}@{module.version}\t{module.filename}\t{','.join(used_by[module])}\n" for module in modules
        )
    with (manifests / "downloaded-go-toolchain-sources.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(toolchain_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
