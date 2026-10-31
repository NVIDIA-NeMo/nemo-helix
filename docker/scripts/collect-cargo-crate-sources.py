#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect the source of every crates.io crate compiled into a Rust binary built with cargo-auditable.

The binary's ``.dep-v0`` section lists the crates it was built from; each crates.io crate is
downloaded and checked against the SHA-256 its project's ``Cargo.lock`` records. Workspace crates
(``source: local``) are part of the project's own source archive; they are listed in
``local-cargo-crates.txt``, and the project's archive is collected as a pinned source.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import struct
import sys
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CRATES_IO_DOWNLOAD = "https://static.crates.io/crates"
#: crates.io crates are downloaded; local (workspace) crates are in the project's own pinned archive.
COLLECTED_SOURCES = {"crates.io", "local"}
CRATES_IO_LOCK_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
DOWNLOAD_WORKERS = 8

DOWNLOAD_ATTEMPTS = 5


@dataclass(frozen=True)
class Crate:
    name: str
    version: str

    @property
    def filename(self) -> str:
        return f"{self.name}-{self.version}.crate"


def elf_section(data: bytes, wanted: str) -> bytes:
    """The contents of a named section of a 64-bit little-endian ELF file."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise RuntimeError("not a 64-bit little-endian ELF file")
    (section_offset,) = struct.unpack_from("<Q", data, 0x28)
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 0x3A)

    def header(index: int) -> tuple[int, int, int]:
        name, _type, _flags, _addr, offset, size = struct.unpack_from(
            "<IIQQQQ", data, section_offset + index * entry_size
        )
        return name, offset, size

    _, names_offset, _ = header(names_index)
    for index in range(count):
        name, offset, size = header(index)
        end = data.index(b"\x00", names_offset + name)
        if data[names_offset + name : end].decode("ascii") == wanted:
            return data[offset : offset + size]
    raise RuntimeError(f"no {wanted} section; build the binary with cargo-auditable")


def audited_packages(binary: bytes) -> list[dict]:
    """The packages recorded in a binary's cargo-auditable ``.dep-v0`` section."""
    return json.loads(zlib.decompress(elf_section(binary, ".dep-v0")))["packages"]


def audited_crates(binary: bytes, source: str = "crates.io") -> list[Crate]:
    """Crates from ``source`` recorded in a binary's cargo-auditable ``.dep-v0`` section."""
    return sorted(
        {
            Crate(package["name"], package["version"])
            for package in audited_packages(binary)
            if package.get("source") == source
        },
        key=lambda crate: (crate.name, crate.version),
    )


def uncollectable_crates(binary: bytes) -> list[str]:
    """Crates from a source other than crates.io or the project itself, such as git or another registry."""
    return sorted(
        {
            f"{package['name']} {package['version']} ({package.get('source')})"
            for package in audited_packages(binary)
            if package.get("source") not in COLLECTED_SOURCES
        }
    )


def lock_checksums(cargo_lock: str) -> dict[Crate, str]:
    lock = tomllib.loads(cargo_lock)
    return {
        Crate(package["name"], package["version"]): package["checksum"]
        for package in lock["package"]
        if package.get("source") == CRATES_IO_LOCK_SOURCE and "checksum" in package
    }


def local_lock_packages(cargo_lock: str) -> set[Crate]:
    """The project's own workspace crates: ``Cargo.lock`` packages with no source."""
    return {
        Crate(package["name"], package["version"])
        for package in tomllib.loads(cargo_lock)["package"]
        if "source" not in package
    }


def read_cargo_lock(path: Path) -> str:
    """A ``Cargo.lock`` file, or the top-level one inside a project's source archive (``.tar.gz``)."""
    if not path.name.endswith((".tar.gz", ".tgz")):
        return path.read_text(encoding="utf-8")
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            (
                member
                for member in archive.getmembers()
                if member.name.count("/") == 1 and member.name.endswith("/Cargo.lock")
            ),
            None,
        )
        handle = archive.extractfile(member) if member is not None else None
        if handle is None:
            raise RuntimeError(f"{path} has no top-level Cargo.lock")
        return handle.read().decode("utf-8")


def download_crate(crate: Crate, sha256: str, destination: Path) -> None:
    url = f"{CRATES_IO_DOWNLOAD}/{crate.name}/{crate.filename}"
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            actual = hashlib.sha256(data).hexdigest()
            if actual != sha256:
                raise RuntimeError(f"sha256 mismatch for {crate.filename}: expected {sha256}, got {actual}")
            destination.write_bytes(data)
            return
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {url}: {last_error}")


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
    parser.add_argument("--binary", required=True, help="Rust binary built with cargo-auditable")
    parser.add_argument(
        "--cargo-lock",
        required=True,
        help="Cargo.lock of the project the binary was built from, or the project's .tar.gz source archive",
    )
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    binary = Path(args.binary).read_bytes()
    uncollectable = uncollectable_crates(binary)
    if uncollectable:
        raise RuntimeError(f"{args.binary} uses crates this collector can't fetch: {', '.join(uncollectable)}")
    crates = audited_crates(binary)
    local_crates = audited_crates(binary, source="local")
    cargo_lock = read_cargo_lock(Path(args.cargo_lock))
    checksums = lock_checksums(cargo_lock)
    lock_local = local_lock_packages(cargo_lock)
    mismatched = [f"{crate.name} {crate.version}" for crate in local_crates if crate not in lock_local]
    if mismatched:
        raise RuntimeError(f"{args.cargo_lock} is not the source the binary was built from: {', '.join(mismatched)}")
    unlocked = [f"{crate.name} {crate.version}" for crate in crates if crate not in checksums]
    if unlocked:
        raise RuntimeError(f"{args.cargo_lock} has no checksum for crates the binary uses: {', '.join(unlocked)}")

    crate_dir = output_dir / "crates"
    crate_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(lambda crate: download_crate(crate, checksums[crate], crate_dir / crate.filename), crates))
    with (manifests / "downloaded-cargo-crates.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(f"{crate.name}=={crate.version}\t{crate.filename}\t{args.binary}\n" for crate in crates)
    with (manifests / "local-cargo-crates.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(f"{crate.name}=={crate.version}\t\t{args.binary}\n" for crate in local_crates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
