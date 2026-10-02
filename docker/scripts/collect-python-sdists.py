#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort PyPI source distribution collector for an installed environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FREEZE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")


@dataclass(frozen=True)
class Sdist:
    url: str
    hash_name: str | None = None
    hash_value: str | None = None


def source_collection_enabled() -> bool:
    return os.environ.get("NHX_COLLECT_SOURCES", "0").strip().lower() in {"1", "true", "yes", "on"}


def record_source_collection_disabled(output_dir: Path) -> None:
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "source-collection-disabled.txt").write_text(
        "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable\n",
        encoding="utf-8",
    )


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def uv_bin() -> str:
    return os.environ.get("UV_BIN", "uv")


def purelib_for(python: str) -> Path:
    script = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    return Path(run_text([python, "-c", script]).strip())


def local_direct_url_packages(purelib: Path) -> set[str]:
    packages: set[str] = set()
    for dist_info in purelib.glob("*.dist-info"):
        direct_url = dist_info / "direct_url.json"
        if not direct_url.is_file():
            continue
        try:
            data = json.loads(direct_url.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        url = str(data.get("url", ""))
        if url.startswith("file://"):
            packages.add(normalize_name(dist_info.name.split("-")[0]))
    return packages


def freeze_packages(python: str, purelib: Path) -> dict[str, tuple[str, str]]:
    output = run_text([uv_bin(), "pip", "freeze", "--python", python, "--path", str(purelib), "--exclude-editable"])
    return parse_freeze(output)


def parse_freeze(output: str) -> dict[str, tuple[str, str]]:
    packages: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        line = line.strip()
        match = FREEZE_RE.match(line)
        if match is None:
            continue
        name, version = match.groups()
        packages[normalize_name(name)] = (name, version)
    return packages


def baseline_packages(path: str | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    try:
        return parse_freeze(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}


def manifest_path(manifests: Path, stem: str, label: str | None) -> Path:
    if not label:
        return manifests / f"{stem}.txt"
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    return manifests / f"{stem}-{safe_label}.txt"


def sdists_from_uv_lock(lock_file: Path) -> dict[tuple[str, str], Sdist]:
    try:
        data = tomllib.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}

    sdists: dict[tuple[str, str], Sdist] = {}
    for package in data.get("package", []):
        name = package.get("name")
        version = package.get("version")
        sdist = package.get("sdist")
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(sdist, dict):
            continue
        url = sdist.get("url")
        if not isinstance(url, str):
            continue
        hash_name = None
        hash_value = None
        raw_hash = sdist.get("hash")
        if isinstance(raw_hash, str) and ":" in raw_hash:
            hash_name, hash_value = raw_hash.split(":", 1)
        sdists[(normalize_name(name), version)] = Sdist(url=url, hash_name=hash_name, hash_value=hash_value)
    return sdists


def pypi_sdist(name: str, version: str) -> Sdist | None:
    metadata_url = (
        f"https://pypi.org/pypi/{urllib.parse.quote(normalize_name(name))}/{urllib.parse.quote(version)}/json"
    )
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for file_info in payload.get("urls", []):
        if file_info.get("packagetype") != "sdist":
            continue
        url = file_info.get("url")
        if not isinstance(url, str):
            continue
        digests = file_info.get("digests") or {}
        sha256 = digests.get("sha256")
        return Sdist(url=url, hash_name="sha256" if sha256 else None, hash_value=sha256)
    return None


def download(url: str, destination: Path, hash_name: str | None, hash_value: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hash_name and hash_value:
                digest = hashlib.new(hash_name)
                digest.update(data)
                actual = digest.hexdigest()
                if actual.lower() != hash_value.lower():
                    raise RuntimeError(f"{hash_name} mismatch: expected {hash_value}, got {actual}")
            destination.write_bytes(data)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(str(last_error))


def unique_filename(output_dir: Path, url: str, name: str, version: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    if not filename:
        filename = f"{normalize_name(name)}-{version}.tar.gz"
    destination = output_dir / "pypi" / filename
    if not destination.exists():
        return destination
    if destination.stat().st_size > 0:
        return destination
    return (
        output_dir / "pypi" / f"{normalize_name(name)}-{version}-{hashlib.sha256(url.encode()).hexdigest()[:12]}.tar.gz"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="Python interpreter for the environment to inspect")
    parser.add_argument("--output", required=True, help="Source distribution output directory")
    parser.add_argument("--lock-file", action="append", default=[], help="uv.lock file to use for exact sdist URLs")
    parser.add_argument(
        "--baseline-freeze", help="Freeze output whose matching name==version entries should be skipped"
    )
    parser.add_argument("--label", help="Label to suffix manifest filenames when collecting multiple environments")
    parser.add_argument(
        "--skip-package", action="append", default=[], help="Package name to exclude from PyPI sdist collection"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    manifests = output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    if not source_collection_enabled():
        record_source_collection_disabled(output_dir)
        return 0

    (output_dir / "pypi").mkdir(parents=True, exist_ok=True)

    try:
        purelib = purelib_for(args.python)
        packages = freeze_packages(args.python, purelib)
    except subprocess.CalledProcessError as error:
        (manifests / "missing-pypi-sdists.txt").write_text(
            f"failed to inspect Python environment {args.python}: {error.output}\n",
            encoding="utf-8",
        )
        return 0

    local_packages = local_direct_url_packages(purelib)
    for package_name in local_packages:
        packages.pop(package_name, None)
    for package_name in args.skip_package:
        packages.pop(normalize_name(package_name), None)
    baseline = baseline_packages(args.baseline_freeze)
    for package_name, package in list(packages.items()):
        if baseline.get(package_name) == package:
            packages.pop(package_name)

    lock_sdists: dict[tuple[str, str], Sdist] = {}
    for lock_file in args.lock_file:
        lock_sdists.update(sdists_from_uv_lock(Path(lock_file)))

    downloaded: list[str] = []
    missing: list[str] = []

    for normalized, (name, version) in sorted(packages.items()):
        sdist = lock_sdists.get((normalized, version))
        source = "uv.lock"
        if sdist is None:
            source = "pypi-json"
            try:
                sdist = pypi_sdist(name, version)
            except (OSError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as error:
                missing.append(f"{name}=={version}\tmetadata lookup failed: {error}")
                continue
        if sdist is None:
            missing.append(f"{name}=={version}\tno sdist found")
            continue
        destination = unique_filename(output_dir, sdist.url, name, version)
        try:
            download(sdist.url, destination, sdist.hash_name, sdist.hash_value)
        except RuntimeError as error:
            missing.append(f"{name}=={version}\tdownload failed from {sdist.url}: {error}")
            continue
        downloaded.append(f"{name}=={version}\t{destination.name}\t{source}\t{sdist.url}")

    manifest_path(manifests, "installed-python-packages", args.label).write_text(
        "\n".join(f"{name}=={version}" for name, version in sorted(packages.values())) + "\n",
        encoding="utf-8",
    )
    manifest_path(manifests, "downloaded-pypi-sdists", args.label).write_text(
        "\n".join(downloaded) + ("\n" if downloaded else ""),
        encoding="utf-8",
    )
    manifest_path(manifests, "missing-pypi-sdists", args.label).write_text(
        "\n".join(missing) + ("\n" if missing else ""),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
