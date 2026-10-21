#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Update Python CA bundles and rebuild installed wandb-core with patched Go."""

import argparse
import base64
import csv
import functools
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOTS = (
    "/app/.venv",
    "/opt/venv",
    "/opt/gym-venv",
    "/opt/gym-host-venv",
    "/opt/nemo_rl_venv",
    "/opt/ray_venvs",
    "/opt/gym_venvs",
    "/opt/cpython",
    "/usr/local",
    "/usr/lib/python3/dist-packages",
    "/root/.local/share/uv/python",
)
SOURCE_OUTPUT = Path("/usr/share/nemo-helix/source-distributions/rebuilt-wandb")
CACHE_ROOTS = (Path("/opt/uv_cache"), Path("/home/ubuntu/.cache/uv"))
FIXED_PACKAGES = {"certifi": "2026.10.19", "pip": "26.2.2", "uv": "0.13.4"}


@functools.cache
def fixed_pip_wheel() -> bytes:
    with urllib.request.urlopen("https://pypi.org/pypi/pip/26.2.2/json", timeout=120) as response:
        metadata = json.load(response)
    wheel = next(file for file in metadata["urls"] if file["filename"] == "pip-26.2.2-py3-none-any.whl")
    with urllib.request.urlopen(wheel["url"], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != wheel["digests"]["sha256"]:
        raise ValueError("pip bootstrap wheel checksum mismatch")
    return data


def update_ensurepip(python: Path) -> None:
    origin = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.util; s=importlib.util.find_spec('ensurepip'); print(s.origin if s else '')",
        ],
        text=True,
    ).strip()
    if not origin:
        return
    module = Path(origin)
    bundled = module.parent / "_bundled"
    if not bundled.is_dir():
        return
    content = module.read_text()
    updated, count = re.subn(r'(?m)^_PIP_VERSION = [\'"][^\'"]+[\'"]', '_PIP_VERSION = "26.2.2"', content)
    if count != 1:
        raise ValueError(f"Unrecognized ensurepip version declaration: {module}")
    wheel = bundled / "pip-26.2.2-py3-none-any.whl"
    if content == updated and wheel.exists():
        return
    wheel.write_bytes(fixed_pip_wheel())
    module.write_text(updated)
    for old in bundled.glob("pip-*.whl"):
        if old != wheel:
            old.unlink()


def python_environments(roots: list[Path]) -> list[Path]:
    interpreters = {Path(shutil.which("python3") or shutil.which("python") or "/usr/bin/python3")}
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("pyvenv.cfg"):
            python = config.parent / "bin/python"
            if python.exists():
                interpreters.add(python)
        for python in root.rglob("bin/python3"):
            if python.is_file():
                interpreters.add(python)
    return sorted(interpreters)


def update_bundles(python: Path) -> None:
    probe = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata,json; "
            "print(json.dumps([name.lower() for d in importlib.metadata.distributions() "
            "if (name := d.metadata.get('Name'))]))",
        ],
        text=True,
    )
    installed = set(json.loads(probe))
    requirements = [
        pin
        for name, pin in ((name, f"{name}=={version}") for name, version in FIXED_PACKAGES.items())
        if name in installed
    ]
    if requirements:
        subprocess.run(
            [
                "/usr/local/bin/uv",
                "pip",
                "install",
                "--directory",
                "/tmp",
                "--python",
                str(python),
                "--break-system-packages",
                "--no-deps",
                "--link-mode",
                "copy",
                *requirements,
            ],
            check=True,
        )


def download_source(version: str, destination: Path) -> Path:
    with urllib.request.urlopen(f"https://pypi.org/pypi/wandb/{version}/json", timeout=120) as response:
        metadata = json.load(response)
    source = next(file for file in metadata["urls"] if file["packagetype"] == "sdist")
    with urllib.request.urlopen(source["url"], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != source["digests"]["sha256"]:
        raise ValueError(f"wandb {version} source checksum mismatch")
    archive = destination / source["filename"]
    archive.write_bytes(data)
    if os.environ.get("NHX_COLLECT_SOURCES") == "1":
        shutil.copy2(archive, SOURCE_OUTPUT / archive.name)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(destination, filter="data")
    return next(path for path in destination.iterdir() if path.is_dir())


def rebuild_wandb(version: str, work: Path) -> Path:
    directory = work / version
    directory.mkdir()
    source = download_source(version, directory)
    output = directory / "wandb-core"
    previous = Path.cwd()
    try:
        os.chdir(source)
        hook = runpy.run_path(str(source / "core/hatch.py"))
        hook["build_wandb_core"](
            go_binary=Path("/usr/local/go/bin/go"),
            output_path=output,
            with_code_coverage=False,
            with_race_detection=False,
            with_cgo=False,
            wandb_commit_sha=None,
            target_system="linux",
            target_arch={"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()],
        )
    finally:
        os.chdir(previous)
    version_info = subprocess.check_output(["/usr/local/go/bin/go", "version", "-m", str(output)], text=True)
    if version_info.splitlines()[0].split()[-1] != "go1.27.3":
        raise ValueError(f"Unexpected wandb-core toolchain: {version_info}")
    if os.environ.get("NHX_COLLECT_SOURCES") == "1":
        subprocess.run(
            [
                sys.executable,
                "/tmp/collect-go-module-sources.py",
                "--binary",
                str(output),
                "--output",
                str(SOURCE_OUTPUT),
            ],
            check=True,
        )
    return output


def replace_binary(binary: Path, source: Path, record: Path) -> None:
    # Detach symlinks/hardlinks before replacing a cached or shared wheel file.
    binary.unlink()
    shutil.copyfile(source, binary)
    binary.chmod(0o755)
    if record.is_file():
        rows = list(csv.reader(io.StringIO(record.read_text())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(binary.read_bytes()).digest()).decode().rstrip("=")
        for row in rows:
            if row[0] == "wandb/bin/wandb-core":
                row[1:] = [f"sha256={digest}", str(binary.stat().st_size)]
        # The RECORD can itself be a shared cache symlink.
        record.unlink()
        with record.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument(
        "--bundles-only", action="store_true", help="Update Python bundles without rebuilding Go clients"
    )
    args = parser.parse_args()
    SOURCE_OUTPUT.mkdir(parents=True, exist_ok=True)
    roots = args.root or [Path(root) for root in ROOTS]
    environment_roots = roots.copy()
    for python in python_environments(roots):
        update_bundles(python)
        update_ensurepip(python)
    if args.bundles_only:
        return
    # Include the caches that supply symlinked RL and Gym environments.
    roots.extend(CACHE_ROOTS)
    rebuilt: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="wandb-security-") as temporary:
        for root in roots:
            if not root.exists():
                continue
            for metadata in root.rglob("wandb-*.dist-info/METADATA"):
                distribution = importlib.metadata.PathDistribution(metadata.parent)
                binary = metadata.parent.parent / "wandb/bin/wandb-core"
                if not binary.is_file():
                    continue
                version = distribution.version
                if version not in rebuilt:
                    rebuilt[version] = rebuild_wandb(version, Path(temporary))
                replace_binary(binary, rebuilt[version], metadata.parent / "RECORD")
                print(f"Rebuilt wandb {version}: {binary}", flush=True)

    # Upgrades used copy mode above, so old CA wheels should have no remaining consumers.
    # Remove only these obsolete archives, preserving all unrelated RL cache entries.
    links = [path for root in environment_roots if root.exists() for path in root.rglob("*") if path.is_symlink()]
    for cache in CACHE_ROOTS:
        archive_root = cache / "archive-v0"
        if not archive_root.exists():
            continue
        for metadata in list(archive_root.glob("*/*.dist-info/METADATA")):
            distribution = importlib.metadata.PathDistribution(metadata.parent)
            name = distribution.metadata["Name"].lower()
            if name not in FIXED_PACKAGES:
                continue
            if distribution.version == FIXED_PACKAGES[name]:
                continue
            archive = metadata.parent.parent
            if any(link.resolve().is_relative_to(archive) for link in links):
                raise RuntimeError(f"An environment still references obsolete {name}: {archive}")
            shutil.rmtree(archive)


if __name__ == "__main__":
    main()
