# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installs the ``nemo-relay`` CLI that Relay's Codex and Claude adapters launch, unless one is on ``PATH``.

It downloads the release matching the installed ``nemo-relay`` bindings, for this machine and C library, checks
its published sha256, and makes sure it runs. Standard library only, like the OTLP receiver: it runs in a task
container that has Fabric's interpreter and nothing of this package.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

_ARCHITECTURES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


def _fetch(url: str, timeout: float, delays: list[float]) -> bytes:
    for attempt, delay in enumerate([*delays, None], start=1):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or delay is None:
                raise SystemExit(f"could not download {url}: {error} (attempts: {attempt})") from error
        except OSError as error:
            if delay is None:
                raise SystemExit(f"could not download {url}: {error} (attempts: {attempt})") from error
        time.sleep(delay)
    raise AssertionError("unreachable")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            f"usage: {argv[0] if argv else 'relay-cli-install'} <releases URL> <install dir> <retry delays, comma-separated>"
        )
        return 2
    if shutil.which("nemo-relay"):
        return 0
    releases_url, install_dir = argv[1], argv[2]
    delays = [float(delay) for delay in argv[3].split(",") if delay]
    try:
        version = importlib.metadata.version("nemo-relay")
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit("Relay telemetry needs the nemo-relay Python package: install nemo-fabric[relay]") from error
    arch = _ARCHITECTURES.get(platform.machine().lower())
    if arch is None:
        raise SystemExit(f"no nemo-relay CLI release for {platform.machine()}")
    libc = "musl" if glob.glob("/lib/ld-musl-*") else "gnu"
    asset = f"nemo-relay-cli-{arch}-unknown-linux-{libc}-{version}"
    url = f"{releases_url}/{version}/{asset}"
    binary = _fetch(url, 120, delays)
    expected = _fetch(url + ".sha256", 60, delays).decode().split()[0]
    if hashlib.sha256(binary).hexdigest() != expected:
        raise SystemExit(f"{asset} does not match its published sha256")
    os.makedirs(install_dir, exist_ok=True)
    target = os.path.join(install_dir, "nemo-relay")
    with open(target, "wb") as handle:
        handle.write(binary)
    os.chmod(target, 0o755)
    check = subprocess.run([target, "--version"], capture_output=True, text=True)
    if check.returncode != 0:
        os.remove(target)
        raise SystemExit(f"{asset} does not run in this image: {(check.stderr or check.stdout).strip()[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
