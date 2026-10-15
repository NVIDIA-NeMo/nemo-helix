# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch installed W&B cores and retained caches after final dependency installs.

Fail closed on SDK/architecture mismatches; no runtime Go toolchain required.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import tempfile
from importlib.metadata import Distribution
from pathlib import Path

CORE = "wandb/bin/wandb-core"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def replace(path: Path, data: bytes, mode: int) -> None:
    # Follow retained-cache symlinks, but break hardlinks to build-only caches.
    path = path.resolve(strict=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        tmp = Path(directory) / path.name
        tmp.write_bytes(data)
        tmp.chmod(mode)
        tmp.replace(path)


def install(artifact: Path, roots: list[Path], *, optional: bool = False, verify_only: bool = False) -> int:
    manifest = json.loads((artifact / "manifest.json").read_text())
    binary = (artifact / "wandb-core").read_bytes()
    digest = hashlib.sha256(binary).hexdigest()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    require(digest == manifest["sha256"], "Patched core does not match build manifest")
    require(
        manifest["arch"] == arch
        and binary[:6] == b"\x7fELF\x02\x01"
        and int.from_bytes(binary[18:20], "little") == {"amd64": 62, "arm64": 183}.get(arch),
        "Patched core architecture does not match this image",
    )
    require(
        manifest["go_version"] == "go1.27.2" and manifest["x_net_version"] == "v0.60.0",
        "Unexpected Go security baseline",
    )
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    expected = [CORE, f"sha256={encoded}", str(len(binary))]
    packages, wheels = set(), set()
    for root in roots:
        for directory, dirs, files in os.walk(root):
            parent = Path(directory)
            # Inspect symlinked package directories without traversing arbitrary symlinks.
            if "wandb" in dirs:
                target = parent / CORE
                if target.is_file():
                    packages.add(parent)
                elif list(parent.glob("wandb-*.dist-info")):
                    raise ValueError(f"Installed W&B core is missing or a broken symlink: {target}")
            wheels.update(parent / name for name in files if name.startswith("wandb-") and name.endswith(".whl"))
    require(bool(packages) or optional, "No installed W&B core found")

    # Preflight ALL copies before modifying any, including retained archives.
    records = []
    resolved_roots = [root.resolve() for root in roots]
    for parent in sorted(packages):
        infos = list(parent.glob("wandb-*.dist-info"))
        require(len(infos) == 1, f"Expected one W&B dist-info next to {parent / 'wandb'}")
        version = Distribution.at(infos[0]).version
        require(version == manifest["wandb_version"], f"W&B version mismatch: {parent} ({version})")
        target, record = parent / CORE, infos[0] / "RECORD"
        for path in (target, record):
            if not any(path.resolve().is_relative_to(root) for root in resolved_roots):
                raise ValueError(f"W&B symlink escapes the retained image roots: {path}")
        rows = list(csv.reader(io.StringIO(record.read_text())))
        require(sum(row[0] == CORE for row in rows) == 1, f"Missing or duplicated core entry in {record}")
        records.append((target, record, rows))

    for target, record, rows in records:
        if not verify_only:
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                replace(target, binary, 0o755)
            output = io.StringIO(newline="")
            csv.writer(output).writerows(expected if row[0] == CORE else row for row in rows)
            replace(record, output.getvalue().encode(), record.stat().st_mode & 0o777)
        patched = hashlib.sha256(target.read_bytes()).hexdigest() == digest and os.access(target, os.X_OK)
        require(patched, f"Unpatched or non-executable core: {target}")
        require(expected in csv.reader(io.StringIO(record.read_text())), f"Stale core RECORD: {record}")
        print(f"Verified patched W&B {manifest['wandb_version']}: {target}")
    # Keep extracted caches (venvs may reference them), but never original wheels.
    for wheel in sorted(wheels):
        require(not verify_only, f"Unpatched retained wheel: {wheel}")
        print(f"Removing original cached wheel: {wheel}")
        wheel.unlink()
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--optional", action="store_true", help="Allow environments without W&B")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    install(args.artifact, args.roots, optional=args.optional, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
