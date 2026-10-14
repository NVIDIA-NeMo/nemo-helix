# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install and verify the shared core in venvs AND retained extracted uv caches.

Run after the final dependency install, before switching to the runtime user.
Version mismatches (including in retained caches) fail closed; never install a
0.30.0 core alongside a different Python SDK. Does not require Go at runtime.
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
from email.parser import Parser
from pathlib import Path


def replace(path: Path, data: bytes, mode: int) -> None:
    # Break hardlinks to build-only uv caches; follow symlinks to retained caches.
    path = path.resolve(strict=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        tmp = Path(file.name)
        try:
            file.write(data)
            file.flush()
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def install(artifact: Path, roots: list[Path], *, optional: bool = False, verify_only: bool = False) -> int:
    manifest = json.loads((artifact / "manifest.json").read_text())
    binary = (artifact / "wandb-core").read_bytes()
    digest = hashlib.sha256(binary).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Patched core does not match build manifest")
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    elf_machine = {"amd64": 62, "arm64": 183}.get(arch)
    if (
        manifest["arch"] != arch
        or binary[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(binary[18:20], "little") != elf_machine
    ):
        raise ValueError("Patched core architecture does not match this image")
    if manifest["go_version"] != "go1.27.2" or manifest["x_net_version"] != "v0.60.0":
        raise ValueError("Unexpected Go security baseline")

    packages: set[Path] = set()
    wheels: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for directory, dirs, files in os.walk(root):
            parent = Path(directory)
            # Also handle symlinked wandb directories without following arbitrary
            # directory symlinks (RL's venvs share a read-only extracted cache).
            if "wandb" in dirs:
                target = parent / "wandb/bin/wandb-core"
                if target.is_file():
                    packages.add(parent)
                elif list(parent.glob("wandb-*.dist-info")):
                    raise ValueError(f"Installed W&B core is missing or a broken symlink: {target}")
            wheels.update(parent / name for name in files if name.startswith("wandb-") and name.endswith(".whl"))
    if not packages and not optional:
        raise ValueError("No installed W&B core found")

    # Validate ALL versions before changing anything, including stale archives.
    records = []
    resolved_roots = [root.resolve() for root in roots]
    for parent in sorted(packages):
        infos = list(parent.glob("wandb-*.dist-info"))
        if len(infos) != 1:
            raise ValueError(f"Expected one W&B dist-info next to {parent / 'wandb'}")
        metadata = Parser().parsestr((infos[0] / "METADATA").read_text())
        if metadata["Version"] != manifest["wandb_version"]:
            raise ValueError(
                f"W&B version mismatch at {parent}: {metadata['Version']} != {manifest['wandb_version']}; upgrade installed SDK or remove unused cache"
            )
        record = infos[0] / "RECORD"
        for path in (parent / "wandb/bin/wandb-core", record):
            if not any(path.resolve().is_relative_to(root) for root in resolved_roots):
                raise ValueError(f"W&B symlink escapes the retained image roots: {path}")
        rows = list(csv.reader(io.StringIO(record.read_text())))
        matches = [row for row in rows if row[0] == "wandb/bin/wandb-core"]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicated core entry in {record}")
        records.append((parent / "wandb/bin/wandb-core", record, rows))

    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    for target, record, rows in records:
        if not verify_only:
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                replace(target, binary, 0o755)
            for row in rows:
                if row[0] == "wandb/bin/wandb-core":
                    row[1:] = [f"sha256={encoded}", str(len(binary))]
            output = io.StringIO(newline="")
            csv.writer(output).writerows(rows)
            replace(record, output.getvalue().encode(), record.stat().st_mode & 0o777)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest or not os.access(target, os.X_OK):
            raise ValueError(f"Unpatched or non-executable core: {target}")
        installed_rows = list(csv.reader(io.StringIO(record.read_text())))
        if ["wandb/bin/wandb-core", f"sha256={encoded}", str(len(binary))] not in installed_rows:
            raise ValueError(f"Stale core RECORD: {record}")
        print(f"Verified patched W&B {manifest['wandb_version']}: {target}")
    # Compressed wheels cannot be left with the old binary. Extracted caches are
    # patched above, not deleted: prefetched venvs may symlink into them.
    for wheel in sorted(wheels):
        if verify_only:
            raise ValueError(f"Unpatched retained wheel: {wheel}")
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
