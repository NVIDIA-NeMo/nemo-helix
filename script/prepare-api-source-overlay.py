#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage current checkout sources on top of an installed nightly Python environment.

Usage: uv run --no-project script/prepare-api-source-overlay.py REPOSITORY DESTINATION

The build helper copies DESTINATION/overlay into the image's site-packages and
stores source-manifest.json alongside it for provenance. This includes uncommitted
source edits; it does not build wheels, install dependencies, or register plugin
entrypoints. A dependency change therefore needs a full build instead of only an
overlay. See /Users/ngoncharenko/code/nvidia/env_files/helix/build-api-local.sh
and docker/Dockerfile.local-source.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    overlay = args.destination / "overlay"
    overlay.mkdir(parents=True)
    # Merge each src/ tree at its Python import root, matching site-packages. The
    # temporary scaled-evals runtime is included without registering its service.
    sources = sorted(
        directory
        for pattern in ("packages/*/src", "plugins/*/src", "services/*/src", "services/*/*/src")
        for directory in root.glob(pattern)
        if directory.is_dir()
    )
    hashes = {}
    for directory in sources:
        for path in sorted(directory.rglob("*")):
            # Copy source/data files, not local environments, bytecode, or .env
            # credentials. Dependencies and other installed assets stay in the base.
            if not path.is_file() or any(part in {"__pycache__", ".venv"} for part in path.parts):
                continue
            if path.suffix in {".pyc", ".pyo"} or path.name.startswith(".env"):
                continue
            destination = overlay / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # HEAD identifies the base commit; file hashes capture working-tree edits too.
    # Stable ordering makes source_digest comparable with the deployed manifest.
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {"revision": revision, "files": hashes}
    manifest["source_digest"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (args.destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Collected {len(hashes)} runtime files; source digest {manifest['source_digest']}")


if __name__ == "__main__":
    main()
