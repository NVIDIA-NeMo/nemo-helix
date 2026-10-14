# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the pinned W&B release without losing its custom vendored Arrow patches."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

VERSION = "0.30.0"
COMMIT = "e5ec9dc9236d1a4fbb9fe7c0795bbccdff1e0e77"
GO_VERSION = "go1.27.2"
NET_VERSION = "v0.60.0"


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> None:
    root = Path.cwd()
    core = root / "core"
    assert f'__version__ = "{VERSION}"' in (root / "wandb/__init__.py").read_text()
    assert run("go", "env", "GOVERSION") == GO_VERSION
    assert os.environ["GOARCH"] in {"amd64", "arm64"}

    # Download ONLY x/net using Go's checksum database. Do not run `go mod vendor`:
    # W&B's vendor tree contains unmerged parquet_read_only changes (PR 10712).
    module = json.loads(run("go", "mod", "download", "-json", f"golang.org/x/net@{NET_VERSION}", cwd=Path("/tmp")))
    assert module["Sum"] == "h1:79p50tfZlm0J9YfoDsSi639qSXNGVwEzOPLCxM2FsYU="
    vendor = core / "vendor"
    original = {
        str(p.relative_to(vendor)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in vendor.rglob("*")
        if p.is_file() and "golang.org/x/net/" not in str(p.relative_to(vendor)) and p.name != "modules.txt"
    }
    net = vendor / "golang.org/x/net"
    shutil.rmtree(net)
    shutil.copytree(module["Dir"], net)
    # Keep the upstream package list and explicit/go directive intact. Only the
    # version changes: v0.60.0 has the same minimum Go and dependency requirements.
    modules = vendor / "modules.txt"
    text = modules.read_text()
    assert text.count("# golang.org/x/net v0.59.0\n") == 1
    modules.write_text(text.replace("# golang.org/x/net v0.59.0\n", f"# golang.org/x/net {NET_VERSION}\n"))
    run("go", "mod", "edit", f"-require=golang.org/x/net@{NET_VERSION}", cwd=core)
    for name, checksum in original.items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == checksum, name

    out = Path("/out")
    out.mkdir(exist_ok=True)
    binary = out / "wandb-core"
    # Match core/hatch.py: same tags, vendor mode, CGO disabled, and commit ldflag.
    run(
        "go",
        "build",
        "-mod=vendor",
        "-trimpath",
        "-tags=disable_grpc_modules,parquet_read_only",
        f"-ldflags=-s -w -X main.commit={COMMIT}",
        "-o",
        str(binary),
        "./cmd/wandb-core/main.go",
        cwd=core,
    )
    metadata = run("go", "version", "-m", str(binary))
    assert f": {GO_VERSION}" in metadata
    assert f"dep\tgolang.org/x/net\t{NET_VERSION}" in metadata
    assert "CGO_ENABLED=0" in metadata
    assert f"GOARCH={os.environ['GOARCH']}" in metadata
    (out / "go-version-m.txt").write_text(metadata + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "wandb_version": VERSION,
                "commit": COMMIT,
                "go_version": GO_VERSION,
                "x_net_version": NET_VERSION,
                "x_net_sum": module["Sum"],
                "arch": os.environ["GOARCH"],
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "vendor_preserved_files": len(original),
            },
            indent=2,
        )
        + "\n"
    )
    print(metadata)


if __name__ == "__main__":
    main()
