# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patch only x/net, then delegate the core build to the pinned W&B release."""

import hashlib
import json
import os
import runpy
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
    core = Path("core")
    assert f'__version__ = "{VERSION}"' in Path("wandb/__init__.py").read_text()
    assert run("go", "env", "GOVERSION") == GO_VERSION
    assert os.environ["GOARCH"] in {"amd64", "arm64"}
    module = json.loads(run("go", "mod", "download", "-json", f"golang.org/x/net@{NET_VERSION}", cwd=Path("/tmp")))
    assert module["Sum"] == "h1:79p50tfZlm0J9YfoDsSi639qSXNGVwEzOPLCxM2FsYU="

    # Replace ONLY x/net. Regenerating vendor would discard W&B's custom Arrow
    # patches (PR 10712); every other vendored module remains untouched.
    net = core / "vendor/golang.org/x/net"
    shutil.rmtree(net)
    shutil.copytree(module["Dir"], net)
    modules = core / "vendor/modules.txt"
    text = modules.read_text()
    assert text.count("# golang.org/x/net v0.59.0\n") == 1
    modules.write_text(text.replace("# golang.org/x/net v0.59.0\n", f"# golang.org/x/net {NET_VERSION}\n"))
    run("go", "mod", "edit", f"-require=golang.org/x/net@{NET_VERSION}", cwd=core)

    out = Path("/out")
    out.mkdir(exist_ok=True)
    binary = out / "wandb-core"
    # Reuse upstream vendor mode, build tags, linker flags, and ELF fixups.
    runpy.run_path(str(core / "hatch.py"))["build_wandb_core"](
        go_binary=Path("go"),
        output_path=binary,
        with_code_coverage=False,
        with_race_detection=False,
        with_cgo=False,
        wandb_commit_sha=COMMIT,
        target_system="linux",
        target_arch=os.environ["GOARCH"],
    )
    metadata = run("go", "version", "-m", str(binary))
    for expected in (
        f": {GO_VERSION}",
        f"dep\tgolang.org/x/net\t{NET_VERSION}",
        "CGO_ENABLED=0",
        f"GOARCH={os.environ['GOARCH']}",
    ):
        assert expected in metadata, expected
    manifest = {
        "wandb_version": VERSION,
        "commit": COMMIT,
        "go_version": GO_VERSION,
        "x_net_version": NET_VERSION,
        "x_net_sum": module["Sum"],
        "arch": os.environ["GOARCH"],
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    (out / "go-version-m.txt").write_text(metadata + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
