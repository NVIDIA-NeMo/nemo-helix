#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy NeMo-RL's CUDA extension pins into the wheel Dockerfile.

RUN THIS WHEN THE PIN FOR RL CHANGES DEPS.
Then regenerate Dockerfile.python-wheels.

Checks out NEMO_RL_REF from docker/rl/Dockerfile.nhx-rl-base, reads the git
revisions and version pins from that commit's pyproject.toml, and writes them
into docker/base/Dockerfile.python-wheels. NeMo-RL's own files stay as they
are. nhx-rl-base installs the built wheels with uv pip, outside uv.lock.

    docker/rl/scripts/sync-rl-cuda-wheels.py [path-to-NeMo-RL]

Default path is NeMo-RL next to this repo. Rebuild rl-cuda-ext-wheel, then
nhx-rl-base.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # Assumes NeMo-RL is a sibling of this repo.
RL_DOCKERFILE = REPO / "docker" / "rl" / "Dockerfile.nhx-rl-base"
WHEELS_DOCKERFILE = REPO / "docker" / "base" / "Dockerfile.python-wheels"

# (pyproject source name, Dockerfile env, clone URL as written in the Dockerfile)
GIT = (
    ("causal-conv1d", "CAUSAL_CONV1D_REV"),
    ("mamba-ssm", "MAMBA_SSM_REV"),
    ("nv-grouped-gemm", "NV_GROUPED_GEMM_REV"),
    ("flash_mla", "FLASH_MLA_REV"),
    ("fast-hadamard-transform", "FAST_HADAMARD_REV"),
    ("deep_ep", "DEEP_EP_REV"),
    ("deep_gemm", "DEEP_GEMM_REV"),
)
SDIST = (
    ("flash-attn", "flash-attn"),
    ("transformer-engine-torch", "transformer-engine"),
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def one(pattern: str, text: str, what: str) -> str:
    found = set(re.findall(pattern, text, flags=re.M))
    if len(found) != 1:
        raise SystemExit(f"{what}: expected one value, found {sorted(found) or 'none'}")
    return found.pop()


def commit_for(url: str, rev: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    lines = subprocess.run(
        ["git", "ls-remote", url, f"refs/tags/{rev}"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.splitlines()
    peeled = [line.split()[0] for line in lines if line.endswith("^{}")]
    commits = peeled or [line.split()[0] for line in lines]
    if len(commits) != 1:
        raise SystemExit(f"{url} tag {rev}: expected one commit, found {commits or 'none'}")
    return commits[0]


def source(text: str, name: str) -> tuple[str, str]:
    table = re.search(
        rf'^{re.escape(name)} = \{{ git = "([^"]+)", (?:rev|tag) = "([^"]+)" \}}',
        text,
        flags=re.M,
    )
    if table:
        return table.group(1), commit_for(table.group(1), table.group(2))
    inline = set(re.findall(rf"{re.escape(name)} @ git\+(https://[^@\s\"]+)@([0-9a-f]{{40}})", text))
    if len(inline) != 1:
        raise SystemExit(f"{name}: expected one git pin, found {sorted(inline) or 'none'}")
    return inline.pop()


def version_of(text: str, name: str) -> str:
    return one(rf"{re.escape(name)}(?:\[[^\]]+\])?==([^\s\"';]+)", text, name)


def replace(text: str, pattern: str, repl: str, what: str) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"expected one {what} in {WHEELS_DOCKERFILE.name}, found {count}")
    return updated


def checkout(repo: Path, url: str, ref: str) -> str:
    if not (repo / ".git").exists():
        raise SystemExit(f"{repo} is not a git checkout")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all")
    if dirty:
        raise SystemExit(f"{repo} has uncommitted changes:\n{dirty}")
    print(f"+ git fetch {url} {ref}", flush=True)
    git(repo, "fetch", "--quiet", url, ref)
    sha = git(repo, "rev-parse", "FETCH_HEAD^{commit}")
    if git(repo, "rev-parse", "HEAD") != sha:
        print(f"+ git checkout --detach {sha}", flush=True)
        git(repo, "checkout", "--quiet", "--detach", sha)
    return sha


def main() -> int:
    nemo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO.parent / "NeMo-RL"
    rl = RL_DOCKERFILE.read_text()
    url = one(r"^ARG NEMO_RL_REPO=(\S+)", rl, "NEMO_RL_REPO")
    ref = one(r"^ARG NEMO_RL_REF=(\S+)", rl, "NEMO_RL_REF")
    sha = checkout(nemo, url, ref)
    print(f"NeMo-RL {ref} -> {sha}")

    pyproject = (nemo / "pyproject.toml").read_text()
    wheels = WHEELS_DOCKERFILE.read_text()
    wheels = replace(wheels, r"^(# NeMo-RL commit: )\S+", rf"\g<1>{sha}", "NeMo-RL commit line")
    for name, env in GIT:
        src, rev = source(pyproject, name)
        print(f"  {name:26} {rev}")
        wheels = replace(wheels, rf"^([ \t]*{env}=)\S+", rf"\g<1>{rev}", env)
        wheels = replace(
            wheels,
            rf'(build-rl-cuda-ext-wheel\.sh git )\S+( "\${env}")',
            rf"\g<1>{src}\2",
            f"{name} clone",
        )
    for wheel_name, pin_name in SDIST:
        version = version_of(pyproject, pin_name)
        print(f"  {wheel_name:26} {version}")
        wheels = replace(
            wheels,
            rf'(build-rl-cuda-ext-wheel\.sh sdist "{re.escape(wheel_name)}==)[^"]+"',
            rf"\g<1>{version}" + '"',
            f"{wheel_name} sdist",
        )

    if wheels == WHEELS_DOCKERFILE.read_text():
        print(f"{WHEELS_DOCKERFILE.relative_to(REPO)} is up to date")
    else:
        WHEELS_DOCKERFILE.write_text(wheels)
        print(f"updated {WHEELS_DOCKERFILE.relative_to(REPO)}")
        print("rebuild rl-cuda-ext-wheel, then nhx-rl-base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
