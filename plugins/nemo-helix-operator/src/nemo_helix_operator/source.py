# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packs a Helix checkout into the archive the builder builds from."""

import io
import subprocess
import tarfile
from pathlib import Path

DOCKERFILE_PATH = ".helix-operator/Dockerfile"


def render_dockerfile(base_image: str, revision: str) -> str:
    # ponytail: overlays every src tree on the base, so files deleted from the source stay in the image, and
    # packages the base doesn't install are copied but never imported. Fixing both needs a diff against the base.
    return (
        f"FROM {base_image}\n"
        "COPY --chown=1000:1000 services/*/src/ services/*/*/src/ packages/*/src/ plugins/*/src/ "
        "/app/.venv/lib/python3.13/site-packages/\n"
        "RUN python -m compileall -q /app/.venv/lib/python3.13/site-packages\n"
        f"ENV NHX_CODE_REVISION={revision}\n"
    )


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True).stdout


def pack(repo_dir: Path, base_image: str) -> tuple[bytes, dict[str, str | bool]]:
    """Return ``source.tar.gz``'s bytes and ``source.json``'s content for the checkout at ``repo_dir``."""
    if not (repo_dir / "docker-bake.hcl").is_file():
        raise ValueError(f"{repo_dir} is not a NeMo Helix checkout: it has no docker-bake.hcl")
    commit = _git(repo_dir, "rev-parse", "HEAD")[:12]
    dirty = bool(_git(repo_dir, "status", "--porcelain").strip())
    names = set(_git(repo_dir, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0"))
    dockerfile = render_dockerfile(base_image, commit + ("-dirty" if dirty else "")).encode()

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in sorted(names - {""}):
            path = repo_dir / name
            if path.is_symlink() or path.is_file():  # Tracked files deleted from the checkout are skipped.
                tar.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo(DOCKERFILE_PATH)
        info.size = len(dockerfile)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(dockerfile))
    return buffer.getvalue(), {"commit": commit, "dirty": dirty}
