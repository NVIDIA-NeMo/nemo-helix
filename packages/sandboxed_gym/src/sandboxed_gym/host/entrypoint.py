# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default entrypoint for the sandboxed Gym host.

Paths are parameterized via constructor args / environment variables so the same
package works in NeMo-RL training images and platform ``nhx-gym-runtime`` images.

Every path here describes the layout of the *host* image. A caller that is not itself running in
that image must pass ``git_root`` explicitly: the packaged-file shortcuts below resolve against this
process's own installation, which is the orchestrator's container, not the sandbox's. These helpers
build an explicit entrypoint for a compatible image that needs the shell bootstrap.
"""

from __future__ import annotations

import os
from pathlib import Path

# Image layout defaults (overridable).
DEFAULT_IMAGE_GIT_ROOT = "/opt/nemo-rl"
DEFAULT_GYM_ACTOR_VENV = "/opt/ray_venvs/sandboxed_gym.ray.gym_actor.SandboxedGymActor"
DEFAULT_GYM_UV_CACHE_DIR = "/home/ubuntu/.cache/uv"
DEFAULT_GYM_UV_VENV_DIR = "/opt/gym_venvs"
DEFAULT_GYM_WRITABLE_SRC = "/tmp/gym-src/Gym"
DEFAULT_GYM_TREE_RELPATH = "3rdparty/Gym-workspace/Gym"

_GYM_HOST_SCRIPT_NAME = "gym_host.sh"


def gym_uv_cache_dir() -> str:
    return os.environ.get("SANDBOXED_GYM_UV_CACHE_DIR", DEFAULT_GYM_UV_CACHE_DIR)


def gym_uv_venv_dir() -> str:
    return os.environ.get("NEMO_GYM_VENV_DIR") or os.environ.get("SANDBOXED_GYM_UV_VENV_DIR") or DEFAULT_GYM_UV_VENV_DIR


def gym_writable_src_dir() -> str:
    return os.environ.get("SANDBOXED_GYM_SRC_DIR", DEFAULT_GYM_WRITABLE_SRC)


def packaged_gym_host_script() -> Path:
    """Path to ``gym_host.sh`` shipped inside this package."""
    return Path(__file__).resolve().parent / _GYM_HOST_SCRIPT_NAME


def gym_host_script_path(*, git_root: str | None = None) -> str:
    """Prefer the packaged script; fall back to an image-relative path."""
    packaged = packaged_gym_host_script()
    if packaged.is_file():
        return str(packaged)
    root = git_root or os.environ.get("SANDBOXED_GYM_IMAGE_ROOT", DEFAULT_IMAGE_GIT_ROOT)
    return f"{root}/packages/sandboxed_gym/src/sandboxed_gym/host/{_GYM_HOST_SCRIPT_NAME}"


def gym_host_runtime_path(*, git_root: str | None = None) -> str:
    """Absolute path to ``gym_host_runtime.py`` for the shell entrypoint."""
    packaged = Path(__file__).resolve().parent.parent / "runtime" / "gym_host_runtime.py"
    if packaged.is_file():
        return str(packaged)
    root = git_root or os.environ.get("SANDBOXED_GYM_IMAGE_ROOT", DEFAULT_IMAGE_GIT_ROOT)
    return f"{root}/packages/sandboxed_gym/src/sandboxed_gym/runtime/gym_host_runtime.py"


def default_gym_host_entrypoint(
    *,
    venv: str | None = None,
    git_root: str | None = None,
    writable_gym_src: str | None = None,
    runtime: str | None = None,
) -> list[str]:
    """Shell entrypoint that starts ``gym_host_runtime`` in the runtime image."""
    root = git_root or os.environ.get("SANDBOXED_GYM_IMAGE_ROOT", DEFAULT_IMAGE_GIT_ROOT)
    return [
        "/bin/sh",
        gym_host_script_path(git_root=root),
        venv or os.environ.get("SANDBOXED_GYM_VENV", DEFAULT_GYM_ACTOR_VENV),
        root,
        writable_gym_src or gym_writable_src_dir(),
        runtime or gym_host_runtime_path(git_root=root),
    ]
