# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for the three step binaries.

These run inside job pods, not in the API process, so nothing here imports the service or the
compiler -- a step reads its config and does one job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from nemo_platform_plugin.jobs.constants import (
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)

logger = logging.getLogger(__name__)


def read_step_config() -> dict[str, Any]:
    """The step's ``config`` dict, as the jobs controller delivered it.

    It arrives as a ConfigMap mounted into the pod, with this env var pointing at the file.
    """
    path = os.environ.get(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR)
    if not path:
        raise RuntimeError(
            f"{NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR} is not set; this binary only runs as a platform job step"
        )
    return json.loads(Path(path).read_text())


def job_identity() -> tuple[str, str]:
    """``(workspace, job_id)`` from the pod's environment.

    ``supervise`` needs these to reproduce the work-volume layout when it mounts the PVC into a
    sandbox -- it has no mount of its own to infer them from.
    """
    workspace = os.environ.get(NEMO_JOB_WORKSPACE_ENVVAR)
    job_id = os.environ.get(NEMO_JOB_ID_ENVVAR)
    if not workspace or not job_id:
        raise RuntimeError(f"{NEMO_JOB_WORKSPACE_ENVVAR} and {NEMO_JOB_ID_ENVVAR} must both be set")
    return workspace, job_id


def work_mount() -> Path:
    """Where the work volume is mounted in this pod.

    Set by the compiler as the *value* of the storage env var, which is also what asks the
    Kubernetes backend for the mount in the first place. Absent in the ``build`` step, on purpose.
    """
    mount = os.environ.get(PERSISTENT_JOB_STORAGE_PATH_ENVVAR)
    if not mount:
        raise RuntimeError(
            f"{PERSISTENT_JOB_STORAGE_PATH_ENVVAR} is not set; this step was compiled without a work volume"
        )
    return Path(mount)


def context_hash(root: Path) -> str:
    """A stable hash of a fetched context tree.

    Sorted relative paths plus each file's content digest, so it does not depend on walk order,
    inode order, or mtimes. Symlinks are hashed as their *target string* rather than followed --
    following them here would make the hash depend on something outside the tree.

    **Advisory, never identity.** This describes the input a build was given; what names the
    output is the digest the registry serves. See the note in ``fetch.py`` about why it does not
    currently reach the image.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"L")
            digest.update(os.readlink(path).encode())
        else:
            digest.update(b"F")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def split_fileset_ref(ref: str, default_workspace: str) -> tuple[str, str]:
    """``workspace/name`` or a bare ``name`` scoped to the job's workspace.

    A bare name resolves in the *job's* workspace rather than anywhere broader, so the shorthand
    cannot quietly widen what a request reaches.
    """
    if "/" in ref:
        workspace, _, name = ref.partition("/")
        return workspace, name
    return default_workspace, ref
