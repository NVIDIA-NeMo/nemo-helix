# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path helpers for customization job storage mounts."""

import os
from pathlib import Path

from nmp.common.jobs.constants import (
    DEFAULT_JOB_STORAGE_PATH,
    DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)

CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR = "NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH"
LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS = ("NMP_JOB_PERSISTENT_JOB_STORAGE_PATH",)

CURRENT_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR = "NEMO_JOB_STEP_CONFIG_FILE_PATH"
LEGACY_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVARS = ("NMP_JOB_STEP_CONFIG_FILE_PATH",)

CURRENT_JOB_STORAGE_PATH = Path("/var/run/scratch/job")
LEGACY_JOB_STORAGE_PATHS = (Path("/run/scratch/job"),)


def _first_env_path(names: tuple[str, ...], default: str) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(default)


def get_job_storage_path_from_env() -> Path:
    """Return the mounted persistent job storage path.

    The explicit current env name is checked before the imported constant so a
    task image still works if another installed package has an older constant.
    """

    return _first_env_path(
        (
            CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            *LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS,
        ),
        DEFAULT_JOB_STORAGE_PATH,
    )


def get_job_step_config_path_from_env() -> Path:
    """Return the mounted step config file path."""

    return _first_env_path(
        (
            CURRENT_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
            NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
            *LEGACY_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVARS,
        ),
        DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
    )


def _known_job_storage_roots(storage_path: Path) -> tuple[Path, ...]:
    roots = (
        storage_path,
        Path(DEFAULT_JOB_STORAGE_PATH),
        CURRENT_JOB_STORAGE_PATH,
        *LEGACY_JOB_STORAGE_PATHS,
    )
    return tuple(dict.fromkeys(roots))


def remap_job_storage_path(storage_path: Path, user_path: str | Path) -> Path:
    """Remap known absolute job-storage roots to the mounted storage path.

    Older customizer task configs used ``/run/scratch/job`` while the current
    Jobs runner mounts persistent storage at ``/var/run/scratch/job``. This
    keeps absolute paths produced by either side usable when API and task images
    are briefly out of sync. Unknown absolute paths are returned unchanged so
    callers can reject them with their normal traversal checks.
    """

    raw_path = Path(user_path)
    if not raw_path.is_absolute():
        return raw_path

    for root in _known_job_storage_roots(storage_path):
        try:
            return storage_path / raw_path.relative_to(root)
        except ValueError:
            continue
    return raw_path
