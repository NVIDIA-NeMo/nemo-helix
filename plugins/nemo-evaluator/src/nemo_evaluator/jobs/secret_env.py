# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared task-environment construction for evaluator plugin job compilers.

Both ``compile_evaluate_job`` and ``compile_agent_eval_job`` turn a spec's metric/endpoint secret
references into ``from_secret`` task environment variables. The per-job code differs only in *how*
it walks its own schema to collect ``(env_name, secret_name)`` pairs; the assembly — the
persistent-storage path, the reserved-name guard, and conflict detection — is identical and lives
here.
"""

from __future__ import annotations

from collections.abc import Iterable

from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable, EnvironmentVariableFromSecret
from nemo_helix_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR

#: Env names a job sets itself, so they cannot be sourced from a secret ref.
RESERVED_SECRET_ENV_NAMES = frozenset({PERSISTENT_JOB_STORAGE_PATH_ENVVAR})


def build_task_environment(secret_refs: Iterable[tuple[str, str]]) -> list[EnvironmentVariable]:
    """Build a task's environment: the persistent-storage path plus ``from_secret`` variables.

    ``secret_refs`` yields ``(env_name, secret_name)`` pairs. Raises if a pair targets a reserved
    env name, or if two pairs map the same env name to different secrets.
    """
    environment = [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]
    resolved: dict[str, str] = {}
    for env_name, secret_name in secret_refs:
        if env_name in RESERVED_SECRET_ENV_NAMES:
            raise ValueError(f"{env_name!r} is reserved and cannot be sourced from secret refs")
        existing = resolved.get(env_name)
        if existing is not None and existing != secret_name:
            raise ValueError(f"conflicting secret references for environment variable {env_name!r}")
        resolved[env_name] = secret_name

    environment.extend(
        EnvironmentVariable(name=env_name, from_secret=EnvironmentVariableFromSecret(name=secret_name))
        for env_name, secret_name in sorted(resolved.items())
    )
    return environment
