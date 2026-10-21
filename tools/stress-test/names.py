# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-name rules for a stress run.

A job without ``output_location`` owns ``job-fileset-<name>``, and that
fileset name must fit in 63 characters. The job name itself is therefore
limited to 51.
"""

from __future__ import annotations

import re

JOB_NAME_MAX_LENGTH = 51
_TOKEN = re.compile(r"^[a-z](?:[a-z0-9]|-(?=[a-z0-9]))*$")


class NameError(ValueError):
    """A run key, step id, or job name violates the stress naming rules."""


def check_token(value: str, label: str) -> None:
    """Reject a token that is not a single-hyphen lowercase name."""
    if not _TOKEN.fullmatch(value):
        raise NameError(f"{label} {value!r} must be a lowercase name with single hyphens")


def job_name(run_key: str, step_id: str, index: int, count: int) -> str:
    """Return ``{run_key}-{step_id}-{index}`` padded so names sort in submit order."""
    width = max(len(str(count)), 1)
    name = f"{run_key}-{step_id}-{index:0{width}d}"
    check_token(name, "job name")
    if len(name) > JOB_NAME_MAX_LENGTH:
        raise NameError(f"job name {name!r} is {len(name)} characters; the maximum is {JOB_NAME_MAX_LENGTH}")
    return name


def step_prefix(run_key: str, step_id: str) -> str:
    """Prefix that matches only this step's job names."""
    return f"{run_key}-{step_id}-"


def run_prefix(run_key: str) -> str:
    """Prefix that matches every job in the run."""
    return f"{run_key}-"


def resource_name(run_key: str) -> str:
    """Workspace and fileset name for a run key. Both are kept across reruns."""
    name = f"stress-{run_key}"
    check_token(name, "workspace")
    if len(name) > 63:
        raise NameError(f"workspace {name!r} exceeds 63 characters")
    return name
