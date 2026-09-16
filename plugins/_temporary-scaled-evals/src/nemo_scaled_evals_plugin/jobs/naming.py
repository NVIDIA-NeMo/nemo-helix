# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic names for scaled-evals Platform Jobs."""

from __future__ import annotations

import hashlib
import re

from nemo_platform_plugin.entity_naming import NAME_MAX_LENGTH

_JOB_NAME_MAX_LENGTH = NAME_MAX_LENGTH - len("job-fileset-")


def task_image_build_job_name(task_id: str, revision: int, build_attempt: int) -> str:
    """Return the stable Platform Job name for a task build attempt."""
    return _bounded_name(f"scaled-evals-build-{task_id}-r{revision}-a{build_attempt}")


def evaluation_execution_job_name(evaluation_id: str, execution_number: int) -> str:
    """Return the stable Platform Job name for an evaluation execution."""
    return _bounded_name(f"scaled-evals-evaluation-{evaluation_id}-e{execution_number}")


def _bounded_name(value: str) -> str:
    normalized = re.sub(r"-+", "-", re.sub(r"[^a-z0-9@.+_-]+", "-", value.lower())).strip("-")
    if len(normalized) <= _JOB_NAME_MAX_LENGTH:
        return normalized
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{normalized[: _JOB_NAME_MAX_LENGTH - len(digest) - 1].rstrip('-')}-{digest}"
